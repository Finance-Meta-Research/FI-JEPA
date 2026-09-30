"""Saved-evidence reader regressions; NumPy only, no training or checkpoint access."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.check_paper_evidence_gate import load_artifact, validate_artifact

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "experiments/paper_results.json"
BASELINE_STAT = "full_latent_minus_raw_context_ridge_mse"


class PaperEvidenceReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_bytes = ARTIFACT.read_bytes()
        cls.source = json.loads(cls.original_bytes)

    def setUp(self):
        self.data = copy.deepcopy(self.source)

    def assert_rejected(self, data):
        with self.assertRaisesRegex(SystemExit, "PAPER_EVIDENCE_GATE_FAIL"):
            validate_artifact(data)

    def test_retained_artifact_passes_unchanged(self):
        self.assertEqual(load_artifact(ARTIFACT), self.source)
        self.assertEqual(ARTIFACT.read_bytes(), self.original_bytes)

    def test_required_metrics_are_numeric_finite_and_in_range(self):
        for value in (True, False, None, "NaN", "0.25", [], {}, float("nan"), float("inf"), -1.0):
            with self.subTest(value=repr(value)):
                data = copy.deepcopy(self.source)
                data["runs"][0]["metrics"]["probe_regression_mse"] = value
                self.assert_rejected(data)
        for field in self.data["runs"][0]["metrics"]:
            with self.subTest(missing=field):
                data = copy.deepcopy(self.source)
                del data["runs"][0]["metrics"][field]
                self.assert_rejected(data)
        self.data["runs"][0]["metrics"]["probe_directional_accuracy"] = 1.1
        self.assert_rejected(self.data)

    def test_matrix_and_frozen_protocol_cannot_drift(self):
        mutations = {
            "protocol": lambda d: d.update(protocol_id="OTHER"),
            "status": lambda d: d.update(status="UNEXECUTED"),
            "different seeds": lambda d: d.update(seed_list=[7, 17, 28]),
            "duplicate seed": lambda d: d.update(seed_list=[7, 17, 17]),
            "boolean seed": lambda d: d.update(seed_list=[True, 17, 27]),
            "non-scalar seed": lambda d: d.update(seed_list=[{}, 17, 27]),
            "budget": lambda d: d.update(epochs=1),
            "float budget": lambda d: d.update(epochs=15.0),
            "variants": lambda d: d["required_variants"].pop(),
            "baseline declaration": lambda d: d.update(downstream_baseline="other"),
            "missing row": lambda d: d["runs"].pop(),
            "duplicate row": lambda d: d["runs"].append(copy.deepcopy(d["runs"][0])),
            "extra condition": lambda d: d["runs"][0].update(variant="historical"),
            "baseline false": lambda d: d["runs"][-1].update(is_downstream_baseline=False),
            "baseline string": lambda d: d["runs"][-1].update(is_downstream_baseline="true"),
            "model baseline": lambda d: d["runs"][0].update(is_downstream_baseline=True),
            "missing boundary": lambda d: d.pop("claim_boundary"),
            "altered boundary": lambda d: d.update(claim_boundary=["Evidence of trading profitability"]),
            "extra boundary claim": lambda d: d["claim_boundary"].append("SOTA forecasting"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                data = copy.deepcopy(self.source)
                mutate(data)
                self.assert_rejected(data)
        for minimum in (1, 2, 4):
            with self.subTest(minimum=minimum), self.assertRaises(SystemExit):
                validate_artifact(self.source, min_seeds=minimum)

    def test_raw_rows_and_signatures_are_checked(self):
        mutations = {
            "missing raw rows": lambda d: d.pop("raw_runs"),
            "duplicate raw cell": lambda d: d["raw_runs"].append(copy.deepcopy(d["raw_runs"][0])),
            "wrong raw MSE": lambda d: d["raw_runs"][0]["probe_regression"].update(mse=123.0),
            "wrong raw baseline": lambda d: d["raw_runs"][0]["baseline_regression"].update(mse=123.0),
            "wrong raw validation": lambda d: d["raw_runs"][0].update(best_val_total=123.0),
            "missing signature": lambda d: d["runs"][0].pop("ablation_signature"),
            "empty signature": lambda d: d["runs"][0].update(ablation_signature={}),
            "numeric boolean signature": lambda d: d["runs"][0]["ablation_signature"].update(stochastic=1),
            "identical ablation signature": lambda d: d["runs"][3].update(ablation_signature=d["runs"][0]["ablation_signature"]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                data = copy.deepcopy(self.source)
                mutate(data)
                self.assert_rejected(data)

    def test_every_paired_summary_is_recomputed(self):
        for name in self.source["paired_seed_level_statistics"]:
            with self.subTest(missing=name):
                data = copy.deepcopy(self.source)
                del data["paired_seed_level_statistics"][name]
                self.assert_rejected(data)
        changes = {"mean_delta": -99.0, "std_delta": 42.0, "bootstrap_95ci": [0.0, 0.0],
                   "n_seed_pairs": True, "bootstrap_seed": 42, "bootstrap_reps": 10}
        for field, value in changes.items():
            with self.subTest(field=field):
                data = copy.deepcopy(self.source)
                data["paired_seed_level_statistics"][BASELINE_STAT][field] = value
                self.assert_rejected(data)
        for value in (None, {}, [], "NaN"):
            with self.subTest(summary=value):
                data = copy.deepcopy(self.source)
                data["paired_seed_level_statistics"][BASELINE_STAT] = value
                self.assert_rejected(data)
        self.data["paired_seed_level_statistics"]["post_hoc"] = {}
        self.assert_rejected(self.data)

    def test_loader_rejects_malformed_and_ambiguous_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            for content in ('{', '[]', '{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaisesRegex(SystemExit, "PAPER_EVIDENCE_GATE_FAIL"):
                        load_artifact(path)
            with self.assertRaisesRegex(SystemExit, "PAPER_EVIDENCE_GATE_FAIL"):
                load_artifact(Path(tmp) / "absent.json")

    def generate(self, artifact, tex, md, module=False):
        entry = ["-m", "scripts.make_canonical_paper_assets"] if module else ["scripts/make_canonical_paper_assets.py"]
        return subprocess.run([sys.executable, *entry, "--artifact", str(artifact),
                               "--tex", str(tex), "--md", str(md)], cwd=ROOT,
                              capture_output=True, text=True, timeout=20)

    def test_generator_validates_before_writing_either_output(self):
        self.data["paired_seed_level_statistics"][BASELINE_STAT]["mean_delta"] = -99.0
        with tempfile.TemporaryDirectory() as tmp:
            artifact, tex, md = [Path(tmp) / name for name in ("input.json", "table.tex", "summary.md")]
            artifact.write_text(json.dumps(self.data))
            for output in (tex, md):
                output.write_text("KEEP EXISTING OUTPUT")
            result = self.generate(artifact, tex, md)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("PAPER_EVIDENCE_GATE_FAIL", result.stderr)
            for output in (tex, md):
                self.assertEqual(output.read_text(), "KEEP EXISTING OUTPUT")

    def test_generator_is_seed_order_independent_and_reproduces_retained_assets(self):
        # Reverse full rows but leave monolithic rows in their original order.
        # Numerical identity must be checked by seed, not by incidental row order.
        self.data["runs"][:3] = reversed(self.data["runs"][:3])
        with tempfile.TemporaryDirectory() as tmp:
            artifact, tex, md = [Path(tmp) / name for name in ("input.json", "table.tex", "summary.md")]
            artifact.write_text(json.dumps(self.data))
            result = self.generate(artifact, tex, md, module=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(tex.read_bytes(), (ROOT / "paper/generated/canonical_results_table.tex").read_bytes())
            self.assertEqual(md.read_bytes(), (ROOT / "paper/generated/canonical_results_summary.md").read_bytes())
            self.assertIn("exactly identical to full across all three seeds", md.read_text())

    def test_generator_cannot_overwrite_its_input_or_alias_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, tex, md = [Path(tmp) / name for name in ("input.json", "table.tex", "summary.md")]
            artifact.write_bytes(self.original_bytes)
            for first, second in ((artifact, md), (tex, artifact), (tex, tex)):
                result = self.generate(artifact, first, second)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be distinct", result.stderr)
                self.assertEqual(artifact.read_bytes(), self.original_bytes)


    def test_generator_rejects_symlink_and_hardlink_to_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "input.json"
            artifact.write_bytes(self.original_bytes)
            for link_kind in ("symlink", "hardlink"):
                alias = Path(tmp) / link_kind
                if link_kind == "symlink":
                    alias.symlink_to(artifact)
                else:
                    os.link(artifact, alias)
                result = self.generate(artifact, alias, Path(tmp) / "summary.md")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be distinct", result.stderr)
                self.assertEqual(artifact.read_bytes(), self.original_bytes)

    def test_cli_pass_does_not_claim_six_identifying_ablations(self):
        result = subprocess.run([sys.executable, "scripts/check_paper_evidence_gate.py"],
                                cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no_operator_split remains non-identifying", result.stdout)
        self.assertIn("not a publication-readiness gate", result.stdout)
        self.assertNotIn("real mechanism variants", result.stdout)


if __name__ == "__main__":
    unittest.main()
