"""Fixed two-update generated engineering demonstration; no finance evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from fijepa.config import LossConfig, ModelConfig
from fijepa.losses import compute_fijepa_loss
from fijepa.memory import PrototypeMemory
from fijepa.model import FIJEPA


def tensor_identity(mapping):
    digest = hashlib.sha256()
    for name, tensor in sorted(mapping.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(94502)
    cfg = ModelConfig(backbone="mlp", d_model=8, latent_dim=8, num_layers=1,
                      dropout=0.0, predictor_hidden=8, stochastic=False,
                      stage_names=["h1"], memory_size=3, memory_topk=2,
                      memory_write_threshold=0.0)
    loss_cfg = LossConfig(multi_horizon_weights=[1.0])
    model = FIJEPA(input_dim=2, config=cfg)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    inputs = {"context": torch.linspace(-1.0, 1.0, 24).reshape(2, 6, 2),
              "future": torch.linspace(-0.5, 0.5, 8).reshape(2, 1, 2, 2)}
    torch.save(inputs, args.output / "generated_inputs.pt")
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch.save(initial, args.output / "initial_model.pt")
    steps = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs["context"], future=inputs["future"])
        loss = compute_fijepa_loss(output, inputs["context"], inputs["future"], loss_cfg).total
        if not torch.isfinite(loss):
            raise ValueError("Generated loss is nonfinite")
        loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                     if parameter.grad is not None}
        if not gradients or not all(torch.isfinite(value).all() for value in gradients.values()):
            raise ValueError("Generated optimizer gradients are missing/nonfinite")
        torch.save(gradients, args.output / f"gradients_step_{step}.pt")
        optimizer.step()
        model.update_target()
        model.update_memory(output["z_context"], output["z_targets"][:, -1])
        steps.append({"step": step, "loss": loss.item(),
                      "gradient_tensor_sha256": tensor_identity(gradients),
                      "model_tensor_sha256": tensor_identity(model.state_dict()),
                      "filled": model.memory.filled.item(),
                      "memory_tensor_sha256": tensor_identity(model.memory.state_dict())})
    checkpoint = {"model_config": asdict(cfg), "loss_config": asdict(loss_cfg),
                  "model": {name: value.detach().clone() for name, value in model.state_dict().items()},
                  "scope": "Generated engineering fixture; not a study or exact training resume"}
    torch.save(checkpoint, args.output / "final_checkpoint.pt")
    # Decode the actual saved checkpoint, including all existing registered bank buffers.
    restored = FIJEPA(input_dim=2, config=ModelConfig(**checkpoint["model_config"]))
    restored.load_state_dict(torch.load(args.output / "final_checkpoint.pt", weights_only=True)["model"])
    model.eval()
    restored.eval()
    with torch.no_grad():
        actual = model(inputs["context"])["z_context"]
        replay = restored(inputs["context"])["z_context"]
    if not torch.equal(actual, replay):
        raise ValueError("Saved model/bank did not exactly reproduce its generated forward")
    # Retain the original active-slot falsifier as concrete repaired behavior.
    bank = PrototypeMemory(2, 1, memory_size=3, decay=0.5, merge_threshold=0.0)
    bank.update(torch.tensor([-1., 0.]), torch.tensor([2.]), torch.tensor(1.))
    bank.update(torch.tensor([1., 0.]), torch.tensor([4.]), torch.tensor(1.))
    if bank.filled.item() != 2 or not torch.equal(bank.values[:2], torch.tensor([[2.], [4.]])):
        raise ValueError("Active-slot discriminator failed")
    torch.save(bank.state_dict(), args.output / "active_slot_witness.pt")
    artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(args.output.iterdir()) if path.is_file()}
    report = {"schema": "fijepa.generated_active_write_demo.v1",
              "scope": "Engineering execution only; v1 remains CLOSED NEGATIVE/BOUNDARY",
              "torch": str(torch.__version__), "seed": 94502, "threads": torch.get_num_threads(),
              "examples": 2, "context_length": 6, "latent_dim": 8, "optimizer_updates": 2,
              "initial_model_tensor_sha256": tensor_identity(initial),
              "input_tensor_sha256": tensor_identity(inputs), "model_config": asdict(cfg),
              "steps": steps, "checkpoint_replay_exact": True,
              "active_slot_witness": {"filled": bank.filled.item(), "values": bank.values.tolist()},
              "artifact_sha256": artifacts}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
