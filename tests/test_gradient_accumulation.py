"""Generated-data optimizer tests; these do not execute a research benchmark."""

import pytest
import torch

from fijepa.config import FIJEPAConfig
from fijepa.trainer import Trainer, _accumulation_groups


class LinearModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.weight.fill_(0.5)
        self.target_updates = 0

    def update_target(self):
        self.target_updates += 1


class RegressionTrainer(Trainer):
    def _step(self, batch):
        x, y = batch["context"], batch["target"]
        loss = torch.mean((self.model(x) - y) ** 2)
        return loss, {"total": loss.detach()}, {}, x, y


def _batches(sizes):
    x = torch.linspace(-1.0, 2.0, sum(sizes), dtype=torch.float64).reshape(-1, 1)
    y = 1.7 * x + 0.2
    out = []
    start = 0
    for size in sizes:
        out.append({"context": x[start : start + size], "target": y[start : start + size]})
        start += size
    return out


@pytest.mark.parametrize("sizes,accumulate", [([2, 2, 2, 2, 1], 3), ([3, 1], 4), ([1, 4, 2], 3), ([2, 2, 1], 1)])
def test_accumulated_update_equals_actual_concatenated_group(tmp_path, sizes, accumulate):
    cfg = FIJEPAConfig()
    cfg.device = "cpu"
    cfg.train.accumulate_grad_batches = accumulate
    cfg.train.grad_clip_norm = None
    batches = _batches(sizes)
    trained = LinearModel()
    reference = LinearModel()
    trainer = RegressionTrainer(trained, batches, [], cfg, 1, str(tmp_path))
    trainer.optim = torch.optim.SGD(trained.parameters(), lr=0.03)
    reference_optim = torch.optim.SGD(reference.parameters(), lr=0.03)
    for start in range(0, len(batches), accumulate):
        group = batches[start : start + accumulate]
        x = torch.cat([batch["context"] for batch in group])
        y = torch.cat([batch["target"] for batch in group])
        reference_optim.zero_grad()
        torch.mean((reference(x) - y) ** 2).backward()
        reference_optim.step()
    metrics = trainer.train_epoch(1)
    torch.testing.assert_close(trained.weight, reference.weight, rtol=1e-12, atol=1e-12)
    assert trained.target_updates == (len(batches) + accumulate - 1) // accumulate
    assert trainer.global_step == len(batches)
    assert torch.isfinite(torch.tensor(metrics["total"]))


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2", None])
def test_invalid_accumulation_count_is_not_silently_coerced(value):
    with pytest.raises(ValueError, match="positive integer"):
        list(_accumulation_groups(_batches([1]), value))


def test_empty_microbatch_is_rejected_before_an_optimizer_step():
    with pytest.raises(ValueError, match="empty microbatch"):
        list(_accumulation_groups(_batches([2, 0]), 2))


def test_group_weights_include_smaller_last_batch():
    groups = list(_accumulation_groups(_batches([3, 1, 2]), 2))
    assert [weight for _, weight in groups[0]] == [0.75, 0.25]
    assert [weight for _, weight in groups[1]] == [1.0]


def test_group_reader_is_bounded_and_does_not_eagerly_consume_loader():
    seen = []

    def loader():
        for i in range(7):
            seen.append(i)
            yield {"context": [i]}

    groups = _accumulation_groups(loader(), 2)
    next(groups)
    assert seen == [0, 1]
    next(groups)
    assert seen == [0, 1, 2, 3]
