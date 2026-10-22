"""A finite loss must not permit a non-finite optimizer update."""

import pytest
import torch

from fijepa.config import FIJEPAConfig
from fijepa.trainer import Trainer


class SingularTrainer(Trainer):
    def _step(self, batch):
        # sqrt(0) is finite, but its derivative is infinite.
        loss = self.model.weight.sqrt().sum()
        return loss, {"total": loss.detach()}, {}, None, None


@pytest.mark.parametrize("clip", [None, 1.0])
def test_nonfinite_gradient_never_changes_parameters_or_target(tmp_path, clip):
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
    updates = []
    model.update_target = lambda: updates.append("target")
    config = FIJEPAConfig()
    config.device = "cpu"
    config.train.grad_clip_norm = clip
    trainer = SingularTrainer(model, [{"context": torch.zeros(1, 1)}], [], config, 1, str(tmp_path))
    before = model.weight.detach().clone()
    with pytest.raises(FloatingPointError, match="gradient"):
        trainer.train_epoch(1)
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)
    assert not updates
    assert not trainer.optim.state
