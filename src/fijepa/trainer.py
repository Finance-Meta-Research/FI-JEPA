from __future__ import annotations

import json
import os
from dataclasses import asdict
from itertools import islice
from operator import index
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from .losses import compute_fijepa_loss


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_rank0():
    return int(os.environ.get("RANK", "0")) == 0


def _accumulation_groups(loader, accumulation_steps):
    """Yield bounded groups with their actual sample count, including the tail.

    Each microbatch loss is a mean. Its gradient therefore has weight
    ``microbatch_size / group_sample_count``. Batch-dependent regularizers still
    operate on their original microbatch; accumulation does not change that
    scientific choice or pretend to reproduce a larger covariance batch.
    """
    if isinstance(accumulation_steps, bool):
        raise ValueError("accumulate_grad_batches must be a positive integer")
    try:
        accumulation_steps = index(accumulation_steps)
    except TypeError as exc:
        raise ValueError("accumulate_grad_batches must be a positive integer") from exc
    if accumulation_steps < 1:
        raise ValueError("accumulate_grad_batches must be a positive integer")
    batches = iter(loader)
    while group := list(islice(batches, accumulation_steps)):
        sizes = [len(batch["context"]) for batch in group]
        if any(size < 1 for size in sizes):
            raise ValueError("gradient accumulation received an empty microbatch")
        total = sum(sizes)
        yield [(batch, size / total) for batch, size in zip(group, sizes)]


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, feature_dim: int, run_dir: str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.feature_dim = feature_dim
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.optim = AdamW(self.model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
        self.scaler = GradScaler(enabled=config.train.amp and self.device.type == "cuda")
        self.best = float("inf")
        self.global_step = 0
        if is_rank0():
            with open(self.run_dir / "config.json", "w") as f:
                json.dump(asdict(config), f, indent=2, default=str)

    def _resolve_device(self, device: str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _step(self, batch):
        context = batch["context"].to(self.device)
        future = batch["futures"].to(self.device)
        asset_id = batch["asset_id"].to(self.device)
        if asset_id.ndim == 1:
            asset_id = asset_id.unsqueeze(-1)
        with autocast(enabled=self.scaler.is_enabled()):
            out = self.model(context=context, future=future, asset_id=asset_id)
            breakdown = compute_fijepa_loss(
                out,
                context,
                future,
                self.config.loss,
                use_financial_regularizers=self.config.loss.use_financial_regularizers,
            )
            loss = breakdown.total
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training/evaluation loss at global_step={self.global_step}")
        return loss, breakdown.components, out, context, future

    def train_epoch(self, epoch: int):
        self.model.train()
        groups = _accumulation_groups(self.train_loader, self.config.train.accumulate_grad_batches)
        pbar = tqdm(total=len(self.train_loader), desc=f"train {epoch}", leave=False, disable=not is_rank0())
        avg = {}
        self.optim.zero_grad(set_to_none=True)
        step = 0
        try:
            for group in groups:
                for batch, loss_weight in group:
                    loss, comps, out, _context, _future = self._step(batch)
                    self.scaler.scale(loss * loss_weight).backward()
                    for k, v in comps.items():
                        if torch.is_tensor(v) and not torch.isfinite(v):
                            raise FloatingPointError(
                                f"non-finite component {k} at epoch={epoch} step={step} global_step={self.global_step}"
                            )
                        avg[k] = avg.get(k, 0.0) + float(v.item())
                    if step % self.config.train.log_every == 0 and is_rank0():
                        pbar.set_postfix({k: f"{float(v.item()):.4f}" for k, v in comps.items() if torch.is_tensor(v)})
                    self.global_step += 1
                    step += 1
                    pbar.update(1)
                # A finite loss can have an infinite derivative (for example
                # sqrt at zero). Validate before *any* optimizer/EMA mutation,
                # including when optional clipping is disabled.
                self.scaler.unscale_(self.optim)
                for name, parameter in self.model.named_parameters():
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    values = gradient.coalesce().values() if gradient.is_sparse else gradient
                    if not torch.isfinite(values).all():
                        raise FloatingPointError(
                            f"non-finite gradient for {name} at epoch={epoch} "
                            f"step={step} global_step={self.global_step}"
                        )
                if self.config.train.grad_clip_norm is not None:
                    grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(
                            f"non-finite gradient norm at epoch={epoch} step={step} global_step={self.global_step}"
                        )
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
                self.model.update_target()
                if hasattr(self.model, "update_memory"):
                    z_ctx = out["z_context"].detach()
                    z_tgt = out["z_targets"][:, -1, :].detach()
                    self.model.update_memory(z_ctx, z_tgt)
        finally:
            pbar.close()
        if not avg:
            raise RuntimeError("training loader produced zero batches")
        for k in avg:
            avg[k] /= len(self.train_loader)
        return avg

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        avg = {}
        for batch in tqdm(self.val_loader, desc="val", leave=False, disable=not is_rank0()):
            _, comps, _, _, _ = self._step(batch)
            for k, v in comps.items():
                if torch.is_tensor(v) and not torch.isfinite(v):
                    raise FloatingPointError(f"non-finite validation component {k}")
                avg[k] = avg.get(k, 0.0) + float(v.item())
        if not avg:
            raise RuntimeError("validation loader produced zero batches")
        for k in avg:
            avg[k] /= len(self.val_loader)
        return avg

    def fit(self, max_epochs: Optional[int] = None):
        max_epochs = max_epochs or self.config.train.max_epochs
        history = []
        for epoch in range(1, max_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate()
            record = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(record)
            metric = val_metrics.get("total", val_metrics.get("pred", float("inf")))
            if metric < self.best:
                self.best = metric
                if is_rank0():
                    torch.save(
                        {"model": self.model.state_dict(), "config": asdict(self.config), "epoch": epoch, "metric": metric},
                        self.run_dir / "best.pt",
                    )
            if is_rank0():
                with open(self.run_dir / "history.jsonl", "a") as f:
                    f.write(json.dumps(record) + "\n")
        if is_rank0():
            torch.save(
                {"model": self.model.state_dict(), "config": asdict(self.config), "epoch": max_epochs},
                self.run_dir / "last.pt",
            )
        return history
