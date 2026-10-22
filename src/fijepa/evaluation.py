"""Prospective sample-mean latent evaluation; historical evaluation is unchanged."""

from __future__ import annotations

import math

import torch

METRIC_SCHEMA = "fijepa_latent_sample_mean_v2"


@torch.no_grad()
def evaluate_latents(model, loader, device="cpu"):
    """Evaluate the final predictor mean against the final target horizon.

    MSE averages all observed latent coordinates; mean norm averages context
    norms over observations. Short final batches get their actual sample weight.
    Reductions use float64, independently of the model's inference precision.

    This does not select a checkpoint, update memory, or change the stochastic
    predictor policy. With stochastic intermediate stages, draws can still
    depend on batching; retain the seed and batch size in the run record.
    """
    device = torch.device(device)
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    squared_error_sum = norm_sum = 0.0
    sample_count = batch_count = 0
    latent_dim = horizon_count = None
    try:
        for batch in loader:
            context = batch["context"].to(device)
            future = batch["futures"].to(device)
            asset_id = batch["asset_id"].to(device)
            if context.ndim != 3 or 0 in context.shape:
                raise ValueError("context must have nonempty shape [B, T, D]")
            if (
                future.ndim != 4
                or 0 in future.shape
                or future.shape[0] != context.shape[0]
                or future.shape[-1] != context.shape[-1]
            ):
                raise ValueError(
                    "futures must have matching nonempty shape [B, H, T, D]"
                )
            if asset_id.ndim == 1:
                asset_id = asset_id.unsqueeze(-1)
            if asset_id.shape != (context.shape[0], 1):
                raise ValueError("asset_id must have shape [B] or [B, 1]")
            if not torch.isfinite(context).all() or not torch.isfinite(future).all():
                raise ValueError("evaluation inputs must be finite")

            out = model(context=context, future=future, asset_id=asset_id)
            z_context = out["z_context"]
            z_targets = out["z_targets"]
            mu = out["stages"][-1]["mu"]
            batch_size = context.shape[0]
            if (
                z_context.ndim != 2
                or z_context.shape[0] != batch_size
                or z_context.shape[1] == 0
                or z_targets.ndim != 3
                or z_targets.shape != (batch_size, future.shape[1], z_context.shape[1])
                or mu.shape != z_context.shape
            ):
                raise ValueError(
                    "model latent outputs must have matching [B, L] / [B, H, L] shapes"
                )
            if any(
                not torch.isfinite(value).all() for value in (z_context, z_targets, mu)
            ):
                raise FloatingPointError(
                    "non-finite model output during latent evaluation"
                )
            if latent_dim is None:
                latent_dim, horizon_count = z_context.shape[1], z_targets.shape[1]
            elif (latent_dim, horizon_count) != (
                z_context.shape[1],
                z_targets.shape[1],
            ):
                raise ValueError(
                    "latent dimension and horizon count must remain constant"
                )

            # Move only the small latent outputs to CPU. Cast before subtraction,
            # squaring and norms so low-precision reductions cannot overflow early.
            prediction = mu.detach().to(device="cpu", dtype=torch.float64)
            target = z_targets[:, -1, :].detach().to(device="cpu", dtype=torch.float64)
            context_latents = z_context.detach().to(device="cpu", dtype=torch.float64)
            squared_error_sum += float((prediction - target).square().sum())
            norm_sum += float(torch.linalg.vector_norm(context_latents, dim=-1).sum())
            sample_count += batch_size
            batch_count += 1
        if sample_count == 0:
            raise RuntimeError("evaluation loader produced zero examples")
        if not math.isfinite(squared_error_sum) or not math.isfinite(norm_sum):
            raise FloatingPointError("non-finite aggregate during latent evaluation")
        return {
            "metric_schema": METRIC_SCHEMA,
            "latent_mse": squared_error_sum / (sample_count * latent_dim),
            "latent_mean_norm": norm_sum / sample_count,
            "sample_count": sample_count,
            "batch_count": batch_count,
            "latent_dim": latent_dim,
            "target_horizon_index": -1,
        }
    finally:
        # Preserve heterogeneous module modes (e.g. an EMA encoder held in eval).
        for module, training in modes:
            module.training = training
