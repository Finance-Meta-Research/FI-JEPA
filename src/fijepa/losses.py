from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


def gaussian_nll(target, mu, logvar):
    var = torch.exp(logvar).clamp_min(1e-6)
    return 0.5 * ((target - mu) ** 2 / var + logvar + torch.log(torch.tensor(2.0 * torch.pi, device=target.device))).mean()


def variance_covariance_penalty(z, gamma: float = 1.0, eps: float = 1e-4):
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    variance_term = torch.relu(gamma - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / max(1, z.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    covariance_term = (off_diag**2).mean()
    return variance_term + covariance_term


def multi_horizon_prediction_loss(stage_outs, z_targets, horizon_weights):
    losses = []
    n = min(len(stage_outs), z_targets.shape[1], len(horizon_weights))
    for i in range(n):
        mu = stage_outs[i]["mu"]
        tgt = z_targets[:, i, :]
        losses.append(horizon_weights[i] * F.mse_loss(mu, tgt))
    if not losses:
        zero = z_targets.sum() * 0.0
        return zero, []
    return sum(losses), losses


def no_arbitrage_penalty(return_head, z_target, cost=0.0):
    return torch.mean(torch.relu(torch.abs(return_head) - cost))


def volatility_persistence_penalty(stage_outs):
    vols = []
    for stage in stage_outs:
        if stage["logvar"] is not None:
            vols.append(torch.exp(0.5 * stage["logvar"]).mean(dim=-1))
    if len(vols) < 2:
        return stage_outs[0]["mu"].sum() * 0.0
    penalty = 0.0
    for i in range(1, len(vols)):
        penalty = penalty + F.mse_loss(vols[i], vols[i - 1].detach())
    return penalty / (len(vols) - 1)


def liquidity_impact_penalty(liq_head, context, futures):
    if futures.ndim != 4:
        raise ValueError(f"Expected futures with shape [B, H, T, D], got {tuple(futures.shape)}")
    delta = futures[:, :, -1, :] - context[:, None, -1, :]
    proxy = delta.abs().mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)
    return F.mse_loss(liq_head, proxy)


def macro_smoothness_penalty(z_target):
    if z_target.shape[0] < 2:
        return z_target.sum() * 0.0
    return F.mse_loss(z_target[1:], z_target[:-1])


def gate_sparsity_penalty(stage_outs, memory_gate=None):
    gates = [stage["gate"].abs().mean() for stage in stage_outs if stage.get("gate") is not None]
    if gates:
        gate_pen = sum(gates) / len(gates)
    else:
        gate_pen = stage_outs[0]["mu"].sum() * 0.0
    if memory_gate is not None:
        gate_pen = gate_pen + memory_gate.abs().mean()
    return gate_pen


def reconstruction_penalty(recon, future):
    if future.ndim != 4:
        raise ValueError(f"Expected future with shape [B, H, T, D], got {tuple(future.shape)}")
    target = future[:, -1, -1, :]
    return F.mse_loss(recon, target)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    components: Dict[str, torch.Tensor]


def compute_fijepa_loss(model_out, context, future, loss_cfg, use_financial_regularizers: bool = True):
    stage_outs = model_out["stages"]
    z_targets = model_out["z_targets"].detach()

    pred_loss, _ = multi_horizon_prediction_loss(stage_outs, z_targets, loss_cfg.multi_horizon_weights)
    total = pred_loss
    components = {"pred": pred_loss.detach()}

    if stage_outs and stage_outs[-1]["logvar"] is not None and loss_cfg.lambda_nll > 0:
        nll = gaussian_nll(z_targets[:, -1, :], stage_outs[-1]["mu"], stage_outs[-1]["logvar"])
        total = total + loss_cfg.lambda_nll * nll
        components["nll"] = nll.detach()
    else:
        components["nll"] = z_targets.sum() * 0.0

    reg = variance_covariance_penalty(model_out["z_context"])
    total = total + loss_cfg.lambda_reg * reg
    components["reg"] = reg.detach()

    gate = gate_sparsity_penalty(stage_outs, model_out.get("memory_gate"))
    total = total + loss_cfg.lambda_gate * gate
    components["gate"] = gate.detach()

    if model_out.get("memory_occupancy") is not None and loss_cfg.lambda_mem > 0:
        mem = model_out["memory_occupancy"].float()
        total = total + loss_cfg.lambda_mem * mem
        components["mem"] = mem.detach()
    else:
        components["mem"] = z_targets.sum() * 0.0

    if model_out.get("recon_head") is not None and loss_cfg.lambda_recon > 0:
        recon = reconstruction_penalty(model_out["recon_head"], future)
        total = total + loss_cfg.lambda_recon * recon
        components["recon"] = recon.detach()
    else:
        components["recon"] = z_targets.sum() * 0.0

    if use_financial_regularizers:
        noarb = no_arbitrage_penalty(model_out["return_head"], z_targets[:, -1, :])
        vol = volatility_persistence_penalty(stage_outs)
        liq = liquidity_impact_penalty(model_out["liq_head"], context, future)
        macro = macro_smoothness_penalty(z_targets[:, -1, :])

        total = total + loss_cfg.lambda_noarb * noarb + loss_cfg.lambda_vol * vol + loss_cfg.lambda_liq * liq + loss_cfg.lambda_macro * macro
        components.update({"noarb": noarb.detach(), "vol": vol.detach(), "liq": liq.detach(), "macro": macro.detach()})
    else:
        zero = z_targets.sum() * 0.0
        components.update({"noarb": zero, "vol": zero, "liq": zero, "macro": zero})

    components["total"] = total.detach()
    return LossBreakdown(total=total, components=components)
