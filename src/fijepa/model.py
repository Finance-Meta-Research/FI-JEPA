from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn

from .config import ModelConfig
from .memory import PrototypeMemory


def sinusoidal_positional_encoding(length: int, dim: int, device=None):
    pos = torch.arange(length, device=device).float().unsqueeze(1)
    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros(length, dim, device=device)
    i = torch.arange(half_dim, device=device).float().unsqueeze(0)
    angle_rates = 1.0 / torch.pow(10000.0, (2 * i) / max(1, dim))
    angles = pos * angle_rates
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0 : 2 * half_dim : 2] = torch.sin(angles)
    pe[:, 1 : 2 * half_dim : 2] = torch.cos(angles)
    if dim % 2 == 1:
        pe[:, -1] = 0.0
    return pe


class TemporalTransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.asset_embed = nn.Embedding(1024, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=4 * d_model, dropout=dropout, batch_first=True, norm_first=True, activation="gelu")
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, asset_id=None):
        b, t, _ = x.shape
        h = self.input_proj(x)
        pe = sinusoidal_positional_encoding(t, h.shape[-1], device=x.device)
        h = h + pe.unsqueeze(0)
        if asset_id is not None:
            h = h + self.asset_embed(asset_id.squeeze(-1)).unsqueeze(1)
        h = self.dropout(h)
        for blk in self.blocks:
            h = blk(h)
        return self.final_norm(h)


class TemporalConvBackbone(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_ch = input_dim
        for _ in range(num_layers):
            layers += [nn.Conv1d(in_ch, d_model, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(dropout)]
            in_ch = d_model
        self.net = nn.Sequential(*layers)
        self.proj = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, asset_id=None):
        h = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.final_norm(self.proj(h))


class MLPBackbone(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout)]
            in_dim = d_model
        self.net = nn.Sequential(*layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, asset_id=None):
        return self.final_norm(self.net(x.mean(dim=1)))


class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        if config.backbone == "transformer":
            self.backbone = TemporalTransformerBackbone(input_dim, config.d_model, config.num_layers, config.num_heads, config.dropout)
        elif config.backbone == "conv":
            self.backbone = TemporalConvBackbone(input_dim, config.d_model, config.num_layers, config.dropout)
        elif config.backbone == "mlp":
            self.backbone = MLPBackbone(input_dim, config.d_model, config.num_layers, config.dropout)
        else:
            raise ValueError(f"Unknown backbone: {config.backbone}")
        self.pool = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.GELU(), nn.Linear(config.d_model, config.latent_dim))
        self.pool_norm = nn.LayerNorm(config.latent_dim)

    def forward(self, x, asset_id=None):
        h = self.backbone(x, asset_id=asset_id)
        if h.ndim == 2:
            pooled = h
        else:
            pooled = h[:, -1] + h.mean(dim=1)
        return self.pool_norm(self.pool(pooled))


class TargetEncoder(nn.Module):
    def __init__(self, context_encoder: ContextEncoder):
        super().__init__()
        import copy
        self.context = copy.deepcopy(context_encoder)
        for p in self.context.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_from(self, online_encoder: ContextEncoder, ema_decay: float):
        for p_t, p in zip(self.context.parameters(), online_encoder.parameters()):
            p_t.data.mul_(ema_decay).add_(p.data, alpha=1.0 - ema_decay)

    def forward(self, x, asset_id=None):
        return self.context(x, asset_id=asset_id)


class PredictorStage(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, stochastic: bool, name: str):
        super().__init__()
        self.name = name
        self.stochastic = stochastic
        self.trunk = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim) if stochastic else None
        self.gate = nn.Linear(hidden, latent_dim)

    def forward(self, z):
        h = self.trunk(z)
        mu = self.mu(h)
        gate = torch.sigmoid(self.gate(h))
        if self.stochastic:
            logvar = torch.clamp(self.logvar(h), -6.0, 2.0)
            return mu, logvar, gate
        return mu, None, gate


class PredictorBank(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, stage_names: Sequence[str], stochastic: bool = True):
        super().__init__()
        self.stage_names = list(stage_names)
        self.stages = nn.ModuleList([PredictorStage(latent_dim, hidden, stochastic=stochastic, name=n) for n in self.stage_names])

    def forward(self, z):
        stage_outs = []
        cur = z
        for stage in self.stages:
            mu, logvar, gate = stage(cur)
            if logvar is not None:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                cur = mu + gate * std * eps
            else:
                cur = mu
            stage_outs.append({"name": stage.name, "mu": mu, "logvar": logvar, "gate": gate, "z": cur})
        return stage_outs


class FIJEPA(nn.Module):
    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.online_encoder = ContextEncoder(input_dim, config)
        self.target_encoder = TargetEncoder(self.online_encoder)
        stage_names = config.stage_names if config.predictor_mode == "split" else ["monolithic"]
        self.predictor_bank = PredictorBank(config.latent_dim, config.predictor_hidden, stage_names, stochastic=config.stochastic)
        self.latent_to_return = nn.Linear(config.latent_dim, 1)
        self.latent_to_vol = nn.Linear(config.latent_dim, 1)
        self.latent_to_liq = nn.Linear(config.latent_dim, 1)
        self.reconstruction_head = nn.Linear(config.latent_dim, input_dim) if config.use_reconstruction_head else None
        self.memory = PrototypeMemory(config.latent_dim, config.latent_dim, memory_size=config.memory_size, topk=config.memory_topk, temperature=config.memory_temperature, decay=config.memory_decay, write_threshold=config.memory_write_threshold, merge_threshold=config.memory_merge_threshold) if config.use_memory else None
        self.memory_query = nn.Linear(config.latent_dim, config.latent_dim)
        self.memory_value = nn.Linear(config.latent_dim, config.latent_dim)
        self.memory_gate = nn.Linear(config.latent_dim * 2, config.latent_dim)
        self.memory_proj = nn.Linear(config.latent_dim, config.latent_dim)

    @torch.no_grad()
    def update_target(self):
        if self.config.use_target_ema:
            self.target_encoder.update_from(self.online_encoder, self.config.ema_decay)
        else:
            self.target_encoder.context.load_state_dict(self.online_encoder.state_dict())

    def encode_context(self, x, asset_id=None):
        return self.online_encoder(x, asset_id=asset_id)

    def encode_target(self, x, asset_id=None):
        return self.target_encoder(x, asset_id=asset_id)

    def encode_targets(self, futures, asset_id=None):
        b, h, t, d = futures.shape
        flat = futures.reshape(b * h, t, d)
        if asset_id is not None:
            aid = asset_id.unsqueeze(1).expand(b, h, 1).reshape(b * h, 1)
        else:
            aid = None
        z = self.encode_target(flat, asset_id=aid)
        return z.view(b, h, -1)

    def predict_latent(self, z):
        return self.predictor_bank(z)

    def memory_context(self, z):
        if self.memory is None:
            return z, None, None
        q = self.memory_query(z)
        v = self.memory_value(z)
        retrieved, attn = self.memory.retrieve(q)
        gate = torch.sigmoid(self.memory_gate(torch.cat([q, retrieved], dim=-1)))
        z_mem = z + gate * self.memory_proj(retrieved)
        return z_mem, attn, gate

    @torch.no_grad()
    def update_memory(self, z_context, z_target, salience=None):
        if self.memory is None:
            return
        if salience is None:
            salience = (z_target - z_context).pow(2).mean(dim=-1).sqrt()
        self.memory.update(self.memory_query(z_context), self.memory_value(z_target), salience)

    def forward(self, context, future=None, asset_id=None):
        z_context_raw = self.encode_context(context, asset_id=asset_id)
        z_context, mem_attn, mem_gate = self.memory_context(z_context_raw)
        stages = self.predict_latent(z_context)
        out = {
            "z_context": z_context,
            "z_context_raw": z_context_raw,
            "stages": stages,
            "latent": stages[-1]["z"],
            "return_head": self.latent_to_return(stages[-1]["z"]),
            "vol_head": self.latent_to_vol(stages[-1]["z"]),
            "liq_head": self.latent_to_liq(stages[-1]["z"]),
            "memory_attention": mem_attn,
            "memory_gate": mem_gate,
            "memory_occupancy": torch.tensor(self.memory.occupancy() if self.memory is not None else 0.0, device=context.device),
        }
        if self.reconstruction_head is not None:
            out["recon_head"] = self.reconstruction_head(stages[-1]["z"])
        if future is not None:
            if future.ndim == 4:
                out["z_targets"] = self.encode_targets(future, asset_id=asset_id)
            else:
                out["z_targets"] = self.encode_target(future, asset_id=asset_id).unsqueeze(1)
        return out

    def rollout(self, context, steps: int, asset_id=None):
        z = self.encode_context(context, asset_id=asset_id)
        if self.memory is not None:
            z, _, _ = self.memory_context(z)
        traj = []
        cur = z
        for _ in range(steps):
            stage_outs = self.predict_latent(cur)
            cur = stage_outs[-1]["z"]
            traj.append(cur)
        return torch.stack(traj, dim=1)
