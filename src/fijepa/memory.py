from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class MemoryState:
    keys: torch.Tensor
    values: torch.Tensor
    salience: torch.Tensor


class PrototypeMemory(nn.Module):
    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        memory_size: int = 64,
        topk: int = 4,
        temperature: float = 0.2,
        decay: float = 0.99,
        write_threshold: float = 0.15,
        merge_threshold: float = 0.9,
    ):
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.memory_size = memory_size
        self.topk = topk
        self.temperature = temperature
        self.decay = decay
        self.write_threshold = write_threshold
        self.merge_threshold = merge_threshold
        self.register_buffer("keys", torch.zeros(memory_size, key_dim))
        self.register_buffer("values", torch.zeros(memory_size, value_dim))
        self.register_buffer("salience", torch.zeros(memory_size))
        self.register_buffer("age", torch.zeros(memory_size))
        self.register_buffer("filled", torch.zeros(1))

    def _active_mask(self):
        return self.salience > 0

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if query.ndim != 2:
            raise ValueError(f"Expected query [B, D], got {tuple(query.shape)}")
        if self.filled.item() < 1 or not self._active_mask().any():
            return torch.zeros(query.shape[0], self.value_dim, device=query.device, dtype=query.dtype), torch.zeros(query.shape[0], 1, device=query.device, dtype=query.dtype)
        keys = self.keys
        values = self.values
        active = self._active_mask()
        keys = keys[active]
        values = values[active]
        salience = self.salience[active].unsqueeze(0)
        sim = query @ keys.t()
        sim = sim / max(self.temperature, 1e-6)
        sim = sim + torch.log(salience + 1e-6)
        k = min(self.topk, sim.shape[-1])
        topv, topi = torch.topk(sim, k=k, dim=-1)
        weights = torch.softmax(topv, dim=-1)
        gathered = values[topi]
        retrieved = (weights.unsqueeze(-1) * gathered).sum(dim=1)
        attention = weights.mean(dim=-1, keepdim=True)
        return retrieved, attention

    @torch.no_grad()
    def update(self, key: torch.Tensor, value: torch.Tensor, salience: torch.Tensor):
        if key.ndim == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            salience = salience.unsqueeze(0)
        key = key.detach()
        value = value.detach()
        salience = salience.detach().flatten()
        if key.shape[-1] != self.key_dim or value.shape[-1] != self.value_dim:
            raise ValueError("Memory dimensions do not match.")
        for i in range(key.shape[0]):
            s = float(salience[i].item())
            if s < self.write_threshold:
                continue
            q = key[i]
            v = value[i]
            if self._active_mask().any():
                sims = F.cosine_similarity(self.keys, q.unsqueeze(0), dim=-1)
                best = int(torch.argmax(sims).item())
                if float(sims[best].item()) >= self.merge_threshold:
                    self.keys[best].mul_(self.decay).add_(q, alpha=1.0 - self.decay)
                    self.values[best].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
                    self.salience[best].mul_(self.decay).add_(torch.tensor(s, device=self.salience.device), alpha=1.0 - self.decay)
                    self.age[best] = 0.0
                    continue
            if int(self.filled.item()) < self.memory_size:
                slot = int(self.filled.item())
                self.filled += 1
            else:
                slot = int(torch.argmin(self.salience).item())
            self.keys[slot].copy_(q)
            self.values[slot].copy_(v)
            self.salience[slot].copy_(torch.tensor(s, device=self.salience.device))
            self.age[slot] = 0.0
        self.age.add_(1.0)

    def diversity_penalty(self):
        active = self._active_mask()
        if active.sum() < 2:
            return self.keys.sum() * 0.0
        keys = self.keys[active]
        keys = keys - keys.mean(dim=0, keepdim=True)
        cov = (keys.t() @ keys) / max(1, keys.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag ** 2).mean()

    def occupancy(self):
        return float(self._active_mask().float().mean().item())
