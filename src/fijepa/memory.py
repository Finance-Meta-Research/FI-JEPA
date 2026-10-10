from __future__ import annotations

from dataclasses import dataclass
import math
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
        """Apply one owned, validated batch or leave every live buffer unchanged.

        Matching is against active prototypes only. All arithmetic is staged in
        candidate buffers so a malformed later row cannot partially write memory.
        This does not provide isolation from concurrent calls or process failure.
        """
        for name, lower, upper in (
            ("decay", 0.0, 1.0),
            ("merge_threshold", -1.0, 1.0),
            ("write_threshold", 0.0, math.inf),
        ):
            setting = getattr(self, name)
            if isinstance(setting, bool) or not isinstance(setting, (int, float)):
                raise ValueError(f"{name} must be a finite real number")
            if not math.isfinite(setting) or not lower <= setting <= upper:
                raise ValueError(f"{name} is outside its valid range")
        shapes = {
            "keys": (self.memory_size, self.key_dim),
            "values": (self.memory_size, self.value_dim),
            "salience": (self.memory_size,),
            "age": (self.memory_size,),
            "filled": (1,),
        }
        if self.memory_size < 1:
            raise ValueError("Memory capacity must be positive")
        for name, shape in shapes.items():
            tensor = getattr(self, name)
            if (
                tensor.layout != torch.strided
                or tensor.shape != shape
                or not tensor.is_floating_point()
                or tensor.device != self.keys.device
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid {name} memory buffer")
        filled = float(self.filled.item())
        if not filled.is_integer() or not 0 <= filled <= self.memory_size:
            raise ValueError("filled must be an exact count within memory capacity")
        if not all(isinstance(tensor, torch.Tensor) for tensor in (key, value, salience)):
            raise TypeError("Memory writes require tensor inputs")
        if key.ndim == 1:
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        if key.ndim != 2 or key.shape[1] != self.key_dim:
            raise ValueError("Expected keys with shape [B, key_dim]")
        if value.ndim != 2 or value.shape != (key.shape[0], self.value_dim):
            raise ValueError("Expected values with shape [B, value_dim]")
        for name, tensor, buffer in (("key", key, self.keys), ("value", value, self.values)):
            if (
                tensor.layout != torch.strided
                or not tensor.is_floating_point()
                or tensor.dtype != buffer.dtype
                or tensor.device != buffer.device
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"{name} must be finite and match its memory dtype/device")
        if (
            salience.layout != torch.strided
            or salience.is_complex()
            or salience.dtype == torch.bool
            or salience.numel() != key.shape[0]
            or not torch.isfinite(salience).all()
            or (salience < 0).any()
        ):
            raise ValueError("Expected one finite nonnegative salience per row")
        # Own inputs before writes, including views into this bank or each other.
        key = key.detach().clone()
        value = value.detach().clone()
        input_salience = salience.detach().flatten().clone()
        salience = input_salience.to(self.salience).clone()
        if not torch.isfinite(salience).all():
            raise ValueError("Salience cannot be represented in the memory dtype")
        candidate = {name: getattr(self, name).clone() for name in shapes}
        keys, values = candidate["keys"], candidate["values"]
        scores, ages = candidate["salience"], candidate["age"]
        for i in range(key.shape[0]):
            s = float(input_salience[i].item())
            if s < self.write_threshold:
                continue
            q = key[i]
            v = value[i]
            active = torch.nonzero(scores > 0, as_tuple=False).flatten()
            if active.numel():
                active_keys = keys[active]
                # Refuse overflow in the existing cosine normalization rather
                # than interpreting a finite zero from an infinite norm as data.
                if not torch.isfinite(active_keys.norm(dim=-1)).all() or not torch.isfinite(q.norm()):
                    raise ValueError("Cosine normalization exceeds the memory dtype")
                sims = F.cosine_similarity(active_keys, q.unsqueeze(0), dim=-1)
                if not torch.isfinite(sims).all():
                    raise ValueError("Nonfinite prototype similarity")
                best_active = int(torch.argmax(sims).item())
                best = int(active[best_active].item())
                if float(sims[best_active].item()) >= self.merge_threshold:
                    keys[best].mul_(self.decay).add_(q, alpha=1.0 - self.decay)
                    values[best].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
                    scores[best].mul_(self.decay).add_(salience[i], alpha=1.0 - self.decay)
                    ages[best] = 0.0
                    continue
            if filled < self.memory_size:
                slot = int(filled)
                filled += 1
            else:
                slot = int(torch.argmin(scores).item())
            keys[slot].copy_(q)
            values[slot].copy_(v)
            scores[slot].copy_(salience[i])
            ages[slot] = 0.0
        ages.add_(1.0)
        candidate["filled"].fill_(filled)
        if not all(torch.isfinite(tensor).all() for tensor in candidate.values()):
            raise ValueError("Prototype update exceeds the finite memory dtype")
        for name, tensor in candidate.items():
            getattr(self, name).copy_(tensor)

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
