"""Generated write-policy witnesses; no financial data or scientific outcomes."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from fijepa.config import LossConfig, ModelConfig
from fijepa.losses import compute_fijepa_loss
from fijepa.memory import PrototypeMemory
from fijepa.model import FIJEPA


def bank(**kwargs):
    return PrototypeMemory(2, 1, memory_size=3, decay=0.5, **kwargs)


def snapshot(memory):
    return {name: value.clone() for name, value in memory.state_dict().items()}


def assert_same(memory, before):
    assert set(memory.state_dict()) == set(before)
    for name, tensor in memory.state_dict().items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0, equal_nan=True)


def test_inactive_zero_cannot_outrank_negative_active_prototype():
    memory = bank(merge_threshold=0.0)
    memory.update(torch.tensor([-1.0, 0.0]), torch.tensor([2.0]), torch.tensor(1.0))
    memory.update(torch.tensor([1.0, 0.0]), torch.tensor([4.0]), torch.tensor(1.0))
    assert memory.filled.item() == 2
    torch.testing.assert_close(memory.keys[:2], torch.tensor([[-1.0, 0.0], [1.0, 0.0]]))
    torch.testing.assert_close(memory.values[:2], torch.tensor([[2.0], [4.0]]))
    torch.testing.assert_close(memory.age, torch.tensor([2.0, 1.0, 2.0]))


def test_active_ties_use_first_active_slot_and_full_bank_replaces_least_salient():
    memory = bank(merge_threshold=0.0)
    memory.keys.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0], [0.0, -1.0]]))
    memory.values.copy_(torch.tensor([[2.0], [9.0], [6.0]]))
    memory.salience.copy_(torch.tensor([1.0, 0.0, 1.0]))
    memory.filled.fill_(3)
    memory.update(torch.tensor([1.0, 0.0]), torch.tensor([4.0]), torch.tensor(1.0))
    torch.testing.assert_close(memory.values[:, 0], torch.tensor([3.0, 9.0, 6.0]))
    memory.merge_threshold = 1.0
    memory.update(torch.tensor([-1.0, 0.0]), torch.tensor([7.0]), torch.tensor(2.0))
    torch.testing.assert_close(memory.keys[1], torch.tensor([-1.0, 0.0]))
    assert memory.values[1].item() == 7.0
    assert memory.filled.item() == 3


def test_no_active_slots_zero_salience_and_empty_call_keep_declared_age_semantics():
    memory = bank(write_threshold=0.0)
    memory.update(torch.tensor([1.0, 0.0]), torch.tensor([2.0]), torch.tensor(0.0))
    assert memory.filled.item() == 1 and memory.occupancy() == 0
    memory.update(torch.tensor([0.0, 1.0]), torch.tensor([3.0]), torch.tensor(1.0))
    assert memory.filled.item() == 2 and memory.salience[1].item() == 1
    before = snapshot(memory)
    memory.update(torch.empty(0, 2), torch.empty(0, 1), torch.empty(0))
    before["age"] += 1
    assert_same(memory, before)


def test_pre_call_views_are_owned_before_any_candidate_write():
    memory = PrototypeMemory(2, 1, memory_size=2, decay=0.5)
    memory.keys.copy_(torch.eye(2))
    memory.values.copy_(torch.tensor([[2.0], [10.0]]))
    memory.salience.fill_(1)
    memory.filled.fill_(2)
    buffer_addresses = {name: value.data_ptr() for name, value in memory.named_buffers()}
    memory.update(memory.keys.flip(0), memory.values, memory.salience)
    torch.testing.assert_close(memory.values, torch.tensor([[6.0], [6.0]]))
    assert {name: value.data_ptr() for name, value in memory.named_buffers()} == buffer_addresses


def test_external_inputs_and_autograd_are_not_written():
    memory = bank()
    key = torch.eye(2, requires_grad=True)
    value = torch.tensor([[2.0], [3.0]], requires_grad=True)
    salience = torch.ones(2, requires_grad=True)
    before = [x.detach().clone() for x in (key, value, salience)]
    memory.update(key, value, salience)
    for tensor, original in zip((key, value, salience), before):
        torch.testing.assert_close(tensor, original)
        assert tensor.grad is None
    assert all(not tensor.requires_grad for tensor in memory.buffers())
    key.data.fill_(99)
    assert memory.keys.max().item() == 1.0


@pytest.mark.parametrize("case", [
    "short_values", "short_salience", "extra_salience", "key_rank", "value_rank",
    "wrong_key_dim", "wrong_value_dim", "nan_key", "inf_value", "nan_salience",
    "negative_salience", "complex_salience", "boolean_salience", "integer_key",
    "double_key", "double_value", "salience_cast_overflow", "meta_key", "meta_value",
])
def test_invalid_complete_batch_is_rejected_without_partial_mutation(case):
    memory = bank()
    memory.update(torch.tensor([1.0, 0.0]), torch.tensor([2.0]), torch.tensor(1.0))
    before = snapshot(memory)
    key, value, salience = torch.eye(2), torch.tensor([[4.0], [5.0]]), torch.ones(2)
    if case == "short_values": value = value[:1]
    elif case == "short_salience": salience = salience[:1]
    elif case == "extra_salience": salience = torch.ones(3)
    elif case == "key_rank": key = key.unsqueeze(0)
    elif case == "value_rank": value = value.unsqueeze(0)
    elif case == "wrong_key_dim": key = torch.ones(2, 3)
    elif case == "wrong_value_dim": value = torch.ones(2, 2)
    elif case == "nan_key": key[1, 0] = float("nan")
    elif case == "inf_value": value[1, 0] = float("inf")
    elif case == "nan_salience": salience[1] = float("nan")
    elif case == "negative_salience": salience[1] = -1
    elif case == "complex_salience": salience = salience.to(torch.complex64)
    elif case == "boolean_salience": salience = salience.bool()
    elif case == "integer_key": key = key.long()
    elif case == "double_key": key = key.double()
    elif case == "double_value": value = value.double()
    elif case == "salience_cast_overflow": salience = torch.tensor([1.0, 1e100], dtype=torch.float64)
    elif case == "meta_key": key = key.to("meta")
    elif case == "meta_value": value = value.to("meta")
    with pytest.raises((ValueError, TypeError)):
        memory.update(key, value, salience)
    assert_same(memory, before)


@pytest.mark.parametrize("name,invalid", [
    ("filled", 0.5), ("filled", 4.0), ("filled", -1.0), ("keys", float("nan")),
    ("values", float("inf")), ("age", float("nan")), ("salience", float("inf")),
])
def test_invalid_stored_state_is_not_mutated(name, invalid):
    memory = bank()
    getattr(memory, name).flatten()[0] = invalid
    before = snapshot(memory)
    with pytest.raises(ValueError):
        memory.update(torch.eye(2), torch.ones(2, 1), torch.ones(2))
    assert_same(memory, before)


@pytest.mark.parametrize("setting,invalid", [
    ("decay", -0.1), ("decay", 1.1), ("decay", float("nan")),
    ("merge_threshold", -1.1), ("merge_threshold", 1.1),
    ("write_threshold", -1.0), ("write_threshold", float("inf")),
])
def test_invalid_write_policy_is_rejected_before_mutation(setting, invalid):
    memory = bank()
    setattr(memory, setting, invalid)
    before = snapshot(memory)
    with pytest.raises(ValueError):
        memory.update(torch.eye(2), torch.ones(2, 1), torch.ones(2))
    assert_same(memory, before)


def test_later_cosine_overflow_rolls_back_an_earlier_valid_candidate_row():
    memory = bank()
    before = snapshot(memory)
    with pytest.raises(ValueError, match="Cosine normalization"):
        memory.update(torch.tensor([[1.0, 0.0], [1e30, 1e30]]), torch.ones(2, 1), torch.ones(2))
    assert_same(memory, before)


def test_salience_threshold_uses_supplied_precision_and_double_write_is_not_rounded_through_float32():
    memory = bank(write_threshold=1.0)
    memory.update(torch.tensor([1.0, 0.0]), torch.tensor([2.0]), torch.tensor(1.0 - 1e-10, dtype=torch.float64))
    assert memory.filled.item() == 0
    memory = bank().double()
    precise = torch.tensor(1.0 + 2 ** -40, dtype=torch.float64)
    memory.update(torch.tensor([1.0, 0.0], dtype=torch.float64), torch.tensor([2.0], dtype=torch.float64), precise)
    assert memory.salience[0].item() == precise.item()


def test_default_valid_updates_are_identical_to_actual_retained_parent():
    source = Path(__file__).resolve().parents[1] / "research/verification/active_prototype_writes_20261010/parent_memory.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "f0c7904b80373e4180c79d53ea35061efaa5e55adf3226b77a42cb7c127b6cd1"
    spec = importlib.util.spec_from_file_location("_retained_fijepa_memory_950bbed", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    old = module.PrototypeMemory(2, 1, memory_size=3)
    new = PrototypeMemory(2, 1, memory_size=3)
    # Insert, merge, fill, replace, skip, and empty update, with no inactive
    # candidate meeting the default positive merge threshold.
    batches = [
        ([[1.0, 0.0], [0.0, 1.0]], [[2.0], [4.0]], [1.0, 2.0]),
        ([[1.0, 0.01], [-1.0, 0.0]], [[6.0], [8.0]], [3.0, 1.0]),
        ([[0.0, -1.0], [1.0, 1.0]], [[3.0], [9.0]], [4.0, 0.01]),
    ]
    for key, value, salience in batches:
        args = tuple(torch.tensor(x) for x in (key, value, salience))
        old.update(*args)
        new.update(*args)
        assert_same(new, old.state_dict())
    for memory in (old, new):
        memory.update(torch.empty(0, 2), torch.empty(0, 1), torch.empty(0))
    assert_same(new, old.state_dict())
    query = torch.tensor([[0.25, -0.75]])
    for actual, expected in zip(new.retrieve(query), old.retrieve(query)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_actual_model_loss_optimizer_ema_and_memory_update_then_checkpoint_restore():
    torch.manual_seed(94502)
    config = ModelConfig(backbone="mlp", d_model=8, latent_dim=8, num_layers=1,
                         dropout=0.0, predictor_hidden=8, stochastic=False,
                         stage_names=["h1"], memory_size=3, memory_topk=2,
                         memory_write_threshold=0.0)
    model = FIJEPA(input_dim=2, config=config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.linspace(-1.0, 1.0, 24).reshape(2, 6, 2)
    future = torch.linspace(-0.5, 0.5, 8).reshape(2, 1, 2, 2)
    output = model(x, future=future)
    loss = compute_fijepa_loss(output, x, future, LossConfig(multi_horizon_weights=[1.0])).total
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters())
    optimizer.step()
    model.update_target()
    model.update_memory(output["z_context"], output["z_targets"][:, -1])
    assert model.memory.filled.item() > 0
    assert all(torch.isfinite(tensor).all() for tensor in model.memory.buffers())
    restored = FIJEPA(input_dim=2, config=config)
    restored.load_state_dict({name: value.clone() for name, value in model.state_dict().items()})
    model.eval()
    restored.eval()
    torch.testing.assert_close(restored(x)["z_context"], model(x)["z_context"], rtol=0, atol=0)
