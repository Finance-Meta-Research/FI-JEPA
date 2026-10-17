"""Generated witnesses against the retained parent before implementation."""
import json

import torch

from fijepa.memory import PrototypeMemory


def state(memory):
    return {name: value.clone() for name, value in memory.state_dict().items()}


def unchanged(before, memory):
    return all(torch.equal(before[name], value) for name, value in memory.state_dict().items())


torch.set_num_threads(1)
cases = []
memory = PrototypeMemory(2, 1, memory_size=3, topk=2, decay=0.5, merge_threshold=0.0)
memory.update(torch.tensor([-1.0, 0.0]), torch.tensor([2.0]), torch.tensor(1.0))
memory.update(torch.tensor([1.0, 0.0]), torch.tensor([4.0]), torch.tensor(1.0))
cases.append({'case': 'inactive_slot_must_not_win_merge', 'filled': memory.filled.item(),
              'keys': memory.keys.tolist(), 'values': memory.values.tolist(),
              'expected_filled': 2, 'contract_passed': memory.filled.item() == 2})

memory = PrototypeMemory(2, 1, memory_size=3)
before = state(memory)
try:
    memory.update(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([[2.0]]), torch.ones(2))
    error = None
except Exception as exc:
    error = type(exc).__name__
cases.append({'case': 'shape_failure_must_roll_back', 'exception': error,
              'contract_passed': error is not None and unchanged(before, memory)})

memory = PrototypeMemory(2, 1, memory_size=3)
before = state(memory)
try:
    memory.update(torch.tensor([[1.0, 0.0], [float('nan'), 1.0]]), torch.tensor([[2.0], [3.0]]), torch.ones(2))
    error = None
except Exception as exc:
    error = type(exc).__name__
cases.append({'case': 'nonfinite_later_row_must_roll_back', 'exception': error,
              'contract_passed': error is not None and unchanged(before, memory)})

memory = PrototypeMemory(2, 1, memory_size=2, decay=0.5)
memory.update(torch.eye(2), torch.tensor([[2.0], [10.0]]), torch.ones(2))
memory.update(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), memory.values, torch.ones(2))
cases.append({'case': 'aliased_values_use_pre_call_snapshot', 'values': memory.values.tolist(),
              'expected_values': [[6.0], [6.0]],
              'contract_passed': torch.equal(memory.values, torch.tensor([[6.0], [6.0]]))})

result = {'scope': 'generated implementation witnesses only', 'torch': str(torch.__version__),
          'cases': cases, 'contract_failures': sum(not case['contract_passed'] for case in cases)}
print(json.dumps(result, indent=2, allow_nan=False))
raise SystemExit(1 if result['contract_failures'] else 0)
