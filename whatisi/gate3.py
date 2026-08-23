from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from .gate1 import _silent
from .gate2 import Gate2Config, GenericQueryAgent, generate_query_sequences, train_agent


@dataclass
class Gate3Config:
    train_labels: int = 256
    test_labels: int = 5000
    head_steps: int = 700
    head_lr: float = 2e-2


class DeicticBitHead(nn.Module):
    """Tiny new-task head. It never receives owner labels."""
    def __init__(self, state_dim: int, n_channels: int):
        super().__init__()
        self.slot = nn.Linear(state_dim, n_channels)

    def forward(self, state, bits):
        w = torch.softmax(self.slot(state), dim=-1)
        return torch.sum(w * bits, dim=-1)


def fit_head(state: np.ndarray, bits: np.ndarray, target: np.ndarray, n_channels: int, cfg: Gate3Config, seed: int):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    s = torch.from_numpy(state.astype(np.float32))
    b = torch.from_numpy(bits.astype(np.float32))
    y = torch.from_numpy(target.astype(np.float32))
    head = DeicticBitHead(state.shape[1], n_channels)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.head_lr, weight_decay=1e-4)
    for _ in range(cfg.head_steps):
        pred = head(s, b)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return head


def accuracy(head, state: np.ndarray, bits: np.ndarray, target: np.ndarray) -> float:
    with torch.no_grad():
        pred = head(torch.from_numpy(state.astype(np.float32)), torch.from_numpy(bits.astype(np.float32))).numpy()
    return float(np.mean(np.sign(pred) == np.sign(target)))


def collect_states(model: GenericQueryAgent, seed: int, cfg2: Gate2Config):
    sensory, values, _, owners = generate_query_sequences(seed, 420, cfg2.world)
    with torch.no_grad():
        _, h = model(torch.from_numpy(sensory), torch.from_numpy(values))
    h = h.numpy()

    random_model = GenericQueryAgent(cfg2.world.n_channels, cfg2.hidden_size)
    with torch.no_grad():
        _, rh = random_model(torch.from_numpy(sensory), torch.from_numpy(values))
    rh = rh.numpy()

    t = np.arange(cfg2.world.steps)
    valid = t >= 12
    silent = np.array([_silent(int(i), cfg2.world.silent_windows) for i in t]) & valid

    return {
        "self_core": h[:, valid].reshape(-1, cfg2.hidden_size),
        "random_core": rh[:, valid].reshape(-1, cfg2.hidden_size),
        "sensory": sensory[:, valid].reshape(-1, sensory.shape[-1]),
        "owners": owners[:, valid].reshape(-1),
        "silent_mask": np.tile(silent[valid], sensory.shape[0]),
    }


def run_one(seed: int, cfg3: Gate3Config, cfg2: Gate2Config):
    model = train_agent(seed, cfg2)
    data = collect_states(model, seed + 20000, cfg2)
    rng = np.random.default_rng(seed + 30000)

    n = len(data["owners"])
    bits = rng.choice([-1.0, 1.0], size=(n, cfg2.world.n_channels)).astype(np.float32)
    target = bits[np.arange(n), data["owners"]]

    idx = rng.permutation(n)
    train_idx = idx[:cfg3.train_labels]
    silent_idx = np.where(data["silent_mask"])[0]
    rng.shuffle(silent_idx)
    test_idx = silent_idx[:cfg3.test_labels]

    out = {}
    for key in ("self_core", "random_core", "sensory"):
        head = fit_head(data[key][train_idx], bits[train_idx], target[train_idx], cfg2.world.n_channels, cfg3, seed + len(key))
        out[key] = accuracy(head, data[key][test_idx], bits[test_idx], target[test_idx])
    out["oracle"] = 1.0
    return out


def run_gate3(seeds=(501, 502, 503), cfg3: Gate3Config | None = None, cfg2: Gate2Config | None = None):
    cfg3 = cfg3 or Gate3Config()
    cfg2 = cfg2 or Gate2Config()
    rows = [{"seed": int(s), **run_one(int(s), cfg3, cfg2)} for s in seeds]

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {
        "train_labels_for_new_task": cfg3.train_labels,
        "silent_test_accuracy_self_core": avg("self_core"),
        "silent_test_accuracy_random_core": avg("random_core"),
        "silent_test_accuracy_current_sensory": avg("sensory"),
        "oracle": 1.0,
        "rows": rows,
    }
    checks = {
        "new_task_reuses_self_state": summary["silent_test_accuracy_self_core"] >= 0.90,
        "beats_random_recurrence": summary["silent_test_accuracy_self_core"] - summary["silent_test_accuracy_random_core"] >= 0.20,
        "beats_current_sensory": summary["silent_test_accuracy_self_core"] - summary["silent_test_accuracy_current_sensory"] >= 0.20,
    }
    summary["checks"] = checks
    summary["pass"] = bool(all(checks.values()))
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_gate3(), indent=2, sort_keys=True))
