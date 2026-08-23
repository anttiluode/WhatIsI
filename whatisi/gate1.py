from __future__ import annotations

from dataclasses import dataclass, replace
import random
import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:
    raise ImportError("Gate 1 requires torch. Install with: pip install -e '.[torch]'") from e

from .gate0 import CausalSelfPointer


@dataclass
class SequenceConfig:
    n_channels: int = 4
    steps: int = 96
    control_gain: float = 2.8
    noise_std: float = 1.0
    distractor_std: float = 3.0
    transfer_step: int = 52
    silent_windows: tuple[tuple[int, int], ...] = ((28, 38), (72, 82))


def _silent(t: int, windows) -> bool:
    return any(a <= t < b for a, b in windows)


def generate_sequences(seed: int, n_seq: int, cfg: SequenceConfig):
    rng = np.random.default_rng(seed)
    xs = np.zeros((n_seq, cfg.steps, 1 + cfg.n_channels), np.float32)
    ys = np.zeros((n_seq, cfg.steps), np.int64)
    for b in range(n_seq):
        self_slot = int(rng.integers(cfg.n_channels))
        distractor = (self_slot + 1) % cfg.n_channels
        for t in range(cfg.steps):
            if t == cfg.transfer_step:
                choices = [i for i in range(cfg.n_channels) if i != self_slot]
                self_slot = int(rng.choice(choices))
                distractor = (self_slot + 1) % cfg.n_channels
            action = 0.0 if _silent(t, cfg.silent_windows) else float(rng.choice([-1.0, 1.0]))
            delta = rng.normal(0.0, cfg.noise_std, cfg.n_channels)
            delta[distractor] += rng.normal(0.0, cfg.distractor_std)
            delta[self_slot] += cfg.control_gain * action
            xs[b, t, 0] = action
            xs[b, t, 1:] = delta
            ys[b, t] = self_slot
    return xs, ys


class TinyGenericMemory(nn.Module):
    """Attacker: four persistent floats, no explicit self semantics."""
    def __init__(self, n_channels: int):
        super().__init__()
        self.gru = nn.GRU(input_size=1 + n_channels, hidden_size=n_channels, batch_first=True)
        self.readout = nn.Linear(n_channels, n_channels)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.readout(h)


def pointer_predictions(xs: np.ndarray, cfg: SequenceConfig) -> np.ndarray:
    out = np.zeros((xs.shape[0], xs.shape[1]), np.int64)
    for b in range(xs.shape[0]):
        p = CausalSelfPointer(cfg.n_channels)
        for t in range(xs.shape[1]):
            belief = p.update(float(xs[b, t, 0]), xs[b, t, 1:])
            out[b, t] = int(np.argmax(belief))
    return out


def masked_accuracy(pred: np.ndarray, y: np.ndarray, cfg: SequenceConfig, mask: str = "all") -> float:
    t = np.arange(cfg.steps)
    valid = t >= 12
    if mask == "silent":
        valid &= np.array([_silent(int(i), cfg.silent_windows) for i in t])
    elif mask == "post_transfer":
        valid &= (t >= cfg.transfer_step) & (t < cfg.transfer_step + 24)
    return float(np.mean(pred[:, valid] == y[:, valid]))


def train_generic(seed: int, train_cfg: SequenceConfig, epochs: int = 12):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    x_np, y_np = generate_sequences(seed + 1000, 900, train_cfg)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    model = TinyGenericMemory(train_cfg.n_channels)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    batch = 64
    for _ in range(epochs):
        order = torch.randperm(x.shape[0])
        for lo in range(0, x.shape[0], batch):
            idx = order[lo:lo + batch]
            logits = model(x[idx])
            loss = loss_fn(logits[:, 12:].reshape(-1, train_cfg.n_channels), y[idx, 12:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def eval_arm(model, seed: int, cfg: SequenceConfig, n_seq: int = 240) -> dict:
    x, y = generate_sequences(seed, n_seq, cfg)
    pp = pointer_predictions(x, cfg)
    with torch.no_grad():
        gp = model(torch.from_numpy(x)).argmax(-1).cpu().numpy()
    return {
        "pointer_all": masked_accuracy(pp, y, cfg, "all"),
        "generic_all": masked_accuracy(gp, y, cfg, "all"),
        "pointer_silent": masked_accuracy(pp, y, cfg, "silent"),
        "generic_silent": masked_accuracy(gp, y, cfg, "silent"),
        "pointer_post_transfer": masked_accuracy(pp, y, cfg, "post_transfer"),
        "generic_post_transfer": masked_accuracy(gp, y, cfg, "post_transfer"),
    }


def run_gate1(seeds=(301, 302, 303)) -> dict:
    train_cfg = SequenceConfig()
    rows = []
    for seed in seeds:
        model = train_generic(int(seed), train_cfg)
        in_dist = eval_arm(model, int(seed) + 20000, train_cfg)
        ood_cfg = replace(train_cfg, control_gain=2.15, distractor_std=4.0, silent_windows=((24, 42), (68, 86)))
        ood = eval_arm(model, int(seed) + 30000, ood_cfg)
        rows.append({"seed": int(seed), "in": in_dist, "ood": ood})

    def avg(split, key):
        return float(np.mean([r[split][key] for r in rows]))

    summary = {
        "persistent_state_floats": train_cfg.n_channels,
        "generic_slow_parameters": int(sum(p.numel() for p in model.parameters())),
        "in": {k: avg("in", k) for k in rows[0]["in"]},
        "ood": {k: avg("ood", k) for k in rows[0]["ood"]},
    }
    generic_matches = (
        summary["ood"]["generic_all"] >= summary["ood"]["pointer_all"] - 0.03
        and summary["ood"]["generic_silent"] >= summary["ood"]["pointer_silent"] - 0.03
    )
    summary["verdict"] = {
        "generic_memory_matches_pointer": bool(generic_matches),
        "special_self_primitive_survives": bool(not generic_matches),
        "interpretation": "GENERIC_MEMORY_SUFFICIENT" if generic_matches else "STRUCTURED_POINTER_HAS_OOD_ADVANTAGE",
    }
    summary["rows"] = rows
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_gate1(), indent=2, sort_keys=True))
