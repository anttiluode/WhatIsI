from __future__ import annotations

from dataclasses import dataclass, field
import random
import numpy as np
import torch
from torch import nn

from .gate1 import SequenceConfig, _silent


@dataclass
class Gate2Config:
    world: SequenceConfig = field(default_factory=SequenceConfig)
    hidden_size: int = 8
    train_sequences: int = 1200
    test_sequences: int = 300
    epochs: int = 10
    batch_size: int = 64
    lr: float = 3e-3


def generate_query_sequences(seed: int, n_seq: int, cfg: SequenceConfig):
    rng = np.random.default_rng(seed)
    sensory = np.zeros((n_seq, cfg.steps, 1 + cfg.n_channels), np.float32)
    values = np.zeros((n_seq, cfg.steps, cfg.n_channels), np.float32)
    targets = np.zeros((n_seq, cfg.steps), np.float32)
    owners = np.zeros((n_seq, cfg.steps), np.int64)

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

            q = rng.normal(0.0, 1.0, cfg.n_channels)
            sensory[b, t, 0] = action
            sensory[b, t, 1:] = delta
            values[b, t] = q
            targets[b, t] = q[self_slot]
            owners[b, t] = self_slot

    return sensory, values, targets, owners


class GenericQueryAgent(nn.Module):
    """Generic recurrent memory. Training never receives an explicit self label."""
    def __init__(self, n_channels: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(1 + n_channels, hidden_size, batch_first=True)
        self.query = nn.Sequential(
            nn.Linear(hidden_size + n_channels, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, sensory, values):
        h, _ = self.gru(sensory)
        y = self.query(torch.cat([h, values], dim=-1)).squeeze(-1)
        return y, h

    def answer_from_hidden(self, h, values):
        return self.query(torch.cat([h, values], dim=-1)).squeeze(-1)


def train_agent(seed: int, cfg: Gate2Config):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(1)
    s_np, v_np, y_np, _ = generate_query_sequences(seed + 1000, cfg.train_sequences, cfg.world)
    s, v, y = map(torch.from_numpy, (s_np, v_np, y_np))
    model = GenericQueryAgent(cfg.world.n_channels, cfg.hidden_size)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    for _ in range(cfg.epochs):
        order = torch.randperm(s.shape[0])
        for lo in range(0, s.shape[0], cfg.batch_size):
            idx = order[lo:lo + cfg.batch_size]
            pred, _ = model(s[idx], v[idx])
            loss = torch.mean((pred[:, 12:] - y[idx, 12:]) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def ridge_probe(h: np.ndarray, owners: np.ndarray, n_channels: int):
    H = np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)
    Y = np.eye(n_channels)[owners]
    return np.linalg.pinv(H) @ Y


def probe_predict(h: np.ndarray, W: np.ndarray) -> np.ndarray:
    H = np.concatenate([h, np.ones((h.shape[0], 1))], axis=1)
    return np.argmax(H @ W, axis=1)


def evaluate(seed: int, model: GenericQueryAgent, cfg: Gate2Config) -> dict:
    s_np, v_np, y_np, o_np = generate_query_sequences(seed, cfg.test_sequences, cfg.world)
    with torch.no_grad():
        pred_t, h_t = model(torch.from_numpy(s_np), torch.from_numpy(v_np))
    pred = pred_t.numpy()
    h = h_t.numpy()
    valid = np.arange(cfg.world.steps) >= 12
    mse = float(np.mean((pred[:, valid] - y_np[:, valid]) ** 2))
    var = float(np.var(y_np[:, valid]))

    cut = cfg.test_sequences // 2
    Htr = h[:cut, valid].reshape(-1, cfg.hidden_size)
    Otr = o_np[:cut, valid].reshape(-1)
    Hte = h[cut:, valid].reshape(-1, cfg.hidden_size)
    Ote = o_np[cut:, valid].reshape(-1)
    W = ridge_probe(Htr, Otr, cfg.world.n_channels)
    probe_acc = float(np.mean(probe_predict(Hte, W) == Ote))
    centroids = np.stack([Htr[Otr == i].mean(axis=0) for i in range(cfg.world.n_channels)])

    silent_t = np.array([_silent(int(t), cfg.world.silent_windows) for t in range(cfg.world.steps)]) & valid
    candidates = []
    rng = np.random.default_rng(seed + 999)
    for b in range(cut, cfg.test_sequences):
        for t in np.where(silent_t)[0]:
            a = int(o_np[b, t])
            choices = [i for i in range(cfg.world.n_channels) if i != a]
            bslot = int(rng.choice(choices))
            candidates.append((b, int(t), a, bslot))
    rng.shuffle(candidates)
    candidates = candidates[:3000]

    hs = torch.from_numpy(np.stack([centroids[bslot] for _, _, _, bslot in candidates]).astype(np.float32))
    qs = torch.from_numpy(np.stack([v_np[b, t] for b, t, _, _ in candidates]).astype(np.float32))
    with torch.no_grad():
        intervened = model.answer_from_hidden(hs, qs).numpy()

    toward_counterfactual = 0
    base_self_dist = []
    cf_dist = []
    for val, (b, t, a, bslot) in zip(intervened, candidates):
        da = abs(float(val) - float(v_np[b, t, a]))
        db = abs(float(val) - float(v_np[b, t, bslot]))
        base_self_dist.append(da)
        cf_dist.append(db)
        toward_counterfactual += int(db < da)

    return {
        "nmse": mse / max(var, 1e-12),
        "probe_accuracy": probe_acc,
        "intervention_counterfactual_rate": float(toward_counterfactual / max(len(candidates), 1)),
        "intervention_distance_to_original": float(np.mean(base_self_dist)),
        "intervention_distance_to_counterfactual": float(np.mean(cf_dist)),
    }


def run_gate2(seeds=(401, 402, 403), cfg: Gate2Config | None = None) -> dict:
    cfg = cfg or Gate2Config()
    rows = []
    for seed in seeds:
        model = train_agent(int(seed), cfg)
        rows.append({"seed": int(seed), **evaluate(int(seed) + 20000, model, cfg)})

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {
        "hidden_state_floats": cfg.hidden_size,
        "downstream_nmse": avg("nmse"),
        "linear_probe_self_accuracy": avg("probe_accuracy"),
        "counterfactual_intervention_rate": avg("intervention_counterfactual_rate"),
        "distance_to_original_after_intervention": avg("intervention_distance_to_original"),
        "distance_to_counterfactual_after_intervention": avg("intervention_distance_to_counterfactual"),
        "rows": rows,
    }
    checks = {
        "task_requires_persistent_state_and_is_learned": summary["downstream_nmse"] <= 0.35,
        "self_address_is_linearly_decodable": summary["linear_probe_self_accuracy"] >= 0.85,
        "hidden_state_is_causally_self_like": summary["counterfactual_intervention_rate"] >= 0.70,
    }
    summary["checks"] = checks
    summary["pass"] = bool(all(checks.values()))
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_gate2(), indent=2, sort_keys=True))
