from __future__ import annotations

from dataclasses import dataclass
import json
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .gate4b import Config, train as train_gate4b, rollout

EXTERNAL = 0
SELF_PRODUCED = 1
I_PRONOUN = 0
YOU_PRONOUN = 1


@dataclass
class GenericBinderConfig:
    labels: int = 256
    updates: int = 800
    batch_size: int = 128
    layers: int = 2
    lr: float = 3e-3


class GenericLateBinder(nn.Module):
    """Unfactorized late lexical binder.

    Unlike Gate 4C, this module is not handed three referent distributions
    (causal address / speaker / addressee), and it has no entity-selector
    readout. It receives:

      * the frozen Gate-4B current agent representations,
      * the raw 20-float persistent state as one generic token,
      * source + I/YOU lexical metadata,
      * visible speaker/addressee name IDs when the situation provides them.

    A generic transformer must combine these inputs and emit an 8-way current
    name class through an unconstrained MLP readout.
    """

    def __init__(self, c: Config, bc: GenericBinderConfig):
        super().__init__()
        d = c.d_model
        self.c = c
        self.bc = bc
        self.mem_proj = nn.Linear(c.memory_dim, d)
        self.source = nn.Embedding(2, d)
        self.pronoun = nn.Embedding(2, d)
        self.role_name = nn.Embedding(c.pool + 1, d)  # final id is UNKNOWN
        self.speaker_proj = nn.Linear(d, d, bias=False)
        self.addressee_proj = nn.Linear(d, d, bias=False)
        self.kind = nn.Embedding(4, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=c.n_heads,
            dim_feedforward=c.ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=bc.layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, c.pool))

    def forward(self, agent_h, memory, source, pronoun, speaker, addressee):
        B = len(memory)
        cls = self.cls.expand(B, -1, -1) + self.kind.weight[0].view(1, 1, -1)
        agents = agent_h + self.kind.weight[1].view(1, 1, -1)
        mem = self.mem_proj(memory).unsqueeze(1) + self.kind.weight[2].view(1, 1, -1)
        query = (
            self.source(source)
            + self.pronoun(pronoun)
            + self.speaker_proj(self.role_name(speaker))
            + self.addressee_proj(self.role_name(addressee))
            + self.kind.weight[3]
        ).unsqueeze(1)
        h = self.norm(self.encoder(torch.cat([cls, agents, mem, query], dim=1)))
        return self.out(h[:, 0])


def _mature_times(c: Config):
    return [t for t in range(4, c.steps) if t != c.transfer_step]


def _collect(core, c: Config, seed: int, n: int):
    core.eval()
    result = rollout(core, c, seed, n, collect=True)
    return result["hidden"], result["data"]


def _all_query_rows(hidden: np.ndarray, data: dict, c: Config, seed: int):
    """Generate all four lexical situations for each mature scene.

    Row fields:
      life, time, source, pronoun, visible_speaker_name_or_-1,
      visible_addressee_name_or_-1, target_name
    """
    rng = np.random.default_rng(seed)
    buckets = {(s, p): [] for s in (EXTERNAL, SELF_PRODUCED) for p in (I_PRONOUN, YOU_PRONOUN)}
    for life in range(len(hidden)):
        for t in _mature_times(c):
            names = data["name"][life, t]
            shapes = data["shape"][life, t]
            self_shape = int(data["self_shape"][life, t])
            self_idx = int(np.where(shapes == self_shape)[0][0])

            # External I: visible speaker is the referent.
            speaker_idx = int(rng.integers(c.n_agents))
            speaker_name = int(names[speaker_idx])
            buckets[(EXTERNAL, I_PRONOUN)].append(
                (life, t, EXTERNAL, I_PRONOUN, speaker_name, -1, speaker_name)
            )

            # External YOU: visible addressee is the referent.
            speaker_idx = int(rng.integers(c.n_agents))
            options = [j for j in range(c.n_agents) if j != speaker_idx]
            addressee_idx = int(rng.choice(options))
            buckets[(EXTERNAL, YOU_PRONOUN)].append(
                (
                    life,
                    t,
                    EXTERNAL,
                    YOU_PRONOUN,
                    int(names[speaker_idx]),
                    int(names[addressee_idx]),
                    int(names[addressee_idx]),
                )
            )

            # Self-produced I: speaker/body identity is deliberately unavailable.
            buckets[(SELF_PRODUCED, I_PRONOUN)].append(
                (life, t, SELF_PRODUCED, I_PRONOUN, -1, -1, int(names[self_idx]))
            )

            # Self-produced YOU: self speaker remains unavailable; addressee is visible.
            options = [j for j in range(c.n_agents) if j != self_idx]
            addressee_idx = int(rng.choice(options))
            buckets[(SELF_PRODUCED, YOU_PRONOUN)].append(
                (
                    life,
                    t,
                    SELF_PRODUCED,
                    YOU_PRONOUN,
                    -1,
                    int(names[addressee_idx]),
                    int(names[addressee_idx]),
                )
            )
    return buckets


def _balanced_rows(hidden, data, c: Config, seed: int, labels: int):
    if labels < 4:
        raise ValueError("labels must be >= 4")
    buckets = _all_query_rows(hidden, data, c, seed)
    rng = np.random.default_rng(seed + 1)
    per = labels // 4
    rows = []
    for key in ((EXTERNAL, I_PRONOUN), (EXTERNAL, YOU_PRONOUN), (SELF_PRODUCED, I_PRONOUN), (SELF_PRODUCED, YOU_PRONOUN)):
        candidates = buckets[key]
        take = min(per, len(candidates))
        idx = rng.choice(len(candidates), take, replace=False)
        rows.extend(candidates[int(i)] for i in idx)
    rng.shuffle(rows)
    return rows


def _eval_rows(hidden, data, c: Config, seed: int):
    buckets = _all_query_rows(hidden, data, c, seed)
    rows = []
    for key in ((EXTERNAL, I_PRONOUN), (EXTERNAL, YOU_PRONOUN), (SELF_PRODUCED, I_PRONOUN), (SELF_PRODUCED, YOU_PRONOUN)):
        rows.extend(buckets[key])
    return rows


@torch.no_grad()
def _inputs(core, c: Config, hidden, data, rows, *, zero_memory=False, memory_override=None):
    device = next(core.parameters()).device
    life = np.asarray([r[0] for r in rows])
    time = np.asarray([r[1] for r in rows])

    def take(name):
        return torch.as_tensor(data[name][life, time], device=device)

    # Reuse the frozen Gate-4B current-scene encoder. Its old role query is a
    # nuisance input; the new binder never receives Gate-4B's pronoun readout.
    agent_h, _ = core.scene(
        take("shape"),
        take("name"),
        take("voice"),
        take("pos"),
        take("motor"),
        take("sp_name"),
        take("ad_name"),
        take("pron"),
    )

    if memory_override is not None:
        memory = torch.as_tensor(memory_override, dtype=torch.float32, device=device)
    elif zero_memory:
        memory = torch.zeros(len(rows), c.memory_dim, device=device)
    else:
        memory = torch.as_tensor(hidden[life, time], dtype=torch.float32, device=device)

    unknown = c.pool
    source = torch.tensor([r[2] for r in rows], dtype=torch.long, device=device)
    pronoun = torch.tensor([r[3] for r in rows], dtype=torch.long, device=device)
    speaker = torch.tensor([r[4] if r[4] >= 0 else unknown for r in rows], dtype=torch.long, device=device)
    addressee = torch.tensor([r[5] if r[5] >= 0 else unknown for r in rows], dtype=torch.long, device=device)
    target = torch.tensor([r[6] for r in rows], dtype=torch.long, device=device)
    return agent_h.detach(), memory, source, pronoun, speaker, addressee, target


def train_binder(core, c: Config, seed: int, bc: GenericBinderConfig | None = None):
    bc = bc or GenericBinderConfig()
    for p in core.parameters():
        p.requires_grad_(False)
    core.eval()

    # Enough frozen worlds for balanced late labels, including stress runs.
    n_lives = max(128, int(np.ceil(bc.labels / max(1, len(_mature_times(c))))))
    hidden, data = _collect(core, c, 60000 + seed * 1000, n_lives)
    rows = _balanced_rows(hidden, data, c, seed * 100 + 34, bc.labels)
    cached = _inputs(core, c, hidden, data, rows)

    torch.manual_seed(seed * 100 + 34)
    np.random.seed(seed * 100 + 34)
    random.seed(seed * 100 + 34)
    binder = GenericLateBinder(c, bc).to(next(core.parameters()).device)
    opt = torch.optim.AdamW(binder.parameters(), lr=bc.lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed * 100 + 34)
    N = len(rows)
    agent_h, memory, source, pronoun, speaker, addressee, target = cached

    binder.train(True)
    for _ in range(bc.updates):
        idx = torch.as_tensor(
            rng.integers(0, N, min(bc.batch_size, N)),
            dtype=torch.long,
            device=target.device,
        )
        logits = binder(
            agent_h[idx], memory[idx], source[idx], pronoun[idx], speaker[idx], addressee[idx]
        )
        loss = F.cross_entropy(logits, target[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(binder.parameters(), 1.0)
        opt.step()
    return binder


def _predict(binder, tensors, chunk=1024):
    binder.eval()
    out = []
    with torch.no_grad():
        for lo in range(0, len(tensors[-1]), chunk):
            hi = min(len(tensors[-1]), lo + chunk)
            out.append(binder(*(x[lo:hi] for x in tensors[:-1])).argmax(dim=1).cpu())
    return torch.cat(out).numpy()


def _acc(pred, target, mask):
    return float(np.mean(pred[mask] == target[mask]))


@torch.no_grad()
def _counterfactual_attack(core, binder, c: Config, hidden, data, seed: int):
    """Same scene and literal SELF+I; replace only the old persistent state."""
    rng = np.random.default_rng(seed)
    half = len(hidden) // 2
    train_h = []
    train_shape = []
    for life in range(half):
        for t in _mature_times(c):
            train_h.append(hidden[life, t])
            train_shape.append(int(data["self_shape"][life, t]))
    train_h = np.asarray(train_h)
    train_shape = np.asarray(train_shape)
    centroids = {
        s: train_h[train_shape == s].mean(axis=0).astype("float32")
        for s in range(c.pool)
        if np.any(train_shape == s)
    }

    rows = []
    overrides = []
    for life in range(half, len(hidden)):
        for t in _mature_times(c):
            shapes = data["shape"][life, t]
            names = data["name"][life, t]
            true_shape = int(data["self_shape"][life, t])
            options = [int(s) for s in shapes if int(s) != true_shape and int(s) in centroids]
            if not options:
                continue
            cf_shape = int(rng.choice(options))
            cf_name = int(names[np.where(shapes == cf_shape)[0][0]])
            rows.append((life, t, SELF_PRODUCED, I_PRONOUN, -1, -1, cf_name))
            overrides.append(centroids[cf_shape])

    tensors = _inputs(
        core,
        c,
        hidden,
        data,
        rows,
        memory_override=np.asarray(overrides, dtype="float32"),
    )
    pred = _predict(binder, tensors)
    target = tensors[-1].cpu().numpy()
    return {
        "counterfactual_I_switch_rate": float(np.mean(pred == target)),
        "counterfactual_cases": int(len(rows)),
    }


def evaluate(core, binder, c: Config, seed: int, n: int = 128):
    hidden, data = _collect(core, c, seed, n)
    rows = _eval_rows(hidden, data, c, seed + 1)
    factual = _inputs(core, c, hidden, data, rows)
    zeroed = _inputs(core, c, hidden, data, rows, zero_memory=True)
    pred = _predict(binder, factual)
    pred_zero = _predict(binder, zeroed)
    target = factual[-1].cpu().numpy()
    src = np.asarray([r[2] for r in rows])
    pr = np.asarray([r[3] for r in rows])
    self_i = (src == SELF_PRODUCED) & (pr == I_PRONOUN)
    external_i = (src == EXTERNAL) & (pr == I_PRONOUN)
    you = pr == YOU_PRONOUN

    out = {
        "self_I_accuracy": _acc(pred, target, self_i),
        "zero_memory_self_I_accuracy": _acc(pred_zero, target, self_i),
        "external_I_accuracy": _acc(pred, target, external_i),
        "zero_memory_external_I_accuracy": _acc(pred_zero, target, external_i),
        "you_accuracy": _acc(pred, target, you),
        "zero_memory_you_accuracy": _acc(pred_zero, target, you),
    }
    out.update(_counterfactual_attack(core, binder, c, hidden, data, seed + 1000))
    out["checks"] = {
        "generic_binder_learns_visible_roles": out["external_I_accuracy"] > 0.95 and out["you_accuracy"] > 0.95,
        "generic_binder_learns_self_I": out["self_I_accuracy"] > 0.90,
        "self_I_selectively_requires_memory": out["zero_memory_self_I_accuracy"] < 0.40
        and out["zero_memory_external_I_accuracy"] > 0.95
        and out["zero_memory_you_accuracy"] > 0.95,
        "state_swap_moves_linguistic_I": out["counterfactual_I_switch_rate"] > 0.80,
    }
    out["pass"] = bool(all(out["checks"].values()))
    return out


def run(seed: int = 3, c: Config | None = None, device: str | None = None, bc: GenericBinderConfig | None = None):
    c = c or Config()
    core, c = train_gate4b(seed, c, device)
    binder = train_binder(core, c, seed, bc)
    return evaluate(core, binder, c, seed=90000 + seed * 100)


def run_many(seeds=(3, 4, 5), c: Config | None = None, device: str | None = None, bc: GenericBinderConfig | None = None):
    bc = bc or GenericBinderConfig()
    rows = []
    for seed in seeds:
        rows.append({"seed": int(seed), **run(int(seed), c, device, bc)})
    keys = [
        "self_I_accuracy",
        "zero_memory_self_I_accuracy",
        "external_I_accuracy",
        "zero_memory_external_I_accuracy",
        "you_accuracy",
        "zero_memory_you_accuracy",
        "counterfactual_I_switch_rate",
    ]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary["binder"] = {
        "labels": bc.labels,
        "updates": bc.updates,
        "batch_size": bc.batch_size,
        "layers": bc.layers,
    }
    summary["rows"] = rows
    summary["pass"] = bool(all(r["pass"] for r in rows))
    summary["verdict"] = (
        "GENERIC_BINDER_REDISTCOVERS_DEICTIC_JOIN"
        if summary["pass"]
        else "GENERIC_BINDER_INSUFFICIENT_IN_MATCHED_LOW_DATA_REGIME"
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(run_many(), indent=2, sort_keys=True))
