from __future__ import annotations

from dataclasses import asdict
import json
import math
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
CAUSAL_CHANNEL = 0
SPEAKER_CHANNEL = 1
ADDRESSEE_CHANNEL = 2


class LexicalRouter(nn.Module):
    """Late lexical binder over three already-available address sources.

    The router never sees body identity directly. It sees only whether speech was
    externally heard or self-produced, plus I/YOU, and learns how much to trust:

      0 causal address carried by the frozen Gate-4B state
      1 current visible speaker
      2 current visible addressee
    """

    def __init__(self):
        super().__init__()
        self.source = nn.Embedding(2, 8)
        self.pronoun = nn.Embedding(2, 8)
        self.net = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 3))

    def weights(self, source: torch.Tensor, pronoun: torch.Tensor) -> torch.Tensor:
        q = torch.cat([self.source(source), self.pronoun(pronoun)], dim=-1)
        return torch.softmax(self.net(q), dim=-1)

    def forward(self, source: torch.Tensor, pronoun: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        w = self.weights(source, pronoun)
        return (w.unsqueeze(-1) * channels).sum(dim=1).clamp_min(1e-8)


def _mature_times(c: Config):
    return [t for t in range(4, c.steps) if t != c.transfer_step]


def make_late_queries(hidden: np.ndarray, data: dict, c: Config, seed: int, limit: int | None = None):
    """Create late I/YOU examples after the causal core already exists.

    A self-produced I query contains no externally supplied speaker identity.
    Its target is the current visible name of the body identified by Gate 4B's
    persistent causal state. External I and YOU use ordinary visible roles.
    """

    rng = np.random.default_rng(seed)
    rows = []
    for life in range(len(hidden)):
        for t in _mature_times(c):
            names = data["name"][life, t]
            shapes = data["shape"][life, t]
            self_shape = int(data["self_shape"][life, t])
            self_idx = int(np.where(shapes == self_shape)[0][0])
            source = int(rng.integers(2))
            pronoun = int(rng.integers(2))

            if source == SELF_PRODUCED:
                speaker = -1  # deliberately unavailable
                if pronoun == I_PRONOUN:
                    addressee = -1
                    target = int(names[self_idx])
                else:
                    options = [j for j in range(c.n_agents) if j != self_idx]
                    addressee_idx = int(rng.choice(options))
                    addressee = int(names[addressee_idx])
                    target = addressee
            else:
                speaker_idx = int(rng.integers(c.n_agents))
                speaker = int(names[speaker_idx])
                if pronoun == I_PRONOUN:
                    addressee = -1
                    target = speaker
                else:
                    options = [j for j in range(c.n_agents) if j != speaker_idx]
                    addressee_idx = int(rng.choice(options))
                    addressee = int(names[addressee_idx])
                    target = addressee

            rows.append((life, t, source, pronoun, speaker, addressee, target))

    if limit is not None and len(rows) > limit:
        take = rng.choice(len(rows), limit, replace=False)
        rows = [rows[int(i)] for i in take]
    return rows


def _collect(core, c: Config, seed: int, n: int):
    core.eval()
    result = rollout(core, c, seed, n, collect=True)
    return result["hidden"], result["data"]


def _address_channels(
    core,
    c: Config,
    hidden: np.ndarray,
    data: dict,
    rows,
    *,
    zero_memory: bool = False,
    memory_override: np.ndarray | None = None,
    chunk: int = 512,
):
    """Return [causal, visible-speaker, visible-addressee] distributions."""

    device = next(core.parameters()).device
    all_channels = []
    for lo in range(0, len(rows), chunk):
        part = rows[lo : lo + chunk]
        life = np.asarray([r[0] for r in part])
        time = np.asarray([r[1] for r in part])

        def take(name):
            return torch.as_tensor(data[name][life, time], device=device)

        shape = take("shape")
        name = take("name")
        voice = take("voice")
        pos = take("pos")
        motor = take("motor")
        # The old Gate-4B role query remains merely a nuisance input to its
        # frozen transformer scene encoder. Gate 4C's lexical router never sees it.
        sp = take("sp_name")
        ad = take("ad_name")
        old_pron = take("pron")
        ah, qh = core.scene(shape, name, voice, pos, motor, sp, ad, old_pron)

        if memory_override is not None:
            mem = torch.as_tensor(memory_override[lo : lo + len(part)], dtype=torch.float32, device=device)
        elif zero_memory:
            mem = torch.zeros(len(part), c.memory_dim, device=device)
        else:
            mem = torch.as_tensor(hidden[life, time], dtype=torch.float32, device=device)

        actor_logits, _ = core.heads(ah, qh, name, mem)
        causal = torch.softmax(actor_logits, dim=-1)
        speaker = torch.zeros(len(part), c.pool, device=device)
        addressee = torch.zeros(len(part), c.pool, device=device)
        for i, row in enumerate(part):
            if row[4] >= 0:
                speaker[i, row[4]] = 1.0
            if row[5] >= 0:
                addressee[i, row[5]] = 1.0
        all_channels.append(torch.stack([causal, speaker, addressee], dim=1))
    return torch.cat(all_channels, dim=0)


def train_router(core, c: Config, seed: int, labels: int = 256, steps: int = 300):
    """Freeze Gate 4B, then introduce the lexical router late."""

    for p in core.parameters():
        p.requires_grad_(False)
    core.eval()
    hidden, data = _collect(core, c, seed + 60000, 128)
    rows = make_late_queries(hidden, data, c, seed + 7, labels)
    device = next(core.parameters()).device
    with torch.no_grad():
        channels = _address_channels(core, c, hidden, data, rows)
    source = torch.tensor([r[2] for r in rows], dtype=torch.long, device=device)
    pronoun = torch.tensor([r[3] for r in rows], dtype=torch.long, device=device)
    target = torch.tensor([r[6] for r in rows], dtype=torch.long, device=device)

    router = LexicalRouter().to(device)
    opt = torch.optim.AdamW(router.parameters(), lr=3e-2)
    for _ in range(steps):
        probs = router(source, pronoun, channels)
        loss = F.nll_loss(torch.log(probs), target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return router


def _accuracy(pred, target, mask):
    return float(np.mean(pred[mask] == target[mask]))


@torch.no_grad()
def evaluate(core, router: LexicalRouter, c: Config, seed: int = 90000, n: int = 128):
    core.eval()
    router.eval()
    hidden, data = _collect(core, c, seed, n)
    rows = make_late_queries(hidden, data, c, seed + 1)
    device = next(core.parameters()).device
    source = torch.tensor([r[2] for r in rows], dtype=torch.long, device=device)
    pronoun = torch.tensor([r[3] for r in rows], dtype=torch.long, device=device)
    target = np.asarray([r[6] for r in rows])

    factual = _address_channels(core, c, hidden, data, rows)
    zeroed = _address_channels(core, c, hidden, data, rows, zero_memory=True)
    pred = router(source, pronoun, factual).argmax(dim=1).cpu().numpy()
    pred_zero = router(source, pronoun, zeroed).argmax(dim=1).cpu().numpy()
    src = np.asarray([r[2] for r in rows])
    pr = np.asarray([r[3] for r in rows])
    self_i = (src == SELF_PRODUCED) & (pr == I_PRONOUN)
    external_i = (src == EXTERNAL) & (pr == I_PRONOUN)
    you = pr == YOU_PRONOUN

    out = {
        "self_I_accuracy": _accuracy(pred, target, self_i),
        "zero_memory_self_I_accuracy": _accuracy(pred_zero, target, self_i),
        "external_I_accuracy": _accuracy(pred, target, external_i),
        "zero_memory_external_I_accuracy": _accuracy(pred_zero, target, external_i),
        "you_accuracy": _accuracy(pred, target, you),
        "zero_memory_you_accuracy": _accuracy(pred_zero, target, you),
    }

    for sname, s in (("external", EXTERNAL), ("self", SELF_PRODUCED)):
        for pname, p in (("I", I_PRONOUN), ("you", YOU_PRONOUN)):
            w = router.weights(
                torch.tensor([s], device=device), torch.tensor([p], device=device)
            )[0].cpu().numpy()
            out[f"router_{sname}_{pname}"] = [float(x) for x in w]

    out.update(_counterfactual_attack(core, router, c, hidden, data, seed + 1000))
    out["checks"] = {
        "late_self_I_works": out["self_I_accuracy"] > 0.90,
        "self_I_requires_causal_state": out["self_I_accuracy"] > 0.90
        and out["zero_memory_self_I_accuracy"] < 0.40,
        "external_deixis_survives_state_ablation": out["external_I_accuracy"] > 0.98
        and out["zero_memory_external_I_accuracy"] > 0.98
        and out["you_accuracy"] > 0.98
        and out["zero_memory_you_accuracy"] > 0.98,
        "lexical_router_attaches_roles": out["router_self_I"][CAUSAL_CHANNEL] > 0.80
        and out["router_external_I"][SPEAKER_CHANNEL] > 0.80
        and out["router_external_you"][ADDRESSEE_CHANNEL] > 0.80
        and out["router_self_you"][ADDRESSEE_CHANNEL] > 0.80,
        "state_swap_moves_linguistic_I": out["counterfactual_I_switch_rate"] > 0.80,
    }
    out["pass"] = bool(all(out["checks"].values()))
    return out


@torch.no_grad()
def _counterfactual_attack(core, router, c: Config, hidden, data, seed: int):
    """Same SELF+I query and same scene; swap only the old causal state."""

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
            # Literal query remains SELF-PRODUCED + I. No speaker/addressee ID appears.
            rows.append((life, t, SELF_PRODUCED, I_PRONOUN, -1, -1, cf_name))
            overrides.append(centroids[cf_shape])

    override = np.asarray(overrides, dtype="float32")
    channels = _address_channels(core, c, hidden, data, rows, memory_override=override)
    device = next(core.parameters()).device
    source = torch.full((len(rows),), SELF_PRODUCED, dtype=torch.long, device=device)
    pronoun = torch.full((len(rows),), I_PRONOUN, dtype=torch.long, device=device)
    target = np.asarray([r[6] for r in rows])
    pred = router(source, pronoun, channels).argmax(dim=1).cpu().numpy()
    return {
        "counterfactual_I_switch_rate": float(np.mean(pred == target)),
        "counterfactual_cases": int(len(rows)),
    }


def run(seed: int = 3, c: Config | None = None, device: str | None = None, labels: int = 256):
    c = c or Config()
    core, c = train_gate4b(seed, c, device)
    router = train_router(core, c, seed, labels=labels)
    return evaluate(core, router, c, seed=90000 + seed * 100)


def run_many(seeds=(3, 4, 5), c: Config | None = None, device: str | None = None, labels: int = 256):
    rows = []
    for seed in seeds:
        rows.append({"seed": int(seed), **run(int(seed), c, device, labels)})
    scalar_keys = [
        "self_I_accuracy",
        "zero_memory_self_I_accuracy",
        "external_I_accuracy",
        "zero_memory_external_I_accuracy",
        "you_accuracy",
        "zero_memory_you_accuracy",
        "counterfactual_I_switch_rate",
    ]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in scalar_keys}
    summary["rows"] = rows
    summary["pass"] = bool(all(r["pass"] for r in rows))
    return summary


if __name__ == "__main__":
    print(json.dumps(run_many(), indent=2, sort_keys=True))
