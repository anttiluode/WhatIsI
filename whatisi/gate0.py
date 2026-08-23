from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class Gate0Config:
    n_channels: int = 4
    steps: int = 720
    control_gain: float = 2.8
    noise_std: float = 1.0
    distractor_std: float = 3.0
    ema: float = 0.94
    temperature: float = 0.55
    transfer_steps: tuple[int, ...] = (240, 480)
    silent_windows: tuple[tuple[int, int], ...] = ((130, 160), (360, 390), (600, 630))


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = x / max(temperature, 1e-6)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


class CausalSelfPointer:
    """Persistent deictic binding learned from action -> consequence contingency."""

    def __init__(self, n_channels: int, ema: float = 0.94, temperature: float = 0.55):
        self.n = n_channels
        self.ema = ema
        self.temperature = temperature
        self.score = np.zeros(n_channels, dtype=np.float64)

    def update(self, action: float, delta: np.ndarray) -> np.ndarray:
        if abs(action) > 1e-12:
            evidence = action * delta
            self.score = self.ema * self.score + (1.0 - self.ema) * evidence
        return self.belief

    @property
    def belief(self) -> np.ndarray:
        return softmax(self.score, self.temperature)


class InstantaneousCausal:
    """Attacker: correct cue, no persistent state."""

    def __init__(self, n_channels: int, temperature: float = 0.55):
        self.n = n_channels
        self.temperature = temperature
        self._belief = np.ones(n_channels) / n_channels

    def update(self, action: float, delta: np.ndarray) -> np.ndarray:
        if abs(action) <= 1e-12:
            self._belief = np.ones(self.n) / self.n
        else:
            self._belief = softmax(action * delta, self.temperature)
        return self._belief


class ActivityMemory:
    """Attacker: same-size persistent state, but tracks salience not agency."""

    def __init__(self, n_channels: int, ema: float = 0.94, temperature: float = 0.55):
        self.n = n_channels
        self.ema = ema
        self.temperature = temperature
        self.score = np.zeros(n_channels, dtype=np.float64)

    def update(self, action: float, delta: np.ndarray) -> np.ndarray:
        evidence = delta * delta
        self.score = self.ema * self.score + (1.0 - self.ema) * evidence
        return softmax(self.score, self.temperature)


def in_silent_window(t: int, windows: tuple[tuple[int, int], ...]) -> bool:
    return any(lo <= t < hi for lo, hi in windows)


def run_lifetime(seed: int, cfg: Gate0Config | None = None) -> dict:
    cfg = cfg or Gate0Config()
    rng = np.random.default_rng(seed)

    causal = CausalSelfPointer(cfg.n_channels, cfg.ema, cfg.temperature)
    instant = InstantaneousCausal(cfg.n_channels, cfg.temperature)
    activity = ActivityMemory(cfg.n_channels, cfg.ema, cfg.temperature)

    self_slot = int(rng.integers(cfg.n_channels))
    distractor_slot = (self_slot + 1) % cfg.n_channels

    names = np.arange(cfg.n_channels)
    appearances = np.arange(cfg.n_channels)

    rec = []
    transfer_recovery_active_steps = []
    recovering = False
    active_since_transfer = 0
    recovered_at = None

    for t in range(cfg.steps):
        if t in cfg.transfer_steps:
            old = self_slot
            choices = [i for i in range(cfg.n_channels) if i != old]
            self_slot = int(rng.choice(choices))
            distractor_slot = (self_slot + 1) % cfg.n_channels
            recovering = True
            active_since_transfer = 0
            recovered_at = None

        if t % 53 == 0 and t > 0:
            rng.shuffle(names)
        if t % 71 == 0 and t > 0:
            rng.shuffle(appearances)

        silent = in_silent_window(t, cfg.silent_windows)
        action = 0.0 if silent else float(rng.choice([-1.0, 1.0]))

        delta = rng.normal(0.0, cfg.noise_std, size=cfg.n_channels)
        delta[distractor_slot] += rng.normal(0.0, cfg.distractor_std)
        delta[self_slot] += cfg.control_gain * action

        values = rng.normal(0.0, 1.0, size=cfg.n_channels)
        target = float(values[self_slot])

        b_c = causal.update(action, delta)
        b_i = instant.update(action, delta)
        b_a = activity.update(action, delta)
        b_u = np.ones(cfg.n_channels) / cfg.n_channels

        pred_c = float(b_c @ values)
        pred_i = float(b_i @ values)
        pred_a = float(b_a @ values)
        pred_u = float(b_u @ values)

        hit_c = int(np.argmax(b_c) == self_slot)
        hit_i = int(np.argmax(b_i) == self_slot)
        hit_a = int(np.argmax(b_a) == self_slot)

        if recovering and not silent:
            active_since_transfer += 1
            if hit_c and b_c[self_slot] > 0.6 and recovered_at is None:
                recovered_at = active_since_transfer
                transfer_recovery_active_steps.append(recovered_at)
                recovering = False

        rec.append({
            "t": t,
            "self_slot": self_slot,
            "silent": silent,
            "action": action,
            "causal_hit": hit_c,
            "instant_hit": hit_i,
            "activity_hit": hit_a,
            "causal_conf": float(b_c[self_slot]),
            "mse_causal": (pred_c - target) ** 2,
            "mse_instant": (pred_i - target) ** 2,
            "mse_activity": (pred_a - target) ** 2,
            "mse_uniform": (pred_u - target) ** 2,
        })

    while len(transfer_recovery_active_steps) < len(cfg.transfer_steps):
        transfer_recovery_active_steps.append(cfg.steps)

    def mean(key, mask=None):
        rows = rec if mask is None else [r for r in rec if mask(r)]
        return float(np.mean([r[key] for r in rows]))

    burn = 50
    stable = lambda r: r["t"] >= burn and all(abs(r["t"] - ts) > 25 for ts in cfg.transfer_steps)
    silent_mask = lambda r: r["silent"] and r["t"] >= burn and all(abs(r["t"] - ts) > 25 for ts in cfg.transfer_steps)

    return {
        "seed": seed,
        "accuracy_causal": mean("causal_hit", stable),
        "accuracy_instant": mean("instant_hit", stable),
        "accuracy_activity": mean("activity_hit", stable),
        "silent_accuracy_causal": mean("causal_hit", silent_mask),
        "silent_accuracy_instant": mean("instant_hit", silent_mask),
        "mse_causal": mean("mse_causal", stable),
        "mse_instant": mean("mse_instant", stable),
        "mse_activity": mean("mse_activity", stable),
        "mse_uniform": mean("mse_uniform", stable),
        "recovery_active_steps": transfer_recovery_active_steps,
    }


def run_gate0(seeds=range(100, 140), cfg: Gate0Config | None = None) -> dict:
    rows = [run_lifetime(int(s), cfg) for s in seeds]

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    recovery = [x for r in rows for x in r["recovery_active_steps"]]
    summary = {
        "n_seeds": len(rows),
        "accuracy_causal": avg("accuracy_causal"),
        "accuracy_instant": avg("accuracy_instant"),
        "accuracy_activity": avg("accuracy_activity"),
        "silent_accuracy_causal": avg("silent_accuracy_causal"),
        "silent_accuracy_instant": avg("silent_accuracy_instant"),
        "mse_causal": avg("mse_causal"),
        "mse_instant": avg("mse_instant"),
        "mse_activity": avg("mse_activity"),
        "mse_uniform": avg("mse_uniform"),
        "mse_ratio_vs_uniform": avg("mse_causal") / avg("mse_uniform"),
        "median_recovery_active_steps": float(np.median(recovery)),
        "p90_recovery_active_steps": float(np.quantile(recovery, 0.9)),
    }

    checks = {
        "tracks_causal_owner": summary["accuracy_causal"] >= 0.90,
        "persists_without_new_action_evidence": summary["silent_accuracy_causal"] >= 0.90,
        "persistence_beats_instantaneous": summary["silent_accuracy_causal"] - summary["silent_accuracy_instant"] >= 0.50,
        "beats_salience_memory": summary["accuracy_causal"] - summary["accuracy_activity"] >= 0.35,
        "useful_for_private_value_query": summary["mse_ratio_vs_uniform"] <= 0.40,
        "recovers_after_agency_transfer": summary["median_recovery_active_steps"] <= 20.0,
    }
    summary["checks"] = checks
    summary["pass"] = bool(all(checks.values()))
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run_gate0(), indent=2, sort_keys=True))
