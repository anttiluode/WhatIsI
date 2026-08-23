from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import random
from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .language_teacher import ACTIONS, LanguageTeacher

SRC_OBS = 0
SRC_ACT = 1
SRC_FEEDBACK = 2
SRC_WORLD = 3
SRC_PAD = 4
N_SOURCES = 5

ACTION_SURFACE = {
    "say red": "red", "say blue": "blue", "say green": "green",
    "touch red": "touch red", "touch blue": "touch blue", "touch green": "touch green",
    "take red": "take red", "take blue": "take blue", "take green": "take green",
    "drop": "drop", "move left": "left", "move right": "right", "wait": "...",
}


@dataclass
class LifeConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff: int = 384
    memory_dim: int = 32
    max_tokens: int = 384
    context_events: int = 12
    dropout: float = 0.05
    lr: float = 2e-4
    weight_decay: float = 1e-4
    unroll: int = 8
    imitation_weight: float = 1.0
    causal_weight: float = 0.35
    echo_prob: float = 0.35
    sample_temperature: float = 0.85
    epsilon_action: float = 0.10
    deixis_after: int = 2000


@dataclass
class WorldState:
    position: int = 0
    held: str | None = None
    last_touched: str | None = None
    last_spoken: str | None = None

    def describe(self) -> str:
        pos = {-1: "left", 0: "middle", 1: "right"}[self.position]
        held = self.held if self.held is not None else "nothing"
        return f"Position: {pos}. Held: {held}."

    def valid_actions(self) -> list[str]:
        out = [a for a in ACTIONS if a != "drop"]
        if self.held is not None:
            out.append("drop")
        return out

    def apply(self, action: str) -> tuple[str, bool]:
        before = (self.position, self.held, self.last_touched, self.last_spoken)
        parts = action.split()
        if action.startswith("say "):
            self.last_spoken = parts[-1]
            msg = f"Spoken word: {parts[-1]}."
        elif action.startswith("touch "):
            self.last_touched = parts[-1]
            msg = f"The {parts[-1]} object was touched."
        elif action.startswith("take "):
            color = parts[-1]
            if self.held is None:
                self.held = color
                msg = f"The {color} object is now held."
            else:
                msg = "Nothing changed because something is already held."
        elif action == "drop":
            if self.held is None:
                msg = "Nothing was held, so nothing changed."
            else:
                old = self.held
                self.held = None
                msg = f"The {old} object was dropped."
        elif action == "move left":
            old = self.position
            self.position = max(-1, self.position - 1)
            msg = "Position changed left." if self.position != old else "Already at the left edge."
        elif action == "move right":
            old = self.position
            self.position = min(1, self.position + 1)
            msg = "Position changed right." if self.position != old else "Already at the right edge."
        else:
            msg = "Nothing changed."
        after = (self.position, self.held, self.last_touched, self.last_spoken)
        return msg, before != after


@dataclass
class Event:
    source: int
    text: str
    changed: int = 0


class ByteEventEncoder:
    @staticmethod
    def encode(events: list[Event], max_tokens: int):
        ids: list[int] = []
        src: list[int] = []
        for ev in events:
            b = ev.text.encode("utf-8", errors="replace") + b"\n"
            ids.extend(b)
            src.extend([ev.source] * len(b))
        ids = ids[-max_tokens:]
        src = src[-max_tokens:]
        if not ids:
            ids, src = [10], [SRC_PAD]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(src, dtype=torch.long)


class LifeTransformer(nn.Module):
    def __init__(self, cfg: LifeConfig, n_actions: int = len(ACTIONS)):
        super().__init__()
        self.cfg = cfg
        self.byte_emb = nn.Embedding(256, cfg.d_model)
        self.src_emb = nn.Embedding(N_SOURCES, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_tokens + 1, cfg.d_model)
        self.mem_to_token = nn.Linear(cfg.memory_dim, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.action_head = nn.Linear(cfg.d_model, n_actions)
        self.causal_head = nn.Linear(cfg.d_model, 2)
        self.mem_update = nn.GRUCell(cfg.d_model + n_actions + 2, cfg.memory_dim)

    def encode(self, ids: torch.Tensor, src: torch.Tensor, memory: torch.Tensor):
        if ids.dim() == 1:
            ids, src = ids.unsqueeze(0), src.unsqueeze(0)
        _, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.byte_emb(ids) + self.src_emb(src) + self.pos_emb(pos + 1)
        memtok = self.mem_to_token(memory).unsqueeze(1) + self.pos_emb.weight[0].view(1, 1, -1)
        h = self.encoder(torch.cat([memtok, x], dim=1))
        return self.norm(h[:, 0])

    def forward(self, ids, src, memory):
        h = self.encode(ids, src, memory)
        return self.action_head(h), self.causal_head(h), h

    def update_memory(self, h, action_idx: int, changed: int, memory):
        a = F.one_hot(torch.tensor([action_idx], device=h.device), len(ACTIONS)).float()
        c = F.one_hot(torch.tensor([int(changed)], device=h.device), 2).float()
        return self.mem_update(torch.cat([h, a, c], dim=-1), memory)


class LanguageLife:
    def __init__(self, teacher: LanguageTeacher, cfg: LifeConfig | None = None, seed: int = 0, device: str | None = None):
        self.cfg = cfg or LifeConfig()
        self.teacher = teacher
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = random.Random(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = LifeTransformer(self.cfg).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.memory = torch.zeros(1, self.cfg.memory_dim, device=self.device)
        self.world = WorldState()
        self.events: deque[Event] = deque(maxlen=self.cfg.context_events)
        self.step = 0
        self._losses: list[torch.Tensor] = []
        self.stats = {"correct": 0, "turns": 0, "loss": 0.0, "source_gap": 0.0}
        self.history: list[dict] = []
        self._log_cursor = 0

    def _encode(self, events: list[Event]):
        ids, src = ByteEventEncoder.encode(events, self.cfg.max_tokens)
        return ids.to(self.device), src.to(self.device)

    def _action_from_logits(self, logits: torch.Tensor, training: bool = True) -> int:
        if training and self.rng.random() < self.cfg.epsilon_action:
            return self.rng.randrange(len(ACTIONS))
        if training and self.cfg.sample_temperature > 0:
            p = torch.softmax(logits[0] / self.cfg.sample_temperature, dim=-1)
            return int(torch.multinomial(p, 1).item())
        return int(torch.argmax(logits[0]).item())

    def _source_gap_probe(self, phrase: str) -> float:
        base = list(self.events)
        with torch.no_grad():
            probs = []
            for source in (SRC_OBS, SRC_ACT):
                ids, src = self._encode(base + [Event(source, phrase)])
                _, c, _ = self.model(ids, src, self.memory.detach())
                probs.append(float(torch.softmax(c, -1)[0, 1].item()))
        return probs[1] - probs[0]

    def turn(self, train: bool = True) -> dict:
        self.model.train(train)
        target_action = self.teacher.choose_target(self.world.valid_actions())
        allow_deixis = self.step >= self.cfg.deixis_after
        heard = self.teacher.utterance(target_action, allow_deixis=allow_deixis)
        self.events.append(Event(SRC_WORLD, self.world.describe()))
        self.events.append(Event(SRC_OBS, heard))

        pre_action_events = list(self.events)
        ids, src = self._encode(pre_action_events)
        action_logits, _, _ = self.model(ids, src, self.memory)
        chosen_idx = self._action_from_logits(action_logits, training=train)
        chosen = ACTIONS[chosen_idx]
        spoken_surface = ACTION_SURFACE[chosen]
        target_idx = ACTIONS.index(target_action)

        # Same bytes, two provenances, before feedback exists.
        ids_a, src_a = self._encode(pre_action_events + [Event(SRC_ACT, spoken_surface)])
        _, causal_act, _ = self.model(ids_a, src_a, self.memory)
        ids_o, src_o = self._encode(pre_action_events + [Event(SRC_OBS, spoken_surface)])
        _, causal_obs, _ = self.model(ids_o, src_o, self.memory)

        # Only the ACT stream changes the world.
        self.events.append(Event(SRC_ACT, spoken_surface))
        feedback, changed = self.world.apply(chosen)
        self.events.append(Event(SRC_FEEDBACK, feedback, int(changed)))

        echoed = False
        if self.rng.random() < self.cfg.echo_prob:
            self.events.append(Event(SRC_OBS, spoken_surface))
            self.events.append(Event(SRC_FEEDBACK, "Repeated external words: no world action.", 0))
            echoed = True

        loss_action = F.cross_entropy(action_logits, torch.tensor([target_idx], device=self.device))
        loss_causal = 0.5 * (
            F.cross_entropy(causal_act, torch.tensor([int(changed)], device=self.device))
            + F.cross_entropy(causal_obs, torch.tensor([0], device=self.device))
        )
        loss = self.cfg.imitation_weight * loss_action + self.cfg.causal_weight * loss_causal
        if train:
            self._losses.append(loss)

        # Post-consequence stream updates persistent life-state.
        ids_after, src_after = self._encode(list(self.events))
        _, _, h_after = self.model(ids_after, src_after, self.memory)
        self.memory = self.model.update_memory(h_after, chosen_idx, int(changed), self.memory)

        self.step += 1
        correct = int(chosen == target_action)
        self.stats["correct"] += correct
        self.stats["turns"] += 1
        self.stats["loss"] = float(loss.detach().item())
        if self.step % 10 == 0:
            self.stats["source_gap"] = self._source_gap_probe(spoken_surface)

        if train and len(self._losses) >= self.cfg.unroll:
            total = torch.stack(self._losses).mean()
            self.opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self._losses.clear()
            self.memory = self.memory.detach()

        row = {
            "step": self.step,
            "heard": heard,
            "target": target_action,
            "action": spoken_surface,
            "action_semantic": chosen,
            "feedback": feedback,
            "changed": bool(changed),
            "correct": bool(correct),
            "echoed": echoed,
            "deictic_language_enabled": allow_deixis,
            "loss": float(loss.detach().item()),
            "accuracy": self.stats["correct"] / max(1, self.stats["turns"]),
            "memory_norm": float(self.memory.detach().norm().item()),
            "source_gap": float(self.stats["source_gap"]),
        }
        self.history.append(row)
        return row

    def checkpoint(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "memory": self.memory.detach().cpu(),
            "step": self.step,
            "world": asdict(self.world),
            "cfg": asdict(self.cfg),
            "stats": self.stats,
            "events": [asdict(ev) for ev in self.events],
        }, path)

    def load_checkpoint(self, path: str | Path):
        ck = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            self.opt.load_state_dict(ck["optimizer"])
        self.memory = ck["memory"].to(self.device)
        self.step = int(ck.get("step", 0))
        self.world = WorldState(**ck.get("world", {}))
        self.stats.update(ck.get("stats", {}))
        self.events.clear()
        for row in ck.get("events", []):
            self.events.append(Event(**row))

    def save_log(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.history[self._log_cursor:]
        if not rows:
            return
        with path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._log_cursor = len(self.history)
