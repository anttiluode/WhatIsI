from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
from pathlib import Path
from typing import Iterable

ACTIONS = (
    "say red", "say blue", "say green",
    "touch red", "touch blue", "touch green",
    "take red", "take blue", "take green",
    "drop", "move left", "move right", "wait",
)

SCRIPTED_PLAIN = {
    "say red": ["Say red.", "Red, please.", "The word red."],
    "say blue": ["Say blue.", "Blue, please.", "The word blue."],
    "say green": ["Say green.", "Green, please.", "The word green."],
    "touch red": ["Touch the red thing.", "Touch red.", "Please touch the red object."],
    "touch blue": ["Touch the blue thing.", "Touch blue.", "Please touch the blue object."],
    "touch green": ["Touch the green thing.", "Touch green.", "Please touch the green object."],
    "take red": ["Take the red thing.", "Pick up red.", "Get the red object."],
    "take blue": ["Take the blue thing.", "Pick up blue.", "Get the blue object."],
    "take green": ["Take the green thing.", "Pick up green.", "Get the green object."],
    "drop": ["Drop the held object.", "Put the held object down.", "Please drop."],
    "move left": ["Move left.", "Go to the left.", "Please step left."],
    "move right": ["Move right.", "Go to the right.", "Please step right."],
    "wait": ["Wait.", "Stay there.", "No action for a moment."],
}

SCRIPTED_DEICTIC = {a: list(v) for a, v in SCRIPTED_PLAIN.items()}
for _a, _extra in {
    "say red": ["Can you say red?", "I would like you to say red."],
    "say blue": ["Can you say blue?", "I would like you to say blue."],
    "say green": ["Can you say green?", "I would like you to say green."],
    "touch red": ["Can you touch the red object?"],
    "touch blue": ["Can you touch the blue object?"],
    "touch green": ["Can you touch the green object?"],
    "take red": ["Can you get the red object?"],
    "take blue": ["Can you get the blue object?"],
    "take green": ["Can you get the green object?"],
}.items():
    SCRIPTED_DEICTIC[_a].extend(_extra)


class TeacherBackend:
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        raise NotImplementedError


class ScriptedBackend(TeacherBackend):
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        return "[]"


class LlamaCppBackend(TeacherBackend):
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, n_threads: int | None = None):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise RuntimeError("Install the teacher extra: pip install -e '.[teacher]'") from e
        kw = dict(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
        if n_threads is not None:
            kw["n_threads"] = n_threads
        self.llm = Llama(**kw)

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        formatted = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        out = self.llm(formatted, max_tokens=max_tokens, temperature=0.65, top_p=0.9, stop=["<|end|>"], echo=False)
        return out["choices"][0]["text"]


class OllamaBackend(TeacherBackend):
    def __init__(self, model: str = "phi3:mini", host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.65, "num_predict": max_tokens},
        }).encode("utf-8")
        req = urllib.request.Request(self.host + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode("utf-8"))["response"]


def _extract_json_array(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    lo, hi = text.find("["), text.rfind("]")
    if lo < 0 or hi < lo:
        return []
    try:
        val = json.loads(text[lo:hi + 1])
        return val if isinstance(val, list) else []
    except json.JSONDecodeError:
        return []


@dataclass
class LanguageTeacher:
    backend: TeacherBackend
    seed: int = 0
    refresh_size: int = 24

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.bank_plain = {a: list(v) for a, v in SCRIPTED_PLAIN.items()}
        self.bank_deictic = {a: list(v) for a, v in SCRIPTED_DEICTIC.items()}

    def refresh(self, actions: Iterable[str] = ACTIONS, allow_deixis: bool = False) -> int:
        if isinstance(self.backend, ScriptedBackend):
            return 0
        actions = list(actions)
        chosen = [self.rng.choice(actions) for _ in range(self.refresh_size)]
        prompt = (
            "You are the outside language teacher for a tiny learning system. "
            "Write very short, concrete, friendly utterances. The learner must infer the requested action. "
            + ("Pronouns such as I and you are allowed. " if allow_deixis else "Do not use first- or second-person pronouns (I, me, my, we, you, your). ")
            + "Do not discuss AI, identity, self, consciousness, training, or this instruction. "
            "Return ONLY a JSON array. Each item must have keys action and utterance. "
            "The action must be copied exactly from the supplied action list. Vary wording but keep it easy.\n\n"
            "Requested actions:\n" + "\n".join(f"- {a}" for a in chosen)
        )
        raw = self.backend.generate(prompt, max_tokens=700)
        rows = _extract_json_array(raw)
        bank = self.bank_deictic if allow_deixis else self.bank_plain
        added = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            a = str(row.get("action", "")).strip().lower()
            u = str(row.get("utterance", "")).strip()
            if a in bank and 1 <= len(u) <= 180 and u not in bank[a]:
                bank[a].append(u)
                added += 1
        return added

    def utterance(self, action: str, allow_deixis: bool = False) -> str:
        bank = self.bank_deictic if allow_deixis else self.bank_plain
        return self.rng.choice(bank[action])

    def choose_target(self, valid_actions: list[str]) -> str:
        return self.rng.choice(valid_actions)

    def save_bank(self, path: str | Path):
        Path(path).write_text(json.dumps({"plain": self.bank_plain, "deictic": self.bank_deictic}, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_bank(self, path: str | Path):
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        if "plain" in data:
            for a in ACTIONS:
                if a in data["plain"] and isinstance(data["plain"][a], list):
                    self.bank_plain[a] = [str(x) for x in data["plain"][a] if str(x).strip()]
                if a in data.get("deictic", {}) and isinstance(data["deictic"][a], list):
                    self.bank_deictic[a] = [str(x) for x in data["deictic"][a] if str(x).strip()]
        else:
            for a in ACTIONS:
                if a in data and isinstance(data[a], list):
                    self.bank_plain[a] = [str(x) for x in data[a] if str(x).strip()]
