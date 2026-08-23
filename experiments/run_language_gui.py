from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path

from whatisi.language_teacher import LanguageTeacher, ScriptedBackend, LlamaCppBackend, OllamaBackend
from whatisi.language_life import LanguageLife, LifeConfig


class App:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.q = queue.Queue()
        self.running = False
        root.title("WhatIsI — Language Life")

        top = ttk.Frame(root, padding=8)
        top.pack(fill="x")
        self.status = tk.StringVar(value="ready")
        ttk.Button(top, text="Start", command=self.start).pack(side="left")
        ttk.Button(top, text="Stop", command=self.stop).pack(side="left", padx=4)
        ttk.Button(top, text="Save", command=self.save).pack(side="left")
        ttk.Label(top, textvariable=self.status).pack(side="left", padx=12)

        self.text = tk.Text(root, width=112, height=34, wrap="word")
        self.text.pack(fill="both", expand=True, padx=8, pady=8)
        self.life = self._build()
        self.root.after(100, self.poll)

    def _build(self):
        if self.args.teacher == "llama":
            backend = LlamaCppBackend(self.args.model, n_gpu_layers=self.args.gpu_layers)
        elif self.args.teacher == "ollama":
            backend = OllamaBackend(self.args.ollama_model)
        else:
            backend = ScriptedBackend()

        teacher = LanguageTeacher(backend, seed=self.args.seed)
        if Path(self.args.bank).exists():
            teacher.load_bank(self.args.bank)
        if self.args.teacher != "scripted":
            try:
                teacher.refresh(allow_deixis=False)
                teacher.save_bank(self.args.bank)
            except Exception as e:
                self.q.put(f"teacher refresh warning: {e}\n")

        cfg = LifeConfig(deixis_after=(10**12 if self.args.deixis_after < 0 else self.args.deixis_after))
        life = LanguageLife(teacher, cfg, seed=self.args.seed, device=self.args.device)
        if self.args.resume and Path(self.args.checkpoint).exists():
            life.load_checkpoint(self.args.checkpoint)
        return life

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self.worker, daemon=True).start()

    def stop(self):
        self.running = False

    def worker(self):
        while self.running:
            row = self.life.turn(True)
            self.q.put(
                f"\n[{row['step']}] heard: {row['heard']}\n"
                f"    action: {row['action']}  target: {row['target']}  {'✓' if row['correct'] else '·'}\n"
                f"  feedback: {row['feedback']}\n"
                f"  acc={row['accuracy']:.3f} loss={row['loss']:.3f} memory={row['memory_norm']:.2f} source-gap={row['source_gap']:+.3f}\n"
            )
            if row["step"] % 100 == 0:
                self.life.checkpoint(self.args.checkpoint)

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self.text.insert("end", msg)
                self.text.see("end")
        except queue.Empty:
            pass
        self.status.set(f"step {self.life.step} | {'running' if self.running else 'stopped'}")
        self.root.after(100, self.poll)

    def save(self):
        self.life.checkpoint(self.args.checkpoint)
        self.life.save_log(self.args.log)
        self.life.teacher.save_bank(self.args.bank)
        self.status.set(f"saved step {self.life.step}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=["scripted", "llama", "ollama"], default="scripted")
    ap.add_argument("--model")
    ap.add_argument("--ollama-model", default="phi3:mini")
    ap.add_argument("--gpu-layers", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--device")
    ap.add_argument("--deixis-after", type=int, default=2000)
    ap.add_argument("--checkpoint", default="runs/language_life.pt")
    ap.add_argument("--log", default="runs/language_life.jsonl")
    ap.add_argument("--bank", default="runs/teacher_bank.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.teacher == "llama" and not args.model:
        raise SystemExit("--model GGUF_PATH required")
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
