from __future__ import annotations

import argparse
import json
from pathlib import Path

from whatisi.language_teacher import LanguageTeacher, ScriptedBackend, LlamaCppBackend, OllamaBackend
from whatisi.language_life import LanguageLife, LifeConfig


def make_teacher(args):
    if args.teacher == "scripted":
        backend = ScriptedBackend()
    elif args.teacher == "llama":
        if not args.model:
            raise SystemExit("--model PATH_TO_GGUF is required for --teacher llama")
        try:
            backend = LlamaCppBackend(args.model, n_gpu_layers=args.gpu_layers, n_threads=args.threads)
        except OSError as e:
            msg = str(e)
            extra = ""
            if "0xc000001d" in msg.lower() or "-1073741795" in msg:
                extra = (
                    "\n\nWindows raised 0xc000001d (illegal CPU instruction) while loading the "
                    "prebuilt llama-cpp-python binary. This is below WhatIsI and usually means "
                    "that the wheel was compiled for an instruction/backend combination that this "
                    "machine cannot execute.\n\n"
                    "Recommended Windows fallback: import the same GGUF into Ollama with the repo's "
                    "Modelfile.phi3, then run:\n"
                    "  ollama create whatisi-phi -f Modelfile.phi3\n"
                    "  python experiments/gate4a_language_life.py --teacher ollama "
                    "--ollama-model whatisi-phi --steps 5000\n"
                )
            raise SystemExit(f"Could not initialize llama-cpp teacher: {e}{extra}") from e
    else:
        backend = OllamaBackend(args.ollama_model, args.ollama_host)
    teacher = LanguageTeacher(backend, seed=args.seed, refresh_size=args.refresh_size)
    if args.bank and Path(args.bank).exists():
        teacher.load_bank(args.bank)
    return teacher


def main():
    ap = argparse.ArgumentParser(description="Long-running language provenance / deictic-life experiment")
    ap.add_argument("--teacher", choices=["scripted", "llama", "ollama"], default="scripted")
    ap.add_argument("--model", help="GGUF path for llama-cpp teacher")
    ap.add_argument("--ollama-model", default="phi3:mini")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--gpu-layers", type=int, default=-1)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--device")
    ap.add_argument("--refresh-every", type=int, default=250)
    ap.add_argument("--refresh-size", type=int, default=24)
    ap.add_argument("--deixis-after", type=int, default=2000, help="allow I/you teacher phrasing after this many turns; -1 disables")
    ap.add_argument("--epsilon-action", type=float, default=0.10, help="forced random-action probability during life; use 0 for a no-exploration run")
    ap.add_argument("--sample-temperature", type=float, default=0.85, help="student action sampling temperature; use 0 for greedy actions")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--checkpoint", default="runs/language_life.pt")
    ap.add_argument("--log", default="runs/language_life.jsonl")
    ap.add_argument("--bank", default="runs/teacher_bank.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    teacher = make_teacher(args)
    if args.teacher != "scripted":
        try:
            n = teacher.refresh(allow_deixis=False)
            print(f"teacher refresh: +{n} paraphrases")
            teacher.save_bank(args.bank)
        except Exception as e:
            print(f"teacher refresh failed; existing/scripted bank remains usable: {e}")

    cfg = LifeConfig(
        deixis_after=(10**12 if args.deixis_after < 0 else args.deixis_after),
        epsilon_action=max(0.0, min(1.0, args.epsilon_action)),
        sample_temperature=max(0.0, args.sample_temperature),
    )
    life = LanguageLife(teacher, cfg, seed=args.seed, device=args.device)
    if args.resume and Path(args.checkpoint).exists():
        life.load_checkpoint(args.checkpoint)
        print(f"resumed step {life.step}")

    last = None
    for _ in range(args.steps):
        row = life.turn(train=True)
        last = row
        if row["step"] <= 20 or row["step"] % 20 == 0:
            print(
                f"{row['step']:6d} recent={row['recent_accuracy']:.3f} all={row['accuracy']:.3f} "
                f"loss={row['loss']:.3f} mem={row['memory_norm']:.2f} srcgap={row['source_gap']:+.3f} | "
                f"OBS {row['heard']!r} -> ACT {row['action']!r} -> {row['feedback']}"
            )
        if args.teacher != "scripted" and args.refresh_every > 0 and row["step"] % args.refresh_every == 0:
            try:
                n = teacher.refresh(allow_deixis=row["deictic_language_enabled"])
                teacher.save_bank(args.bank)
                print(f"teacher refresh at {row['step']}: +{n}")
            except Exception as e:
                print(f"teacher refresh failed: {e}")
        if args.save_every > 0 and row["step"] % args.save_every == 0:
            life.checkpoint(args.checkpoint)
            life.save_log(args.log)

    life.checkpoint(args.checkpoint)
    life.save_log(args.log)
    teacher.save_bank(args.bank)
    summary = {"step": life.step, **life.stats}
    if last is not None:
        summary["recent_accuracy"] = last["recent_accuracy"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
