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
        backend = LlamaCppBackend(args.model, n_gpu_layers=args.gpu_layers, n_threads=args.threads)
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

    cfg = LifeConfig(deixis_after=(10**12 if args.deixis_after < 0 else args.deixis_after))
    life = LanguageLife(teacher, cfg, seed=args.seed, device=args.device)
    if args.resume and Path(args.checkpoint).exists():
        life.load_checkpoint(args.checkpoint)
        print(f"resumed step {life.step}")

    for _ in range(args.steps):
        row = life.turn(train=True)
        if row["step"] <= 20 or row["step"] % 20 == 0:
            print(
                f"{row['step']:6d} acc={row['accuracy']:.3f} loss={row['loss']:.3f} "
                f"mem={row['memory_norm']:.2f} srcgap={row['source_gap']:+.3f} | "
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
    print(json.dumps({"step": life.step, **life.stats}, indent=2))


if __name__ == "__main__":
    main()
