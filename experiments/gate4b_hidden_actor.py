from __future__ import annotations

import argparse
import json

from whatisi.gate4b import Config, run_many


def main():
    ap = argparse.ArgumentParser(description="Gate 4B: hidden causal actor + deictic language")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--device", default=None, help="cpu, cuda, or omit for auto")
    ap.add_argument("--quick", action="store_true", help="smaller smoke run; not the recorded gate")
    args = ap.parse_args()

    cfg = Config()
    if args.quick:
        cfg.train_lives = 128
        cfg.test_lives = 64
        cfg.epochs = 4

    result = run_many(tuple(args.seeds), cfg=cfg, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
