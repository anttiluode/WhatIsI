from __future__ import annotations

import argparse
import json

from whatisi.gate4b import Config
from whatisi.gate4d import GenericBinderConfig, run_many


def main():
    ap = argparse.ArgumentParser(description="Gate 4D: generic late binder without factorized address channels")
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--device", default=None, help="cpu, cuda, or omit for auto")
    ap.add_argument("--labels", type=int, default=256, help="late lexical labels")
    ap.add_argument("--updates", type=int, default=800, help="generic-binder optimizer updates")
    ap.add_argument("--layers", type=int, default=2, help="generic late transformer layers")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--quick", action="store_true", help="smaller smoke run; not the recorded gate")
    args = ap.parse_args()

    cfg = Config()
    if args.quick:
        cfg.train_lives = 128
        cfg.test_lives = 64
        cfg.epochs = 4

    bc = GenericBinderConfig(
        labels=args.labels,
        updates=args.updates,
        batch_size=args.batch_size,
        layers=args.layers,
    )
    result = run_many(tuple(args.seeds), c=cfg, device=args.device, bc=bc)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
