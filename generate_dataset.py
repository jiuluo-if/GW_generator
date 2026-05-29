"""
generate_dataset.py
===================
Command-line entry point.

Examples
--------
# Quick smoke test (200 samples, 1 worker)
python generate_dataset.py --total 200 --workers 1 --out ./tiny_dataset

# Full-scale generation, real glitches, 16 workers
python generate_dataset.py --total 30000 --workers 16 \
    --glitch-source gravityspy \
    --glitch-dir /path/to/gravityspy_timeseries \
    --out ./gw_yolo_real
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

from config import DEFAULT, Config
from pipeline import run_serial, run_parallel


def main(argv=None):
    ap = argparse.ArgumentParser(description="GW-YOLO synthetic data generator")
    ap.add_argument("--out", default=DEFAULT.dataset.output_root,
                    help="output dataset root")
    ap.add_argument("--total", type=int, default=DEFAULT.dataset.total_samples,
                    help="total number of samples to generate")
    ap.add_argument("--workers", type=int, default=DEFAULT.dataset.n_workers,
                    help="parallel workers (1 = serial mode)")
    ap.add_argument("--seed", type=int, default=None,
                    help="random seed (default: random based on time)")
    ap.add_argument("--glitch-source", choices=("simulated", "gravityspy"),
                    default=DEFAULT.glitch.source)
    ap.add_argument("--glitch-dir", default=DEFAULT.glitch.gravityspy_dir,
                    help="only used with --glitch-source gravityspy")
    ap.add_argument("--sample-rate", type=int, default=DEFAULT.strain.sample_rate)
    ap.add_argument("--duration", type=float, default=DEFAULT.strain.duration)
    ap.add_argument("--image-size", type=int, default=DEFAULT.image.width,
                    help="square image side in pixels")
    args = ap.parse_args(argv)

    # 随机种子
    if args.seed is None:
        args.seed = int(time.time() * 1000) % (2**31 - 1)
        print(f"Using random seed: {args.seed}")

    cfg = DEFAULT
    cfg = replace(
        cfg,
        strain=replace(cfg.strain,
                       sample_rate=args.sample_rate,
                       duration=args.duration),
        glitch=replace(cfg.glitch,
                       source=args.glitch_source,
                       gravityspy_dir=args.glitch_dir),
        image=replace(cfg.image,
                      width=args.image_size,
                      height=args.image_size),
        dataset=replace(cfg.dataset,
                        output_root=args.out,
                        seed=args.seed,
                        n_workers=args.workers,
                        total_samples=args.total),
    )

    os.makedirs(args.out, exist_ok=True)
    print(f"Output -> {os.path.abspath(args.out)}")
    print(f"Total samples: {cfg.dataset.total_samples}")

    if args.workers <= 1:
        run_serial(cfg)
    else:
        run_parallel(cfg)

    print("done.")


if __name__ == "__main__":
    sys.exit(main())
