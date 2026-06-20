"""Step 4 — generate the (features, rate) dataset for one or all cities.

Usage:
    python scripts/generate_dataset.py --city Nafplio --n 3000
    python scripts/generate_dataset.py --city Nafplio --n 200 --jobs 1   # serial profile
    python scripts/generate_dataset.py --all --n 3000                    # full dataset
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common.data_loader import CITY_REGISTRY, generate_town_eubucco
from student_a.dataset import generate_city_dataset, save_city_dataset


def run_city(city, args):
    t_load = time.perf_counter()
    buildings, _, roads = generate_town_eubucco(
        city, area_size=args.area, fetch_roads=True, num_users=1)
    load_dt = time.perf_counter() - t_load

    t0 = time.perf_counter()
    df = generate_city_dataset(city, n_samples=args.n, n_mc=args.n_mc,
                               area_size=args.area, n_jobs=args.jobs, seed=args.seed,
                               buildings=buildings, roads=roads)
    dt = time.perf_counter() - t0
    path = save_city_dataset(df, city)

    r = df["rate"].to_numpy()
    print(f"[{city}] {len(df)} samples | gen {dt:.1f}s ({dt/len(df)*1000:.1f} ms/sample, "
          f"jobs={args.jobs}) | load {load_dt:.1f}s | {len(buildings)} buildings")
    print(f"        rate b/s/Hz: mean {r.mean():.2f} | min {r.min():.2f} | "
          f"max {r.max():.2f} | frac<0.1: {(r < 0.1).mean():.2%}")
    print(f"        saved -> {path}")
    return dt, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Nafplio", choices=list(CITY_REGISTRY))
    ap.add_argument("--all", action="store_true", help="generate all 20 cities")
    ap.add_argument("--n", type=int, default=3000, help="samples per city")
    ap.add_argument("--n-mc", type=int, default=30, dest="n_mc",
                    help="Monte-Carlo fading samples per link")
    ap.add_argument("--area", type=float, default=1000.0)
    ap.add_argument("--jobs", type=int, default=1,
                    help="1 = serial (fastest here); -1 = all cores (only helps heavy configs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cities = list(CITY_REGISTRY) if args.all else [args.city]
    total_dt, total_n = 0.0, 0
    t_all = time.perf_counter()
    for city in cities:
        dt, n = run_city(city, args)
        total_dt += dt; total_n += n
    if args.all:
        wall = time.perf_counter() - t_all
        print(f"\nALL: {total_n} samples across {len(cities)} cities | "
              f"wall {wall/60:.1f} min | gen {total_dt/60:.1f} min")


if __name__ == "__main__":
    main()
