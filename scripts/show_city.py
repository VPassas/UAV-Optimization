"""Load one city from EUBUCCO + OSM and show what came back.

Usage (with the uav env active):
    python scripts/show_city.py --city Nafplio
    python scripts/show_city.py --city Ermoupoli --map
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from common.data_loader import generate_town_eubucco, CITY_REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Nafplio", choices=list(CITY_REGISTRY),
                    help="which city to load")
    ap.add_argument("--area", type=float, default=1000.0, help="scene size in metres")
    ap.add_argument("--map", action="store_true", help="also save a PNG map")
    args = ap.parse_args()

    print(f"Loading {args.city} ...")
    buildings, users, roads = generate_town_eubucco(args.city, area_size=args.area)

    heights = np.array([b[4] for b in buildings])
    print("-" * 50)
    print(f"City:           {args.city}  (scene {args.area:.0f} x {args.area:.0f} m)")
    print(f"Buildings:      {len(buildings)}")
    print(f"  height (m):   mean {heights.mean():.1f} | "
          f"median {np.median(heights):.1f} | p90 {np.percentile(heights,90):.1f} | "
          f"max {heights.max():.1f}")
    print(f"Road segments:  {len(roads)}")
    print(f"Users:          {len(users)}")
    print("-" * 50)

    if args.map:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.collections import PatchCollection

        fig, ax = plt.subplots(figsize=(9, 9))
        patches = [Rectangle((bx - bw/2, by - bd/2), bw, bd)
                   for bx, by, bw, bd, _ in buildings]
        pc = PatchCollection(patches, cmap="viridis")
        pc.set_array(heights)
        ax.add_collection(pc)
        for road in roads:
            rx = [p[0] for p in road]; ry = [p[1] for p in road]
            ax.plot(rx, ry, color="0.6", lw=0.6, zorder=1)
        if len(users):
            ax.scatter(users[:, 0], users[:, 1], c="red", s=18, zorder=3, label="users")
        fig.colorbar(pc, ax=ax, label="building height (m)")
        ax.set_xlim(0, args.area); ax.set_ylim(0, args.area); ax.set_aspect("equal")
        ax.set_title(f"{args.city} — {len(buildings)} buildings")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.legend(loc="upper right")
        out = Path(__file__).resolve().parents[1] / "docs" / "figures" / \
            f"city_{args.city.lower().replace(' ', '_')}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"map saved -> {out}")


if __name__ == "__main__":
    main()
