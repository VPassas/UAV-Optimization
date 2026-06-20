"""Step 3 deliverable: visualise the features for a sample UAV->user link.

Left panel : the city scene with the link drawn (buildings the ray passes
             through are highlighted red).
Right panel: the 14 feature values for that link.

Usage:
    python scripts/show_features.py --city Nafplio
    python scripts/show_features.py --city Heidelberg --seed 3
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from common.data_loader import generate_town_eubucco, CITY_REGISTRY
from common.channel import has_line_of_sight, compute_user_rate, dbm_to_watts
from student_a.features import (
    build_city_geometry, extract_features, FEATURE_NAMES,
)


def _pick_link(buildings, users, rng, area):
    """Pick a user and a UAV offset so the link is interesting (often NLoS)."""
    user = users[rng.integers(len(users))] if len(users) else \
        np.array([area / 2, area / 2, 1.5])
    ang = rng.uniform(0, 2 * np.pi)
    horiz = rng.uniform(150, 350)
    uav = np.array([np.clip(user[0] + horiz * np.cos(ang), 0, area),
                    np.clip(user[1] + horiz * np.sin(ang), 0, area),
                    rng.uniform(60, 140)])
    return uav, user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Nafplio", choices=list(CITY_REGISTRY))
    ap.add_argument("--area", type=float, default=1000.0)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    buildings, users, roads = generate_town_eubucco(args.city, area_size=args.area)
    geo = build_city_geometry(buildings)
    uav, user = _pick_link(buildings, users, rng, args.area)

    feats = extract_features(uav, user, geo, as_dataclass=True)
    fvec = feats.to_array()
    los = has_line_of_sight(tuple(uav), tuple(user), buildings)
    rate = compute_user_rate(tuple(uav), tuple(user), dbm_to_watts(30),
                             buildings=buildings, rng=rng, n_mc=30)

    # which buildings does the ray cross (for highlighting)
    from student_a.features import _segment_aabb_hits
    hit, _, _ = _segment_aabb_hits(np.asarray(uav, float), np.asarray(user, float), geo)

    fig, (axm, axb) = plt.subplots(1, 2, figsize=(17, 8),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # --- Map panel ---
    heights = np.array([b[4] for b in buildings])
    patches = [Rectangle((bx - bw/2, by - bd/2), bw, bd) for bx, by, bw, bd, _ in buildings]
    pc = PatchCollection(patches, cmap="viridis", alpha=0.85)
    pc.set_array(heights)
    axm.add_collection(pc)
    if hit.any():  # highlight blocking buildings
        hp = [patches[i] for i in np.where(hit)[0]]
        axm.add_collection(PatchCollection(hp, facecolor="none", edgecolor="red", lw=1.8, zorder=4))
    for road in roads:
        axm.plot([p[0] for p in road], [p[1] for p in road], color="0.6", lw=0.4, zorder=1)
    link_color = "lime" if los else "red"
    axm.plot([uav[0], user[0]], [uav[1], user[1]], "-", color=link_color, lw=2, zorder=5)
    axm.plot(*uav[:2], "^", color="white", mec="k", ms=14, zorder=6, label=f"UAV (z={uav[2]:.0f} m)")
    axm.plot(*user[:2], "o", color="cyan", mec="k", ms=11, zorder=6, label="user")
    fig.colorbar(pc, ax=axm, label="building height (m)", fraction=0.046)
    axm.set_xlim(0, args.area); axm.set_ylim(0, args.area); axm.set_aspect("equal")
    axm.set_title(f"{args.city}: sample link — {'LoS' if los else 'NLoS'}, "
                  f"true rate {rate:.2f} b/s/Hz")
    axm.set_xlabel("X (m)"); axm.set_ylabel("Y (m)"); axm.legend(loc="upper right")

    # --- Feature bar panel ---
    y = np.arange(len(FEATURE_NAMES))[::-1]
    axb.barh(y, fvec, color="steelblue")
    axb.set_yticks(y); axb.set_yticklabels(FEATURE_NAMES, fontsize=9)
    for yi, v in zip(y, fvec):
        axb.text(v, yi, f" {v:.2f}", va="center", fontsize=8)
    axb.set_title("extract_features(...) -> 14 values")
    axb.set_xlabel("feature value (raw units, pre-normalisation)")
    axb.grid(axis="x", alpha=0.3)

    fig.suptitle("Step 3 — urban-geometry features for one UAV->user link", fontsize=14)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "figures" / \
        f"features_{args.city.lower().replace(' ', '_')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")
    print(f"link: {'LoS' if los else 'NLoS'} | true rate {rate:.2f} b/s/Hz | "
          f"ray crosses {int(feats.ray_n_intersections)} buildings")


if __name__ == "__main__":
    main()
