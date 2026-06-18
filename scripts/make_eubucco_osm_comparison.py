"""Step 2 deliverable: EUBUCCO-vs-OSM comparison plot.

Shows why EUBUCCO is the right primary source: OSM building footprints are
mostly missing a usable height tag (they fall back to a constant ~10 m), while
EUBUCCO provides a realistic, continuous height distribution.

Usage:
    conda run -n uav python scripts/make_eubucco_osm_comparison.py --city Ermoupoli
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

import osmnx as ox
import geopandas as gpd

from common.data_loader import generate_town_eubucco, CITY_REGISTRY, _to_3035

OSM_FALLBACK_H = 10.0  # constant used when a building carries no height/levels tag


def _building_patches(buildings):
    return [Rectangle((bx - bw / 2, by - bd / 2), bw, bd) for bx, by, bw, bd, _ in buildings]


def _osm_buildings_aligned(center_lonlat, x0, y0, area):
    """Fetch OSM buildings on the SAME patch of ground as the EUBUCCO scene.

    Returns (buildings, frac_with_height_tag). Heights use the OSM `height` tag,
    else `building:levels`*3, else the 10 m fallback (this is exactly the gap
    EUBUCCO fills). frac_with_height_tag = share that had a real height/levels tag.
    """
    lon, lat = center_lonlat
    gdf = ox.features_from_point((lat, lon), tags={"building": True},
                                 dist=area / 2 * 1.5)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].to_crs("EPSG:3035")
    buildings, tagged = [], 0
    for _, row in gdf.iterrows():
        minx, miny, maxx, maxy = row.geometry.bounds
        bx, by = (minx + maxx) / 2 - x0, (miny + maxy) / 2 - y0
        bw, bd = maxx - minx, maxy - miny
        if not (0 < bx < area and 0 < by < area and bw > 1 and bd > 1):
            continue
        bh, has_tag = OSM_FALLBACK_H, False
        for attr, scale in (("height", 1.0), ("building:levels", 3.0)):
            val = row.get(attr)
            if val is not None and gpd.pd.notna(val):
                try:
                    bh = float(str(val).replace("m", "").strip()) * scale
                    has_tag = True
                    break
                except (ValueError, TypeError):
                    pass
        tagged += int(has_tag)
        buildings.append((bx, by, bw, bd, bh))
    frac = tagged / len(buildings) if buildings else float("nan")
    return buildings, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Ermoupoli", choices=list(CITY_REGISTRY))
    ap.add_argument("--area", type=float, default=1000.0)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- EUBUCCO scene (defines the centre + bbox both panels share) ---
    center = CITY_REGISTRY[args.city]["center"]
    cx, cy = _to_3035.transform(*center)
    x0, y0 = cx - args.area / 2, cy - args.area / 2
    eub_b, _, eub_roads = generate_town_eubucco(
        args.city, area_size=args.area, fetch_roads=True, num_users=1)
    eub_h = np.array([b[4] for b in eub_b])

    # --- OSM scene on the SAME patch of ground ---
    osm_b, osm_tag_frac = _osm_buildings_aligned(center, x0, y0, args.area)
    osm_h = np.array([b[4] for b in osm_b]) if len(osm_b) else np.array([])
    osm_missing = 1.0 - osm_tag_frac  # share relying on the 10 m fallback

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    vmax = np.percentile(eub_h, 99)

    # Panel 1: EUBUCCO footprints colored by height
    ax = axes[0]
    pc = PatchCollection(_building_patches(eub_b), cmap="viridis")
    pc.set_array(eub_h); pc.set_clim(0, vmax)
    ax.add_collection(pc)
    for road in eub_roads:
        rx = [p[0] for p in road]; ry = [p[1] for p in road]
        ax.plot(rx, ry, color="0.6", lw=0.5, zorder=1)
    fig.colorbar(pc, ax=ax, label="Building height (m)")
    ax.set_xlim(0, args.area); ax.set_ylim(0, args.area); ax.set_aspect("equal")
    ax.set_title(f"EUBUCCO v0.2 — {args.city}\n{len(eub_b)} buildings, real heights")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # Panel 2: OSM footprints colored by height (mostly constant fallback)
    ax = axes[1]
    if len(osm_b):
        pc2 = PatchCollection(_building_patches(osm_b), cmap="viridis")
        pc2.set_array(osm_h); pc2.set_clim(0, vmax)
        ax.add_collection(pc2)
        fig.colorbar(pc2, ax=ax, label="Building height (m)")
    ax.set_xlim(0, args.area); ax.set_ylim(0, args.area); ax.set_aspect("equal")
    ax.set_title(f"OSM (osmnx) — {args.city}\n{len(osm_b)} buildings, "
                 f"{osm_missing*100:.0f}% at {OSM_FALLBACK_H:.0f} m fallback")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # Panel 3: height distributions
    ax = axes[2]
    bins = np.linspace(0, max(vmax, 25), 30)
    ax.hist(eub_h, bins=bins, alpha=0.7, label=f"EUBUCCO (n={len(eub_h)})", color="tab:green")
    if len(osm_h):
        ax.hist(osm_h, bins=bins, alpha=0.5, label=f"OSM (n={len(osm_h)})", color="tab:red")
    ax.axvline(OSM_FALLBACK_H, color="tab:red", ls="--", lw=1, label="OSM fallback 10 m")
    ax.set_xlabel("Building height (m)"); ax.set_ylabel("count")
    ax.set_title("Height distribution\n(OSM lacks real heights)")
    ax.legend()

    fig.suptitle(f"EUBUCCO vs OSM building heights — {args.city} "
                 f"({args.area:.0f} m scene)", fontsize=14)
    fig.tight_layout()
    out = out_dir / f"eubucco_vs_osm_{args.city.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    print(f"EUBUCCO: {len(eub_b)} buildings, height mean {eub_h.mean():.1f} m, "
          f"p90 {np.percentile(eub_h,90):.1f} m")
    print(f"OSM:     {len(osm_b)} buildings, {osm_missing*100:.0f}% at fallback height")


if __name__ == "__main__":
    main()
