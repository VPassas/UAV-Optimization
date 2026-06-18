"""EUBUCCO data loader (Step 2 — co-owned by Student A and Student B).

``generate_town_eubucco(city_name)`` returns a town scene in the *same format*
as the reference simulator's ``generate_town_osm``::

    buildings : list[(cx, cy, w, d, h)]   axis-aligned boxes, local metres
    users     : np.ndarray (N, 3)         (x, y, z=0) on the road network
    roads     : list[list[(x, y)]]        polylines, local metres

Buildings + heights come from **EUBUCCO v0.2** (per-NUTS-region GeoPackage/
Parquet on the project S3, EPSG:3035, ~100% height coverage incl. Greece).
Roads come from **OpenStreetMap** via osmnx (EUBUCCO has no road network).
Everything is cropped to a square ``area_size`` scene around the city centre
and translated so coordinates live in ``[0, area_size]``.

Data source: Milojevic-Dupont et al., "EUBUCCO", Nature Scientific Data 2023.
License: ODbL. API: https://api.eubucco.com  (1-hour presigned S3 URLs).
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer

try:
    import osmnx as ox
    _OSMNX_OK = True
except Exception:  # pragma: no cover
    _OSMNX_OK = False

# --- Paths / constants ------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _REPO_ROOT / "data" / "raw" / "eubucco"
_API = "https://api.eubucco.com"
_EUBUCCO_CRS = "EPSG:3035"  # ETRS89-LAEA Europe, metres
_WGS84 = "EPSG:4326"

# WGS84 (lon, lat) -> EUBUCCO metres. always_xy => (easting, northing).
_to_3035 = Transformer.from_crs(_WGS84, _EUBUCCO_CRS, always_xy=True)


# --- City registry ----------------------------------------------------------
# center = (lon, lat) of the town centre; nuts = EUBUCCO v0.2 NUTS-2 partition.
# split: which side of the 15/5 train/test divide the city is on.
CITY_REGISTRY: dict[str, dict] = {
    # Mediterranean, low density
    "Nafplio":            {"center": (22.8069, 37.5676), "nuts": "EL65", "split": "train"},
    "Rethymno":           {"center": (24.4737, 35.3650), "nuts": "EL43", "split": "train"},
    "Karpenisi":          {"center": (21.7903, 38.9125), "nuts": "EL64", "split": "train"},
    "Kastoria":           {"center": (21.2683, 40.5167), "nuts": "EL53", "split": "train"},
    "Ermoupoli":          {"center": (24.9419, 37.4450), "nuts": "EL42", "split": "test"},
    # European historic centres
    "Bruges":             {"center": (3.2247, 51.2093),  "nuts": "BE25", "split": "train"},
    "Cesky Krumlov":      {"center": (14.3175, 48.8127), "nuts": "CZ03", "split": "train"},
    "Sintra":             {"center": (-9.3817, 38.7979), "nuts": "PT17", "split": "train"},
    "Heidelberg":         {"center": (8.6724, 49.3988),  "nuts": "DE12", "split": "train"},
    "Annecy":             {"center": (6.1294, 45.8992),  "nuts": "FRK2", "split": "test"},
    # Modern small cities
    "Eindhoven":          {"center": (5.4697, 51.4416),  "nuts": "NL41", "split": "train"},
    "Aarhus":             {"center": (10.2039, 56.1572), "nuts": "DK04", "split": "train"},
    "Tampere":            {"center": (23.7610, 61.4978), "nuts": "FI19", "split": "train"},
    "Tallinn":            {"center": (24.7536, 59.4370), "nuts": "EE00", "split": "test"},
    # Island / tourist towns
    "Mykonos":            {"center": (25.3289, 37.4467), "nuts": "EL42", "split": "train"},
    "Split":              {"center": (16.4402, 43.5081), "nuts": "HR03", "split": "train"},
    "Chania":             {"center": (24.0180, 35.5138), "nuts": "EL43", "split": "test"},
    # Denser urban
    "Patras":             {"center": (21.7346, 38.2466), "nuts": "EL63", "split": "train"},
    "Thessaloniki":       {"center": (22.9447, 40.6403), "nuts": "EL52", "split": "train"},
    "Athens":             {"center": (23.7448, 37.9795), "nuts": "EL30", "split": "test"},
    "Kolonaki":           {"center": (23.7430, 37.9785), "nuts": "EL30", "split": "train"},
}

TRAIN_CITIES = [c for c, m in CITY_REGISTRY.items() if m["split"] == "train"]
TEST_CITIES = [c for c, m in CITY_REGISTRY.items() if m["split"] == "test"]


# --- EUBUCCO region download / cache ---------------------------------------
def _region_parquet_url(nuts_id: str, version: str = "v0.2") -> str:
    """Ask the EUBUCCO API for a fresh presigned parquet URL for a NUTS region."""
    url = f"{_API}/v1/datalake/nuts/{version}/{nuts_id}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        info = json.load(resp)
    for f in info["files"]:
        if f["key"].endswith(".parquet"):
            return f["presigned_url"]
    raise RuntimeError(f"No parquet file listed for NUTS region {nuts_id}")


def download_region(nuts_id: str, version: str = "v0.2",
                    cache_dir: Path | str = _CACHE_DIR,
                    force: bool = False) -> Path:
    """Download (and cache) the EUBUCCO parquet for one NUTS region."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{nuts_id}.parquet"
    if dest.exists() and not force:
        return dest
    src = _region_parquet_url(nuts_id, version)
    print(f"[EUBUCCO] downloading {nuts_id} -> {dest} ...")
    tmp = dest.with_suffix(".parquet.tmp")
    urllib.request.urlretrieve(src, tmp)
    tmp.replace(dest)
    return dest


def _load_region_gdf(nuts_id: str, version: str = "v0.2",
                     cache_dir: Path | str = _CACHE_DIR) -> gpd.GeoDataFrame:
    path = download_region(nuts_id, version, cache_dir)
    cols = ["id", "height", "floors", "type", "city_id", "geometry"]
    gdf = gpd.read_parquet(path, columns=cols)
    if gdf.crs is None:
        gdf = gdf.set_crs(_EUBUCCO_CRS)
    return gdf


# --- Height parsing ---------------------------------------------------------
def _numeric_heights(gdf: gpd.GeoDataFrame, default_floor_height: float = 3.0,
                     fallback_height: float = 10.0) -> np.ndarray:
    """EUBUCCO stores height/floors as strings. Coerce; fall back floors*3, then 10 m."""
    h = pd.to_numeric(gdf["height"], errors="coerce")
    floors = pd.to_numeric(gdf.get("floors"), errors="coerce")
    h = h.where(h > 0)
    h = h.fillna(floors * default_floor_height)
    h = h.fillna(fallback_height)
    return h.to_numpy(dtype=float)


# --- Roads (OSM) ------------------------------------------------------------
def _fetch_roads_local(center_lonlat, x0, y0, area_size):
    """Roads from OSM around the city centre, projected to EPSG:3035 and shifted
    into local [0, area_size] coordinates. Returns list of polylines."""
    if not _OSMNX_OK:
        return []
    lon, lat = center_lonlat
    try:
        ox.settings.use_cache = True
        ox.settings.log_console = False
        G = ox.graph_from_point((lat, lon), dist=area_size / 2 * 1.2,
                                 network_type="drive")
        G = ox.projection.project_graph(G, to_crs=_EUBUCCO_CRS)
        _, edges = ox.graph_to_gdfs(G)
        roads = []
        for geom in edges.geometry:
            if geom is None or geom.is_empty:
                continue
            pts = [(x - x0, y - y0) for x, y in geom.coords]
            if any(0 <= px <= area_size and 0 <= py <= area_size for px, py in pts):
                roads.append(pts)
        return roads
    except Exception as exc:  # pragma: no cover
        print(f"[EUBUCCO] road fetch failed ({exc}); no roads.")
        return []


# --- Main entry point -------------------------------------------------------
def generate_town_eubucco(city_name: str | None = None, bbox=None, *,
                          center_lonlat=None, nuts_id: str | None = None,
                          area_size: float = 1000.0, num_users: int = 20,
                          seed: int = 42, fetch_roads: bool = True,
                          version: str = "v0.2",
                          cache_dir: Path | str = _CACHE_DIR):
    """Build a town scene from EUBUCCO buildings + OSM roads.

    Parameters
    ----------
    city_name : one of CITY_REGISTRY (preferred), or None to use center/nuts.
    bbox : optional (min_lon, min_lat, max_lon, max_lat); overrides area_size.
    center_lonlat, nuts_id : manual override when city_name is None.
    area_size : scene side length in metres (keep < ~1000 per the brief).

    Returns (buildings, users, roads) in local metres, origin at the SW corner.
    """
    rng = np.random.default_rng(seed)

    # 1. Resolve centre + region
    if city_name is not None:
        if city_name not in CITY_REGISTRY:
            raise KeyError(f"Unknown city '{city_name}'. Known: {list(CITY_REGISTRY)}")
        meta = CITY_REGISTRY[city_name]
        center_lonlat = meta["center"]
        nuts_id = meta["nuts"]
    if center_lonlat is None or nuts_id is None:
        raise ValueError("Provide city_name, or both center_lonlat and nuts_id.")

    # 2. Scene bbox in EPSG:3035 metres
    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        xa, ya = _to_3035.transform(min_lon, min_lat)
        xb, yb = _to_3035.transform(max_lon, max_lat)
        x0, y0, x1, y1 = min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)
        area_w, area_h = x1 - x0, y1 - y0
    else:
        cx, cy = _to_3035.transform(*center_lonlat)
        half = area_size / 2.0
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
        area_w = area_h = area_size

    # 3. Load region, crop to bbox
    gdf = _load_region_gdf(nuts_id, version, cache_dir)
    scene = gdf.cx[x0:x1, y0:y1]
    if len(scene) == 0:
        raise RuntimeError(
            f"No EUBUCCO buildings in scene for {city_name or center_lonlat} "
            f"(region {nuts_id}). Check the city centre coordinates.")

    # 4. Buildings -> axis-aligned boxes in local metres
    heights = _numeric_heights(scene)
    bounds = scene.geometry.bounds.to_numpy()  # minx, miny, maxx, maxy
    buildings = []
    for (bminx, bminy, bmaxx, bmaxy), h in zip(bounds, heights):
        w = bmaxx - bminx
        d = bmaxy - bminy
        if w < 1.0 or d < 1.0:
            continue
        bx = (bminx + bmaxx) / 2 - x0
        by = (bminy + bmaxy) / 2 - y0
        buildings.append((float(bx), float(by), float(w), float(d), float(h)))

    # 5. Roads from OSM (local coords)
    roads = _fetch_roads_local(center_lonlat, x0, y0, max(area_w, area_h)) \
        if fetch_roads else []

    # 6. Users on roads, rejecting building interiors
    def _inside_building(px, py):
        return any(abs(px - bx) < bw / 2 and abs(py - by) < bd / 2
                   for bx, by, bw, bd, _ in buildings)

    road_pts = [(px, py) for road in roads for (px, py) in road
                if 0 < px < area_w and 0 < py < area_h]
    users, attempts = [], 0
    while len(users) < num_users and attempts < 20000:
        attempts += 1
        if road_pts:
            px, py = road_pts[rng.integers(len(road_pts))]
            px += rng.normal(0, 5)
            py += rng.normal(0, 5)
        else:
            px, py = rng.uniform(0, area_w), rng.uniform(0, area_h)
        px = float(np.clip(px, 0, area_w))
        py = float(np.clip(py, 0, area_h))
        if not _inside_building(px, py):
            users.append((px, py, 0.0))

    print(f"[EUBUCCO] {city_name or nuts_id}: {len(buildings)} buildings, "
          f"{len(roads)} road segments, {len(users)} users.")
    return buildings, np.array(users), roads


def list_cities():
    """Return (train_cities, test_cities)."""
    return TRAIN_CITIES, TEST_CITIES
