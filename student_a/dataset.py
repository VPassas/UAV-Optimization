"""Step 4 — data-generation pipeline.

For one city we draw ~3000 (UAV, user) links and, for each, compute:
  * inputs : the 14 urban-geometry features (student_a.features)
  * target : expected ergodic capacity (b/s/Hz), Monte-Carlo averaged over
             ``n_mc`` fading realisations by the full Al-Hourani channel model.

Sampling (brief 4.1):
  * UAV positions : Latin Hypercube over the 3-D box
    [0,area] x [0,area] x [alt_min, alt_max].
  * user positions: uniform along the OSM street network (by arc length),
    rejecting building interiors, antenna at 1.5 m.

The expensive part is the channel Monte-Carlo (per link: one ray-AABB LoS test
over all buildings + n_mc fading draws), so that is what we parallelise with
joblib. Feature extraction is ~0.1 ms and runs in the main process.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from joblib import Parallel, delayed

from common.data_loader import generate_town_eubucco, CITY_REGISTRY
from common.channel import compute_user_rate, dbm_to_watts, DEFAULT_ENV
from student_a.features import (
    build_city_geometry, extract_features, FEATURE_NAMES, USER_HEIGHT_M,
)

_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
DEFAULT_PTX_DBM = 30.0


# --- Sampling ---------------------------------------------------------------
def sample_uav_positions(n, area, alt_min=50.0, alt_max=200.0, seed=0):
    """Latin Hypercube sample of n UAV (x, y, z) positions over the 3-D box."""
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    unit = sampler.random(n)  # (n, 3) in [0,1)
    lo = np.array([0.0, 0.0, alt_min])
    hi = np.array([area, area, alt_max])
    return qmc.scale(unit, lo, hi)


def _inside_building(px, py, buildings):
    for bx, by, bw, bd, _ in buildings:
        if abs(px - bx) < bw / 2 and abs(py - by) < bd / 2:
            return True
    return False


def sample_user_positions(n, roads, buildings, area, seed=0):
    """Uniform points along the road network (weighted by segment length).

    Falls back to uniform-in-scene if no roads are available. Returns (n, 3)
    with z = USER_HEIGHT_M; rejects points inside building footprints.
    """
    rng = np.random.default_rng(seed)
    # Flatten roads into individual segments with their lengths.
    segs = []
    for road in roads:
        pts = np.asarray(road, dtype=float)
        for a, b in zip(pts[:-1], pts[1:]):
            L = float(np.hypot(*(b - a)))
            if L > 1e-6:
                segs.append((a, b, L))
    users = []
    if segs:
        seg_len = np.array([s[2] for s in segs])
        seg_p = seg_len / seg_len.sum()
        attempts = 0
        while len(users) < n and attempts < 50 * n:
            attempts += 1
            i = rng.choice(len(segs), p=seg_p)
            a, b, _ = segs[i]
            t = rng.random()
            px, py = a + t * (b - a)
            px += rng.normal(0, 3); py += rng.normal(0, 3)
            px = float(np.clip(px, 0, area)); py = float(np.clip(py, 0, area))
            if not _inside_building(px, py, buildings):
                users.append((px, py, USER_HEIGHT_M))
    while len(users) < n:  # fallback / top-up
        px, py = rng.uniform(0, area), rng.uniform(0, area)
        if not _inside_building(px, py, buildings):
            users.append((float(px), float(py), USER_HEIGHT_M))
    return np.array(users[:n])


# --- Target computation (parallel) -----------------------------------------
def _rate_batch(pairs, buildings, n_mc, env_key, p_tx_watts, seed):
    """Compute target rates for a chunk of (uav, user) pairs in one worker."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(pairs), dtype=np.float64)
    for i, (uav, user) in enumerate(pairs):
        out[i] = compute_user_rate(tuple(uav), tuple(user), p_tx_watts,
                                   env_key=env_key, buildings=buildings,
                                   rng=rng, n_mc=n_mc)
    return out


def generate_city_dataset(city_name, n_samples=3000, n_mc=30, *,
                          area_size=1000.0, alt_min=50.0, alt_max=200.0,
                          env_key=DEFAULT_ENV, p_tx_dbm=DEFAULT_PTX_DBM,
                          n_jobs=1, seed=0, buildings=None, roads=None):
    # NOTE: n_jobs=1 (serial) is fastest here -- each link costs ~0.8 ms, so
    # joblib's process-spawn + pickling overhead dominates. Raise n_jobs only
    # for much heavier configs (large n_mc or bigger scenes).
    """Generate the (features, rate) dataset for one city as a DataFrame."""
    if buildings is None or roads is None:
        buildings, _, roads = generate_town_eubucco(
            city_name, area_size=area_size, fetch_roads=True, num_users=1)
    geo = build_city_geometry(buildings)
    p_tx_watts = dbm_to_watts(p_tx_dbm)

    uavs = sample_uav_positions(n_samples, area_size, alt_min, alt_max, seed=seed)
    users = sample_user_positions(n_samples, roads, buildings, area_size, seed=seed + 1)

    # Features (cheap, main process)
    feats = np.stack([extract_features(uavs[i], users[i], geo)
                      for i in range(n_samples)])

    # Targets (expensive, parallel over chunks)
    pairs = list(zip(uavs, users))
    n_jobs_eff = (n_jobs if n_jobs and n_jobs > 0 else None)
    n_chunks = max(1, (n_jobs_eff or 8))
    chunks = np.array_split(np.arange(n_samples), n_chunks)
    ss = np.random.SeedSequence(seed + 1234)
    child_seeds = ss.spawn(len(chunks))
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_rate_batch)([pairs[j] for j in idx], buildings, n_mc,
                             env_key, p_tx_watts, cs)
        for idx, cs in zip(chunks, child_seeds))
    rates = np.empty(n_samples)
    for idx, r in zip(chunks, results):
        rates[idx] = r

    df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    df.insert(0, "city", city_name)
    df.insert(1, "split", CITY_REGISTRY.get(city_name, {}).get("split", "train"))
    df["uav_x"], df["uav_y"], df["uav_z"] = uavs[:, 0], uavs[:, 1], uavs[:, 2]
    df["user_x"], df["user_y"] = users[:, 0], users[:, 1]
    df["rate"] = rates
    return df


def save_city_dataset(df, city_name, out_dir=_PROCESSED):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{city_name.lower().replace(' ', '_')}.parquet"
    df.to_parquet(path, index=False)
    return path
