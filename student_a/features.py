"""Step 3 — urban-geometry feature extraction.

The neural network cannot see the whole city; we compress the geometry of one
UAV->user link into a small fixed-length vector. ``extract_features`` returns a
14-D ``np.ndarray`` (matching ``ChannelSurrogate(in_features=14)``).

Design notes
------------
* Buildings are axis-aligned boxes ``(cx, cy, w, d, h)`` in local metres, exactly
  as produced by :func:`common.data_loader.generate_town_eubucco`.
* ``CityGeometry`` precomputes vectorised arrays once per city so each
  ``extract_features`` call is pure NumPy (fast enough for bulk data generation).
* Features are grouped (link geometry / density / ray / height / relative) and
  named, so dropping a feature for the Step 9 ablation is trivial.

Feature vector (order = ``FEATURE_NAMES``)
    0  d2d                  horizontal UAV-user distance (m)
    1  d3d                  3-D UAV-user distance (m)
    2  elevation_deg        elevation angle user->UAV (0=horizon, 90=overhead)
    3  azimuth_deg          bearing user->UAV in the ground plane [0,360)
    4  density_50           building-footprint area fraction within 50 m of user
    5  density_100          ... within 100 m
    6  density_200          ... within 200 m
    7  ray_n_intersections  # buildings the UAV->user ray passes through (0 = LoS)
    8  ray_sum_blocked_h    sum of heights of intersected buildings (m)
    9  ray_max_blocked_h    tallest intersected building (m)
    10 ray_block_length     total metres of building the ray passes through
    11 nbhd_mean_h          mean building height within 100 m of user (m)
    12 nbhd_p90_h           90th-pct building height within 100 m of user (m)
    13 uav_clearance        UAV altitude minus tallest building within 100 m (m)
"""
from __future__ import annotations

from dataclasses import dataclass, astuple, fields

import numpy as np

USER_HEIGHT_M = 1.5  # ground-user antenna height (brief 4.1)

FEATURE_NAMES = [
    "d2d", "d3d", "elevation_deg", "azimuth_deg",
    "density_50", "density_100", "density_200",
    "ray_n_intersections", "ray_sum_blocked_h", "ray_max_blocked_h",
    "ray_block_length",
    "nbhd_mean_h", "nbhd_p90_h", "uav_clearance",
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class LinkFeatures:
    """Named view of one link's features (ablation-friendly)."""
    d2d: float
    d3d: float
    elevation_deg: float
    azimuth_deg: float
    density_50: float
    density_100: float
    density_200: float
    ray_n_intersections: float
    ray_sum_blocked_h: float
    ray_max_blocked_h: float
    ray_block_length: float
    nbhd_mean_h: float
    nbhd_p90_h: float
    uav_clearance: float

    def to_array(self) -> np.ndarray:
        return np.array(astuple(self), dtype=np.float32)

    @staticmethod
    def names() -> list[str]:
        return [f.name for f in fields(LinkFeatures)]


class CityGeometry:
    """Vectorised building arrays for one city scene (precompute once)."""

    def __init__(self, buildings):
        if len(buildings) == 0:
            z = np.zeros(0, dtype=float)
            self.cx = self.cy = self.hw = self.hd = self.h = self.area = z
            self.xmin = self.xmax = self.ymin = self.ymax = z
            return
        b = np.asarray(buildings, dtype=float)        # (B, 5): cx,cy,w,d,h
        self.cx, self.cy = b[:, 0], b[:, 1]
        self.hw, self.hd = b[:, 2] / 2.0, b[:, 3] / 2.0
        self.h = b[:, 4]
        self.area = b[:, 2] * b[:, 3]
        self.xmin, self.xmax = self.cx - self.hw, self.cx + self.hw
        self.ymin, self.ymax = self.cy - self.hd, self.cy + self.hd

    def __len__(self):
        return len(self.h)


def build_city_geometry(buildings) -> CityGeometry:
    """Convenience wrapper: list of (cx,cy,w,d,h) -> CityGeometry."""
    return CityGeometry(buildings)


def _segment_aabb_hits(p0, p1, geo: CityGeometry):
    """Vectorised 3-D segment vs axis-aligned-box test over all buildings.

    Boxes span x:[xmin,xmax], y:[ymin,ymax], z:[0,h]. Returns
    (hit_mask, enter_t, exit_t) with t in [0,1] along the segment p0->p1.
    """
    B = len(geo)
    if B == 0:
        m = np.zeros(0, dtype=bool)
        return m, np.zeros(0), np.zeros(0)
    d = p1 - p0
    big = 1e18
    t_near = np.full(B, -np.inf)
    t_far = np.full(B, np.inf)

    for axis, lo, hi in (
        (0, geo.xmin, geo.xmax),
        (1, geo.ymin, geo.ymax),
        (2, np.zeros(B), geo.h),
    ):
        di = d[axis]
        if abs(di) < 1e-12:
            # Ray parallel to this slab: miss unless origin is inside the slab.
            inside = (p0[axis] >= lo) & (p0[axis] <= hi)
            t_near = np.where(inside, t_near, big)
            t_far = np.where(inside, t_far, -big)
        else:
            t1 = (lo - p0[axis]) / di
            t2 = (hi - p0[axis]) / di
            tmin = np.minimum(t1, t2)
            tmax = np.maximum(t1, t2)
            t_near = np.maximum(t_near, tmin)
            t_far = np.minimum(t_far, tmax)

    enter = np.clip(t_near, 0.0, 1.0)
    exit_ = np.clip(t_far, 0.0, 1.0)
    hit = (t_near <= t_far) & (t_far >= 0.0) & (t_near <= 1.0) & (exit_ > enter)
    return hit, enter, exit_


def extract_features(uav_pos, user_pos, city_geom, *, as_dataclass: bool = False):
    """Compute the 14-D feature vector for a single UAV->user link.

    Parameters
    ----------
    uav_pos  : (x, y, z) UAV position, metres.
    user_pos : (x, y[, z]) ground-user position; z defaults to USER_HEIGHT_M.
    city_geom: CityGeometry (or a raw list of buildings, wrapped on the fly).
    """
    if not isinstance(city_geom, CityGeometry):
        city_geom = CityGeometry(city_geom)
    geo = city_geom

    uav = np.asarray(uav_pos, dtype=float)
    user = np.asarray(user_pos, dtype=float)
    if user.shape[0] < 3:
        user = np.array([user[0], user[1], USER_HEIGHT_M], dtype=float)

    # --- Link geometry ---
    dx, dy, dz = uav[0] - user[0], uav[1] - user[1], uav[2] - user[2]
    d2d = float(np.hypot(dx, dy))
    d3d = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    elevation_deg = float(np.degrees(np.arctan2(dz, max(d2d, 1e-9))))
    azimuth_deg = float(np.degrees(np.arctan2(dy, dx)) % 360.0)

    # --- Distances from user to building centres (2-D) ---
    if len(geo) > 0:
        dc = np.hypot(geo.cx - user[0], geo.cy - user[1])
    else:
        dc = np.zeros(0)

    def _density(radius):
        if len(geo) == 0:
            return 0.0
        m = dc <= radius
        covered = float(geo.area[m].sum())
        return float(np.clip(covered / (np.pi * radius * radius), 0.0, 1.0))

    density_50 = _density(50.0)
    density_100 = _density(100.0)
    density_200 = _density(200.0)

    # --- Neighbourhood height stats within 100 m of the user ---
    if len(geo) > 0:
        near = dc <= 100.0
        near_h = geo.h[near]
    else:
        near_h = np.zeros(0)
    if near_h.size > 0:
        nbhd_mean_h = float(near_h.mean())
        nbhd_p90_h = float(np.percentile(near_h, 90))
        tallest = float(near_h.max())
    else:
        nbhd_mean_h = nbhd_p90_h = tallest = 0.0
    uav_clearance = float(uav[2] - tallest)

    # --- Ray-blockage statistics (UAV -> user) ---
    hit, enter, exit_ = _segment_aabb_hits(uav, user, geo)
    if hit.any():
        seg_len = d3d
        lengths = (exit_ - enter) * seg_len
        blocked_h = geo.h[hit]
        ray_n = float(int(hit.sum()))
        ray_sum_blocked_h = float(blocked_h.sum())
        ray_max_blocked_h = float(blocked_h.max())
        ray_block_length = float(lengths[hit].sum())
    else:
        ray_n = ray_sum_blocked_h = ray_max_blocked_h = ray_block_length = 0.0

    feats = LinkFeatures(
        d2d=d2d, d3d=d3d, elevation_deg=elevation_deg, azimuth_deg=azimuth_deg,
        density_50=density_50, density_100=density_100, density_200=density_200,
        ray_n_intersections=ray_n, ray_sum_blocked_h=ray_sum_blocked_h,
        ray_max_blocked_h=ray_max_blocked_h, ray_block_length=ray_block_length,
        nbhd_mean_h=nbhd_mean_h, nbhd_p90_h=nbhd_p90_h, uav_clearance=uav_clearance,
    )
    return feats if as_dataclass else feats.to_array()
