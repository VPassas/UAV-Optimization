"""Unit tests for the Step 4 data-generation pipeline (no network needed)."""
import numpy as np

from student_a.dataset import (
    sample_uav_positions, sample_user_positions, generate_city_dataset,
)
from student_a.features import FEATURE_NAMES, USER_HEIGHT_M


def test_uav_lhs_bounds():
    uavs = sample_uav_positions(500, area=1000, alt_min=50, alt_max=200, seed=0)
    assert uavs.shape == (500, 3)
    assert (uavs[:, 0] >= 0).all() and (uavs[:, 0] <= 1000).all()
    assert (uavs[:, 1] >= 0).all() and (uavs[:, 1] <= 1000).all()
    assert (uavs[:, 2] >= 50).all() and (uavs[:, 2] <= 200).all()
    # Latin Hypercube => good altitude coverage (roughly uniform spread)
    assert uavs[:, 2].std() > 30


def test_users_on_roads_avoid_buildings():
    roads = [[(0, 500), (1000, 500)], [(500, 0), (500, 1000)]]  # a cross
    buildings = [(200, 500, 40, 40, 20)]  # sits on the horizontal road
    users = sample_user_positions(200, roads, buildings, area=1000, seed=1)
    assert users.shape == (200, 3)
    assert np.allclose(users[:, 2], USER_HEIGHT_M)
    # none inside the building footprint
    inside = (np.abs(users[:, 0] - 200) < 20) & (np.abs(users[:, 1] - 500) < 20)
    assert not inside.any()
    # most users hug one of the two roads (within a few metres of x=500 or y=500)
    near = (np.abs(users[:, 0] - 500) < 12) | (np.abs(users[:, 1] - 500) < 12)
    assert near.mean() > 0.8


def test_users_fallback_without_roads():
    users = sample_user_positions(50, roads=[], buildings=[], area=500, seed=2)
    assert users.shape == (50, 3)


def test_end_to_end_synthetic_city():
    # Synthetic scene -> no EUBUCCO/OSM download required.
    rng = np.random.default_rng(0)
    buildings = [(float(rng.uniform(100, 900)), float(rng.uniform(100, 900)),
                  float(rng.uniform(15, 40)), float(rng.uniform(15, 40)),
                  float(rng.uniform(10, 40))) for _ in range(40)]
    roads = [[(0, 300), (1000, 300)], [(300, 0), (300, 1000)],
             [(0, 700), (1000, 700)]]
    df = generate_city_dataset("TestTown", n_samples=60, n_mc=4,
                               n_jobs=1, seed=0, buildings=buildings, roads=roads)
    assert len(df) == 60
    for col in FEATURE_NAMES + ["rate", "uav_x", "uav_z", "user_x", "city"]:
        assert col in df.columns
    r = df["rate"].to_numpy()
    assert np.isfinite(r).all()
    assert (r >= 0).all()
    assert r.max() > 0  # at least some links carry signal
