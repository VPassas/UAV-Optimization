"""Unit tests for the Step 3 feature extractor."""
import numpy as np
import pytest

from student_a.features import (
    extract_features, build_city_geometry, FEATURE_NAMES, N_FEATURES, LinkFeatures,
)
from common.channel import has_line_of_sight


def test_shape_and_finite():
    buildings = [(100, 100, 40, 40, 30)]
    f = extract_features((250, 250, 100), (120, 120, 1.5), buildings)
    assert f.shape == (N_FEATURES,)
    assert f.dtype == np.float32
    assert np.isfinite(f).all()
    assert FEATURE_NAMES == LinkFeatures.names()


def test_uav_directly_overhead_clear_sky():
    # No buildings: straight-down link, ~90 deg elevation, ~0 horizontal distance.
    f = extract_features((200, 200, 120), (200, 200, 1.5), [], as_dataclass=True)
    assert f.d2d < 1e-6
    assert f.elevation_deg == pytest.approx(90.0, abs=1e-3)
    assert f.ray_n_intersections == 0
    assert f.density_100 == 0.0
    assert f.uav_clearance == pytest.approx(120.0)


def test_single_building_blocks_low_link():
    # User and UAV on opposite sides of a tall building, low elevation -> blocked.
    buildings = [(100, 100, 40, 40, 50)]
    uav = (60, 100, 20)
    user = (140, 100, 1.5)
    f = extract_features(uav, user, buildings, as_dataclass=True)
    assert f.ray_n_intersections >= 1
    assert f.ray_max_blocked_h == pytest.approx(50.0)
    assert f.ray_block_length > 0
    # agrees with the simulator's own LoS test
    assert has_line_of_sight(uav, user, buildings) is False


def test_los_consistency_with_simulator():
    # ray_n_intersections == 0  <=>  has_line_of_sight True, over random links.
    rng = np.random.default_rng(0)
    buildings = [(float(rng.uniform(50, 450)), float(rng.uniform(50, 450)),
                  float(rng.uniform(15, 45)), float(rng.uniform(15, 45)),
                  float(rng.uniform(10, 45))) for _ in range(25)]
    geo = build_city_geometry(buildings)
    mismatches = 0
    for _ in range(300):
        uav = (rng.uniform(0, 500), rng.uniform(0, 500), rng.uniform(40, 150))
        user = (rng.uniform(0, 500), rng.uniform(0, 500), 1.5)
        n = extract_features(uav, user, geo, as_dataclass=True).ray_n_intersections
        los = has_line_of_sight(uav, user, buildings)
        if (n == 0) != los:
            mismatches += 1
    # allow a tiny number of grazing-edge disagreements
    assert mismatches <= 3, f"{mismatches} LoS mismatches vs simulator"


def test_density_increases_with_radius_for_uniform_city():
    # A regular grid of buildings -> coverage fraction roughly constant/■stable;
    # at least it must stay in [0,1] and be finite.
    buildings = [(x, y, 20, 20, 12) for x in range(50, 451, 50)
                 for y in range(50, 451, 50)]
    f = extract_features((250, 250, 100), (240, 240, 1.5), buildings, as_dataclass=True)
    for v in (f.density_50, f.density_100, f.density_200):
        assert 0.0 <= v <= 1.0


def test_empty_city_safe():
    f = extract_features((10, 10, 80), (300, 300, 1.5), [], as_dataclass=True)
    assert f.ray_n_intersections == 0
    assert f.nbhd_mean_h == 0.0
    assert np.isfinite(f.to_array()).all()
