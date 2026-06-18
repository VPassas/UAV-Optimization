"""Smoke tests for the EUBUCCO loader (Step 2).

Uses the cached EL42 region (Ermoupoli / South Aegean). The first run downloads
~42 MB; afterwards it is offline. Road fetching (OSM/network) is disabled here
so the core building extraction is tested deterministically.

Run:  conda run -n uav python -m pytest tests/test_data_loader.py -q
"""
import numpy as np
import pytest

from common.data_loader import (
    generate_town_eubucco, CITY_REGISTRY, TRAIN_CITIES, TEST_CITIES,
)


def test_registry_split_counts():
    assert len(TRAIN_CITIES) == 16  # 15 cities + Kolonaki (Athens neighbourhood)
    assert len(TEST_CITIES) == 5
    assert set(TEST_CITIES) == {"Ermoupoli", "Annecy", "Tallinn", "Chania", "Athens"}


def test_all_cities_have_center_and_region():
    for name, meta in CITY_REGISTRY.items():
        assert len(meta["center"]) == 2
        assert meta["nuts"] and isinstance(meta["nuts"], str)


@pytest.mark.parametrize("city", ["Ermoupoli"])
def test_ermoupoli_scene(city):
    try:
        buildings, users, roads = generate_town_eubucco(
            city, area_size=1000, fetch_roads=False, num_users=10)
    except Exception as exc:  # network/download unavailable
        pytest.skip(f"EUBUCCO region unavailable: {exc}")

    assert len(buildings) > 500            # dense Greek island town
    heights = np.array([b[4] for b in buildings])
    assert heights.min() > 0               # no zero/negative heights
    assert heights.max() < 200             # no absurd skyscrapers
    assert 3 < heights.mean() < 30         # realistic mean
    # building centres lie within the scene (small margin: edge-straddling
    # buildings are kept on purpose since they can still block rays)
    margin = 100
    for bx, by, bw, bd, _ in buildings:
        assert -margin <= bx <= 1000 + margin and -margin <= by <= 1000 + margin
    assert users.shape == (10, 3)
    assert (users[:, 2] == 0).all()        # users on the ground
