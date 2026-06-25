"""Tests for the shared config system."""
import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ValidationError

from common.config import load_config, default_config, resolve_cities
from common.data_loader import TRAIN_CITIES, TEST_CITIES, CITY_REGISTRY


def test_defaults_present():
    cfg = default_config()
    assert cfg.surrogate.model.in_features == 14
    assert cfg.shared.scene.area_size == 1000.0
    assert cfg.policy.model.arch == "deepsets"


def test_cli_overrides_apply_and_typecheck():
    cfg = load_config(overrides=["shared.seed=7", "surrogate.data.n_mc=50"])
    assert cfg.shared.seed == 7
    assert cfg.surrogate.data.n_mc == 50


def test_wrong_type_rejected():
    with pytest.raises((ValidationError, Exception)):
        load_config(overrides=["surrogate.data.n_mc=not_an_int"])


def test_unknown_key_rejected():
    # structured schema is closed -> unknown keys raise
    with pytest.raises(Exception):
        load_config(overrides=["surrogate.data.nonexistent=1"])


@pytest.mark.parametrize("spec,expected", [
    ("train", TRAIN_CITIES),
    ("test", TEST_CITIES),
    ("all", list(CITY_REGISTRY)),
    ("Nafplio", ["Nafplio"]),
])
def test_resolve_cities_keywords(spec, expected):
    assert resolve_cities(spec) == list(expected)


def test_resolve_cities_explicit_list():
    assert resolve_cities(["Nafplio", "Athens"]) == ["Nafplio", "Athens"]


def test_resolve_cities_bad_name():
    with pytest.raises(ValueError):
        resolve_cities("Atlantis")


def test_cities_override_from_cli_resolves():
    cfg = load_config(overrides=["shared.cities=test"])
    assert resolve_cities(cfg.shared.cities) == list(TEST_CITIES)
