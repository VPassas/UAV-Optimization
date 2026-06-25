"""Shared config system (co-owned by Student A and Student B).

One typed, reproducible interface for both projects, built on OmegaConf
(Hydra-compatible — Hydra can be layered on later for multirun if needed).

Usage
-----
    from common.config import load_config, resolve_cities

    cfg = load_config(overrides=["shared.seed=1", "surrogate.data.n_samples=5000"])
    cities = resolve_cities(cfg.shared.cities)        # "train" -> [list of cities]
    print(cfg.surrogate.model.hidden)                 # typed access

CLI style (same for both projects)::

    python scripts/show_config.py shared.seed=1 surrogate.data.n_mc=50

The schema below is the *contract*: shared params are co-owned; `surrogate.*`
is Student A; `policy.*` is Student B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, DictConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --- Shared (co-owned) ------------------------------------------------------
@dataclass
class Paths:
    repo_root: str = str(_REPO_ROOT)
    data_raw: str = str(_REPO_ROOT / "data" / "raw")
    data_processed: str = str(_REPO_ROOT / "data" / "processed")
    checkpoints: str = str(_REPO_ROOT / "checkpoints")
    figures: str = str(_REPO_ROOT / "docs" / "figures")


@dataclass
class Scene:
    area_size: float = 1000.0          # metres; keep < ~1000 (ray-AABB cost)


@dataclass
class Channel:
    env: str = "Urban"                 # AG_ENV key in common.channel
    ptx_dbm: float = 30.0              # UAV transmit power


@dataclass
class Shared:
    seed: int = 0
    cities: Any = "train"              # "train"|"test"|"all"|city name|[list]
    scene: Scene = field(default_factory=Scene)
    channel: Channel = field(default_factory=Channel)
    paths: Paths = field(default_factory=Paths)


# --- Student A: channel surrogate ------------------------------------------
@dataclass
class SurrogateData:
    n_samples: int = 3000              # (UAV, user) links per city
    n_mc: int = 30                     # Monte-Carlo fading samples per link
    alt_min: float = 50.0
    alt_max: float = 200.0


@dataclass
class SurrogateModel:
    in_features: int = 14
    hidden: int = 128
    depth: int = 3                     # hidden layers in the backbone
    heteroscedastic: bool = True       # mean + log-variance heads


@dataclass
class SurrogateTrain:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    epochs: int = 150
    val_fraction: float = 0.2          # per-city holdout from the train cities
    early_stop_patience: int = 20


@dataclass
class Surrogate:
    data: SurrogateData = field(default_factory=SurrogateData)
    model: SurrogateModel = field(default_factory=SurrogateModel)
    train: SurrogateTrain = field(default_factory=SurrogateTrain)


# --- Student B: trajectory policy (stub — B fills in) -----------------------
@dataclass
class PolicyExpert:
    horizon: int = 5                   # look-ahead steps for the max-min planner


@dataclass
class PolicyModel:
    arch: str = "deepsets"             # "deepsets" | "attention"
    vehicle_dim: int = 8
    global_dim: int = 5
    embed_dim: int = 64
    hidden: int = 128


@dataclass
class PolicyTrain:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    n_max_vehicles: int = 10


@dataclass
class Policy:
    expert: PolicyExpert = field(default_factory=PolicyExpert)
    model: PolicyModel = field(default_factory=PolicyModel)
    train: PolicyTrain = field(default_factory=PolicyTrain)


@dataclass
class Config:
    shared: Shared = field(default_factory=Shared)
    surrogate: Surrogate = field(default_factory=Surrogate)
    policy: Policy = field(default_factory=Policy)


DEFAULT_YAML = _REPO_ROOT / "conf" / "config.yaml"


def default_config() -> DictConfig:
    """The full schema with built-in defaults (no YAML needed)."""
    return OmegaConf.structured(Config)


def load_config(yaml_path: str | Path | None = DEFAULT_YAML,
                overrides: list[str] | None = None) -> DictConfig:
    """Compose: schema defaults <- YAML file <- CLI dotlist overrides.

    Type-checked against the dataclass schema (typos / wrong types raise).
    """
    cfg = default_config()
    if yaml_path is not None and Path(yaml_path).exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(yaml_path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def resolve_cities(spec: Any) -> list[str]:
    """Turn cfg.shared.cities into a concrete list of city names.

    Accepts "train" | "test" | "all" | a single city name | an explicit list.
    """
    # Imported here to avoid a circular import at module load.
    from common.data_loader import TRAIN_CITIES, TEST_CITIES, CITY_REGISTRY

    if isinstance(spec, str):
        key = spec.lower()
        if key == "train":
            return list(TRAIN_CITIES)
        if key == "test":
            return list(TEST_CITIES)
        if key == "all":
            return list(CITY_REGISTRY)
        if spec in CITY_REGISTRY:
            return [spec]
        raise ValueError(f"Unknown cities spec '{spec}'. Use train/test/all, "
                         f"a city name, or a list.")
    # OmegaConf list or plain list
    cities = list(spec)
    unknown = [c for c in cities if c not in CITY_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown cities in list: {unknown}")
    return cities


def save_config(cfg: DictConfig, path: str | Path) -> Path:
    """Persist the resolved config next to an experiment's outputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
    return path
