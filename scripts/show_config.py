"""Show the composed experiment config (schema <- conf/config.yaml <- CLI).

Demonstrates the shared config interface used by both projects.

Examples:
    python scripts/show_config.py
    python scripts/show_config.py shared.seed=7 surrogate.data.n_mc=50
    python scripts/show_config.py shared.cities=test
    python scripts/show_config.py shared.cities=[Nafplio,Athens]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf
from common.config import load_config, resolve_cities


def main():
    overrides = sys.argv[1:]                 # OmegaConf dotlist, e.g. a.b=1
    cfg = load_config(overrides=overrides)

    print("=== composed config ===")
    print(OmegaConf.to_yaml(cfg))
    cities = resolve_cities(cfg.shared.cities)
    print(f"resolved cities ({len(cities)}): {cities}")
    if overrides:
        print(f"applied overrides: {overrides}")


if __name__ == "__main__":
    main()
