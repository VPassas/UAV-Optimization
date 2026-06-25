# Shared Config System (co-owned)

Status: ✅ done — the third shared-infrastructure piece (with the data loader and
channel model).

## Choice: OmegaConf + structured (dataclass) configs

The brief said "Hydra or OmegaConf". We use **OmegaConf with a dataclass schema**:
typed, reproducible, CLI overrides — but without Hydra's working-directory magic
and decorators that confuse beginners. It is **Hydra-compatible** (Hydra is built
on OmegaConf), so Hydra multirun can be added later for big sweeps without
rewriting anything.

## The contract

Schema + defaults: [`common/config.py`](../common/config.py). Human-editable
overrides: [`conf/config.yaml`](../conf/config.yaml).

```
shared.*      co-owned: seed, cities, scene.area_size, channel.{env,ptx_dbm}, paths.*
surrogate.*   Student A: data / model / train
policy.*      Student B: expert / model / train
```

`shared.cities` accepts `train` | `test` | `all` | a city name | an explicit list;
`resolve_cities()` turns it into concrete names (validated against the registry).

## Usage (same interface for both projects)

```python
from common.config import load_config, resolve_cities, save_config

cfg = load_config(overrides=["shared.seed=1", "surrogate.data.n_mc=50"])
cities = resolve_cities(cfg.shared.cities)
# ... run experiment ...
save_config(cfg, f"{cfg.shared.paths.checkpoints}/exp1/config.yaml")  # reproducibility
```

Composition order: **schema defaults  <-  conf/config.yaml  <-  CLI overrides**.
Wrong types and unknown keys raise (the schema is closed) — typos fail fast.

## CLI

```bash
python scripts/show_config.py                                  # show defaults
python scripts/show_config.py shared.seed=7 surrogate.data.n_mc=50
python scripts/show_config.py shared.cities=[Nafplio,Athens]
```

Each project's training scripts (Step 7) will take the same dotlist overrides, and
`save_config` writes the resolved config beside every checkpoint so any run is
reproducible.

## Tests

`tests/test_config.py` (11): defaults, overrides, type/unknown-key rejection,
city resolution for all spec forms.
