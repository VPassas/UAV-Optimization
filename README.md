# Learning End-to-End UAV Networking

Channel surrogates (Student A) + learned policies (Student B).

**Student A goal:** replace the slow physical UAV-to-ground channel simulator with a
small neural network (≤100k params) that predicts expected link capacity (with an
uncertainty estimate) from urban geometry, trained on 15 European cities and
evaluated on 5 unseen ones, then plugged into the placement optimizer for a real
wall-clock speedup.

## Repository layout

```
common/        # co-owned with Student B
  channel.py       # full Al-Hourani channel model (re-exported from project/)
  data_loader.py   # generate_town_eubucco(city, bbox) -> buildings + roads + heights  [Step 2]
student_a/     # the channel-surrogate project
  features.py      # extract_features(uav, user, city_geom) -> np.ndarray(~14)         [Step 3]
  dataset.py       # (UAV, user) sampling + parallel rate generation                  [Step 4]
  model.py         # ChannelSurrogate (heteroscedastic Gaussian MLP)                  [Step 7]
  baselines.py     # mean / linear / RandomForest / KNN                               [Step 6]
  config/          # Hydra configs (reproducible experiments)
project/       # original reference simulator (uav_extended_simulation.py)
data/          # raw/ (EUBUCCO, OSM cache) + processed/ (datasets) — gitignored
docs/          # reading_log.md, paper drafts
scripts/       # CLI entrypoints (generate data, train, evaluate)
tests/         # unit tests (feature edge cases, etc.)
```

## Setup

Pick **one** of the two options. Both give an importable project (`import common`,
`import student_a` work from anywhere) and have been verified on Windows.

### Option A — pip + venv (no conda needed)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate  on macOS/Linux)
pip install -e ".[dev]"          # installs deps + the project as a package
```

`pip install -e .` works because geopandas/osmnx/pyproj/rtree now ship
self-contained wheels — GDAL no longer needs conda. `torch` installs as the CPU
build by default; for a CUDA GPU follow https://pytorch.org/get-started (e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu124`).

[`uv`](https://docs.astral.sh/uv/) works too and is faster: `uv venv` then
`uv pip install -e ".[dev]"`.

### Option B — conda (most robust geospatial stack)

```bash
conda env create -f environment.yml
conda activate uav
pip install -e .                 # make common/ and student_a/ importable
```

### Sanity check (either option)

```bash
python -c "from common.channel import compute_user_rate; print('channel OK')"
python -m pytest tests -q
```

## Cities (morphological diversity → forces generalization)

| Category                | Training (15)                              | Test (5)        |
|-------------------------|--------------------------------------------|-----------------|
| Mediterranean low-dens. | Nafplio, Rethymno, Karpenisi, Kastoria     | Ermoupoli       |
| European historic       | Bruges, Cesky Krumlov, Sintra, Heidelberg  | Annecy          |
| Modern small            | Eindhoven, Aarhus, Tampere                 | Tallinn         |
| Island / tourist        | Mykonos town, Split                        | Chania          |
| Denser urban            | Patras centre, Thessaloniki centre, Athens Kolonaki | Athens   |

Keep each scene < 1×1 km² (ray-AABB cost).

## Workflow (16 steps, 4 phases)

- **I Foundation:** 1 setup · 2 EUBUCCO loader · 3 features · 4 data pipeline
- **II Modeling:** 5 full dataset+EDA · 6 baselines · 7 MLP training · 8 evaluation
- **III Ablations/Integration:** 9 feature ablation · 10 city-count ablation · 11 DE optimizer integration · 12 join with Student B
- **IV Writing:** 13 figures · 14 draft · 15 revisions/merge · 16 submission

See `student_a_1.docx` for the full brief.
