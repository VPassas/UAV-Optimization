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

```bash
# 1. Create the environment (conda-forge; best for the geospatial stack on Windows)
conda env create -f environment.yml
conda activate uav

# 2. Sanity check
python -c "from common.channel import compute_user_rate; print('channel OK')"
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
