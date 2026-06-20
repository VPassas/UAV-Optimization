# Step 4 — Data-Generation Pipeline

**Deliverable status:** pipeline ✅ · 3000-sample smoke test ✅ · profiled ✅ ·
parallelization evaluated ✅ · tests ✅ (4 new, 13 total). Full-dataset run is
Step 5.

## What it does

For one city, [`student_a/dataset.py`](../student_a/dataset.py) draws
`n_samples` (UAV, user) links and computes, for each:
- **inputs:** the 14 geometry features (`extract_features`)
- **target:** expected ergodic capacity (b/s/Hz), Monte-Carlo over `n_mc=30`
  fading realisations via the full Al-Hourani model (`compute_user_rate`).

Sampling (brief §4.1):
- **UAV:** Latin Hypercube over `[0,area] x [0,area] x [50,200] m` (good 3-D
  coverage with few points).
- **User:** uniform along the OSM road network (weighted by segment length),
  rejecting building interiors, antenna at 1.5 m.

Output: one `data/processed/<city>.parquet` with columns
`city, split, <14 features>, uav_x/y/z, user_x/y, rate`.

## Performance (the headline)

Measured on Nafplio (1259 buildings):

| config | time | per sample |
|--------|------|------------|
| serial, 3000 samples | **2.3 s** | **0.8 ms** |
| parallel (`--jobs -1`), 3000 | 3.9 s | 1.3 ms |

The brief targeted < 100 ms/sample and < 6 h for the full dataset. We are at
**0.8 ms/sample** — ~125× under budget. The full 20-city × 3000 dataset is
~2 minutes of compute (plus a few seconds of road-fetch per city).

**Honest finding — parallelization is not needed (and hurts here).** Each link is
so cheap (~0.8 ms) that joblib's process-spawn + building-pickling overhead makes
`--jobs -1` *slower* than serial. So the default is `--jobs 1`. The joblib path is
kept for genuinely heavy configs (large `n_mc`, bigger scenes). Likewise an Rtree
spatial index (brief's contingency) is unnecessary at this scene size — the
vectorised NumPy ray test already runs in microseconds.

## Reproduce

```bash
# one city
conda run -n uav python scripts/generate_dataset.py --city Nafplio --n 3000
# serial profile on a small batch
conda run -n uav python scripts/generate_dataset.py --city Nafplio --n 200 --jobs 1
# full dataset (Step 5)
conda run -n uav python scripts/generate_dataset.py --all --n 3000
conda run -n uav python -m pytest tests/test_dataset.py -q
```

Datasets are gitignored (they go to Zenodo for the paper, not git).
