# Step 2 — EUBUCCO Data Loader

**Deliverable status:** working loader ✅ · EUBUCCO-vs-OSM comparison ✅ · manual
height validation ⏳ (procedure below — needs a human + Google Earth).

## What was built

[`common/data_loader.py`](../common/data_loader.py) — `generate_town_eubucco(city_name, ...)`
returns a scene in the same format as the reference simulator's `generate_town_osm`:

```python
from common.data_loader import generate_town_eubucco
buildings, users, roads = generate_town_eubucco("Ermoupoli", area_size=1000)
#   buildings : list[(cx, cy, w, d, h)]  axis-aligned boxes, local metres
#   users     : np.ndarray (N, 3)        on the road network, z = 0
#   roads     : list[list[(x, y)]]       OSM polylines, local metres
```

- **Buildings + heights:** EUBUCCO **v0.2**, per-NUTS-2-region Parquet on the
  project S3 (EPSG:3035, metres). The API (`api.eubucco.com/v1/datalake/nuts/v0.2/{region}`)
  returns 1-hour presigned URLs; `download_region()` caches the parquet under
  `data/raw/eubucco/`. Heights are stored as **strings** → coerced numeric,
  falling back to `floors*3`, then 10 m (fallbacks are essentially never needed
  in v0.2 — Greek coverage is ~100%).
- **Roads:** OpenStreetMap via osmnx (EUBUCCO has no road network), projected to
  EPSG:3035 and shifted into the local frame.
- **Cities:** all 20 (15 train + 5 test, plus Athens/Kolonaki) are in
  `CITY_REGISTRY` with centre lon/lat and NUTS-2 region. All 18 distinct region
  codes were verified present in the live v0.2 partition list.

## Important finding (be honest in the paper)

The brief assumes "<5% of OSM buildings have a height tag". That is a **global
average and not true per city**. For Ermoupoli (Syros), a same-ground comparison
(`scripts/make_eubucco_osm_comparison.py`) gives:

| Source   | Buildings (1 km² scene) | Height info |
|----------|--------------------------|-------------|
| EUBUCCO  | 3428 | continuous estimates, mean 7.6 m, p90 10.3 m |
| OSM      | 3337 | ~98% had a `building:levels` tag, but **quantized** at multiples of 3 m |

So for *well-mapped* towns OSM footprint **coverage** is comparable. The real,
defensible EUBUCCO advantages are: (1) **consistency** across all 20 cities — no
per-city tag-coverage lottery; (2) **continuous** height estimates vs coarse
`levels*3` quantization (see the histogram in
`docs/figures/eubucco_vs_osm_ermoupoli.png`); (3) coverage where OSM is sparse.
Run the comparison on a few cities to confirm the pattern before writing.

## Manual height validation (the remaining deliverable)

The brief asks: *compare heights against Google Earth for 10 buildings per city,
on 3 cities*. This needs a human (no programmatic Google Earth height API):

1. Pick 3 cities spanning the morphology range, e.g. **Ermoupoli** (island, low),
   **Heidelberg** (historic mid-rise), **Thessaloniki** (denser urban).
2. For each, load the scene and pick 10 buildings (note their `(cx, cy)` and
   EUBUCCO height).
3. In Google Earth Pro, navigate to each building, read floors/height from the
   3D model or street view, record the reference height.
4. Tabulate EUBUCCO vs reference; report mean abs error and bias. Add the table
   + a scatter plot to the paper's data section.

A helper to dump 10 candidate buildings per city for this check can be added when
you are ready to do the manual pass.

## Reproduce

```bash
conda run -n uav python scripts/make_eubucco_osm_comparison.py --city Ermoupoli
conda run -n uav python -m pytest tests/test_data_loader.py -q
```
