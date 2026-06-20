# Step 3 — Urban-Geometry Features

**Deliverable status:** `extract_features` ✅ · unit tests ✅ (6 passing, incl. LoS
consistency vs the simulator) · per-link visualization ✅.

## API

```python
from common.data_loader import generate_town_eubucco
from student_a.features import build_city_geometry, extract_features

buildings, users, roads = generate_town_eubucco("Nafplio")
geo = build_city_geometry(buildings)          # precompute once per city
x = extract_features(uav_pos, user_pos, geo)  # -> np.float32[14]
```

`build_city_geometry` turns the building list into vectorised NumPy arrays so each
`extract_features` call is pure NumPy — **0.115 ms/call** on a 1259-building city
(2000-call benchmark). Feature extraction is *not* the data-generation bottleneck;
the Monte-Carlo channel sim is.

## The 14 features (`student_a.features.FEATURE_NAMES`)

| # | name | meaning |
|---|------|---------|
| 0 | `d2d` | horizontal UAV-user distance (m) |
| 1 | `d3d` | 3-D UAV-user distance (m) |
| 2 | `elevation_deg` | elevation angle user->UAV (0 horizon, 90 overhead) |
| 3 | `azimuth_deg` | ground bearing user->UAV [0,360) |
| 4-6 | `density_50/100/200` | building-footprint area fraction within 50/100/200 m of user |
| 7 | `ray_n_intersections` | # buildings the UAV->user ray crosses (**0 = LoS**) |
| 8 | `ray_sum_blocked_h` | sum of heights of crossed buildings (m) |
| 9 | `ray_max_blocked_h` | tallest crossed building (m) |
| 10 | `ray_block_length` | total metres of building the ray passes through |
| 11 | `nbhd_mean_h` | mean building height within 100 m of user (m) |
| 12 | `nbhd_p90_h` | 90th-pct building height within 100 m of user (m) |
| 13 | `uav_clearance` | UAV altitude minus tallest building within 100 m (m) |

Grouped exactly per brief §4.2 (link geometry / density / ray stats / height
distribution / relative geometry). The `LinkFeatures` dataclass + `FEATURE_NAMES`
make the Step 9 drop-one-out ablation a one-liner.

## Correctness

`test_features.py::test_los_consistency_with_simulator` checks that
`ray_n_intersections == 0` matches the simulator's own `has_line_of_sight` over
300 random links (the ray-AABB blockage logic mirrors the physics the surrogate
must learn). Edge cases covered: UAV directly overhead, single blocking building,
empty city, uniform grid.

## Caveats / ablation candidates

- **`azimuth_deg`** has no canonical city orientation and wraps at 360°; likely a
  weak feature. Kept to be faithful to the brief — the Step 9 ablation will show
  whether it earns its place.
- **`density_*`** is an approximation: footprint area of buildings whose *centre*
  lies within the radius, over the disk area, clipped to [0,1]. Good enough as a
  local-clutter proxy; revisit only if the ablation says density matters a lot.

## Reproduce

```bash
conda run -n uav python scripts/show_features.py --city Nafplio --seed 2
conda run -n uav python -m pytest tests/test_features.py -q
```
Figure: `docs/figures/features_<city>.png`.
