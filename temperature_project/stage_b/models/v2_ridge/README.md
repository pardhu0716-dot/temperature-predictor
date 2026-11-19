# Stage B v2 Production Model (`v2_ridge`)

Portable, pickle-free production export of the audited **Stage B v2** shared per-core physics-informed Ridge regression model (`alpha = 0.01`, `63` features).

## Artifact Inventory

- `model_v2.npz`: `3,028` bytes (`allow_pickle=False` NumPy archive)
  - SHA-256: `edbae799105b57134ab99cfe3a68db2e4afd9c7aeb0ae78b3ec49476899c2175`
  - Contents (`float64`): `coef` `(63,)`, `intercept` `()`, `x_mean` `(63,)`, `x_scale` `(63,)`, `y_mean` `()`, `y_scale` `()` (`512 B` model weights+intercept; `1,536 B` total raw float64 arrays)
- `metadata.json`: feature ordering, groups, hyperparameters, split floorplan IDs (`53` train / `12` val / `12` test, `seed=42`), integrity hashes, and audited metrics.
- `export_v2_production.py`: reproducible export and `< 1e-12 °C` equivalence verifier against the untouched experimental artifact (`stage_b/experiments/stage_b_v2/candidate_model/model_v2.joblib`, `4,267 B`, `SHA-256: e2c39c6d1ca55351e67e051b5e35f2c17a107389fd6102b12badb0b22446507a`).

## Standalone Loading (Fresh Python Process, No Training Code)

```python
import numpy as np
from stage_b.ml.v2_model import load_v2_production_model

model = load_v2_production_model()  # verifies SHA-256 and loads model_v2.npz (allow_pickle=False)
# geometry_mm: (8, 4) [w, h, x, y], powers_w: (8,), die_mm: (2,) [W, H]
temps_c = model.predict(geometry_mm, powers_w, die_mm)  # shape (1, 8) in °C
```

## Independently Reproduced Benchmark (v1 vs v2)

| Dataset | Stage B v1 MAE | Stage B v2 MAE | Stage B v2 RMSE | Stage B v2 Hottest-Core Acc |
| :--- | :---: | :---: | :---: | :---: |
| Validation (`12` FPs / `240` wl) | `1.144337 °C` | `0.419293 °C` | `0.572021 °C` | `82.92%` |
| Held-out test (`12` FPs / `240` wl) | `1.070281 °C` | `0.350588 °C` | `0.445882 °C` | `84.17%` |
| External 23 Stage A FPs (`460` wl) | `1.491469 °C` | `0.460754 °C` | `0.643017 °C` | `79.78%` |

> **Evaluation Disclosure**: The 12 held-out Stage B test floorplans and 23 external Stage A floorplans were evaluated during earlier v1/v2 development and should be treated as reused reference benchmarks, not untouched blind test sets or accuracy guarantees.
