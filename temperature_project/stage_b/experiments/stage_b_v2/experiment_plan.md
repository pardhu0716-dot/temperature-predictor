# Stage B v2 Experimental Plan

## 1. Scientific Hypothesis (Grounded in Phase 1 Error Analysis)

Phase 1 showed that Stage B v1 (8 independent Lasso models over 252 slot-indexed features) suffers from:
1. **Core-slot overfitting**: `Core_0_L0` validation MAE is `0.61 °C` while `Core_6_L0` is `1.63 °C`, despite identical physical laws across all 8 cores.
2. **Missing power × geometry multiplicative physics**: Steady-state heat conduction satisfies $\Delta T_i = \sum_j R_{ij}(\text{geom}) P_j$. Additive geometry terms independent of power cannot scale with total power or die area.
3. **Underprediction on compact / high-aspect-ratio dies**: Missing global spreading terms ($P_{\text{total}}/A_{\text{die}}$, $P_{\text{total}}/\sqrt{A_{\text{die}}}$, aspect-ratio scaling) and adiabatic die-wall reflection (method of images) cause `-1.7` to `-2.7 °C` negative bias on compact layouts (`6×14 mm`, `12×6 mm`, `10×6 mm`).
4. **Within-chip relative coupling error (`1.05–1.17 °C` centered MAE)**: Because the hottest and second-hottest cores are typically separated by only `0.85–1.17 °C`, reducing within-chip relative error is essential to lift hottest-core accuracy above `65%`.

## 2. Leakage-Prevention Protocol

- **Original Dataset**: `stage_b/stage_b_training_dataset.csv` (`77` floorplans, `1,540` workloads, SHA-256 `ad78ac96...`) — preserved read-only.
- **Exact Original Split** (`seed=42`, from `stage_b/models/final_lasso/manifest.json`):
  - **Train**: `53` floorplans (`1,060` workloads; `8,480` core observations)
  - **Validation**: `12` floorplans (`240` workloads; `1,920` core observations)
  - **Test**: `12` floorplans (`240` workloads; `1,920` core observations) — **never accessed during feature ablation or model selection**; evaluated once after freezing the selected candidate.
- **External 23 Stage A Floorplans (`460` HotSpot workloads)**: Never used for training, scaling, or hyperparameter tuning. Disclosed as a non-untouched external benchmark because v1 errors were previously inspected.

## 3. Controlled Feature Ablation Plan (Phase 2)

We evaluate a **permutation-equivariant shared per-core feature representation** ($\phi_i$ for target core $i \in \{0,\dots,7\}$, predicting $\Delta T_i = T_i - 45.0\,^\circ\text{C}$), adding feature groups one by one on the 53-train / 12-validation split:

- **F0 (Baseline Multi-Output v1)**: Original 252 slot-indexed features (`stage_b/ml/features.py`), 8 separate regressors.
- **F1 (Shared Self + Global Power)**: Target core power $P_i$, core area $A_i$, power density $P_i/A_i$, `is_big` flag, $P_i \cdot \text{is\_big}$, total chip power $P_{\text{tot}}$, other-core power $P_{\text{tot}} - P_i$, Big/Little total power split.
- **F2 (+ Multi-Scale Pairwise Coupling)**: Aggregated neighbor coupling $\sum_{j \ne i} P_j K(d_{ij})$ using physically motivated 2D heat kernels: $1/(d+c)$, $1/(d^2+c)$, $\exp(-d/\lambda)$ for $\lambda \in \{2, 4, 8\}\,\text{mm}$, $\ln(1 + 10/(d+0.5))$, edge-gap coupling $P_j/(g_{ij}+0.5)$, Big vs Little neighbor coupling, and nearest-neighbor coupling/distance.
- **F3 (+ Die Spreading & Aspect-Ratio Physics)**: Global spreading resistance proxies $P_{\text{tot}}/A_{\text{die}}$, $P_{\text{tot}}/\sqrt{A_{\text{die}}}$, $P_{\text{tot}} \cdot \text{AR}$, $P_{\text{tot}} \cdot \text{whitespace}$, normalized die dimensions, die perimeter, and $P_i / \sqrt{A_{\text{die}}}$.
- **F4 (+ Die-Boundary & Method-of-Images Reflections)**: Distances from core $i$ to all 4 die edges (`left`, `right`, `bottom`, `top`), min/second-min edge distance, relative normalized position $(x_c/W, y_c/H)$, center offset from die center, self-reflection coupling $P_i \cdot \sum_{\text{4 walls}} K(2 d_{\text{wall}})$ (adiabatic Neumann mirror images), and neighbor-reflection coupling $\sum_{j \ne i} P_j K(d_{\text{image},ij})$ across the 4 adiabatic die edges.
- **F5 (+ Self-Power × Local Geometry Interactions)**: Interactions of $P_i$ and $P_i/A_i$ with core boundary proximity, whitespace fraction, and local power concentration (power within 4 mm / 6 mm radius).

## 4. Lightweight Model Comparison Plan (Phase 3)

On the best validation-selected feature set (and compared against v1's 252 features):
1. Ordinary Linear Regression (`OLS`)
2. `Ridge` with `alpha` in `[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]`
3. `Lasso` with `alpha` in `[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]`
4. `ElasticNet` with `alpha` in `[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]` × `l1_ratio` in `[0.2, 0.5, 0.8, 0.9, 0.95]`

All models use training-only `StandardScaler` on features and target temperature rise, with convergence checks (`max_iter=20000`, `tol=1e-5`). Selection criterion: lowest validation MAE (with RMSE, hottest-core accuracy, model size `< 25 KB`, and pipeline latency `< 2 ms` as tie-breakers/constraints).
