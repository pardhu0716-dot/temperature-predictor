# Fixed validation-only optimizer/regularization plan

Source commit: 39b5494b5d9da5a6956aa572fb6b58c399a6f5ef.
No framework or experiment-definition changes. Source configuration groups are
used verbatim; the additional matrix is external run-only configuration JSON.

- Existing optimizers: 6 entries (linear/MLP x batch GD/SGD/Adam).
- Existing regularization: 10 entries, unchanged.
- Additional combinations: linear/MLP x batch GD/SGD/Adam x none/L1/L2/L1+L2 = 24.
- Dense combination strengths fixed before inspecting results: L1=0.0001,
  L2=0.001; combined uses both. These are not a tuned strength grid.
- Rates follow existing optimizer group: GD=.01, SGD=.001, Adam=.003.
- Epoch budget=300, configured batch_size=32, patience=20, tolerance=1e-7,
  gradient_clip=10, seed=42. Effective batch: all 1060 training rows for GD,
  1 for SGD, 32 for Adam. MLP hidden layer=32 tanh, output layer=8 linear.
- Exact duplicate effective estimator configurations reuse a single new fit,
  preserving its timing/model bytes and recording the canonical run name.
- Same already-saved seed-42 preparation; no split regeneration/scaler refit.
- Test arrays prohibited by runtime guard; old artifact writes prohibited.
- Full validation metrics, fit time, inference latency (50 repetitions after
  warmup), compressed bundle size, warnings and actual epoch counts recorded.

Dense objective: mean over rows AND 8 outputs of squared scaled-target error,
plus l1*sum(abs(weights)) + l2*sum(weights**2). Biases are not penalized.
L1 uses a sign subgradient: technically not differentiable at zero. Combined
L1+L2 is an elastic-net-style penalty, not the sklearn ElasticNet estimator,
alpha convention, or coordinate-descent solver. Dense models have no alpha
argument; alpha is N/A and l1/l2 are reported explicitly. Native sklearn Lasso,
Ridge and ElasticNet alpha values cannot be directly equated with dense l1/l2.

Supported and unsupported coverage:

| Implementation | Native solver | GD/SGD/Adam selectable? | Penalty/control |
|---|---|---|---|
| NumPy linear/MLP | GD, SGD, Adam | All three, all tested | none, L1, L2, simultaneous L1+L2 |
| sklearn OLS | least squares | No | no explicit penalty |
| sklearn Ridge | auto linear solver | No | L2 alpha .1, 1 |
| sklearn Lasso | coordinate descent | No | L1 alpha .0001, .001 |
| sklearn ElasticNet | coordinate descent | No | alpha .001, l1_ratio .5 |
| Random Forest | tree construction | No: not gradient-based | depth/leaf controls, NOT L1/L2 |
| XGBoost | gradient tree boosting | No: framework has no SGD/Adam switch | Existing core comparison retained; no XGB entry in existing regularization group |

RF/boosted-tree x batch-GD/SGD/Adam are unsupported; tree split/boosting algorithms
must not be labeled Adam/SGD. sklearn OLS/Ridge/Lasso/ElasticNet do not accept the
framework's optimizer selector. Dense linear models are the separate supported
alternative; they do not convert those native solvers. XGBoost native reg_alpha/
reg_lambda could be exposed via estimator kwargs, but are outside these existing
groups and this fixed dense combination matrix; no extra tree penalty sweep is
silently implied. Trees with versus without depth constraints are not native
L1-versus-L2 comparisons. Existing RF shallow/default controls are compared;
a new fully unconstrained forest is not claimed to have been tested.

Interpretation: rates and update counts differ among optimizers. This is a
comparison of the defined configurations, not an equal-compute search proving
one optimizer universally superior. Hitting 300 epochs is recorded as an epoch
budget limit, not presumed convergence. Early stopping uses TRAINING loss only;
selection uses validation RMSE only. Repeated validation selection can introduce
selection optimism; the held-out test remains reserved and unevaluated.
