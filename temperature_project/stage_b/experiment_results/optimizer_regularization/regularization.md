# Regularization comparison

| Configuration | MAE C | RMSE C | Max C | Fit s | Median ms | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| ols | 2.472769 | 3.116586 | 10.258925 | 0.019717 | 0.057389 | 21,006 |
| ridge_weak | 2.320226 | 2.864986 | 8.936102 | 0.005953 | 0.059860 | 19,951 |
| ridge | 1.851581 | 2.248562 | 6.226269 | 0.004817 | 0.062513 | 19,882 |
| lasso_weak | 1.572389 | 1.995122 | 6.667774 | 3.176018 | 0.074416 | 14,216 |
| lasso | 1.144337 | 1.463817 | 5.068595 | 0.544267 | 0.067633 | 11,197 |
| elastic_net | 1.256918 | 1.613383 | 5.211665 | 0.850404 | 0.061702 | 12,464 |
| rf_shallow | 4.345949 | 5.502127 | 17.543137 | 5.684812 | 6.390382 | 560,594 |
| random_forest | 3.843019 | 4.868436 | 14.760765 | 10.301781 | 6.992583 | 4,015,608 |
| mlp_no_penalty | 2.972676 | 3.664374 | 12.429757 | 0.884102 | 0.034283 | 69,984 |
| mlp_l2 | 2.744256 | 3.436108 | 10.922784 | 0.982698 | 0.035388 | 69,862 |

All metrics measured on the same 240 validation rows / 1,920 temperatures.
See PLAN.md for objective conventions, unsupported combinations and exact penalty strengths.
Rows marked reused in CSV share one identical effective configuration/fit with the canonical run.
