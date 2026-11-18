# Optimizers comparison

| Configuration | MAE C | RMSE C | Max C | Fit s | Median ms | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| linear_gradient_batch_gd | 4.913047 | 6.131626 | 19.753439 | 0.470295 | 0.031195 | 22,783 |
| linear_gradient_sgd | 2.609734 | 3.339236 | 11.450524 | 14.884705 | 0.031707 | 22,794 |
| linear_gradient_adam | 2.042145 | 2.624305 | 8.210626 | 0.591499 | 0.032216 | 21,534 |
| mlp_batch_gd | 4.560223 | 5.641336 | 18.663507 | 0.854258 | 0.034923 | 71,615 |
| mlp_sgd | 3.487289 | 4.344457 | 13.000188 | 38.104134 | 0.035047 | 71,680 |
| mlp_adam | 2.972676 | 3.664374 | 12.429757 | 0.884102 | 0.034283 | 69,984 |

All metrics measured on the same 240 validation rows / 1,920 temperatures.
See PLAN.md for objective conventions, unsupported combinations and exact penalty strengths.
Rows marked reused in CSV share one identical effective configuration/fit with the canonical run.
