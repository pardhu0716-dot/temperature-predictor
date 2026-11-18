# Combinations comparison

| Configuration | MAE C | RMSE C | Max C | Fit s | Median ms | Bytes |
|---|---:|---:|---:|---:|---:|---:|
| linear_gradient_batch_gd_none | 4.913047 | 6.131626 | 19.753439 | 0.470295 | 0.031195 | 22,783 |
| linear_gradient_batch_gd_l1 | 4.906363 | 6.122905 | 19.704412 | 0.640547 | 0.031027 | 22,801 |
| linear_gradient_batch_gd_l2 | 4.895775 | 6.109899 | 19.680248 | 0.428349 | 0.032512 | 22,808 |
| linear_gradient_batch_gd_l1_l2 | 4.889140 | 6.101253 | 19.631416 | 0.427506 | 0.030246 | 22,799 |
| linear_gradient_sgd_none | 2.609734 | 3.339236 | 11.450524 | 14.884705 | 0.031707 | 22,794 |
| linear_gradient_sgd_l1 | 2.278589 | 2.903142 | 10.535188 | 15.951352 | 0.031811 | 22,841 |
| linear_gradient_sgd_l2 | 2.321036 | 2.941863 | 10.660859 | 14.500878 | 0.031147 | 22,780 |
| linear_gradient_sgd_l1_l2 | 2.162696 | 2.715251 | 9.685952 | 15.158118 | 0.032546 | 22,789 |
| linear_gradient_adam_none | 2.042145 | 2.624305 | 8.210626 | 0.591499 | 0.032216 | 21,534 |
| linear_gradient_adam_l1 | 1.348842 | 1.739552 | 6.026972 | 0.501287 | 0.031745 | 21,566 |
| linear_gradient_adam_l2 | 2.184941 | 2.728169 | 9.023959 | 0.362902 | 0.031258 | 21,225 |
| linear_gradient_adam_l1_l2 | 1.992796 | 2.500118 | 8.384366 | 0.362256 | 0.030875 | 21,212 |
| mlp_batch_gd_none | 4.560223 | 5.641336 | 18.663507 | 0.854258 | 0.034923 | 71,615 |
| mlp_batch_gd_l1 | 4.554133 | 5.634341 | 18.650753 | 0.841162 | 0.035269 | 71,610 |
| mlp_batch_gd_l2 | 4.544467 | 5.622682 | 18.627883 | 1.053503 | 0.035544 | 71,648 |
| mlp_batch_gd_l1_l2 | 4.538443 | 5.615776 | 18.615274 | 0.870853 | 0.035238 | 71,613 |
| mlp_sgd_none | 3.487289 | 4.344457 | 13.000188 | 38.104134 | 0.035047 | 71,680 |
| mlp_sgd_l1 | 2.877274 | 3.632460 | 11.379377 | 36.425068 | 0.035571 | 71,995 |
| mlp_sgd_l2 | 2.559415 | 3.259206 | 10.930467 | 35.056840 | 0.034666 | 71,626 |
| mlp_sgd_l1_l2 | 2.210666 | 2.818907 | 9.448932 | 36.147255 | 0.035365 | 72,069 |
| mlp_adam_none | 2.972676 | 3.664374 | 12.429757 | 0.884102 | 0.034283 | 69,984 |
| mlp_adam_l1 | 2.197660 | 2.798260 | 10.514853 | 1.016105 | 0.035222 | 70,076 |
| mlp_adam_l2 | 2.744256 | 3.436108 | 10.922784 | 0.982698 | 0.035388 | 69,862 |
| mlp_adam_l1_l2 | 2.079679 | 2.672208 | 9.208479 | 0.832363 | 0.035376 | 70,071 |

All metrics measured on the same 240 validation rows / 1,920 temperatures.
See PLAN.md for objective conventions, unsupported combinations and exact penalty strengths.
Rows marked reused in CSV share one identical effective configuration/fit with the canonical run.
