# Score analysis: `results/paper_run_percentile`

## Warm-up score percentiles, per run

| run | model | n | min | p1 | p5 | p10 |
|---|---|---|---|---|---|---|
| run_01 | flow | 426 | -0.1211 | -0.0637 | 0.0000 | 0.0437 |
| run_01 | host-pair | 232 | -0.1726 | -0.1084 | -0.0000 | 0.0526 |
| run_02 | flow | 424 | -0.1342 | -0.0701 | 0.0000 | 0.0329 |
| run_02 | host-pair | 232 | -0.1259 | -0.1089 | 0.0000 | 0.0396 |
| run_03 | flow | 426 | -0.1120 | -0.0613 | 0.0000 | 0.0336 |
| run_03 | host-pair | 232 | -0.1702 | -0.1075 | 0.0000 | 0.0258 |
| run_04 | flow | 424 | -0.1148 | -0.0913 | 0.0000 | 0.0368 |
| run_04 | host-pair | 232 | -0.1111 | -0.0863 | 0.0000 | 0.0227 |
| run_05 | flow | 427 | -0.1191 | -0.0721 | 0.0000 | 0.0291 |
| run_05 | host-pair | 232 | -0.1812 | -0.0811 | -0.0000 | 0.0493 |

## Per-class score distribution (pooled_flow_score_distribution / pooled_pair_score_distribution)

| class | flow n_rows | flow min | flow median | flow max | pair n_rows | pair min | pair median | pair max |
|---|---|---|---|---|---|---|---|---|
| benign | 3973 | -0.1497 | 0.1353 | 0.2469 | 0 | n/a | n/a | n/a |
| null_scan | 2560 | -0.0566 | 0.0127 | 0.1316 | 242 | -0.0977 | 0.0170 | 0.0643 |
| rate_spike | 722 | -0.1304 | -0.0874 | 0.1446 | 644 | -0.1207 | 0.0600 | 0.2316 |
| sqli | 1280 | -0.0240 | 0.0781 | 0.2320 | 162 | -0.1207 | -0.0075 | 0.2030 |
| syn_scan | 2560 | -0.0566 | 0.0127 | 0.1316 | 246 | -0.0977 | -0.0176 | 0.1436 |

## Per-class ROC-AUC (class vs. benign, by -score)

| class | flow-level AUC | pair-level AUC |
|---|---|---|
| null_scan | 0.7500 | 0.7724 |
| rate_spike | 0.9612 | 0.4504 |
| sqli | 0.6180 | 0.7078 |
| syn_scan | 0.7527 | 0.7529 |

_AUC 0.5 = score carries no separating information; 1.0 = perfect separation. `n/a` means one class (positive or benign) had no matching rows to compare against._