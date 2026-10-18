# Score analysis: `results/paper_run_D`

## Warm-up score percentiles, per run

| run | model | n | min | p1 | p5 | p10 |
|---|---|---|---|---|---|---|
| run_01 | flow | 424 | -0.0565 | -0.0389 | -0.0000 | 0.0326 |
| run_01 | host-pair | 234 | -0.1348 | -0.1067 | -0.0000 | 0.0626 |
| run_02 | flow | 381 | -0.0993 | -0.0613 | 0.0000 | 0.0334 |
| run_02 | host-pair | 232 | -0.1476 | -0.1155 | -0.0000 | 0.0617 |
| run_03 | flow | 404 | -0.1024 | -0.0398 | -0.0000 | 0.0380 |
| run_03 | host-pair | 232 | -0.1545 | -0.0990 | 0.0000 | 0.0564 |
| run_04 | flow | 428 | -0.0648 | -0.0488 | -0.0000 | 0.0327 |
| run_04 | host-pair | 236 | -0.1431 | -0.0827 | 0.0000 | 0.0663 |
| run_05 | flow | 426 | -0.0844 | -0.0486 | 0.0000 | 0.0371 |
| run_05 | host-pair | 232 | -0.1491 | -0.0779 | 0.0000 | 0.0906 |

## Per-class score distribution (pooled_flow_score_distribution / pooled_pair_score_distribution)

| class | flow n_rows | flow min | flow median | flow max | pair n_rows | pair min | pair median | pair max |
|---|---|---|---|---|---|---|---|---|
| benign | 3937 | -0.0994 | 0.1324 | 0.2474 | 0 | n/a | n/a | n/a |
| null_scan | 2400 | -0.0566 | 0.0059 | 0.0565 | 242 | -0.1545 | -0.0557 | 0.1793 |
| rate_spike | 714 | -0.0804 | -0.0538 | -0.0103 | 594 | -0.1436 | 0.1323 | 0.2559 |
| sqli | 1284 | -0.0404 | 0.1753 | 0.2498 | 204 | -0.1256 | 0.0445 | 0.2559 |
| syn_scan | 2490 | -0.0566 | 0.0017 | 0.0778 | 243 | -0.1832 | -0.0557 | 0.1552 |

## Per-class ROC-AUC (class vs. benign, by -score)

| class | flow-level AUC | pair-level AUC |
|---|---|---|
| null_scan | 0.8108 | 0.7446 |
| rate_spike | 0.9429 | 0.4190 |
| sqli | 0.3278 | 0.6563 |
| syn_scan | 0.8133 | 0.7943 |

_AUC 0.5 = score carries no separating information; 1.0 = perfect separation. `n/a` means one class (positive or benign) had no matching rows to compare against._