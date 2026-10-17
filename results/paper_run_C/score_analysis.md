# Score analysis: `results/paper_run_C`

## Warm-up score percentiles, per run

| run | model | n | min | p1 | p5 | p10 |
|---|---|---|---|---|---|---|
| run_01 | flow | 430 | -0.0793 | -0.0469 | 0.0000 | 0.0281 |
| run_01 | host-pair | 232 | -0.1204 | -0.0814 | 0.0000 | 0.0152 |
| run_02 | flow | 448 | -0.0721 | -0.0479 | -0.0000 | 0.0265 |
| run_02 | host-pair | 232 | -0.1934 | -0.0809 | -0.0000 | 0.0325 |
| run_03 | flow | 432 | -0.0874 | -0.0368 | 0.0000 | 0.0404 |
| run_03 | host-pair | 232 | -0.0856 | -0.0574 | 0.0000 | 0.0833 |
| run_04 | flow | 404 | -0.0702 | -0.0596 | 0.0000 | 0.0485 |
| run_04 | host-pair | 232 | -0.1529 | -0.1413 | 0.0000 | 0.0169 |
| run_05 | flow | 436 | -0.0616 | -0.0442 | -0.0000 | 0.0341 |
| run_05 | host-pair | 232 | -0.0909 | -0.0637 | 0.0000 | 0.0629 |

## Per-class score distribution (pooled_flow_score_distribution / pooled_pair_score_distribution)

| class | flow n_rows | flow min | flow median | flow max | pair n_rows | pair min | pair median | pair max |
|---|---|---|---|---|---|---|---|---|
| benign | 3938 | -0.0888 | 0.1287 | 0.2559 | 0 | n/a | n/a | n/a |
| null_scan | 2556 | -0.0223 | 0.0404 | 0.1057 | 189 | -0.1854 | -0.0412 | 0.1248 |
| rate_spike | 721 | -0.0750 | -0.0438 | -0.0077 | 582 | -0.0748 | 0.1232 | 0.2265 |
| sqli | 1279 | 0.0072 | 0.1698 | 0.2342 | 218 | -0.0866 | 0.0335 | 0.2210 |
| syn_scan | 2554 | -0.0598 | 0.0374 | 0.1261 | 201 | -0.1854 | -0.0415 | 0.1248 |

## Per-class ROC-AUC (class vs. benign, by -score)

| class | flow-level AUC | pair-level AUC |
|---|---|---|
| null_scan | 0.7229 | 0.7832 |
| rate_spike | 0.9541 | 0.3207 |
| sqli | 0.3811 | 0.6740 |
| syn_scan | 0.7328 | 0.8313 |

_AUC 0.5 = score carries no separating information; 1.0 = perfect separation. `n/a` means one class (positive or benign) had no matching rows to compare against._