# Score analysis: `results/paper_run_full/run_01`

## Warm-up score percentiles

| model | n | min | p1 | p5 | p10 |
|---|---|---|---|---|---|
| flow | 424 | -0.1190 | n/a | -0.0000 | n/a |
| host-pair | 232 | -0.1108 | n/a | 0.0000 | n/a |

_p1/p10 unavailable: this run predates controller/model.py persisting them (warmup_scores/warmup_p1/warmup_p10 in model.json)._

## Per-class flow-level score distribution (flow_score_distribution)

| class | n_rows | min | median | max |
|---|---|---|---|---|
| benign | 793 | -0.1417 | 0.1140 | 0.2253 |
| null_scan | 512 | -0.0009 | 0.0541 | 0.1208 |
| rate_spike | 144 | -0.1141 | -0.0805 | -0.0544 |
| sqli | 257 | -0.0312 | 0.0656 | 0.2159 |
| syn_scan | 512 | -0.0009 | 0.0544 | 0.1189 |

## Per-class ROC-AUC (class vs. benign, by -score)

| class | flow-level AUC | pair-level AUC |
|---|---|---|
| null_scan | 0.6922 | 0.7237 |
| rate_spike | 0.9698 | 0.4402 |
| sqli | 0.6390 | 0.7670 |
| syn_scan | 0.6924 | 0.7304 |

_AUC 0.5 = score carries no separating information; 1.0 = perfect separation. `n/a` means one class (positive or benign) had no matching rows to compare against._