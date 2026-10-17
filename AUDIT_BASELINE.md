# Audit baseline (before `paper-impl` changes)

This file records how the code behaved **before** the paper-implementation work, so
later stages can be diffed against it. All `file:line` references are to commit
`282dccc` ("Snapshot pre-existing uncommitted working tree"). That commit holds the
edits that were uncommitted on `main` when this branch was created; `main` is unchanged.

The brief does not number its gaps. Below, items 1–9 follow the order of the brief
(Stage 1 → Stage 5). Items 10–12 cover the evaluation and results gaps in Stages 6–7.

## 1. Mirroring session 100 is never configured
- `controller/dataplane.py:243-252` `set_mirroring_session()` and
  `controller/dataplane.py:319-327` `install_mirroring()` exist, but nothing calls them
  (`grep -rn install_mirroring` matches only the definition).
- `p4/adaptmon.p4:11-12` defines `CPU_PORT 255` / `CLONE_SESSION 100`, but no switch
  binds port 255: `topo/namespace_topo.py:134-145` (`SW_IFACES`) gives each switch only
  ports 1–3.
- Result: `clone_preserving_field_list(..., 100, ...)` targets an unconfigured session,
  so bmv2 makes no clones.

## 2. `cpu_hdr` is set on the forwarded original
- `p4/adaptmon.p4:326-334` (egress): when `meta.do_clone == 1`, it calls
  `hdr.cpu_hdr.setValid()` on the packet being processed, without checking
  `standard_metadata.instance_type`, and only then requests the E2E clone.
- The deparser emits `cpu_hdr` first (`p4/adaptmon.p4:371`), so every forwarded packet
  of a cloned flow leaves with 13 extra bytes before its Ethernet header. The receiving
  host cannot parse those packets.

## 3. `@field_list(0)` is empty
- `p4/adaptmon.p4:76` puts `@field_list(0)` on the `metadata_t` **struct**, not on its
  fields; `p4/adaptmon.p4:333` clones with field list 0.
- p4c warns about this (`p4/build/compile.log`: "no user metadata fields tagged with
  @field_list(0)"), and the compiled JSON has `field_lists: [{"name": "empty", "elements": []}]`.
  A clone would therefore carry no flow index or level.

## 4. No per-flow `flow_level` register; clone/drop decisions are per host pair
- Registers: `p4/adaptmon.p4:86-93`. There is no `flow_level` register.
- Cloning is driven by `clone_table`, keyed on (srcAddr, dstAddr)
  (`p4/adaptmon.p4:221-232`). Dropping uses `drop_table`, keyed on
  (srcAddr, dstAddr, protocol) (`p4/adaptmon.p4:207-219`). Both act on every flow
  between two hosts, not on one 5-tuple.
- The flow hash uses `max = MAX_FLOWS-1` (`p4/adaptmon.p4:277`), so only slots 0..4094
  are reachable. Slot 4095 is never used.

## 5. Detection is switch-global, not per flow
- Each poll builds one 10-feature vector per switch (`controller/monitor.py:150`,
  `controller/features.py:204-258`; features listed at `controller/model.py:20-31`).
  It is scored once, and the resulting level applies to the whole switch
  (`controller/monitor.py:166-177`).
- Per-flow features exist (`controller/features.py:260-283`, `extract_per_flow_features`)
  but nothing calls them. `PerFlowDetector` is constructed (`controller/monitor.py:385`)
  but never used.
- Entropy is the switch-wide entropy of per-slot cumulative packet counts
  (`controller/features.py:140-159`). There is no payload entropy.

## 6. Isolation Forest training
- `controller/model.py:106-116`: Pipeline(log1p → StandardScaler → IsolationForest
  `n_estimators=100, contamination=0.05, random_state=42`). `max_samples` is not set
  (sklearn default `"auto"`, i.e. min(256, n)).
- With `--train-until` (which `experiments/run_series.py:166` passes), the model trains
  once on samples taken before attack start − 1 poll (`controller/model.py:167-203`).
  Warm-up lasts ~30 s (`experiments/run_series.py:37`, plus `CTRL_STARTUP_S=4` at line 32).
  Without `--train-until` it retrains every 30 samples (`controller/monitor.py:377-384`).
- One model per switch. Training samples are switch-windows (~29 per run), not flows.
- `requirements.txt` has `scikit-learn>=1.3.0` (unpinned); installed version: 1.7.2.

## 7. Thresholds and the z-score trigger
- Thresholds come from the warm-up: min, min−σ, min−2σ (`controller/policy.py:78-101`).
  This already matches the paper, but it is applied to the 3-window **mean** score
  (`controller/policy.py:200-202`), not to each window's score.
- The z-score trigger is active: `controller/monitor.py:164` computes `z_max`, and
  `controller/policy.py:216` takes `max(if_level, z_level)`, so z-scores can escalate
  even when the IF score would not (thresholds at `controller/policy.py:51-53`).

## 8. Hysteresis is time-based and switch-global
- Escalation is immediate (`controller/policy.py:220-222`). De-escalation happens only
  after a time in state: DROP 30 s, MONITOR 60 s, otherwise 5 s
  (`controller/policy.py:223-234`). There is no "k=2 windows up / 3 windows down"
  counting, and no per-flow state.

## 9. Enforcement, DPI, traffic, and controller start-up
- **Enforcement:** at DROP, drop rules go to the top-5 talkers; at CLONE, clone rules go
  to the top-3 talkers (`controller/monitor.py:296-332`). Rules are never removed:
  `delete_drop_rule` (`controller/dataplane.py:189`) has no callers, and no clone-rule
  delete exists.
- **DPI:** there is no `controller/dpi.py`, no CPU-port sniffer and no alert output.
- **Traffic:** benign traffic is 4 Python-socket TCP senders
  (`traffic/ns_traffic.py:287-310`). The only attack is an hping3 `--rand-source` SYN
  flood (`traffic/ns_traffic.py:225-243`). There is no Nmap SYN/NULL scan, no SQLi,
  no iperf3 and no HTTP. iperf3 and nmap are not installed (`traffic/normal.py` wraps
  iperf3, but the real-mode path does not use it). The attack runs 30–90 s of a
  120 s run (`experiments/run_series.py:36-38`).
- **Ground truth:** none. Only `attack_window.json` (start/end epoch) is written
  (`experiments/run_series.py:189-206`).
- **`import json` in `main()`:** the original upload had a local `import json` at
  `controller/controller.py:185` (commit `ee4154c`), which makes `json` local to all of
  `main()` and raises `UnboundLocalError` at `json.load` (line 113) whenever
  `--switches` is given. Snapshot `282dccc` has already removed it; only the
  module-level import (`controller/controller.py:13`) remains.

## 10. Evaluation
- `experiments/metrics.py:149-218` and `paper/make_result_figs.py` evaluate per
  **switch-window**, with ground truth from the attack time window. There are no
  flow-level metrics, Clopper-Pearson bounds, DPI metrics or inspection reduction.

## 11. Simulated benchmarks
- `experiments/metrics.py:221-280` `compute_overhead_vs_flows()` ("Synthetic benchmark")
  runs on random data and fits a fresh IF on random noise. This is what
  `run_scalability_experiment` writes to `results/scalability.json`.
- `analysis/overhead.py:155-280` `benchmark_register_poll` / `benchmark_feature_extraction`
  / `benchmark_model_inference` time loops over `np.random` data, not real switches.
- The mock mode (`experiments/run_series.py:266-301`, `analysis/plots.py:_generate_synthetic_df`)
  produces synthetic CSVs.

## 12. Existing artefacts in `results/`
- `results/*.png`, `results/report.md`, `results/scalability.json` and
  `results/overhead_benchmark.json` come from the switch-global detector and/or the
  simulated benchmarks above. Nothing in this branch's `RESULTS.md` uses them.
