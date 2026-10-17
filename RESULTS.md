# AdaptMon: real-data detection results

**Run C is the final evaluation.** Runs A and B are development results,
kept and labelled below. All three are runs of the real system —
namespaced bmv2 topology, the real controller and DPI processes,
`traffic/paper_traffic.py`'s generators — scored against their own
`traffic/ground_truth.py` output by `analysis/evaluate.py` and
`analysis/score_analysis.py`. **None installs the `host_clone_table`
shortcut the smoke tests use to guarantee DPI visibility**: detection
happens the way the deployed system would do it, unassisted.

Every flow in a run has a unique 5-tuple, so ground truth joins to
controller output by exact tuple. A flow counts as detected if its own
level, or its host pair's level while it was active, reached CLONE; the
*trigger* is what escalated it (IF-flow, IF-pair, DPI-head, DPI-sample).
CIs are exact (Clopper-Pearson) 95% intervals.

---

## Final evaluation — Run C (`results/paper_run_C`, 5 runs, 1,310 flows)

### What Run C is

Decided before any Run C (or pilot) data existed:

1. **Flow-head cloning, N = 3.** `p4/adaptmon.p4` clones the first 3
   packets of every new flow to the CPU port (`clone_reason` 4), on top of
   level-based cloning. A flow is new on an empty slot or when the slot's
   stored 5-tuple differs (so a hash-colliding flow gets its own head too).
   N is the controller's `--head-clone-n`, written to a `head_clone_n`
   register at startup and logged with the value read back from each
   switch (`{1: 3, 2: 3}` in every run). Priority: level (1) > head (4) >
   sampling (2); a host-pair rule (3) overrides all, as before.
2. **DPI processes head clones like any other clone.** Each confirmed
   alert carries the `clone_reason` that put the flow's packet on the CPU
   port; a confirmation below CLONE force-clones the flow (and, for
   `syn_scan`/`null_scan`, its host pair) and is logged as **DPI-head**
   (reason 4) or **DPI-sample** (reason 2). A confirmation on a packet that
   was cloned because of a host-pair rule (reason 3) is credited to
   whatever escalated that pair.
3. **Detector unchanged from Run B**: IF thresholds MONITOR < warm-up p5,
   CLONE < warm-up p1; K_UP/K_DOWN, IsolationForest hyperparameters and
   `random_state`, sampling rates, DROP rule — all as in Run B.
4. **Inspection reduction = 1 − cloned / forwarded**, cloned = every copy
   to a CPU port (head + sampled + level + host-pair), packets and bytes,
   from the final cumulative data-plane counters (summed over switch hops).
5. **Load budget.** Run B had exceeded it (see "Development results"), and
   the cause was the 1,500 pkt/s UDP rate spike being cloned at CLONE, not
   nmap. So Run C's spike is **1,000 pkt/s** (nmap stays at `--max-rate
   500`), with a rule fixed in advance: if a pilot window exceeded 2,500
   pkt/s or DPI dropped any clone, step down to 750. The 1-run dev pilot
   (`results/dev_pilot_headclone`, dev seed) peaked at 1,209 pkt/s with 0
   drops, so 1,000 stood.
6. **Seeds.** Each run's traffic schedule is a pure function of its seed
   (`experiments/paper_run.py:build_schedule`): per cycle, the 8 scan
   ports (1–1023), the order of the four attack phases, benign phase
   length (6–10 s), iperf3 rate (1–3 Mbit/s), HTTP body choice and gap
   jitter, SQLi body order; plus the warm-up iperf3 rate. Detector
   parameters are not seeded. Run C's five seeds were drawn from
   `secrets.randbits(31)` at launch and recorded in
   `experiments/seeds_used.json`, which `paper_run` checks — the only seed
   used before them was the pilot's. (Runs A and B used the fixed,
   unseeded schedule.)

Verified before the run: `tests/smoke_headclone.py` on real bmv2 (13/13 —
packets 1–3 of a new flow cloned, the 4th not; level CLONE clones every
packet; MONITOR head then 1-in-8; collision and aged slot restart the head;
N = 0 disables; N changed at runtime applies; per-reason packet and byte
counters exact), the other five real-switch suites (40/40), and 80 unit tests.

### Detection

| | TP | FN | FP | TN | TPR | FPR |
|---|---|---|---|---|---|---|
| Run C, pooled | 755 | 85 | 37 | 433 | **0.899** [0.876, 0.918] | **0.079** [0.056, 0.107] |

Per run: TPR 0.899 / 0.905 / 0.857 / 0.905 / 0.929, FPR 0.074 / 0.072 /
0.089 / 0.087 / 0.073.

| class | n | detected | recall [95% CI] | by trigger, recall [95% CI] | DPI-confirmed |
|---|---|---|---|---|---|
| sqli | 160 | 160 | **1.00** [0.98, 1.00] | DPI-head 1.00 [0.98, 1.00] | 160 |
| null_scan | 320 | 320 | **1.00** [0.99, 1.00] | DPI-head 0.97 [0.95, 0.99] · IF-pair 0.03 [0.01, 0.05] | 320 |
| syn_scan | 320 | 248 | **0.78** [0.73, 0.82] | DPI-head 0.72 [0.67, 0.77] · IF-pair 0.05 [0.03, 0.08] | 160 |
| rate_spike | 40 | 27 | **0.68** [0.51, 0.81] | IF-flow 0.68 [0.51, 0.81] | 0 (no DPI rule) |
| benign | 470 | 37 FP | FPR 0.079 | IF-flow 0.062 [0.042, 0.087] · DPI-head 0.015 [0.006, 0.030] · IF-pair 0.002 [0.000, 0.012] | 0 |

DPI-sample triggered nothing: with head cloning, every SQLi and scan flow
is already in front of DPI on its first packets, before any 1-in-64 sample.
No benign flow was ever dropped (DROP still needs a second confirmation on
a flow at CLONE), and **DPI confirmed nothing on any benign flow** in any
run.

What the misses and false positives are:

- **syn_scan misses (72/320) are all one case**: a SYN scan that runs
  first in its cycle, right after the benign phase. In those 9 cycles
  0/72 were detected; in every other position 248/248 were. The benign
  phase runs iperf3 over the same h1→h3 pair; its last cloned packet lands
  ~1.5 s (`PHASE_GAP_S`) before the scan, inside DPI's 3 s SYN-ratio
  window, and with an 8-port scan **one** cloned non-SYN packet from the
  same pair puts the SYN-only ratio at 8/9 = 0.889, under the rule's 0.9.
  So DPI's syn-scan rule can be suppressed by legitimate traffic between
  the same two hosts — a real limitation of the rule, not tuned away
  here. (When a NULL scan precedes the SYN scan, DPI's own syn_scan
  confirmation drops to 16/104 for the same reason, but those probes are
  still all detected: the NULL scan has already force-cloned the pair.)
- **rate_spike (27/40) is seed-sensitive**: 7, 0, 8, 8, 4 of 8 per run.
  At 1,000 pkt/s its flow score sits right at the p1 boundary (≈ −0.04 to
  −0.05 against per-run p1 of −0.037 to −0.060), so small differences in
  each run's trained model decide it; the p1 value alone does not explain
  which runs caught it. This is the cost of the load-budget decision
  (Run B's 1,500 pkt/s spike was caught 40/40, while breaking the budget).
- **All 37 false positives are iperf3 flows; no HTTP benign flow was
  flagged.** 29 are the flow model scoring an iperf3 flow anomalous (Run B
  had 27 of the same). 8 are pair spillover: an iperf3 flow on h1→h3
  active while that pair was force-cloned by a scan confirmation (7,
  DPI-head) or an IF-pair escalation (1) — the flow's own level never
  left NORMAL. This topology carries benign and attack traffic on the
  same host pairs by design, which is what makes spillover visible.

### Latency (detection time − ground-truth flow start)

| | median | p95 |
|---|---|---|
| DPI-head escalations (704) | 0.43 s | 1.01 s |
| IF-pair (24) | 0.67 s | 3.78 s |
| IF-flow (27) | 1.48 s | 2.01 s |
| DPI match itself — sqli / null_scan / syn_scan | 0.02 / 0.11 / 0.12 s | 0.03 / 0.13 / 2.71 s |

Escalation trails the DPI match by up to one 1 s poll: confirmations are
acted on at the controller's next poll. Flow start is when the generator
recorded the flow, so nmap's process start-up is included.

### Inspection reduction, load, resources

| | Run C | Run B (dev) |
|---|---|---|
| inspection reduction, packets | **59.6%** (per run 55.8 / 83.0 / 42.4 / 50.9 / 65.1%) | 59.3% |
| inspection reduction, bytes | **64.9%** | — (not recorded) |
| clones ÷ forwarded: level / host-pair / sampling / head | 30.4% / 4.7% / 3.6% / **1.7%** | — |
| peak load, originals + clones per switch | 2,239 pkt/s, 0 windows > 2,500 | 3,274 pkt/s, 185 windows > 2,500 |
| DPI capture drops | **0** | 5,732 |
| controller poll time, mean / worst-run p95 / max | 108 / 161 / 1,130 ms | 112 / 209 / 1,037 ms |
| controller CPU mean / worst-run p95; RSS | 9.5% / 12.8%; 147 MB | 9.6% / 20.8%; 146 MB |
| DPI CPU mean / worst-run p95; RSS | 8.5% / 77%; 183 MB | 10.7% / 100%; 182 MB |

Head cloning itself is cheap on this workload (1.7% of forwarded
packets); the cost that dominates is level cloning of flows held at CLONE
(chiefly the rate spike when caught while still sending, and iperf3 false
positives), which is also why per-run reduction swings with how many
spikes were caught (run_02 caught none: 83.0%). Head cloning's cost scales
with the new-flow arrival rate (N packets × both directions × switch
hops): on `tests/smoke_controller.py`'s older benign generator, which
opens a new TCP connection per message (~6,500 flows in 55 s), head clones
equalled level clones (14,720 per switch).

### AUC (class vs. benign, by −score; `results/paper_run_C/score_analysis.md`)

| class | flow-level AUC | pair-level AUC |
|---|---|---|
| syn_scan | 0.733 | 0.831 |
| null_scan | 0.723 | 0.783 |
| sqli | 0.381 | 0.674 |
| rate_spike | **0.954** | 0.321 |

Unchanged in kind from Run B: the Isolation Forest scores separate the
UDP spike well at the flow level and the scans moderately at the pair
level; SQLi flows score *less* anomalous than benign at the flow level
(AUC < 0.5). Run C's detection gain over Run B does not come from the
scores — it comes from DPI seeing every new flow's first packets.

### Run C takeaways

1. **Flow-head cloning is what makes DPI's content rules count**: SQLi and
   NULL scans go from 5% / 0% (Run B) to 100% / 100%, at a head-clone cost
   of 1.7% of forwarded packets on this workload, with sub-second median
   latency.
2. **DPI's syn-scan ratio rule is suppressible** by any same-pair traffic
   in its 3 s window; that, not sampling, is now the remaining scan gap.
3. **The volumetric attack is the hard case under a load budget**: the
   IF catches a 1,000 pkt/s spike in 68% of instances, seed-dependently.
4. **False positives are the flow model on iperf3 (6.2%) plus host-pair
   spillover (1.7%)**; DPI itself produced none.

---

## Development results — Runs A and B (kept for the record)

Not the final evaluation: both were run while the pipeline was still
being fixed, with the fixed (unseeded) schedule — 30 s benign warm-up,
then 8 cycles of (iperf3 + HTTP benign) → `nmap -sS` → `nmap -sN` → SQLi
burst → 1,500 pkt/s UDP spike, phases 1.5 s apart — and without
flow-head cloning.

### Corrections to earlier versions of this file

- **Inspection reduction was computed wrongly.** `windows.csv`'s
  data-plane counters are cumulative since the controller started; the
  earlier analysis summed them over every window instead of taking the
  final value. Corrected: Run A 91.9% (was 92.9%), Run B 59.3% (was 72.9%).
  `tests/test_evaluate.py` now pins this.
- **Run B exceeded the load budget** (3,050–3,270 pkt/s per switch, 185
  windows over 2,500) and DPI dropped 5,732 clones — not reported before.

Both runs include this session's three correctness fixes, found while
chasing the originally-reported 71% FPR:
- **veth checksum/segmentation offload** (`topo/namespace_topo.py`) was
  silently dropping every real TCP segment relayed through bmv2 — no RST,
  just a timeout. Every benign TCP flow in every *prior* run never
  actually completed; it was retrying SYNs. This alone is a plausible
  major contributor to the original FPR.
- **Ground-truth mislabeling** (`traffic/paper_traffic.py`): attack
  generators passed `attack_class` into the wrong constructor argument, so
  every attack flow's ground truth silently read `"benign"`.
- **DPI's syn-scan ratio was a lifetime counter**, not a window (`controller/dpi.py`):
  one ordinary TCP session to a host pair permanently diluted that pair's
  SYN-only ratio below threshold, masking a real scan of the same pair
  for the rest of the process. Now a 3 s recency window.

They differ only in Decision A's threshold policy and the DPI
confirmation path, described below.

---

### Run A — original Decision A (`results/paper_run_full/run_01`, 1 run, 261 flows)

CLONE threshold = the single most extreme score seen anywhere in warm-up
(`controller/model.py` before this session's change). A DPI confirmation
only ever acted on a flow already at CLONE.

| | TP | FN | FP | TN | TPR | FPR |
|---|---|---|---|---|---|---|
| overall | 0 | 168 | 1 | 92 | **0.000** [0.000, 0.022] | **0.011** [0.000, 0.058] |

| class | n | detected (≥CLONE) | trigger | DPI-confirmed |
|---|---|---|---|---|
| syn_scan | 64 | 0 | — | 0 |
| null_scan | 64 | 0 | — | 0 |
| sqli | 32 | 0 | — | 0 |
| rate_spike | 8 | 0 | — | 0 |

Inspection reduction (1 − cloned/forwarded, final counter values): **91.9%**
(13,470 cloned / 165,296 forwarded packets). *Corrected* — an earlier
version of this file said 92.9%, computed by summing `windows.csv`'s
cumulative counters over every window (see "Corrections" above).
Peak load 1,783 pkt/s per switch; 0 DPI capture drops.

**Root cause (quantified):** the warm-up's single worst benign sample
scored −0.119 (flow model) / −0.111 (pair model). The *best* score any
attack flow reached, any class, was −0.114 (a 5 s / 1500 pkt/s UDP flood —
about as obvious an anomaly as this traffic set produces). Every other
class fell well short. Beating the single most extreme benign sample ever
seen is not a bar any of these attacks had to be anomalous enough to
clear; it only had to be *less* anomalous than whatever the warm-up's
worst outlier happened to be — a fragile basis for a threshold. See
`results/paper_run_full/run_01/score_analysis.md` for the full percentile
and AUC breakdown.

---

### Run B — revised Decision A + sampled-DPI escalation (`results/paper_run_percentile`, 5 runs, 1305 flows)

Two changes, both scoped to this finding:

1. **CLONE threshold = warm-up 1st percentile**, not the raw minimum
   (`controller/model.py`). MONITOR stays the 5th percentile, unchanged.
2. **A DPI confirmation on a flow/pair below CLONE now escalates it
   straight to CLONE** (`force_clone()`, `controller/policy.py`;
   wired in `controller/monitor.py` section 6), bypassing the up-hysteresis
   — waiting for two more anomalous score windows would just leave a
   confirmed attack's packets uninspected in the meantime. A confirmed
   scan class (`syn_scan`/`null_scan`) also force-clones the whole host
   pair, since each probe is its own short-lived flow that per-flow
   hysteresis alone rarely survives long enough to act on. DROP is
   unchanged: still only a second confirmation on a flow already at CLONE.

| | TP | FN | FP | TN | TPR | FPR |
|---|---|---|---|---|---|---|
| overall | 48 | 792 | 27 | 438 | **0.057** [0.042, 0.075] | **0.058** [0.039, 0.083] |

| class | n | detected (≥CLONE) | recall | trigger | DPI-confirmed |
|---|---|---|---|---|---|
| syn_scan | 320 | 0 | 0.00 [0.00, 0.011] | — | 0 |
| null_scan | 320 | 0 | 0.00 [0.00, 0.011] | — | 0 |
| sqli | 160 | 8 | 0.05 [0.022, 0.096] | IF-pair ×8 | 0 |
| rate_spike | 40 | 40 | **1.00** [0.912, 1.0] | IF-flow ×40 | 0 |
| benign | 465 | 27 escalated (all IF-flow) | — | — | 0 dropped |

Inspection reduction (1 − cloned/forwarded, final counter values): **59.3%**
pooled (335,728 cloned / 824,436 forwarded), per run 54.9 / 48.1 / 53.6 /
91.8 / 48.0%. *Corrected* — an earlier version of this file said 72.9%
(see "Corrections"). Lower than Run A's because the rate spike now reaches
CLONE, and at CLONE every one of its packets is cloned; run_04 is the
exception because there it did not stay at CLONE.

**Run B broke Decision D's load budget.** With the 1,500 pkt/s spike at
CLONE, offered load (originals + clones) reached 3,050–3,270 pkt/s per
switch, over the ~2,500 pkt/s budget in 185 poll windows across 4 of the 5
runs, and DPI's capture sockets dropped **5,732 clones** (0 in run_04 and
in Run A). The drops are indiscriminate, so some clones that could have
mattered may have been lost; Run B's numbers carry that caveat.

**DPI-sample escalation never fired, in either run.** Across Run B's 5
runs, DPI processed 223–672 TCP clones per run but **0 HTTP payload bytes
and 0 confirmations, every run** (`run_0N/ctrl/dpi_stats.json`). A SQLi
POST is 2–3 packets; across 160 ground-truth SQLi flows (~400 packets)
at NORMAL-level adaptive sampling (1-in-64), the expected number of
samples landing on the one packet that actually carries the payload is
close to 1 — getting zero across 5 runs is unlucky but well within
plausible variance, not a sign the mechanism is broken (it is exercised
and passing in `tests/test_monitor_loop.py::test_dpi_confirmation_force_clones_then_force_drops`
and `::test_dpi_confirmation_on_scan_class_also_force_clones_the_host_pair`,
and in `tests/smoke_dpi.py` where the flows are forced to CLONE first so
DPI sees every packet). The bottleneck here is sampling probability over
very few packets per flow, not the escalation logic.

**Why syn_scan/null_scan still detect at 0%, even with p1:** not a
remaining software bug — see `results/paper_run_percentile/score_analysis.md`.
Per-run p1 thresholds range −0.061 to −0.091 (flow) and −0.081 to −0.109
(pair) across the 5 runs. The scan classes' best (pooled) score is −0.057
(flow-level) / −0.098 (pair-level) — still short of p1 in every run. The
pair-level *AUC* for both scan classes is 0.75, meaning the score does
carry real separating information (a scan's host-pair feature vector —
8–9 distinct ports, SYN-only ratio ≈1.0, vs. a trained max of 2–3 ports
and ≤0.2 ratio — really is unusual), but the Isolation Forest's score on
that vector isn't extreme enough to cross even the less strict 1st
percentile. Sqli's pair-level AUC (0.71) is similar, and it *does* cross
occasionally (8/160) — the difference is that the pair model is scoring a
per-window aggregate, and a 4-request SQLi burst in `SQLI_GAP_S=0.1`
produces a denser single-window spike (`new_flows`, `pkt_count`) than a
`--max-rate 500` nmap scan's handful of packets does.

---

### Run B AUC (class vs. benign, by −score; `analysis/score_analysis.py`), pooled

| class | flow-level AUC | pair-level AUC |
|---|---|---|
| syn_scan | 0.753 | 0.753 |
| null_scan | 0.750 | 0.772 |
| sqli | 0.618 | 0.708 |
| rate_spike | **0.961** | 0.450 |

0.5 = no separating information, 1.0 = perfect separation. Rate_spike is
genuinely well-separated at the flow level (and correspondingly detects
at 100%); everything else is only moderately separable, which is the
honest reason recall stays low even after fixing the threshold's fragility.

---

### Development takeaways (A/B)

1. **The original 71% FPR was very likely an artifact of the checksum
   bug**, not of the detection design being too aggressive: real TCP
   traffic (the majority of "benign" traffic) never actually completed in
   *any* prior run, which would make benign flows look structurally
   unlike normal traffic to begin with.
2. **With the checksum/labeling/DPI-window bugs fixed and the system
   evaluated honestly (no clone shortcut), both FPR and TPR are low.** The
   system is not "too aggressive" or "too lax" — it mostly doesn't
   separate this traffic well yet. Only the UDP rate flood (a volumetric
   anomaly with no DPI rule to help it) is reliably caught, at 100%, by
   score alone.
3. **The raw-minimum CLONE threshold (original Decision A) is
   demonstrably fragile** — quantified above — and the 1st-percentile
   revision is a strict improvement (TPR 0%→5.7%) but does not fix the
   deeper issue: scan-class aggregates look unusual on inspection (AUC
   0.75) but the Isolation Forest's score doesn't reflect that margin
   strongly enough to cross even a loosened threshold.
4. **DPI's content-based detection (SQLi, scan-pattern confirmation) is
   bottlenecked by adaptive sampling's hit probability on short flows**,
   not by the confirmation/escalation logic itself (unit- and
   smoke-tested working). A real scan or SQLi attempt that happens to run
   long enough, or escalates by score first, gets inspected; a few-packet
   one-off does not, under the sampling rates this run used
   (Decision C: 1-in-64 NORMAL, 1-in-8 MONITOR). This is what Run C's
   flow-head cloning was designed against.

## What this is not

Not a tuned result. Run C's design — N = 3, DPI-head attribution, the
unchanged p5/p1 thresholds, the inspection-reduction definition, the
1,000 pkt/s spike with its 750 fallback rule, and what the seed varies —
was fixed before any Run C data, and the one pilot only checked the load
rule (which it passed). Nothing was adjusted after seeing Run C's
results; its weak spots (the syn-scan rule's suppression, the rate
spike's seed sensitivity, iperf3 false positives) are reported as found.
Runs A and B are kept as labelled development results: the raw-minimum
threshold's fragility (A) is a finding in its own right.

## Reproducing

```
# Final-style series: fresh seeds (refused if already in experiments/seeds_used.json)
scripts/sandbox.sh python3 -m experiments.paper_run --out results/<name> --runs 5 --cycles 8 \
    --seeds <s1,s2,s3,s4,s5> --purpose final [--head-clone-n 3] [--rate-spike-pps 1000]
python3 -m analysis.evaluate results/<name>
python3 -m analysis.score_analysis results/<name>

# Real-switch checks of flow-head cloning and the rest of the data plane
scripts/sandbox.sh python3 tests/smoke_headclone.py
```

Run C's seeds: 1675701711, 1847364810, 1250895948, 1945352977, 764817343
(`results/paper_run_C/run_0N/run_config.json` has each run's full schedule).
Runs A and B: the unseeded schedule, `--seeds` omitted, at
`--rate-spike-pps 1500 --head-clone-n 0` on the code of their time.
