"""
analysis/replay.py — offline counterfactuals on a run's logs, for the
ablation in analysis/paper_runC.py. Nothing here re-runs the network:

  * replay_levels(): re-derive per-window monitoring levels of every flow
    slot and every host pair from the LOGGED raw IF levels and/or the
    LOGGED DPI confirmations, through controller/policy.FlowPolicy (the
    controller's own hysteresis code), in the controller's own per-poll
    order (score update, then DPI force, then the row is written).
      mode "IF":  raw IF levels only, DPI forces ignored
      mode "DPI": raw level NORMAL every window (no IF); at the poll that
                  processes each given confirmation, the controller's own
                  action for it: dpi_action "CLONE" (Run C: force_clone, or
                  force_drop on a flow already at CLONE) or "DROP" (Run D on:
                  force_drop on the flow, and on the pair for a scan)
      mode "LOG": the logged levels, untouched (for estimator validation)
  * estimate_clones(): how many packets the P4 clone rule
    (p4/adaptmon.p4) would have sent to the CPU port given per-window
    pkt_delta / pkt_count and a level timeline: host-pair clones if a
    host-clone rule for the source was in place (it overrides the reason
    in P4), else level clones if the slot's level (written at the previous
    poll) was >= CLONE, else the flow's first N packets (head), else
    1-in-64 / 1-in-8 sampling. A packet is dropped instead of forwarded
    if the slot's level was DROP (5-tuple drop rule) or a host-drop rule
    for the source was in place (any of its pairs at DROP); p4/adaptmon.p4
    clones a dropped packet exactly as it would a forwarded one.

Limits, stated wherever these numbers are reported: the traffic and the
IF scores are the logged ones; DPI's own behaviour (which confirmations
it makes) cannot change, though with a different clone stream it might;
the clone estimate assumes the slot's packet counter is the flow's own
(true except across a hash collision).
"""

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from controller.policy import FlowPolicy, MonitoringLevel

CLONE = int(MonitoringLevel.CLONE)
DROP = int(MonitoringLevel.DROP)
NORMAL = MonitoringLevel.NORMAL
CONFIRMATION_RETRY_WINDOWS = 10


def _poll_of(windows: pd.DataFrame, ts: float) -> int:
    """Window number of the first poll at/after ts (the poll that processes
    an alert written at ts)."""
    idx = np.searchsorted(windows["ts"].to_numpy(), ts, side="left")
    return int(windows["window"].iloc[min(idx, len(windows) - 1)])


def replay_levels(fw: pd.DataFrame, pw: pd.DataFrame, windows: pd.DataFrame, mode: str,
                  flow_forces: List[Tuple[int, tuple, float]] = (),
                  pair_forces: List[Tuple[int, str, str, float]] = (),
                  dpi_action: str = "CLONE") -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (flow_levels, pair_levels) aligned with fw / pw rows.
    flow_forces: (switch, five_tuple, alert_ts); pair_forces: (switch, src,
    dst, alert_ts) -- only used in mode "DPI".
    """
    assert dpi_action in ("CLONE", "DROP")
    if mode == "LOG":
        return fw["level"].to_numpy(), pw["level"].to_numpy()

    # Flow forces -> (switch, tuple) -> sorted poll windows at which they fire
    ff = defaultdict(list)
    for sw, tup, ts in flow_forces:
        ff[(sw, tup)].append(_poll_of(windows, ts))
    for k in ff:
        ff[k].sort()

    flow_out = np.zeros(len(fw), dtype=int)
    pol = FlowPolicy()
    keyid = 0
    fws = fw.sort_values(["switch", "slot", "window"])
    for (sw, slot), g in fws.groupby(["switch", "slot"], sort=False):
        cur_tup, last_w, key = None, None, None
        pending = []
        for i, r in zip(g.index, g.itertuples(index=False)):
            tup = (r.src, r.dst, int(r.proto), int(r.sport), int(r.dport))
            # New key on a 5-tuple change, or after a gap (the slot was aged
            # out: the controller forgets the policy state of an aged slot)
            if tup != cur_tup or r.window != last_w + 1:
                keyid += 1
                key, cur_tup = keyid, tup
                pending = list(ff.get((sw, tup), []))
            last_w = r.window
            if mode == "IF":
                if r.scored == 1:
                    pol.update(key, MonitoringLevel(int(r.raw_level)))
            else:   # DPI
                pol.update(key, NORMAL)
                # a confirmation is acted on at the first poll at/after it
                # in which the flow is active, within the retry window
                while pending and pending[0] <= r.window:
                    if r.window - pending[0] <= CONFIRMATION_RETRY_WINDOWS:
                        st = pol.state(key)
                        if dpi_action == "DROP" or st.level >= CLONE:
                            pol.force_drop(key)
                        else:
                            pol.force_clone(key)
                    pending.pop(0)
            flow_out[i] = int(pol.state(key).level)   # fw must have a RangeIndex

    # Pairs: never forgotten (the controller never forgets pair state)
    pf = defaultdict(list)
    for sw, src, dst, ts in pair_forces:
        pf[(sw, src, dst)].append(_poll_of(windows, ts))
    pair_out = np.zeros(len(pw), dtype=int)
    ppol = FlowPolicy()
    pws = pw.sort_values(["switch", "src", "dst", "window"])
    for (sw, src, dst), g in pws.groupby(["switch", "src", "dst"], sort=False):
        key = (sw, src, dst)
        pending = sorted(pf.get(key, []))
        for i, r in zip(g.index, g.itertuples(index=False)):
            if mode == "IF":
                ppol.update(key, MonitoringLevel(int(r.raw_level)))
            else:
                ppol.update(key, NORMAL)
                # pair forces need no active row: state is forced regardless,
                # but it only shows on a row; apply every force up to now
                fired = False
                while pending and pending[0] <= r.window:
                    pending.pop(0)
                    fired = True
                if fired:
                    if dpi_action == "DROP":
                        ppol.force_drop(key)
                    else:
                        ppol.force_clone(key)
            pair_out[i] = int(ppol.state(key).level)
    return flow_out, pair_out


def estimate_clones(fw: pd.DataFrame, pw: pd.DataFrame, flow_levels: np.ndarray,
                    pair_levels: np.ndarray, head_n: int, sampling: bool) -> Dict[str, int]:
    """Packets forwarded, dropped and cloned (by reason) the P4 rules would
    produce, summed over every flow row (all windows, both switches)."""
    fw = fw.assign(rlvl=flow_levels)
    pw = pw.assign(rlvl=pair_levels)
    # Host rule for (switch, src) after poll w, from the highest level of
    # any of its pairs after that poll (levels persist between rows): a
    # host-clone rule at >= CLONE, a host-drop rule (which also clones) at
    # DROP
    rule_after: Dict[Tuple[int, str], List[Tuple[int, int]]] = defaultdict(list)
    for (sw, src), g in pw.sort_values("window").groupby(["switch", "src"]):
        last = {}
        for r in g.itertuples(index=False):
            last[r.dst] = r.rlvl
            rule_after[(sw, src)].append((r.window, max(last.values())))

    def src_level(sw, src, w):
        ev = rule_after.get((sw, src))
        if not ev:
            return 0
        ws = [e[0] for e in ev]
        i = np.searchsorted(ws, w, side="right") - 1
        return int(ev[i][1]) if i >= 0 else 0

    tot = {"forwarded": 0, "dropped": 0, "clone_level": 0, "clone_hostpair": 0,
           "clone_head": 0, "clone_sample": 0}
    fws = fw.sort_values(["switch", "slot", "window"])
    for (sw, slot), g in fws.groupby(["switch", "slot"], sort=False):
        prev_lvl, prev_w, prev_tup = 0, None, None
        for r in g.itertuples(index=False):
            tup = (r.src, r.dst, int(r.proto), int(r.sport), int(r.dport))
            if tup != prev_tup or r.window != (prev_w or -2) + 1:
                prev_lvl = 0
            d = max(int(r.pkt_delta), 0)
            pc = int(r.pkt_count)
            lo = max(pc - d, 0)
            src_lvl = src_level(sw, r.src, r.window - 1)
            dropped = prev_lvl >= DROP or src_lvl >= DROP
            tot["dropped" if dropped else "forwarded"] += d
            # p4/adaptmon.p4 applies host_clone_table after the level/head/
            # sampling decision, overriding the reason: host-pair first
            if src_lvl >= CLONE:
                tot["clone_hostpair"] += d
            elif prev_lvl >= CLONE:
                tot["clone_level"] += d
            else:
                head = max(0, min(pc, head_n) - lo)
                tot["clone_head"] += head
                if sampling:
                    mod = 8 if prev_lvl == 1 else 64
                    first = max(lo, head_n) + 1
                    if pc >= first:
                        tot["clone_sample"] += pc // mod - (first - 1) // mod
            prev_lvl, prev_w, prev_tup = int(r.rlvl), r.window, tup
    tot["cloned"] = sum(v for k, v in tot.items() if k.startswith("clone_"))
    return tot
