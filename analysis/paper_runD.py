#!/usr/bin/env python3
"""
analysis/paper_runD.py — the paper numbers and figures of Run D
(results/paper_run_D: DPI confirmation -> DROP) into paper/runD/, with the
same items as paper/runC/ plus item 10 (drops). See analysis/paper_runC.py.

    python3 -m analysis.paper_runD
"""

import sys

from analysis import paper_runC

if __name__ == "__main__":
    paper_runC.main(["--series", "results/paper_run_D", "--out", "paper/runD",
                     "--name", "Run D"] + sys.argv[1:])
