#!/usr/bin/env bash
# install_tools.sh — user-space nmap + iperf3 without root.
# Downloads the distribution's own .deb packages (apt-get download needs no
# root) and unpacks them under ~/.local/adaptmon-tools/root. sandbox.sh puts
# that tree on PATH / LD_LIBRARY_PATH.
set -euo pipefail
T="${ADAPTMON_TOOLS_DIR:-$HOME/.local/adaptmon-tools}"
mkdir -p "$T/debs" "$T/root"
cd "$T/debs"
apt-get download nmap nmap-common iperf3 libiperf0 liblinear4 libsctp1 lua-lpeg
for d in *.deb; do dpkg -x "$d" "$T/root"; done
L="$T/root/usr/lib/x86_64-linux-gnu"
LD_LIBRARY_PATH="$L" "$T/root/usr/bin/nmap"   --version | head -1
LD_LIBRARY_PATH="$L" "$T/root/usr/bin/iperf3" --version | head -1
