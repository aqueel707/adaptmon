#!/usr/bin/env bash
# sandbox.sh — run a command as namespaced root, without sudo.
#
# Creates a private user + network + mount + PID namespace. Inside
# it the command has CAP_NET_ADMIN/CAP_NET_RAW over a fresh, empty network
# stack: `ip netns`, veth pairs, bmv2 port binding, raw sockets, tcpdump and
# nmap all work, and nothing touches the host network. Everything created
# (namespaces, veths, switches, every child process) disappears when the
# command exits.
#
#   scripts/sandbox.sh python3 -m experiments.paper_run --runs 5
#
# Inside the sandbox:
#   * /run/netns is a private tmpfs (for `ip netns add`)
#   * /sys is remounted so it reflects this network namespace
#   * am_dummy holds 192.0.2.1/32: with only loopback configured,
#     getaddrinfo(AI_ADDRCONFIG) returns nothing and simple_switch_CLI cannot
#     resolve localhost
#   * IPv6 is disabled so the kernel does not inject RS/ND packets into the
#     switch ports
#   * user-space nmap/iperf3 from scripts/install_tools.sh are put on PATH

set -euo pipefail
TOOLS="${ADAPTMON_TOOLS:-$HOME/.local/adaptmon-tools/root}"
if [ -d "$TOOLS/usr/bin" ]; then
    export PATH="$TOOLS/usr/bin:$PATH"
    export LD_LIBRARY_PATH="$TOOLS/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export NMAPDIR="$TOOLS/usr/share/nmap"
    export LUA_CPATH="$TOOLS/usr/lib/x86_64-linux-gnu/lua/5.3/?.so;;"
fi

if [ "${ADAPTMON_SANDBOX:-0}" = 1 ]; then
    exec "$@"
fi

exec unshare --user --map-root-user --net --mount --propagation private \
    --pid --fork --mount-proc \
    bash -c '
set -e
mount -t tmpfs tmpfs /run/netns
mount -t sysfs sysfs /sys
ip link set lo up
ip link add am_dummy type dummy
ip addr add 192.0.2.1/32 dev am_dummy
ip link set am_dummy up
sysctl -qw net.ipv6.conf.all.disable_ipv6=1 net.ipv6.conf.default.disable_ipv6=1
export ADAPTMON_SANDBOX=1
exec "$@"' bash "$@"
