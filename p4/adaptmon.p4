/* adaptmon.p4 - Adaptive Monitoring for Programmable Data Planes
 * P4_16 / BMv2 simple_switch
 */

#include <core.p4>
#include <v1model.p4>

/* ========== CONSTANTS ========== */
#define MAX_FLOWS       4096
#define CPU_PORT        255
#define CLONE_SESSION   100
#define CLONE_FIELDS    1      /* @field_list id carried by clones */

/* Per-flow monitoring levels (flow_level register, written by the controller) */
#define LEVEL_NORMAL    0
#define LEVEL_MONITOR   1
#define LEVEL_CLONE     2
#define LEVEL_DROP      3

/* v1model standard_metadata.instance_type values */
#define PKT_INSTANCE_TYPE_NORMAL        0
#define PKT_INSTANCE_TYPE_EGRESS_CLONE  2

/* ========== HEADERS ========== */
typedef bit<48>  MacAddr;
typedef bit<32>  IPv4Addr;
typedef bit<16>  Port;

header ethernet_t {
    MacAddr  dstAddr;
    MacAddr  srcAddr;
    bit<16>  etherType;
}

header ipv4_t {
    bit<4>   version;
    bit<4>   ihl;
    bit<8>   diffserv;
    bit<16>  totalLen;
    bit<16>  identification;
    bit<3>   flags;
    bit<13>  fragOffset;
    bit<8>   ttl;
    bit<8>   protocol;
    bit<16>  hdrChecksum;
    IPv4Addr srcAddr;
    IPv4Addr dstAddr;
}

header tcp_t {
    Port     srcPort;
    Port     dstPort;
    bit<32>  seqNo;
    bit<32>  ackNo;
    bit<4>   dataOffset;
    bit<3>   res;
    bit<3>   ecn;
    bit<6>   ctrl;
    bit<16>  window;
    bit<16>  checksum;
    bit<16>  urgentPtr;
}

header udp_t {
    Port     srcPort;
    Port     dstPort;
    bit<16>  len;
    bit<16>  checksum;
}

/* Prepended to cloned copies only (14 bytes); never on forwarded packets */
header cpu_header_t {
    bit<32>  flow_index;
    bit<8>   flow_level;
    bit<32>  pkt_count;
    bit<32>  byte_count;
    bit<8>   reason;
}

struct headers_t {
    ethernet_t   ethernet;
    ipv4_t       ipv4;
    tcp_t        tcp;
    udp_t        udp;
    cpu_header_t cpu_hdr;
}

struct metadata_t {
    /* Fields tagged CLONE_FIELDS are copied into the E2E clone */
    @field_list(CLONE_FIELDS) bit<32> flow_index;
    @field_list(CLONE_FIELDS) bit<2>  flow_level;
    @field_list(CLONE_FIELDS) bit<32> pkt_count_val;
    @field_list(CLONE_FIELDS) bit<32> byte_count_val;
    /* Why this packet was cloned (set in ingress, read back when the CLONE
     * itself is processed in egress): 1 = full capture at CLONE/DROP level,
     * 2 = per-flow sampling at NORMAL/MONITOR, 3 = host-pair forced clone,
     * 4 = flow head (one of the first head_clone_n packets of a new flow) */
    @field_list(CLONE_FIELDS) bit<8>  clone_reason;
    bit<1>   do_drop;
    bit<1>   do_clone;
    bit<16>  l4_sport;
    bit<16>  l4_dport;
}

/* ========== REGISTERS ========== */
register<bit<32>>(MAX_FLOWS) pkt_counter;
register<bit<32>>(MAX_FLOWS) byte_counter;
register<bit<32>>(MAX_FLOWS) flow_src_ip;
register<bit<32>>(MAX_FLOWS) flow_dst_ip;
register<bit<16>>(MAX_FLOWS) flow_src_port;
register<bit<16>>(MAX_FLOWS) flow_dst_port;
register<bit<8>>(MAX_FLOWS)  flow_protocol;
register<bit<1>>(MAX_FLOWS)  flow_active;
register<bit<48>>(MAX_FLOWS) flow_first_ts;   /* ingress_global_timestamp, us */
register<bit<48>>(MAX_FLOWS) flow_last_ts;
register<bit<2>>(MAX_FLOWS)  flow_level;      /* controller-written (Thrift) */
register<bit<32>>(MAX_FLOWS) flow_syn_pkts;   /* packets with TCP flags == SYN only */
/* Flow-head cloning: packets seen by the slot's *current* 5-tuple since it
 * became new (slot empty, or a different 5-tuple than the one stored). Kept
 * separate from pkt_counter, which keeps accumulating across a hash
 * collision -- a colliding new flow must still get its own head cloned. */
register<bit<32>>(MAX_FLOWS) head_pkts;
/* N, written by the controller at startup (--head-clone-n). 0 = disabled,
 * which is also what a switch no controller has configured does. */
register<bit<32>>(1)         head_clone_n;
/* Packets of the slot's current flow that the data plane dropped (5-tuple
 * drop rule or source-IP drop rule); reset on a new flow like head_pkts.
 * Measurement only -- read by the controller, never used to decide. */
register<bit<32>>(MAX_FLOWS) drop_pkts;

/* ========== PARSER ========== */
parser AdaptMonParser(
    packet_in pkt,
    out headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t std_meta)
{
    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            0x0800: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        pkt.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            6:  parse_tcp;
            17: parse_udp;
            default: accept;
        }
    }

    state parse_tcp {
        pkt.extract(hdr.tcp);
        transition accept;
    }

    state parse_udp {
        pkt.extract(hdr.udp);
        transition accept;
    }
}

/* ========== CHECKSUM VERIFICATION ========== */
control AdaptMonVerifyChecksum(
    inout headers_t hdr,
    inout metadata_t meta)
{
    apply {
        verify_checksum(
            hdr.ipv4.isValid(),
            {
                hdr.ipv4.version,
                hdr.ipv4.ihl,
                hdr.ipv4.diffserv,
                hdr.ipv4.totalLen,
                hdr.ipv4.identification,
                hdr.ipv4.flags,
                hdr.ipv4.fragOffset,
                hdr.ipv4.ttl,
                hdr.ipv4.protocol,
                hdr.ipv4.srcAddr,
                hdr.ipv4.dstAddr
            },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16
        );
    }
}

/* ========== INGRESS ========== */
control AdaptMonIngress(
    inout headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t std_meta)
{
    /* ---- actions ---- */
    action drop_pkt() {
        mark_to_drop(std_meta);
    }

    action forward_l2(bit<9> port) {
        std_meta.egress_spec = port;
    }

    action forward_l3(bit<9> port, MacAddr dst_mac) {
        std_meta.egress_spec  = port;
        hdr.ethernet.dstAddr  = dst_mac;
        hdr.ipv4.ttl          = hdr.ipv4.ttl - 1;
    }

    action do_drop_flow() {
        meta.do_drop = 1;
    }

    /* Host-pair detector escalated: clone every packet from this source IP,
     * regardless of its own per-flow level (catches later probes on new
     * destination ports/flows before any one of them earns its own level) */
    action do_force_clone() {
        meta.do_clone     = 1;
        meta.clone_reason = 3;
    }

    /* A DPI-confirmed scan from this source: drop every packet from it (and
     * still clone it, as for a flow at DROP, so DPI keeps seeing it) */
    action do_host_drop() {
        meta.do_drop      = 1;
        meta.do_clone     = 1;
        meta.clone_reason = 3;
    }

    action nop() {}

    /* ---- tables ---- */
    table ipv4_forward {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            forward_l2;
            forward_l3;
            drop_pkt;
            nop;
        }
        size = 1024;
        default_action = drop_pkt();
    }

    /* Per-flow (5-tuple) drop rules installed at DROP level */
    table drop_table {
        key = {
            hdr.ipv4.srcAddr : exact;
            hdr.ipv4.dstAddr : exact;
            hdr.ipv4.protocol: exact;
            meta.l4_sport    : exact;
            meta.l4_dport    : exact;
        }
        actions = {
            do_drop_flow;
            nop;
        }
        size = MAX_FLOWS;
        default_action = nop();
    }

    /* Installed when the host-pair detector (controller-side second
     * IsolationForest) escalates a (src IP, dst IP) aggregate to CLONE */
    table host_clone_table {
        key = {
            hdr.ipv4.srcAddr: exact;
        }
        actions = {
            do_force_clone;
            nop;
        }
        size = 256;
        default_action = nop();
    }

    /* Installed when a DPI confirmation of a scan class forces the host
     * pair to DROP: every packet from the source IP is dropped */
    table host_drop_table {
        key = {
            hdr.ipv4.srcAddr: exact;
        }
        actions = {
            do_host_drop;
            nop;
        }
        size = 256;
        default_action = nop();
    }

    /* ---- apply ---- */
    apply {
        meta.do_drop      = 0;
        meta.do_clone     = 0;
        meta.clone_reason = 0;
        meta.flow_level   = LEVEL_NORMAL;

        if (std_meta.ingress_port == CPU_PORT) {
            /* Nothing is injected from the CPU port */
            mark_to_drop(std_meta);
        } else if (hdr.ethernet.etherType == 0x0806) {
            // ARP: send out all ports except ingress
            // Simple approach: use multicast or just forward based on ingress
            if (std_meta.ingress_port == 1) {
                std_meta.egress_spec = 2;
            } else if (std_meta.ingress_port == 2) {
                std_meta.egress_spec = 1;
            } else if (std_meta.ingress_port == 3) {
                std_meta.egress_spec = 1;
            } else {
                std_meta.egress_spec = 1;
            }
        } else if (hdr.ipv4.isValid()) {

            /* Compute flow hash index using 5-tuple */
            meta.l4_sport = 0;
            meta.l4_dport = 0;

            if (hdr.tcp.isValid()) {
                meta.l4_sport = hdr.tcp.srcPort;
                meta.l4_dport = hdr.tcp.dstPort;
            } else if (hdr.udp.isValid()) {
                meta.l4_sport = hdr.udp.srcPort;
                meta.l4_dport = hdr.udp.dstPort;
            }

            /* index = crc32(5-tuple) mod MAX_FLOWS, so all slots are usable */
            hash(
                meta.flow_index,
                HashAlgorithm.crc32,
                (bit<32>)0,
                {
                    hdr.ipv4.srcAddr,
                    hdr.ipv4.dstAddr,
                    hdr.ipv4.protocol,
                    meta.l4_sport,
                    meta.l4_dport
                },
                (bit<32>)MAX_FLOWS
            );

            /* Update counters */
            bit<32> pkt_val;
            bit<32> byte_val;
            bit<48> now = std_meta.ingress_global_timestamp;

            pkt_counter.read(pkt_val, meta.flow_index);
            if (pkt_val == 0) {
                /* Empty slot (new or aged-out by the controller) */
                flow_first_ts.write(meta.flow_index, now);
            }

            /* Flow head: a new flow is an empty slot, or one whose stored
             * 5-tuple (read before it is overwritten below) is not this
             * packet's -- a hash collision is a new flow too */
            bit<32> old_src;
            bit<32> old_dst;
            bit<8>  old_proto;
            bit<16> old_sport;
            bit<16> old_dport;
            flow_src_ip.read(old_src, meta.flow_index);
            flow_dst_ip.read(old_dst, meta.flow_index);
            flow_protocol.read(old_proto, meta.flow_index);
            flow_src_port.read(old_sport, meta.flow_index);
            flow_dst_port.read(old_dport, meta.flow_index);
            bit<32> head_val;
            bool new_flow = pkt_val == 0 || old_src != hdr.ipv4.srcAddr
                    || old_dst != hdr.ipv4.dstAddr || old_proto != hdr.ipv4.protocol
                    || old_sport != meta.l4_sport || old_dport != meta.l4_dport;
            if (new_flow) {
                head_val = 1;
            } else {
                head_pkts.read(head_val, meta.flow_index);
                head_val = head_val + 1;
            }
            head_pkts.write(meta.flow_index, head_val);
            bit<32> head_n;
            head_clone_n.read(head_n, 0);
            pkt_val = pkt_val + 1;
            pkt_counter.write(meta.flow_index, pkt_val);
            meta.pkt_count_val = pkt_val;

            byte_counter.read(byte_val, meta.flow_index);
            byte_val = byte_val + (bit<32>)std_meta.packet_length;
            byte_counter.write(meta.flow_index, byte_val);
            meta.byte_count_val = byte_val;
            flow_last_ts.write(meta.flow_index, now);

            /* SYN-only packets (ctrl == 0b000010, no other flag set): feeds
             * the controller's host-pair SYN-only-ratio feature */
            if (hdr.tcp.isValid() && hdr.tcp.ctrl == 2) {
                bit<32> syn_val;
                flow_syn_pkts.read(syn_val, meta.flow_index);
                syn_val = syn_val + 1;
                flow_syn_pkts.write(meta.flow_index, syn_val);
            }

            /* Store flow 5-tuple info */
            flow_src_ip.write(meta.flow_index, hdr.ipv4.srcAddr);
            flow_dst_ip.write(meta.flow_index, hdr.ipv4.dstAddr);
            flow_protocol.write(meta.flow_index, hdr.ipv4.protocol);
            flow_src_port.write(meta.flow_index, meta.l4_sport);
            flow_dst_port.write(meta.flow_index, meta.l4_dport);
            flow_active.write(meta.flow_index, 1);

            /* Monitoring level set by the controller for this flow */
            flow_level.read(meta.flow_level, meta.flow_index);

            /* Adaptive sampling (Decision C): CLONE/DROP captures every
             * packet; MONITOR samples 1-in-8; NORMAL samples 1-in-64.
             * pkt_val is a power-of-two modulus check via bitwise AND.
             * Flow-head cloning sits between level and sampling: the first
             * head_n packets of every new flow are cloned whatever its level
             * (a level that already captures everything keeps reason 1). */
            if (meta.flow_level >= LEVEL_CLONE) {
                meta.do_clone     = 1;
                meta.clone_reason = 1;
            } else if (head_val <= head_n) {
                meta.do_clone     = 1;
                meta.clone_reason = 4;
            } else if (meta.flow_level == LEVEL_MONITOR && (pkt_val & 7) == 0) {
                meta.do_clone     = 1;
                meta.clone_reason = 2;
            } else if (meta.flow_level == LEVEL_NORMAL && (pkt_val & 63) == 0) {
                meta.do_clone     = 1;
                meta.clone_reason = 2;
            }

            /* Host-pair escalation (Decision B): clone every packet from
             * this source IP regardless of its own flow's level/sampling */
            host_clone_table.apply();
            host_drop_table.apply();

            /* A DROP flow is still forwarded to egress, so its clone is made
             * there; egress then discards the original */
            drop_table.apply();

            bit<32> drop_val = 0;
            if (!new_flow) {
                drop_pkts.read(drop_val, meta.flow_index);
            }
            if (meta.do_drop == 1) {
                drop_val = drop_val + 1;
            }
            drop_pkts.write(meta.flow_index, drop_val);

            ipv4_forward.apply();
        } // end ipv4.isValid
    } // end apply
}

/* ========== EGRESS ========== */
control AdaptMonEgress(
    inout headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t std_meta)
{
    /* Index 0 only: data-plane totals for the inspection-reduction metric.
     * bmv2 counts a packet's ingress length in bytes, so a clone's bytes
     * are the original frame's (no cpu_hdr). */
    counter(1, CounterType.packets_and_bytes) fwd_counter;     /* originals sent out */
    counter(1, CounterType.packets_and_bytes) clone_counter;   /* copies sent to CPU */
    counter(1, CounterType.packets_and_bytes) drop_counter;    /* originals dropped  */
    /* Clones broken down by clone_reason (index = reason, 1..4; 0 unused) */
    counter(5, CounterType.packets_and_bytes) clone_reason_counter;

    apply {
        if (std_meta.instance_type == PKT_INSTANCE_TYPE_EGRESS_CLONE) {
            /* The cloned copy: prepend the CPU header from the preserved
             * field list. Only clones ever carry cpu_hdr. */
            hdr.cpu_hdr.setValid();
            hdr.cpu_hdr.flow_index  = meta.flow_index;
            hdr.cpu_hdr.flow_level  = (bit<8>)meta.flow_level;
            hdr.cpu_hdr.pkt_count   = meta.pkt_count_val;
            hdr.cpu_hdr.byte_count  = meta.byte_count_val;
            hdr.cpu_hdr.reason      = meta.clone_reason;
            clone_counter.count(0);
            clone_reason_counter.count((bit<32>)meta.clone_reason);
        } else if (std_meta.instance_type == PKT_INSTANCE_TYPE_NORMAL
                   && hdr.ipv4.isValid()) {
            /* The original: clone per ingress's sampling/host-pair decision,
             * then forward it unchanged (or drop it at DROP) */
            if (meta.do_clone == 1) {
                clone_preserving_field_list(
                    CloneType.E2E, (bit<32>)CLONE_SESSION, CLONE_FIELDS);
            }
            if (meta.do_drop == 1) {
                mark_to_drop(std_meta);
                drop_counter.count(0);
            } else {
                fwd_counter.count(0);
            }
        }
    }
}

/* ========== CHECKSUM COMPUTE ========== */
control AdaptMonComputeChecksum(
    inout headers_t hdr,
    inout metadata_t meta)
{
    apply {
        update_checksum(
            hdr.ipv4.isValid(),
            {
                hdr.ipv4.version,
                hdr.ipv4.ihl,
                hdr.ipv4.diffserv,
                hdr.ipv4.totalLen,
                hdr.ipv4.identification,
                hdr.ipv4.flags,
                hdr.ipv4.fragOffset,
                hdr.ipv4.ttl,
                hdr.ipv4.protocol,
                hdr.ipv4.srcAddr,
                hdr.ipv4.dstAddr
            },
            hdr.ipv4.hdrChecksum,
            HashAlgorithm.csum16
        );
    }
}

/* ========== DEPARSER ========== */
control AdaptMonDeparser(
    packet_out pkt,
    in headers_t hdr)
{
    apply {
        pkt.emit(hdr.cpu_hdr);
        pkt.emit(hdr.ethernet);
        pkt.emit(hdr.ipv4);
        pkt.emit(hdr.tcp);
        pkt.emit(hdr.udp);
    }
}

/* ========== SWITCH ========== */
V1Switch(
    AdaptMonParser(),
    AdaptMonVerifyChecksum(),
    AdaptMonIngress(),
    AdaptMonEgress(),
    AdaptMonComputeChecksum(),
    AdaptMonDeparser()
) main;
