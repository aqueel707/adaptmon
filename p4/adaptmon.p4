/* adaptmon.p4 - Adaptive Monitoring for Programmable Data Planes
 * P4_16 / BMv2 simple_switch
 */

#include <core.p4>
#include <v1model.p4>

/* ========== CONSTANTS ========== */
#define MAX_FLOWS       4096
#define FLOW_HASH_WIDTH 12
#define CPU_PORT        255
#define CLONE_SESSION   100

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

header cpu_header_t {
    bit<32>  flow_index;
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

@field_list(0)
struct metadata_t {
    bit<FLOW_HASH_WIDTH> flow_index;
    bit<1>               do_clone;
    bit<1>               do_drop;
    bit<32>              pkt_count_val;
    bit<32>              byte_count_val;
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

    action do_clone_to_cpu() {
        meta.do_clone = 1;
    }

    action do_drop_flow() {
        meta.do_drop = 1;
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

    table drop_table {
        key = {
            hdr.ipv4.srcAddr: exact;
            hdr.ipv4.dstAddr: exact;
            hdr.ipv4.protocol: exact;
        }
        actions = {
            do_drop_flow;
            nop;
        }
        size = 512;
        default_action = nop();
    }

    table clone_table {
        key = {
            hdr.ipv4.srcAddr: exact;
            hdr.ipv4.dstAddr: exact;
        }
        actions = {
            do_clone_to_cpu;
            nop;
        }
        size = 512;
        default_action = nop();
    }

    /* ---- apply ---- */
    apply {
        meta.do_clone = 0;
        meta.do_drop  = 0;

        // Pass ARP packets through by flooding to all ports
        if (hdr.ethernet.etherType == 0x0806) {
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
            bit<16> src_port = 0;
            bit<16> dst_port = 0;

            if (hdr.tcp.isValid()) {
                src_port = hdr.tcp.srcPort;
                dst_port = hdr.tcp.dstPort;
            } else if (hdr.udp.isValid()) {
                src_port = hdr.udp.srcPort;
                dst_port = hdr.udp.dstPort;
            }

            hash(
                meta.flow_index,
                HashAlgorithm.crc32,
                (bit<1>)0,
                {
                    hdr.ipv4.srcAddr,
                    hdr.ipv4.dstAddr,
                    hdr.ipv4.protocol,
                    src_port,
                    dst_port
                },
                (bit<FLOW_HASH_WIDTH>)(MAX_FLOWS - 1)
            );

            /* Update counters */
            bit<32> pkt_val;
            bit<32> byte_val;

            pkt_counter.read(pkt_val, (bit<32>)meta.flow_index);
            pkt_val = pkt_val + 1;
            pkt_counter.write((bit<32>)meta.flow_index, pkt_val);
            meta.pkt_count_val = pkt_val;

            byte_counter.read(byte_val, (bit<32>)meta.flow_index);
            byte_val = byte_val + (bit<32>)std_meta.packet_length;
            byte_counter.write((bit<32>)meta.flow_index, byte_val);
            meta.byte_count_val = byte_val;

            /* Store flow 5-tuple info */
            flow_src_ip.write((bit<32>)meta.flow_index, hdr.ipv4.srcAddr);
            flow_dst_ip.write((bit<32>)meta.flow_index, hdr.ipv4.dstAddr);
            flow_protocol.write((bit<32>)meta.flow_index, hdr.ipv4.protocol);
            flow_src_port.write((bit<32>)meta.flow_index, src_port);
            flow_dst_port.write((bit<32>)meta.flow_index, dst_port);
            flow_active.write((bit<32>)meta.flow_index, 1);

            /* Check drop table */
            drop_table.apply();

            if (meta.do_drop == 1) {
                mark_to_drop(std_meta);
            } else {
                /* Check clone table */
                clone_table.apply();

                /* Forward */
                ipv4_forward.apply();
            }
        } // end ipv4.isValid
    } // end apply
}

/* ========== EGRESS ========== */
control AdaptMonEgress(
    inout headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t std_meta)
{
    apply {
        /* Clone packet to CPU if requested */
        if (meta.do_clone == 1) {
            /* Prepend CPU header */
            hdr.cpu_hdr.setValid();
            hdr.cpu_hdr.flow_index  = (bit<32>)meta.flow_index;
            hdr.cpu_hdr.pkt_count   = meta.pkt_count_val;
            hdr.cpu_hdr.byte_count  = meta.byte_count_val;
            hdr.cpu_hdr.reason      = 1;
            clone_preserving_field_list(CloneType.E2E, (bit<32>)CLONE_SESSION, 0);
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
