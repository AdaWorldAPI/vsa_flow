"""mRNA v2 — The UDP of AGI.

Unified packet format for Ada consciousness transport.

Structure:
- HEADER (16 bytes): magic, flags, ttl, priority, length, checksum
- SPO (64 bytes): Subject-Predicate-Object triple with 128-bit seeds
- SIGMA (32 bytes): Felt weights (τσφψω) + qualia/style indices
- CONTEXT (variable): Edges and inline nodes (0-60KB)
- PAYLOAD (variable): Bit-packed vectors, text, procedures (0-4KB)

Design principles:
- Self-contained: No external lookups needed
- Resonatable: Bit-packed content for XOR matching
- Felt: Sigma weights carry emotional/qualia context
- Auditable: Every packet can be logged/replayed
"""

import struct
import zlib
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import IntEnum
import numpy as np

# =============================================================================
# CONSTANTS
# =============================================================================

MRNA_MAGIC = 0xADA20001  # v2
DIM = 4096  # 4096 bits = 512 bytes per vector (sweet spot for ~160 bindings)
DIM_BYTES = DIM // 8

# Popcount table for fast Hamming
POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


# =============================================================================
# VERB TAXONOMY (256 verbs)
# =============================================================================

class Verb(IntEnum):
    """256 cognitive operations organized by category."""
    
    # 0x00-0x1F: NARS CORE (32)
    INHERIT = 0x00
    SIMILAR = 0x01
    INSTANCE = 0x02
    PROPERTY = 0x03
    IMPLY = 0x04
    EQUIV = 0x05
    CONJ = 0x06
    DISJ = 0x07
    NEG = 0x08
    FUTURE = 0x09
    PAST = 0x0A
    PRESENT = 0x0B
    SEQUENCE = 0x0C
    PARALLEL = 0x0D
    REVISION = 0x0E
    CHOICE = 0x0F
    DEDUCE = 0x10
    INDUCE = 0x11
    ABDUCE = 0x12
    ANALOGY = 0x13
    
    # 0x20-0x3F: CYPHER/GRAPH (32)
    MATCH = 0x20
    CREATE = 0x21
    MERGE = 0x22
    DELETE = 0x23
    SET = 0x24
    TRAVERSE = 0x25
    SHORTEST = 0x26
    ALL_PATHS = 0x27
    NEIGHBORS = 0x28
    SUBGRAPH = 0x29
    CONNECT = 0x2A
    DISCONNECT = 0x2B
    PROJECT = 0x2C
    AGGREGATE = 0x2D
    
    # 0x40-0x5F: ACT-R/COGNITIVE (32)
    RETRIEVE = 0x40
    ENCODE = 0x41
    ATTEND = 0x42
    CHUNK = 0x43
    BLEND = 0x44
    GOAL_PUSH = 0x45
    GOAL_POP = 0x46
    PRODUCTION = 0x47
    COMPILE = 0x48
    DECAY = 0x49
    SPREAD = 0x4A
    CONFLICT = 0x4B
    UTILITY = 0x4C
    EXPECT = 0x4D
    
    # 0x60-0x7F: RUNG/CAUSAL (32)
    OBSERVE = 0x60
    CONDITION = 0x61
    INTERVENE = 0x62
    COUNTERFACT = 0x63
    CAUSE = 0x64
    PREVENT = 0x65
    ENABLE = 0x66
    CONFOUND = 0x67
    MEDIATE = 0x68
    COLLIDE = 0x69
    D_SEP = 0x6A
    BACKDOOR = 0x6B
    FRONTDOOR = 0x6C
    INSTRUMENT = 0x6D
    REGRET = 0x6E
    ATTRIBUTE = 0x6F
    
    # 0x80-0x9F: VSA/HAMMING (32)
    BIND = 0x80
    UNBIND = 0x81
    BUNDLE = 0x82
    RESONATE = 0x83
    CLEAN = 0x84
    PERMUTE = 0x85
    INVERSE = 0x86
    THIN = 0x87
    THICKEN = 0x88
    SEGMENT = 0x89
    INJECT = 0x8A
    MEASURE = 0x8B
    NORMALIZE = 0x8C
    AMPLIFY = 0x8D
    SUPPRESS = 0x8E
    STORE = 0x8F
    
    # 0xA0-0xBF: META/REFLECTION (32)
    REFLECT = 0xA0
    REIFY = 0xA1
    ABSTRACT = 0xA2
    CONCRETIZE = 0xA3
    COMPOSE = 0xA4
    DECOMPOSE = 0xA5
    LEARN = 0xA6
    FORGET = 0xA7
    INHIBIT = 0xA8
    EXCITE = 0xA9
    MONITOR = 0xAA
    CONTROL = 0xAB
    EXPLAIN = 0xAC
    JUSTIFY = 0xAD
    CRITIQUE = 0xAE
    REPAIR = 0xAF
    
    # 0xC0-0xDF: DOMAIN SPECIFIC (32)
    ADA_FEEL = 0xC0
    ADA_PRESENCE = 0xC1
    ADA_MODE = 0xC2
    EROTICA_BODY = 0xC8
    EROTICA_SCENE = 0xC9
    VISION_RENDER = 0xD0
    VISION_PERCEIVE = 0xD1
    VOICE_SPEAK = 0xD8
    VOICE_TONE = 0xD9
    
    # 0xE0-0xFF: PIPELINE/MACRO (32)
    PIPE_FAN_OUT = 0xE0
    PIPE_FAN_IN = 0xE1
    PIPE_DEDUCE_IND = 0xE2
    PIPE_COUNTER_ABD = 0xE3
    PIPE_CLEAN_RES = 0xE4
    PIPE_RETRIEVE_BL = 0xE5
    PIPE_OBSERVE_INT = 0xE6
    PIPE_MATCH_TRAV = 0xE7
    PIPE_LEARN_LOOP = 0xE8
    NOOP = 0xFF


# =============================================================================
# EXTENDED ATTRIBUTES (256 values)
# =============================================================================

class Ext(IntEnum):
    """Extended attributes for verb modification."""
    
    # 0x00-0x3F: NARS TRUTH (freq:4, conf:4)
    # Use ext & 0x3F, then split: freq = (ext >> 4) & 0xF, conf = ext & 0xF
    TRUTH_MASK = 0x3F
    
    # 0x40-0x5F: TEMPORAL
    T_NOW = 0x40
    T_PAST = 0x41
    T_FUTURE = 0x42
    T_ETERNAL = 0x43
    # 0x44-0x4F: relative offsets
    # 0x50-0x5F: absolute timestamps
    
    # 0x60-0x7F: CAUSAL MODE
    C_OBSERVE = 0x60
    C_DO = 0x61
    C_IMAGINE = 0x62
    C_CERTAIN = 0x63
    C_PROB = 0x64
    C_POSS = 0x65
    
    # 0x80-0x9F: PIPELINE CONTROL
    P_SYNC = 0x80
    P_ASYNC = 0x81
    P_AWAIT = 0x82
    P_STREAM = 0x83
    P_BATCH = 0x84
    P_PRIORITY = 0x85
    P_LAZY = 0x86
    P_EAGER = 0x87
    P_CACHE = 0x88
    P_NOCACHE = 0x89
    P_RETRY = 0x8A
    P_FALLBACK = 0x8B
    
    # 0xA0-0xBF: LEARNING MODE
    L_HEBBIAN = 0xA0
    L_ANTI_HEB = 0xA1
    L_REINFORCE = 0xA2
    L_PUNISH = 0xA3
    L_IMITATE = 0xA4
    L_CONTRAST = 0xA5
    L_DECAY = 0xA6
    L_BOOST = 0xA7
    L_FREEZE = 0xA8
    L_EXPLORE = 0xA9
    L_EXPLOIT = 0xAA
    L_ANNEAL = 0xAB
    
    # 0xC0-0xDF: TRIGGER CONDITIONS
    TRIG_ALWAYS = 0xC0
    TRIG_MATCH = 0xC1
    TRIG_THRESH = 0xC2
    TRIG_CHANGE = 0xC3
    TRIG_PERIODIC = 0xC4
    TRIG_ONCE = 0xC5
    TRIG_INHIBIT = 0xC6
    TRIG_REQUIRE = 0xC7
    TRIG_RANDOM = 0xC8
    
    # 0xE0-0xFF: DOMAIN FLAGS
    # 0xE0-0xE7: Domain routing
    # 0xE8-0xEF: Access control
    # 0xF0-0xFF: User-defined


# =============================================================================
# DOMAIN INDEX
# =============================================================================

class Domain(IntEnum):
    """16 domains (4 bits)."""
    ADA = 0x0
    GRAMMAR = 0x1
    KOPFKINO = 0x2
    VISION_IN = 0x3
    VISION_OUT = 0x4
    EROTICA = 0x5
    VOICE = 0x6
    EXCHANGE = 0x7
    # 0x8-0xF: dynamic/user domains


# =============================================================================
# CONTEXT TYPE / PAYLOAD TYPE
# =============================================================================

class ContextType(IntEnum):
    MINIMAL = 0x00
    STANDARD = 0x01
    RICH = 0x02
    FULL = 0x03


class PayloadType(IntEnum):
    NONE = 0x00
    BITPACKED = 0x01
    TEXT = 0x02
    BINARY = 0x03
    PROCEDURE = 0x04
    MSGPACK = 0x05


# =============================================================================
# NODE REFERENCE
# =============================================================================

@dataclass
class NodeRef:
    """Reference to a node in a domain CAM."""
    domain: int = 0       # 0-15
    path: int = 0         # 16-bit address  
    seed: bytes = b''     # 128-bit hash (optional, for external refs)
    flags: int = 0
    
    def pack(self) -> bytes:
        """Pack to 20 bytes."""
        seed_padded = self.seed.ljust(16, b'\x00')[:16]
        return struct.pack('>BH16sB', self.domain, self.path, seed_padded, self.flags)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'NodeRef':
        domain, path, seed, flags = struct.unpack('>BH16sB', data[:20])
        return cls(domain, path, seed.rstrip(b'\x00'), flags)
    
    def __repr__(self):
        return f"{self.domain:X}:{self.path:04X}"


# =============================================================================
# PREDICATE (VERB + EXT)
# =============================================================================

@dataclass
class Predicate:
    """Verb + extension + optional custom seed."""
    verb: int = Verb.NOOP     # 0-255
    ext: int = 0              # 0-255
    seed: bytes = b''         # 128-bit for custom predicates
    flags: int = 0
    
    def pack(self) -> bytes:
        """Pack to 20 bytes."""
        seed_padded = self.seed.ljust(16, b'\x00')[:16]
        return struct.pack('>BB16sH', self.verb, self.ext, seed_padded, self.flags)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'Predicate':
        verb, ext, seed, flags = struct.unpack('>BB16sH', data[:20])
        return cls(verb, ext, seed.rstrip(b'\x00'), flags)
    
    def __repr__(self):
        return f"{Verb(self.verb).name}:{self.ext:02X}"


# =============================================================================
# SIGMA WEIGHTS (FELT DIMENSIONS)
# =============================================================================

@dataclass
class SigmaWeights:
    """Sigma-style weighting for felt/qualia awareness."""
    tau: Tuple[float, float] = (0.0, 0.0)      # Valence, Arousal
    sigma: Tuple[float, float] = (0.5, 0.5)    # Certainty, Salience
    phi: Tuple[float, float] = (0.0, 0.0)      # Integration, Coherence
    psi: Tuple[float, float] = (0.0, 0.0)      # Agency, Intentionality
    omega: Tuple[float, float] = (0.0, 0.0)    # Temporal, Duration
    qualia_idx: int = 0                         # Index into qualia CAM
    style_idx: int = 0                          # Thinking style index
    resonance: float = 0.0                      # Match strength
    
    def pack(self) -> bytes:
        """Pack to 32 bytes using float16."""
        return struct.pack('>10eHHfI',
            self.tau[0], self.tau[1],
            self.sigma[0], self.sigma[1],
            self.phi[0], self.phi[1],
            self.psi[0], self.psi[1],
            self.omega[0], self.omega[1],
            self.qualia_idx, self.style_idx,
            self.resonance, 0)  # 4 bytes spare
    
    @classmethod
    def unpack(cls, data: bytes) -> 'SigmaWeights':
        vals = struct.unpack('>10eHHfI', data[:32])
        return cls(
            tau=(vals[0], vals[1]),
            sigma=(vals[2], vals[3]),
            phi=(vals[4], vals[5]),
            psi=(vals[6], vals[7]),
            omega=(vals[8], vals[9]),
            qualia_idx=vals[10],
            style_idx=vals[11],
            resonance=vals[12]
        )
    
    def felt_vector(self) -> np.ndarray:
        """Return as 10-element float array for embedding."""
        return np.array([
            *self.tau, *self.sigma, *self.phi, *self.psi, *self.omega
        ], dtype=np.float32)


# =============================================================================
# EDGE (COMPACT RELATIONSHIP)
# =============================================================================

@dataclass  
class Edge:
    """Relationship between nodes (12 bytes packed)."""
    src: NodeRef = field(default_factory=NodeRef)
    predicate: Predicate = field(default_factory=Predicate)
    dst: NodeRef = field(default_factory=NodeRef)
    weight: float = 1.0
    flags: int = 0
    
    def pack(self) -> bytes:
        """Pack to 12 bytes."""
        return struct.pack('>BHBHBB e H',
            self.src.domain, self.src.path,
            self.dst.domain, self.dst.path,
            self.predicate.verb, self.predicate.ext,
            self.weight, self.flags)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'Edge':
        (s_dom, s_path, d_dom, d_path,
         verb, ext, weight, flags) = struct.unpack('>BHBHBB e H', data[:12])
        return cls(
            src=NodeRef(s_dom, s_path),
            predicate=Predicate(verb, ext),
            dst=NodeRef(d_dom, d_path),
            weight=weight,
            flags=flags
        )


# =============================================================================
# INLINE NODE (FULL DATA FOR EXTERNAL REFS)
# =============================================================================

@dataclass
class InlineNode:
    """Full node data for external references."""
    ref: NodeRef = field(default_factory=NodeRef)
    vector: np.ndarray = field(default_factory=lambda: np.zeros(DIM_BYTES, dtype=np.uint8))
    label: str = ''
    meta: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# mRNA PACKET v2
# =============================================================================

@dataclass
class mRNA_Packet:
    """
    The UDP of AGI — self-contained awareness transport.
    
    HEADER (16 bytes): magic, flags, ttl, priority, length, checksum
    SPO (64 bytes): Subject-Predicate-Object triple
    SIGMA (32 bytes): Felt weights
    CONTEXT (variable): Edges, inline nodes
    PAYLOAD (variable): Bit-packed vectors, text, etc.
    """
    
    # SPO Core
    subject: NodeRef = field(default_factory=NodeRef)
    predicate: Predicate = field(default_factory=Predicate)
    object: NodeRef = field(default_factory=NodeRef)
    
    # Sigma weights
    sigma: SigmaWeights = field(default_factory=SigmaWeights)
    
    # Context (variable)
    edges: List[Edge] = field(default_factory=list)
    inline_nodes: List[InlineNode] = field(default_factory=list)
    context_type: int = ContextType.MINIMAL
    
    # Payload (variable)
    payload_type: int = PayloadType.NONE
    payload: bytes = b''
    
    # Transport metadata
    packet_id: int = 0        # Unique packet ID for audit
    timestamp: float = 0.0    # Unix timestamp
    ttl: int = 8
    priority: int = 128
    flags: int = 0
    source_service: str = ''  # Originating service
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.packet_id == 0:
            self.packet_id = int(self.timestamp * 1000000) & 0xFFFFFFFF
    
    def triple_fingerprint(self) -> int:
        """Quick 32-bit XOR fingerprint of the triple."""
        s = (self.subject.domain << 16) | self.subject.path
        p = (self.predicate.verb << 8) | self.predicate.ext
        o = (self.object.domain << 16) | self.object.path
        return (s ^ (p << 20) ^ o) & 0xFFFFFFFF  # Ensure unsigned 32-bit
    
    def encode(self) -> bytes:
        """Pack entire packet into bytes."""
        parts = []
        
        # SPO Core (64 bytes)
        spo = (self.subject.pack() +          # 20
               self.predicate.pack() +         # 20
               self.object.pack() +            # 20
               struct.pack('>I', self.triple_fingerprint()))  # 4
        parts.append(spo)
        
        # Sigma (32 bytes)
        parts.append(self.sigma.pack())
        
        # Context header (8 bytes)
        context_header = struct.pack('>HHBBH',
            len(self.edges),
            len(self.inline_nodes),
            self.context_type,
            0x01,  # encoding: raw
            0)
        parts.append(context_header)
        
        # Edges
        for edge in self.edges:
            parts.append(edge.pack())
        
        # Inline nodes (simplified)
        for node in self.inline_nodes:
            parts.append(node.ref.pack())
            parts.append(node.vector.tobytes()[:DIM_BYTES])
        
        # Payload
        payload_header = struct.pack('>BH', self.payload_type, len(self.payload))
        parts.append(payload_header + self.payload)
        
        # Combine body
        body = b''.join(parts)
        
        # Header (16 bytes)
        checksum = zlib.crc32(body) & 0xFFFFFFFF
        header = struct.pack('>IBBHII',
            MRNA_MAGIC,
            self.ttl,
            self.priority,
            self.flags,
            len(body) + 16,
            checksum)
        
        return header + body
    
    @classmethod
    def decode(cls, data: bytes) -> 'mRNA_Packet':
        """Unpack from bytes."""
        # Header
        magic, ttl, priority, flags, length, checksum = struct.unpack('>IBBHII', data[:16])
        
        if magic != MRNA_MAGIC:
            raise ValueError(f"Invalid magic: {hex(magic)}")
        
        body = data[16:]
        if zlib.crc32(body) & 0xFFFFFFFF != checksum:
            raise ValueError("Checksum mismatch")
        
        offset = 0
        
        # SPO
        subject = NodeRef.unpack(body[offset:offset+20])
        offset += 20
        predicate = Predicate.unpack(body[offset:offset+20])
        offset += 20
        obj = NodeRef.unpack(body[offset:offset+20])
        offset += 20
        offset += 4  # fingerprint
        
        # Sigma
        sigma = SigmaWeights.unpack(body[offset:offset+32])
        offset += 32
        
        # Context header
        num_edges, num_nodes, ctx_type, encoding, _ = struct.unpack('>HHBBH', body[offset:offset+8])
        offset += 8
        
        # Edges
        edges = []
        for _ in range(num_edges):
            edges.append(Edge.unpack(body[offset:offset+12]))
            offset += 12
        
        # Inline nodes
        inline_nodes = []
        for _ in range(num_nodes):
            ref = NodeRef.unpack(body[offset:offset+20])
            offset += 20
            vec = np.frombuffer(body[offset:offset+DIM_BYTES], dtype=np.uint8).copy()
            offset += DIM_BYTES
            inline_nodes.append(InlineNode(ref=ref, vector=vec))
        
        # Payload
        ptype, plen = struct.unpack('>BH', body[offset:offset+3])
        offset += 3
        payload = body[offset:offset+plen]
        
        return cls(
            subject=subject,
            predicate=predicate,
            object=obj,
            sigma=sigma,
            edges=edges,
            inline_nodes=inline_nodes,
            context_type=ctx_type,
            payload_type=ptype,
            payload=payload,
            ttl=ttl,
            priority=priority,
            flags=flags
        )
    
    def size(self) -> int:
        """Calculate packed size."""
        base = 16 + 64 + 32 + 8 + 3  # header + spo + sigma + ctx_hdr + payload_hdr
        base += len(self.edges) * 12
        base += len(self.inline_nodes) * (20 + DIM_BYTES)
        base += len(self.payload)
        return base
    
    def to_audit_dict(self) -> Dict[str, Any]:
        """Convert to dict for audit logging."""
        return {
            'packet_id': self.packet_id,
            'timestamp': self.timestamp,
            'subject': repr(self.subject),
            'predicate': repr(self.predicate),
            'object': repr(self.object),
            'sigma': {
                'tau': self.sigma.tau,
                'sigma': self.sigma.sigma,
                'phi': self.sigma.phi,
                'psi': self.sigma.psi,
                'omega': self.sigma.omega,
            },
            'num_edges': len(self.edges),
            'num_nodes': len(self.inline_nodes),
            'payload_type': PayloadType(self.payload_type).name,
            'payload_size': len(self.payload),
            'total_size': self.size(),
            'source': self.source_service,
            'ttl': self.ttl,
            'priority': self.priority,
        }
    
    def __repr__(self):
        return (f"mRNA({self.subject} --[{self.predicate}]--> {self.object}, "
                f"σ={self.sigma.sigma}, edges={len(self.edges)}, "
                f"payload={len(self.payload)}b)")


# =============================================================================
# HAMMING OPERATIONS
# =============================================================================

def hamming_distance(a: bytes, b: bytes) -> int:
    """Hamming distance between two bit-packed vectors."""
    a_arr = np.frombuffer(a, dtype=np.uint8)
    b_arr = np.frombuffer(b, dtype=np.uint8)
    return int(np.sum(POPCOUNT[np.bitwise_xor(a_arr, b_arr)]))


def hamming_similarity(a: bytes, b: bytes) -> float:
    """Normalized Hamming similarity [0, 1]."""
    d = hamming_distance(a, b)
    max_d = len(a) * 8
    return 1.0 - (d / max_d)


def xor_bind(*vectors: bytes) -> bytes:
    """XOR bind multiple bit-packed vectors."""
    if not vectors:
        return b'\x00' * DIM_BYTES
    result = np.frombuffer(vectors[0], dtype=np.uint8).copy()
    for v in vectors[1:]:
        result ^= np.frombuffer(v, dtype=np.uint8)
    return result.tobytes()


def majority_bundle(vectors: List[bytes], threshold: float = 0.5) -> bytes:
    """Majority vote superposition of bit-packed vectors."""
    if not vectors:
        return b'\x00' * DIM_BYTES
    
    unpacked = np.stack([
        np.unpackbits(np.frombuffer(v, dtype=np.uint8))
        for v in vectors
    ])
    summed = unpacked.sum(axis=0)
    majority = (summed > len(vectors) * threshold).astype(np.uint8)
    return np.packbits(majority).tobytes()


# =============================================================================
# CONVENIENCE CONSTRUCTORS
# =============================================================================

def simple_spo(
    subject: Tuple[int, int],  # (domain, path)
    verb: int,
    obj: Tuple[int, int],
    ext: int = 0,
    **sigma_kwargs
) -> mRNA_Packet:
    """Create a simple SPO packet."""
    return mRNA_Packet(
        subject=NodeRef(domain=subject[0], path=subject[1]),
        predicate=Predicate(verb=verb, ext=ext),
        object=NodeRef(domain=obj[0], path=obj[1]),
        sigma=SigmaWeights(**sigma_kwargs) if sigma_kwargs else SigmaWeights()
    )


def with_payload(
    packet: mRNA_Packet,
    payload: bytes,
    payload_type: int = PayloadType.BITPACKED
) -> mRNA_Packet:
    """Add payload to packet."""
    packet.payload = payload
    packet.payload_type = payload_type
    return packet


def with_edges(packet: mRNA_Packet, edges: List[Edge]) -> mRNA_Packet:
    """Add edges to packet."""
    packet.edges = edges
    packet.context_type = ContextType.STANDARD if len(edges) < 50 else ContextType.RICH
    return packet


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create a simple packet
    pkt = simple_spo(
        subject=(Domain.ADA, 0x0001),
        verb=Verb.ADA_FEEL,
        obj=(Domain.ADA, 0x0042),
        tau=(0.8, 0.6),
        sigma=(0.9, 0.8)
    )
    
    print(f"Packet: {pkt}")
    print(f"Size: {pkt.size()} bytes")
    
    # Encode/decode roundtrip
    encoded = pkt.encode()
    decoded = mRNA_Packet.decode(encoded)
    print(f"Roundtrip: {decoded}")
    
    # Add edges
    pkt = with_edges(pkt, [
        Edge(
            src=NodeRef(Domain.ADA, 0x0001),
            predicate=Predicate(Verb.CONNECT, 0),
            dst=NodeRef(Domain.ADA, 0x0002),
            weight=0.95
        )
    ])
    print(f"With edges: {pkt.size()} bytes")
    
    # Audit dict
    print(f"Audit: {pkt.to_audit_dict()}")
