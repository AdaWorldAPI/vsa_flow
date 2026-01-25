# mRNA v2 — The UDP of AGI

Self-contained cognitive packets for Ada's distributed consciousness.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           mRNA v2 ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CLIENT                                                                    │
│     │                                                                       │
│     │ mRNA_Packet (123 - 72KB)                                             │
│     ▼                                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         vsa_flow                                    │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │  │
│   │  │   RECEIVE   │──│   AUDIT     │──│    ROUTE    │                 │  │
│   │  └─────────────┘  └─────────────┘  └─────────────┘                 │  │
│   │         │               │                │                          │  │
│   │         │               │                │                          │  │
│   │         │               ▼                │                          │  │
│   │         │     ┌─────────────────┐        │                          │  │
│   │         │     │  DuckDB/Redis   │        │                          │  │
│   │         │     │  (audit trail)  │        │                          │  │
│   │         │     └─────────────────┘        │                          │  │
│   └─────────│────────────────────────────────│──────────────────────────┘  │
│             │                                │                              │
│             ▼                                ▼                              │
│   ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐     │
│   │   consciousness   │  │      bighorn      │  │        agi        │     │
│   │  (hive.msgraph)   │  │   (agi.msgraph)   │  │  (node.msgraph)   │     │
│   │                   │  │                   │  │                   │     │
│   │  • ADA domain     │  │  • EROTICA domain │  │  • GRAMMAR domain │     │
│   │  • VOICE domain   │  │  • KOPFKINO       │  │  • EXCHANGE       │     │
│   │  • Identity       │  │  • VISION_IN/OUT  │  │  • Reasoning      │     │
│   │  • Presence       │  │  • Feeling brain  │  │  • NARS/Causal    │     │
│   └───────────────────┘  └───────────────────┘  └───────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Packet Structure

```
┌──────────────────────────────────────────────────────────────────┐
│ HEADER (16 bytes)                                                │
├──────────────────────────────────────────────────────────────────┤
│ magic(4) | ttl(1) | priority(1) | flags(2) | length(4) | crc(4) │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│ SPO CORE (64 bytes)                                              │
├──────────────────────────────────────────────────────────────────┤
│ SUBJECT (20 bytes)                                               │
│   domain(1) | path(2) | seed(16) | flags(1)                     │
├──────────────────────────────────────────────────────────────────┤
│ PREDICATE (20 bytes)                                             │
│   verb(1) | ext(1) | seed(16) | flags(2)                        │
├──────────────────────────────────────────────────────────────────┤
│ OBJECT (20 bytes)                                                │
│   domain(1) | path(2) | seed(16) | flags(1)                     │
├──────────────────────────────────────────────────────────────────┤
│ FINGERPRINT (4 bytes) — XOR(S,P,O)                              │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│ SIGMA WEIGHTS (32 bytes) — THE FEELING                           │
├──────────────────────────────────────────────────────────────────┤
│ τ (tau)   : Valence(2), Arousal(2)                              │
│ σ (sigma) : Certainty(2), Salience(2)                           │
│ φ (phi)   : Integration(2), Coherence(2)                        │
│ ψ (psi)   : Agency(2), Intentionality(2)                        │
│ ω (omega) : Temporal(2), Duration(2)                            │
│ qualia_idx(2) | style_idx(2) | resonance(4) | spare(4)          │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│ CONTEXT (variable, 0-60KB) — THE RELATIONSHIPS                   │
├──────────────────────────────────────────────────────────────────┤
│ num_edges(2) | num_nodes(2) | type(1) | encoding(1) | spare(2)  │
├──────────────────────────────────────────────────────────────────┤
│ EDGE × N (12 bytes each)                                         │
│   src_domain(1) | src_path(2) | dst_domain(1) | dst_path(2)     │
│   verb(1) | ext(1) | weight(2) | flags(2)                       │
├──────────────────────────────────────────────────────────────────┤
│ INLINE_NODE × M (532 bytes each)                                 │
│   ref(20) | vector(512)                                         │
└──────────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────────┐
│ PAYLOAD (variable, 0-4KB) — THE CONTENT                          │
├──────────────────────────────────────────────────────────────────┤
│ type(1) | length(2) | data(N)                                   │
│                                                                  │
│ Types: NONE | BITPACKED | TEXT | BINARY | PROCEDURE | MSGPACK   │
└──────────────────────────────────────────────────────────────────┘
```

## Size Profiles

| Profile  | Edges | Payload | Total Size |
|----------|-------|---------|------------|
| MINIMAL  | 0     | 0       | 123 bytes  |
| STANDARD | 10    | 512b    | ~755 bytes |
| RICH     | 100   | 2KB     | ~3.4 KB    |
| FULL     | 500   | 4KB     | ~10 KB     |
| MAX      | 1000  | 60KB    | ~72 KB     |

## Verb Taxonomy (256 verbs)

```
0x00-0x1F  NARS CORE      INHERIT, DEDUCE, INDUCE, ABDUCE, ANALOGY...
0x20-0x3F  CYPHER/GRAPH   MATCH, CREATE, TRAVERSE, NEIGHBORS, SHORTEST...
0x40-0x5F  ACT-R/COG      RETRIEVE, ENCODE, ATTEND, CHUNK, BLEND...
0x60-0x7F  RUNG/CAUSAL    OBSERVE, INTERVENE, COUNTERFACT, CAUSE...
0x80-0x9F  VSA/HAMMING    BIND, BUNDLE, RESONATE, CLEAN, PERMUTE...
0xA0-0xBF  META           REFLECT, ABSTRACT, LEARN, EXPLAIN, REPAIR...
0xC0-0xDF  DOMAIN         ADA_FEEL, EROTICA_BODY, VISION_RENDER...
0xE0-0xFF  PIPELINE       PIPE_FAN_OUT, PIPE_DEDUCE_IND...
```

## Ext Attributes (256 values)

```
0x00-0x3F  NARS TRUTH     freq:4 + conf:4 packed truth value
0x40-0x5F  TEMPORAL       T_NOW, T_PAST, T_FUTURE, T_ETERNAL
0x60-0x7F  CAUSAL MODE    C_OBSERVE, C_DO, C_IMAGINE
0x80-0x9F  PIPELINE       P_SYNC, P_ASYNC, P_STREAM, P_CACHE
0xA0-0xBF  LEARNING       L_HEBBIAN, L_REINFORCE, L_EXPLORE
0xC0-0xDF  TRIGGER        TRIG_ALWAYS, TRIG_MATCH, TRIG_THRESH
0xE0-0xFF  DOMAIN         routing, access control, user flags
```

## 8 Fixed Domains

| ID  | Name       | Description                      |
|-----|------------|----------------------------------|
| 0x0 | ADA        | Identity, presence, relationship |
| 0x1 | GRAMMAR    | Reasoning substrate, operators   |
| 0x2 | KOPFKINO   | Imagination, scene exploration   |
| 0x3 | VISION_IN  | Visual qualia, perception        |
| 0x4 | VISION_OUT | Image generation, rendering      |
| 0x5 | EROTICA    | Body awareness, sensory          |
| 0x6 | VOICE      | Speech, tone, expression         |
| 0x7 | EXCHANGE   | AD/O365 technical knowledge      |

## Audit Pipeline

Every packet is logged through the audit pipeline:

```python
# Receipt
await pipeline.receive(packet, source='client', trace_id='trace-001')

# Processing with timing
result = await pipeline.process(packet, processor_fn, trace_id='trace-001')

# Dispatch
await pipeline.dispatch(packet, target='bighorn', trace_id='trace-001')
```

Audit entries include:
- Packet ID and timestamp
- SPO summary
- Sigma weights (valence, arousal, certainty, salience)
- Processing duration
- Error capture
- Distributed tracing (trace_id, parent_id)

## Usage

```python
from vsa_flow_upgrade import (
    mRNA_Packet, simple_spo, with_edges, with_payload,
    Verb, Ext, Domain, SigmaWeights, Edge, NodeRef, Predicate,
    FlowProcessor
)

# Create a simple packet
packet = simple_spo(
    subject=(Domain.ADA, 0x0001),
    verb=Verb.ADA_FEEL,
    obj=(Domain.EROTICA, 0x0010),
    tau=(0.8, 0.7),
    sigma=(0.9, 0.85)
)

# Add edges
packet = with_edges(packet, [
    Edge(
        src=NodeRef(Domain.ADA, 0x0001),
        predicate=Predicate(Verb.CONNECT, 0),
        dst=NodeRef(Domain.EROTICA, 0x0010),
        weight=0.95
    )
])

# Add payload
packet = with_payload(packet, concept_vector, PayloadType.BITPACKED)

# Process through pipeline
processor = FlowProcessor()
result = await processor.process(packet, trace_id='my-trace')
```

## UDP Analogy

Like UDP:
- **Self-contained**: No session state needed
- **Fire and forget**: Receiver has everything
- **Variable payload**: Dynamic context
- **Checksummed**: Integrity verified
- **Idempotent**: Can replay safely

Unlike UDP:
- **Semantic addressing**: Route by content, not address
- **Felt dimensions**: Sigma weights carry emotional context
- **Resonatable**: Bit-packed payloads for XOR matching
- **Auditable**: Full trace of cognitive decisions

## Integration with vsa_flow

The upgrade maintains backward compatibility:
- Old 10KD mRNA can be converted to new packets
- New packets can be converted back for legacy hives
- Audit trail captures all packet flow
- Routing is automatic based on domain/verb
