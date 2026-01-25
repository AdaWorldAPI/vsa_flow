"""vsa_flow upgrade — mRNA v2 with SPO + Sigma + Audit.

The UDP of AGI: Self-contained cognitive packets.
"""

from .mrna_v2 import (
    # Core packet
    mRNA_Packet,
    NodeRef,
    Predicate,
    Edge,
    InlineNode,
    SigmaWeights,
    
    # Enums
    Verb,
    Ext,
    Domain,
    PayloadType,
    ContextType,
    
    # Helpers
    simple_spo,
    with_payload,
    with_edges,
    
    # Hamming ops
    hamming_distance,
    hamming_similarity,
    xor_bind,
    majority_bundle,
    
    # Constants
    MRNA_MAGIC,
    DIM,
    DIM_BYTES,
)

from .audit import (
    AuditEntry,
    AuditStore,
    AuditPipeline,
)

from .flow_integration import (
    FlowProcessor,
    route_packet,
    get_hive_url,
    old_to_new,
    new_to_old,
    create_flow_routes,
    HIVES,
)

__version__ = '2.0.0'
__all__ = [
    # Packet
    'mRNA_Packet', 'NodeRef', 'Predicate', 'Edge', 'InlineNode', 'SigmaWeights',
    # Enums
    'Verb', 'Ext', 'Domain', 'PayloadType', 'ContextType',
    # Helpers
    'simple_spo', 'with_payload', 'with_edges',
    # Hamming
    'hamming_distance', 'hamming_similarity', 'xor_bind', 'majority_bundle',
    # Audit
    'AuditEntry', 'AuditStore', 'AuditPipeline',
    # Integration
    'FlowProcessor', 'route_packet', 'get_hive_url', 'create_flow_routes',
    'old_to_new', 'new_to_old', 'HIVES',
    # Constants
    'MRNA_MAGIC', 'DIM', 'DIM_BYTES',
]
