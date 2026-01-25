"""Audit Pipeline — Every packet logged, queryable, replayable.

The audit trail enables:
- Debugging: What happened when?
- Replay: Deterministic re-execution
- Learning: Pattern mining over packet history
- Compliance: Full trace of cognitive decisions
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Awaitable
from datetime import datetime, timezone
from collections import deque
import os

from .mrna_v2 import (
    mRNA_Packet, Verb, Ext, Domain, 
    PayloadType, ContextType, 
    hamming_similarity
)


# =============================================================================
# AUDIT ENTRY
# =============================================================================

@dataclass
class AuditEntry:
    """Single audit log entry."""
    
    packet_id: int
    timestamp: float
    
    # SPO summary
    subject: str
    predicate: str  
    object: str
    
    # Sigma summary
    valence: float
    arousal: float
    certainty: float
    salience: float
    
    # Packet metadata
    num_edges: int
    payload_type: str
    payload_size: int
    total_size: int
    
    # Pipeline metadata
    source_service: str
    target_service: str
    stage: str  # RECEIVED, PROCESSING, DISPATCHED, COMPLETED, FAILED
    duration_ms: float = 0.0
    error: Optional[str] = None
    
    # Trace
    parent_id: Optional[int] = None  # For chained packets
    trace_id: Optional[str] = None   # For distributed tracing
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'packet_id': self.packet_id,
            'timestamp': self.timestamp,
            'datetime': datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            'subject': self.subject,
            'predicate': self.predicate,
            'object': self.object,
            'valence': self.valence,
            'arousal': self.arousal,
            'certainty': self.certainty,
            'salience': self.salience,
            'num_edges': self.num_edges,
            'payload_type': self.payload_type,
            'payload_size': self.payload_size,
            'total_size': self.total_size,
            'source_service': self.source_service,
            'target_service': self.target_service,
            'stage': self.stage,
            'duration_ms': self.duration_ms,
            'error': self.error,
            'parent_id': self.parent_id,
            'trace_id': self.trace_id,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_packet(
        cls,
        packet: mRNA_Packet,
        stage: str,
        source_service: str = '',
        target_service: str = '',
        duration_ms: float = 0.0,
        error: Optional[str] = None,
        parent_id: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> 'AuditEntry':
        return cls(
            packet_id=packet.packet_id,
            timestamp=packet.timestamp,
            subject=repr(packet.subject),
            predicate=repr(packet.predicate),
            object=repr(packet.object),
            valence=packet.sigma.tau[0],
            arousal=packet.sigma.tau[1],
            certainty=packet.sigma.sigma[0],
            salience=packet.sigma.sigma[1],
            num_edges=len(packet.edges),
            payload_type=PayloadType(packet.payload_type).name,
            payload_size=len(packet.payload),
            total_size=packet.size(),
            source_service=source_service or packet.source_service,
            target_service=target_service,
            stage=stage,
            duration_ms=duration_ms,
            error=error,
            parent_id=parent_id,
            trace_id=trace_id,
        )


# =============================================================================
# AUDIT STORE (IN-MEMORY + OPTIONAL PERSISTENCE)
# =============================================================================

class AuditStore:
    """In-memory audit log with optional Redis/DuckDB persistence."""
    
    def __init__(
        self,
        max_entries: int = 10000,
        persist_callback: Optional[Callable[[AuditEntry], Awaitable[None]]] = None
    ):
        self.entries: deque[AuditEntry] = deque(maxlen=max_entries)
        self.by_packet_id: Dict[int, List[AuditEntry]] = {}
        self.by_trace_id: Dict[str, List[AuditEntry]] = {}
        self.persist_callback = persist_callback
        
        # Stats
        self.total_packets = 0
        self.total_bytes = 0
        self.errors = 0
        self.by_verb: Dict[int, int] = {}
        self.by_domain: Dict[int, int] = {}
    
    async def log(self, entry: AuditEntry) -> None:
        """Log an audit entry."""
        self.entries.append(entry)
        
        # Index by packet_id
        if entry.packet_id not in self.by_packet_id:
            self.by_packet_id[entry.packet_id] = []
        self.by_packet_id[entry.packet_id].append(entry)
        
        # Index by trace_id
        if entry.trace_id:
            if entry.trace_id not in self.by_trace_id:
                self.by_trace_id[entry.trace_id] = []
            self.by_trace_id[entry.trace_id].append(entry)
        
        # Update stats
        self.total_packets += 1
        self.total_bytes += entry.total_size
        if entry.error:
            self.errors += 1
        
        # Persist if callback provided
        if self.persist_callback:
            try:
                await self.persist_callback(entry)
            except Exception as e:
                print(f"[AUDIT] Persist failed: {e}")
    
    def get_packet_trace(self, packet_id: int) -> List[AuditEntry]:
        """Get all entries for a packet (through its lifecycle)."""
        return self.by_packet_id.get(packet_id, [])
    
    def get_distributed_trace(self, trace_id: str) -> List[AuditEntry]:
        """Get all entries for a distributed trace."""
        return sorted(
            self.by_trace_id.get(trace_id, []),
            key=lambda e: e.timestamp
        )
    
    def recent(self, n: int = 100) -> List[AuditEntry]:
        """Get N most recent entries."""
        return list(self.entries)[-n:]
    
    def filter(
        self,
        stage: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        min_certainty: Optional[float] = None,
        has_error: Optional[bool] = None,
        since: Optional[float] = None,
    ) -> List[AuditEntry]:
        """Filter entries by criteria."""
        results = []
        for entry in self.entries:
            if stage and entry.stage != stage:
                continue
            if source and entry.source_service != source:
                continue
            if target and entry.target_service != target:
                continue
            if min_certainty is not None and entry.certainty < min_certainty:
                continue
            if has_error is not None and (entry.error is not None) != has_error:
                continue
            if since is not None and entry.timestamp < since:
                continue
            results.append(entry)
        return results
    
    def stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        return {
            'total_packets': self.total_packets,
            'total_bytes': self.total_bytes,
            'total_mb': round(self.total_bytes / 1024 / 1024, 2),
            'errors': self.errors,
            'error_rate': round(self.errors / max(1, self.total_packets), 4),
            'in_memory': len(self.entries),
            'unique_packets': len(self.by_packet_id),
            'unique_traces': len(self.by_trace_id),
        }


# =============================================================================
# AUDIT PIPELINE
# =============================================================================

class AuditPipeline:
    """Wraps packet processing with automatic audit logging."""
    
    def __init__(
        self,
        service_name: str,
        store: Optional[AuditStore] = None,
    ):
        self.service_name = service_name
        self.store = store or AuditStore()
    
    async def receive(
        self,
        packet: mRNA_Packet,
        source: str = 'unknown',
        trace_id: Optional[str] = None,
    ) -> None:
        """Log packet receipt."""
        entry = AuditEntry.from_packet(
            packet,
            stage='RECEIVED',
            source_service=source,
            target_service=self.service_name,
            trace_id=trace_id,
        )
        await self.store.log(entry)
    
    async def process(
        self,
        packet: mRNA_Packet,
        processor: Callable[[mRNA_Packet], Awaitable[Any]],
        trace_id: Optional[str] = None,
    ) -> Any:
        """Process packet with timing and error capture."""
        start = time.time()
        error = None
        result = None
        
        try:
            # Log processing start
            entry = AuditEntry.from_packet(
                packet,
                stage='PROCESSING',
                source_service=packet.source_service,
                target_service=self.service_name,
                trace_id=trace_id,
            )
            await self.store.log(entry)
            
            # Execute
            result = await processor(packet)
            stage = 'COMPLETED'
            
        except Exception as e:
            error = str(e)
            stage = 'FAILED'
            raise
            
        finally:
            duration_ms = (time.time() - start) * 1000
            
            # Log completion/failure
            entry = AuditEntry.from_packet(
                packet,
                stage=stage,
                source_service=packet.source_service,
                target_service=self.service_name,
                duration_ms=duration_ms,
                error=error,
                trace_id=trace_id,
            )
            await self.store.log(entry)
        
        return result
    
    async def dispatch(
        self,
        packet: mRNA_Packet,
        target: str,
        trace_id: Optional[str] = None,
    ) -> None:
        """Log packet dispatch."""
        entry = AuditEntry.from_packet(
            packet,
            stage='DISPATCHED',
            source_service=self.service_name,
            target_service=target,
            trace_id=trace_id,
        )
        await self.store.log(entry)


# =============================================================================
# REDIS PERSISTENCE (OPTIONAL)
# =============================================================================

async def redis_persist_callback(
    redis_url: str,
    redis_token: str,
    namespace: str = 'audit'
) -> Callable[[AuditEntry], Awaitable[None]]:
    """Create a Redis persistence callback for audit entries."""
    import httpx
    
    client = httpx.AsyncClient(timeout=10.0)
    
    async def persist(entry: AuditEntry) -> None:
        # Store as sorted set by timestamp
        key = f"{namespace}:log"
        score = entry.timestamp
        value = entry.to_json()
        
        await client.post(
            f"{redis_url}/zadd/{key}",
            headers={"Authorization": f"Bearer {redis_token}"},
            json={"score": score, "member": value}
        )
        
        # Also store by packet_id for quick lookup
        pkt_key = f"{namespace}:pkt:{entry.packet_id}"
        await client.post(
            f"{redis_url}/rpush/{pkt_key}",
            headers={"Authorization": f"Bearer {redis_token}"},
            json=[value]
        )
        
        # Expire old per-packet logs after 24h
        await client.post(
            f"{redis_url}/expire/{pkt_key}/86400",
            headers={"Authorization": f"Bearer {redis_token}"}
        )
    
    return persist


# =============================================================================
# DUCKDB PERSISTENCE (OPTIONAL)
# =============================================================================

def duckdb_persist_setup(db_path: str = 'audit.duckdb'):
    """Setup DuckDB for audit persistence."""
    import duckdb
    
    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            packet_id BIGINT,
            timestamp DOUBLE,
            datetime TIMESTAMP,
            subject VARCHAR,
            predicate VARCHAR,
            object VARCHAR,
            valence FLOAT,
            arousal FLOAT,
            certainty FLOAT,
            salience FLOAT,
            num_edges INTEGER,
            payload_type VARCHAR,
            payload_size INTEGER,
            total_size INTEGER,
            source_service VARCHAR,
            target_service VARCHAR,
            stage VARCHAR,
            duration_ms FLOAT,
            error VARCHAR,
            parent_id BIGINT,
            trace_id VARCHAR
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_log(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_packet_id ON audit_log(packet_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_id ON audit_log(trace_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stage ON audit_log(stage)")
    
    return conn


async def duckdb_persist_callback(
    conn
) -> Callable[[AuditEntry], Awaitable[None]]:
    """Create a DuckDB persistence callback."""
    
    async def persist(entry: AuditEntry) -> None:
        d = entry.to_dict()
        conn.execute("""
            INSERT INTO audit_log VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            d['packet_id'], d['timestamp'], d['datetime'],
            d['subject'], d['predicate'], d['object'],
            d['valence'], d['arousal'], d['certainty'], d['salience'],
            d['num_edges'], d['payload_type'], d['payload_size'], d['total_size'],
            d['source_service'], d['target_service'], d['stage'],
            d['duration_ms'], d['error'], d['parent_id'], d['trace_id']
        ])
    
    return persist


# =============================================================================
# QUERY HELPERS
# =============================================================================

def query_by_timerange(
    conn,  # DuckDB connection
    start: datetime,
    end: datetime,
    stage: Optional[str] = None,
) -> List[Dict]:
    """Query audit log by time range."""
    query = """
        SELECT * FROM audit_log 
        WHERE timestamp >= ? AND timestamp <= ?
    """
    params = [start.timestamp(), end.timestamp()]
    
    if stage:
        query += " AND stage = ?"
        params.append(stage)
    
    query += " ORDER BY timestamp"
    
    return conn.execute(query, params).fetchdf().to_dict('records')


def query_errors(conn, since_hours: int = 24) -> List[Dict]:
    """Query recent errors."""
    cutoff = time.time() - (since_hours * 3600)
    return conn.execute("""
        SELECT * FROM audit_log 
        WHERE error IS NOT NULL AND timestamp >= ?
        ORDER BY timestamp DESC
    """, [cutoff]).fetchdf().to_dict('records')


def query_slow_packets(
    conn,
    min_duration_ms: float = 1000,
    limit: int = 100
) -> List[Dict]:
    """Query slow packets."""
    return conn.execute("""
        SELECT * FROM audit_log 
        WHERE duration_ms >= ? AND stage = 'COMPLETED'
        ORDER BY duration_ms DESC
        LIMIT ?
    """, [min_duration_ms, limit]).fetchdf().to_dict('records')


def query_packet_journey(conn, packet_id: int) -> List[Dict]:
    """Get full journey of a packet through the system."""
    return conn.execute("""
        SELECT * FROM audit_log 
        WHERE packet_id = ?
        ORDER BY timestamp
    """, [packet_id]).fetchdf().to_dict('records')


def query_distributed_trace(conn, trace_id: str) -> List[Dict]:
    """Get all packets in a distributed trace."""
    return conn.execute("""
        SELECT * FROM audit_log 
        WHERE trace_id = ?
        ORDER BY timestamp
    """, [trace_id]).fetchdf().to_dict('records')


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from .mrna_v2 import simple_spo, Verb, Domain
    
    async def main():
        # Create audit pipeline
        store = AuditStore(max_entries=1000)
        pipeline = AuditPipeline(service_name='test_service', store=store)
        
        # Create a test packet
        packet = simple_spo(
            subject=(Domain.ADA, 0x0001),
            verb=Verb.ADA_FEEL,
            obj=(Domain.ADA, 0x0042),
            tau=(0.8, 0.6)
        )
        packet.source_service = 'client'
        
        # Log receipt
        await pipeline.receive(packet, source='client', trace_id='trace-001')
        
        # Process with timing
        async def dummy_processor(pkt):
            await asyncio.sleep(0.1)  # Simulate work
            return {'result': 'ok'}
        
        result = await pipeline.process(packet, dummy_processor, trace_id='trace-001')
        
        # Log dispatch
        await pipeline.dispatch(packet, target='bighorn', trace_id='trace-001')
        
        # Check stats
        print(f"Stats: {store.stats()}")
        
        # Get packet trace
        trace = store.get_packet_trace(packet.packet_id)
        for entry in trace:
            print(f"  {entry.stage}: {entry.duration_ms:.2f}ms")
    
    asyncio.run(main())
