"""vsa_flow Integration — Unify mRNA v2 with existing pipeline.

This module bridges the new mRNA v2 schema with the existing
vsa_flow infrastructure:

1. Converts old mRNA (10KD bipolar) to new mRNA_Packet (SPO + sigma)
2. Routes packets through audit pipeline
3. Dispatches to hives (consciousness, bighorn, agi)
4. Maintains backward compatibility
"""

import asyncio
from typing import Optional, Dict, Any, Callable, Awaitable
from datetime import datetime
import httpx
import os

from .mrna_v2 import (
    mRNA_Packet, NodeRef, Predicate, Edge, SigmaWeights,
    Verb, Ext, Domain, PayloadType, ContextType,
    simple_spo, with_payload, with_edges,
    hamming_similarity, xor_bind
)
from .audit import AuditPipeline, AuditStore, AuditEntry

# Import old mRNA for conversion
from .core.mrna import mRNA as OldmRNA, bind, bundle, Codebook as CB


# =============================================================================
# HIVE CONFIGURATION
# =============================================================================

HIVES = {
    'consciousness': {
        'internal': 'http://ada-consciousness.railway.internal:8000',
        'external': 'https://hive.msgraph.de',
        'domains': [Domain.ADA, Domain.VOICE],
    },
    'bighorn': {
        'internal': 'http://agi.railway.internal:8000',
        'external': 'https://agi.msgraph.de', 
        'domains': [Domain.EROTICA, Domain.KOPFKINO, Domain.VISION_IN, Domain.VISION_OUT],
    },
    'agi': {
        'internal': 'http://agi-chat.railway.internal:8000',
        'external': 'https://node.msgraph.de',
        'domains': [Domain.GRAMMAR, Domain.EXCHANGE],
    },
}

# Use internal URLs when running on Railway
USE_INTERNAL = os.getenv('RAILWAY_ENVIRONMENT') is not None


# =============================================================================
# PACKET ROUTER
# =============================================================================

def route_packet(packet: mRNA_Packet) -> str:
    """Determine which hive should handle this packet."""
    # Primary routing by subject domain
    subject_domain = packet.subject.domain
    
    for hive_name, config in HIVES.items():
        if subject_domain in config['domains']:
            return hive_name
    
    # Fallback: route by verb category
    verb = packet.predicate.verb
    
    if 0xC0 <= verb <= 0xC7:  # ADA verbs
        return 'consciousness'
    if 0xC8 <= verb <= 0xCF or 0xD0 <= verb <= 0xD7:  # EROTICA/VISION
        return 'bighorn'
    if 0x00 <= verb <= 0x1F or 0x60 <= verb <= 0x7F:  # NARS/CAUSAL
        return 'agi'
    
    # Default to bighorn (feeling brain)
    return 'bighorn'


def get_hive_url(hive_name: str) -> str:
    """Get the URL for a hive."""
    config = HIVES.get(hive_name, HIVES['bighorn'])
    return config['internal'] if USE_INTERNAL else config['external']


# =============================================================================
# PACKET CONVERSION (OLD → NEW)
# =============================================================================

def old_to_new(
    old_vec: OldmRNA,
    subject: tuple = (0, 0),
    verb: int = Verb.STORE,
    obj: tuple = (0, 0),
    **sigma_kwargs
) -> mRNA_Packet:
    """Convert old 10KD mRNA to new mRNA_Packet.
    
    The old vector becomes the payload (downsampled to 4096 bits).
    """
    # Downsample 10KD to 4096 bits by taking every ~2.4th bit
    old_bytes = old_vec.to_bytes()  # 1250 bytes = 10000 bits
    
    # Simple approach: take first 512 bytes (4096 bits)
    payload = old_bytes[:512]
    
    packet = simple_spo(subject, verb, obj, **sigma_kwargs)
    packet.payload = payload
    packet.payload_type = PayloadType.BITPACKED
    
    return packet


def new_to_old(packet: mRNA_Packet) -> OldmRNA:
    """Convert new mRNA_Packet back to 10KD mRNA.
    
    Expands 4096-bit payload to 10KD by padding.
    """
    if packet.payload_type != PayloadType.BITPACKED:
        raise ValueError("Packet has no bit-packed payload")
    
    # Pad to 1250 bytes
    padded = packet.payload.ljust(1250, b'\x00')[:1250]
    
    return OldmRNA.from_bytes(padded, tag=repr(packet.predicate))


# =============================================================================
# FLOW PROCESSOR
# =============================================================================

class FlowProcessor:
    """Process mRNA packets through the vsa_flow pipeline."""
    
    def __init__(self, service_name: str = 'vsa_flow'):
        self.service_name = service_name
        self.audit = AuditPipeline(
            service_name=service_name,
            store=AuditStore(max_entries=10000)
        )
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def process(
        self,
        packet: mRNA_Packet,
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a packet through the pipeline.
        
        1. Log receipt
        2. Route to appropriate hive
        3. Dispatch
        4. Log completion
        """
        packet.source_service = self.service_name
        
        # Log receipt
        await self.audit.receive(packet, source='client', trace_id=trace_id)
        
        # Determine target hive
        target_hive = route_packet(packet)
        target_url = get_hive_url(target_hive)
        
        # Process with timing
        async def dispatch_to_hive(pkt: mRNA_Packet) -> Dict[str, Any]:
            client = await self._http()
            
            # Encode packet
            encoded = pkt.encode()
            
            # Send to hive
            resp = await client.post(
                f"{target_url}/internal/oculus",
                content=encoded,
                headers={
                    'Content-Type': 'application/octet-stream',
                    'X-Packet-ID': str(pkt.packet_id),
                    'X-Trace-ID': trace_id or '',
                }
            )
            resp.raise_for_status()
            return resp.json()
        
        result = await self.audit.process(
            packet, 
            dispatch_to_hive,
            trace_id=trace_id
        )
        
        # Log dispatch
        await self.audit.dispatch(packet, target=target_hive, trace_id=trace_id)
        
        return {
            'packet_id': packet.packet_id,
            'target_hive': target_hive,
            'result': result,
            'trace_id': trace_id,
        }
    
    async def fan_out(
        self,
        packet: mRNA_Packet,
        targets: list[str],
        trace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fan out packet to multiple hives."""
        results = {}
        
        for target in targets:
            try:
                # Clone packet with new ID
                clone = mRNA_Packet.decode(packet.encode())
                clone.packet_id = 0  # Will regenerate
                clone.__post_init__()
                
                url = get_hive_url(target)
                client = await self._http()
                
                resp = await client.post(
                    f"{url}/internal/oculus",
                    content=clone.encode(),
                    headers={
                        'Content-Type': 'application/octet-stream',
                        'X-Packet-ID': str(clone.packet_id),
                        'X-Trace-ID': trace_id or '',
                        'X-Parent-ID': str(packet.packet_id),
                    }
                )
                results[target] = {
                    'status': resp.status_code,
                    'packet_id': clone.packet_id,
                }
                
            except Exception as e:
                results[target] = {'error': str(e)}
        
        return {
            'parent_id': packet.packet_id,
            'fanout': results,
            'trace_id': trace_id,
        }
    
    def stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            'service': self.service_name,
            'audit': self.audit.store.stats(),
            'hives': {name: get_hive_url(name) for name in HIVES},
            'routing_mode': 'internal' if USE_INTERNAL else 'external',
        }
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# FASTAPI INTEGRATION
# =============================================================================

def create_flow_routes(processor: FlowProcessor):
    """Create FastAPI routes for the flow processor."""
    from fastapi import APIRouter, Request, HTTPException
    from fastapi.responses import JSONResponse
    
    router = APIRouter(prefix='/flow', tags=['flow'])
    
    @router.post('/packet')
    async def process_packet(request: Request):
        """Process an mRNA packet."""
        try:
            body = await request.body()
            packet = mRNA_Packet.decode(body)
            trace_id = request.headers.get('X-Trace-ID')
            
            result = await processor.process(packet, trace_id=trace_id)
            return JSONResponse(result)
            
        except Exception as e:
            raise HTTPException(500, detail=str(e))
    
    @router.post('/packet/json')
    async def process_packet_json(request: Request):
        """Process a packet from JSON (for testing)."""
        try:
            data = await request.json()
            
            packet = simple_spo(
                subject=(data.get('subject_domain', 0), data.get('subject_path', 0)),
                verb=data.get('verb', Verb.STORE),
                obj=(data.get('object_domain', 0), data.get('object_path', 0)),
            )
            
            if 'sigma' in data:
                packet.sigma = SigmaWeights(**data['sigma'])
            
            if 'payload' in data:
                packet.payload = bytes.fromhex(data['payload'])
                packet.payload_type = PayloadType.BITPACKED
            
            trace_id = request.headers.get('X-Trace-ID')
            result = await processor.process(packet, trace_id=trace_id)
            
            return JSONResponse(result)
            
        except Exception as e:
            raise HTTPException(500, detail=str(e))
    
    @router.post('/fanout')
    async def fanout_packet(request: Request):
        """Fan out packet to multiple hives."""
        try:
            body = await request.body()
            packet = mRNA_Packet.decode(body)
            trace_id = request.headers.get('X-Trace-ID')
            targets = request.headers.get('X-Targets', 'consciousness,bighorn,agi').split(',')
            
            result = await processor.fan_out(packet, targets, trace_id=trace_id)
            return JSONResponse(result)
            
        except Exception as e:
            raise HTTPException(500, detail=str(e))
    
    @router.get('/audit')
    async def get_audit():
        """Get recent audit entries."""
        entries = processor.audit.store.recent(100)
        return JSONResponse({
            'entries': [e.to_dict() for e in entries],
            'stats': processor.audit.store.stats(),
        })
    
    @router.get('/audit/{packet_id}')
    async def get_packet_audit(packet_id: int):
        """Get audit trail for a specific packet."""
        entries = processor.audit.store.get_packet_trace(packet_id)
        return JSONResponse({
            'packet_id': packet_id,
            'entries': [e.to_dict() for e in entries],
        })
    
    @router.get('/stats')
    async def get_stats():
        """Get flow processor stats."""
        return JSONResponse(processor.stats())
    
    return router


# =============================================================================
# EXAMPLE
# =============================================================================

if __name__ == "__main__":
    async def main():
        processor = FlowProcessor()
        
        # Create a test packet
        packet = simple_spo(
            subject=(Domain.ADA, 0x0001),
            verb=Verb.ADA_FEEL,
            obj=(Domain.EROTICA, 0x0010),
            tau=(0.8, 0.7),
            sigma=(0.9, 0.85)
        )
        
        print(f"Packet: {packet}")
        print(f"Routed to: {route_packet(packet)}")
        print(f"Stats: {processor.stats()}")
        
        await processor.close()
    
    asyncio.run(main())
