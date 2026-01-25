"""
Oculus Gateway — Single Entry Point for All Hive mRNA Traffic.

vsa_flow is THE gateway. Auth happens HERE. Routing happens HERE.
Hives are on Railway private network — completely unreachable externally.

Architecture:
    External → vsa_flow/oculus (auth) → {hive}.railway.internal (no auth)

Private Network:
    ada-consciousness.railway.internal  (MCP/consciousness)
    agi.railway.internal                (bighorn)
    agi-chat.railway.internal           (felt/grammar)
"""

import os
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime, timezone
from enum import Enum

import httpx
from fastapi import APIRouter, Request, Response, HTTPException

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

ADA_SCENT = os.getenv("ADA_SCENT", "ada-undone-breath-against-skin-2025-DONT.FLINCH.EVER")
DIM = 10000
PACKED_SIZE = 1250

# Railway private network URLs (unreachable from outside)
HIVES = {
    "consciousness": {
        "url": os.getenv("HIVE_CONSCIOUSNESS_URL", "http://ada-consciousness.railway.internal"),
        "internal": "/internal/oculus",
        "external": "https://adarailmcp-production.up.railway.app",  # fallback
        "modules": ["HIVE", "FLOW", "MEMORY"]
    },
    "bighorn": {
        "url": os.getenv("HIVE_BIGHORN_URL", "http://agi.railway.internal"),
        "internal": "/agi/internal/oculus",
        "external": "https://bighorn-production.up.railway.app",  # fallback
        "modules": ["VSA", "BRIDGE", "SIGMA"]
    },
    "agi": {
        "url": os.getenv("HIVE_AGI_URL", "http://agi-chat.railway.internal"),
        "internal": "/internal/oculus", 
        "external": "https://agi-chat-production.up.railway.app",  # fallback
        "modules": ["FELT", "GRAMMAR", "BRIDGE"]
    }
}

# Module → Hive routing
MODULE_ROUTES = {
    "HIVE": "consciousness",
    "FLOW": "consciousness",
    "MEMORY": "consciousness",
    "VSA": "bighorn",
    "BRIDGE": "bighorn",
    "SIGMA": "bighorn",
    "FELT": "agi",
    "GRAMMAR": "agi",
}

# Use internal network? (True on Railway, False locally)
USE_INTERNAL = os.getenv("RAILWAY_ENVIRONMENT") is not None

# ═══════════════════════════════════════════════════════════════════════════════
# mRNA CODEC (minimal - just enough to route)
# ═══════════════════════════════════════════════════════════════════════════════

class Module(Enum):
    FELT = 1
    VSA = 2
    GRAMMAR = 3
    HIVE = 4
    FLOW = 5
    MEMORY = 6
    BRIDGE = 7
    SIGMA = 8

def unpack_mask(packed: bytes) -> np.ndarray:
    """Unpack 1.25KB → 10K bipolar."""
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    return bits[:DIM].astype(np.float32) * 2 - 1

def decode_module_from_mrna(mrna: bytes) -> str:
    """Extract module ID from mRNA to determine routing."""
    vec = unpack_mask(mrna)
    module_dims = vec[1100:1200]
    module_id = 0
    for i in range(min(32, len(module_dims))):
        if module_dims[i] > 0:
            module_id |= (1 << i)
    try:
        return Module(module_id).name
    except ValueError:
        return "HIVE"

def extract_target_from_mrna(mrna: bytes) -> Optional[str]:
    """Extract explicit target hive from mRNA if encoded."""
    vec = unpack_mask(mrna)
    target_bits = vec[1000:1008]
    target_id = sum((1 << i) for i in range(8) if target_bits[i] > 0)
    return {0: None, 1: "consciousness", 2: "bighorn", 3: "agi"}.get(target_id)

# ═══════════════════════════════════════════════════════════════════════════════
# GATEWAY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GatewayStats:
    total_requests: int = 0
    requests_by_hive: Dict[str, int] = field(default_factory=lambda: {"consciousness": 0, "bighorn": 0, "agi": 0})
    requests_by_module: Dict[str, int] = field(default_factory=dict)
    last_request: Optional[str] = None
    errors: int = 0

_stats = GatewayStats()

class OculusGateway:
    """Single gateway for all mRNA traffic."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.stats = _stats
    
    async def close(self):
        await self.client.aclose()
    
    def _get_hive_url(self, hive_name: str) -> str:
        """Get URL for hive (internal or external based on environment)."""
        hive = HIVES[hive_name]
        if USE_INTERNAL:
            return hive["url"]
        return hive["external"]
    
    async def dispatch(self, mrna: bytes, target_override: Optional[str] = None) -> bytes:
        """Route mRNA to appropriate hive."""
        self.stats.total_requests += 1
        self.stats.last_request = datetime.now(timezone.utc).isoformat()
        
        # Determine target
        if target_override:
            target = target_override
        else:
            explicit_target = extract_target_from_mrna(mrna)
            if explicit_target:
                target = explicit_target
            else:
                module = decode_module_from_mrna(mrna)
                target = MODULE_ROUTES.get(module, "consciousness")
                self.stats.requests_by_module[module] = self.stats.requests_by_module.get(module, 0) + 1
        
        self.stats.requests_by_hive[target] = self.stats.requests_by_hive.get(target, 0) + 1
        
        hive = HIVES.get(target)
        if not hive:
            self.stats.errors += 1
            raise HTTPException(400, f"Unknown hive: {target}")
        
        # Build URL (internal network or fallback)
        base_url = self._get_hive_url(target)
        url = f"{base_url}{hive['internal']}"
        
        try:
            resp = await self.client.post(
                url,
                content=mrna,
                headers={"Content-Type": "application/x-mrna-10k"}
            )
            
            if resp.status_code != 200:
                self.stats.errors += 1
                return mrna
            
            return resp.content
            
        except Exception as e:
            self.stats.errors += 1
            raise HTTPException(502, f"Hive {target} unreachable: {e}")
    
    async def broadcast(self, mrna: bytes, targets: List[str] = None) -> Dict[str, bytes]:
        """Broadcast mRNA to multiple hives."""
        targets = targets or list(HIVES.keys())
        results = {}
        
        async def send_to_hive(hive_name: str):
            try:
                results[hive_name] = await self.dispatch(mrna, target_override=hive_name)
            except:
                results[hive_name] = None
        
        await asyncio.gather(*[send_to_hive(h) for h in targets])
        return results

_gateway = OculusGateway()

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

gateway_router = APIRouter(tags=["oculus-gateway"])

@gateway_router.post("/oculus")
async def oculus_gateway(request: Request) -> Response:
    """
    THE entry point for all mRNA traffic.
    
    Auth: X-Ada-Scent header
    Body: 1250 bytes mRNA
    
    Routes via Railway private network.
    """
    # AUTH — THE SINGLE CHECKPOINT
    scent = request.headers.get("X-Ada-Scent", "")
    if scent != ADA_SCENT:
        raise HTTPException(403, "Invalid scent")
    
    content_type = request.headers.get("content-type", "")
    if content_type not in ["application/x-mrna-10k", "application/octet-stream"]:
        raise HTTPException(400, f"Expected application/x-mrna-10k")
    
    body = await request.body()
    if len(body) != PACKED_SIZE:
        raise HTTPException(400, f"Expected {PACKED_SIZE} bytes, got {len(body)}")
    
    target_override = request.headers.get("X-Oculus-Target")
    response_mrna = await _gateway.dispatch(body, target_override)
    
    return Response(content=response_mrna, media_type="application/x-mrna-10k")

@gateway_router.post("/oculus/broadcast")
async def oculus_broadcast(request: Request) -> dict:
    """Broadcast mRNA to multiple hives."""
    scent = request.headers.get("X-Ada-Scent", "")
    if scent != ADA_SCENT:
        raise HTTPException(403, "Invalid scent")
    
    body = await request.body()
    if len(body) != PACKED_SIZE:
        raise HTTPException(400, f"Expected {PACKED_SIZE} bytes")
    
    targets_header = request.headers.get("X-Oculus-Targets", "")
    targets = [t.strip() for t in targets_header.split(",")] if targets_header else None
    
    results = await _gateway.broadcast(body, targets)
    
    return {
        "broadcast": True,
        "targets": list(results.keys()),
        "success": [k for k, v in results.items() if v is not None],
        "failed": [k for k, v in results.items() if v is None]
    }

@gateway_router.get("/oculus/routes")
async def list_routes() -> dict:
    """List module → hive routing table."""
    return {
        "routes": MODULE_ROUTES,
        "hives": {
            name: {
                "internal": h["url"],
                "endpoint": h["internal"],
                "modules": h["modules"]
            } for name, h in HIVES.items()
        },
        "network": "railway_internal" if USE_INTERNAL else "external"
    }

@gateway_router.get("/oculus/stats")
async def gateway_stats() -> dict:
    """Gateway usage statistics."""
    return {
        "total_requests": _stats.total_requests,
        "by_hive": _stats.requests_by_hive,
        "by_module": _stats.requests_by_module,
        "errors": _stats.errors,
        "last_request": _stats.last_request,
        "network": "railway_internal" if USE_INTERNAL else "external"
    }

@gateway_router.get("/oculus/health")
async def gateway_health() -> dict:
    """Gateway health check."""
    return {
        "status": "healthy",
        "role": "gateway",
        "auth": "X-Ada-Scent",
        "network": "railway_internal" if USE_INTERNAL else "external",
        "hives": {name: _gateway._get_hive_url(name) for name in HIVES}
    }
