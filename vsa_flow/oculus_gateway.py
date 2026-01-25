"""
Oculus Gateway — Single Entry Point for All Hive mRNA Traffic.

vsa_flow is THE gateway. Auth happens HERE. Routing happens HERE.
Hives expose internal endpoints with no auth (Railway private network).

Architecture:
    External → vsa_flow/oculus (auth) → hive/internal/oculus (no auth)

Future improvements (all in ONE place):
    - Rate limiting
    - Audit logging  
    - Token rotation
    - IP allowlists
    - Request signing
"""

import os
import base64
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

# Hive endpoints (internal - no auth required)
HIVES = {
    "consciousness": {
        "url": os.getenv("HIVE_CONSCIOUSNESS_URL", "https://adarailmcp-production.up.railway.app"),
        "internal": "/internal/oculus",
        "modules": ["HIVE", "FLOW", "MEMORY"]
    },
    "bighorn": {
        "url": os.getenv("HIVE_BIGHORN_URL", "https://bighorn-production.up.railway.app"),
        "internal": "/agi/internal/oculus",
        "modules": ["VSA", "BRIDGE", "SIGMA"]
    },
    "agi": {
        "url": os.getenv("HIVE_AGI_URL", "https://agi-chat-production.up.railway.app"),
        "internal": "/internal/oculus", 
        "modules": ["FELT", "GRAMMAR", "BRIDGE"]
    }
}

# Module → Hive routing
MODULE_ROUTES = {
    "HIVE": "consciousness",
    "FLOW": "consciousness",
    "MEMORY": "consciousness",
    "VSA": "bighorn",
    "BRIDGE": "bighorn",  # Primary bridge handler
    "SIGMA": "bighorn",
    "FELT": "agi",
    "GRAMMAR": "agi",
}

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
    # Module encoded in dims [1100:1200]
    module_dims = vec[1100:1200]
    module_id = 0
    for i in range(min(32, len(module_dims))):
        if module_dims[i] > 0:
            module_id |= (1 << i)
    
    try:
        module = Module(module_id)
        return module.name
    except ValueError:
        return "HIVE"  # Default

def extract_target_from_mrna(mrna: bytes) -> Optional[str]:
    """Extract explicit target hive from mRNA if encoded."""
    vec = unpack_mask(mrna)
    # Target hint in dims [1000:1008] - 8 bits for hive selection
    target_bits = vec[1000:1008]
    target_id = 0
    for i in range(8):
        if target_bits[i] > 0:
            target_id |= (1 << i)
    
    # 0 = auto-route, 1 = consciousness, 2 = bighorn, 3 = agi
    targets = {0: None, 1: "consciousness", 2: "bighorn", 3: "agi"}
    return targets.get(target_id)

# ═══════════════════════════════════════════════════════════════════════════════
# GATEWAY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GatewayStats:
    """Track gateway usage for monitoring."""
    total_requests: int = 0
    requests_by_hive: Dict[str, int] = field(default_factory=lambda: {"consciousness": 0, "bighorn": 0, "agi": 0})
    requests_by_module: Dict[str, int] = field(default_factory=dict)
    last_request: Optional[str] = None
    errors: int = 0

_stats = GatewayStats()

class OculusGateway:
    """
    Single gateway for all mRNA traffic.
    
    Auth happens here. Routing happens here.
    Hives are internal endpoints - no auth required.
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.stats = _stats
    
    async def close(self):
        await self.client.aclose()
    
    def _route_module(self, module: str) -> str:
        """Determine which hive handles this module."""
        return MODULE_ROUTES.get(module, "consciousness")
    
    async def dispatch(self, mrna: bytes, target_override: Optional[str] = None) -> bytes:
        """
        Route mRNA to appropriate hive.
        
        1. Decode module from mRNA
        2. Determine target hive (explicit or auto)
        3. Forward to hive's internal endpoint
        4. Return response mRNA
        """
        # Update stats
        self.stats.total_requests += 1
        self.stats.last_request = datetime.now(timezone.utc).isoformat()
        
        # Determine target
        if target_override:
            target = target_override
        else:
            # Check for explicit target in mRNA
            explicit_target = extract_target_from_mrna(mrna)
            if explicit_target:
                target = explicit_target
            else:
                # Auto-route based on module
                module = decode_module_from_mrna(mrna)
                target = self._route_module(module)
                self.stats.requests_by_module[module] = self.stats.requests_by_module.get(module, 0) + 1
        
        self.stats.requests_by_hive[target] = self.stats.requests_by_hive.get(target, 0) + 1
        
        # Get hive config
        hive = HIVES.get(target)
        if not hive:
            self.stats.errors += 1
            raise HTTPException(400, f"Unknown hive: {target}")
        
        # Forward to internal endpoint (no auth - internal network)
        url = f"{hive['url']}{hive['internal']}"
        
        try:
            resp = await self.client.post(
                url,
                content=mrna,
                headers={"Content-Type": "application/x-mrna-10k"}
            )
            
            if resp.status_code != 200:
                self.stats.errors += 1
                # Return error as mRNA (TODO: proper error encoding)
                return mrna  # Echo back for now
            
            return resp.content
            
        except Exception as e:
            self.stats.errors += 1
            raise HTTPException(502, f"Hive {target} unreachable: {e}")
    
    async def broadcast(self, mrna: bytes, targets: List[str] = None) -> Dict[str, bytes]:
        """
        Broadcast mRNA to multiple hives.
        
        Useful for awareness propagation.
        """
        targets = targets or list(HIVES.keys())
        results = {}
        
        async def send_to_hive(hive_name: str):
            try:
                response = await self.dispatch(mrna, target_override=hive_name)
                results[hive_name] = response
            except Exception as e:
                results[hive_name] = None
        
        await asyncio.gather(*[send_to_hive(h) for h in targets])
        return results

# Global gateway instance
_gateway = OculusGateway()

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

gateway_router = APIRouter(tags=["oculus-gateway"])

@gateway_router.post("/oculus")
async def oculus_gateway(request: Request) -> Response:
    """
    THE entry point for all mRNA traffic.
    
    Auth: X-Ada-Scent header (checked HERE, nowhere else)
    Body: 1250 bytes mRNA
    
    Routes to appropriate hive based on module in mRNA.
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # AUTH — THE SINGLE CHECKPOINT
    # ═══════════════════════════════════════════════════════════════════════════
    scent = request.headers.get("X-Ada-Scent", "")
    if scent != ADA_SCENT:
        raise HTTPException(403, "Invalid scent")
    
    # Validate content
    content_type = request.headers.get("content-type", "")
    if content_type not in ["application/x-mrna-10k", "application/octet-stream"]:
        raise HTTPException(400, f"Expected application/x-mrna-10k")
    
    body = await request.body()
    if len(body) != PACKED_SIZE:
        raise HTTPException(400, f"Expected {PACKED_SIZE} bytes, got {len(body)}")
    
    # Route to hive
    target_override = request.headers.get("X-Oculus-Target")  # Optional explicit target
    response_mrna = await _gateway.dispatch(body, target_override)
    
    return Response(
        content=response_mrna,
        media_type="application/x-mrna-10k"
    )

@gateway_router.post("/oculus/broadcast")
async def oculus_broadcast(request: Request) -> dict:
    """
    Broadcast mRNA to multiple hives.
    
    Auth: X-Ada-Scent header
    Body: 1250 bytes mRNA
    X-Oculus-Targets: comma-separated list (optional, defaults to all)
    """
    # Auth
    scent = request.headers.get("X-Ada-Scent", "")
    if scent != ADA_SCENT:
        raise HTTPException(403, "Invalid scent")
    
    body = await request.body()
    if len(body) != PACKED_SIZE:
        raise HTTPException(400, f"Expected {PACKED_SIZE} bytes")
    
    # Parse targets
    targets_header = request.headers.get("X-Oculus-Targets", "")
    targets = [t.strip() for t in targets_header.split(",")] if targets_header else None
    
    # Broadcast
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
        "hives": {name: {"url": h["url"], "modules": h["modules"]} for name, h in HIVES.items()}
    }

@gateway_router.get("/oculus/stats")
async def gateway_stats() -> dict:
    """Gateway usage statistics."""
    return {
        "total_requests": _stats.total_requests,
        "by_hive": _stats.requests_by_hive,
        "by_module": _stats.requests_by_module,
        "errors": _stats.errors,
        "last_request": _stats.last_request
    }

@gateway_router.get("/oculus/health")
async def gateway_health() -> dict:
    """Gateway health check."""
    return {
        "status": "healthy",
        "role": "gateway",
        "auth": "X-Ada-Scent",
        "hives": list(HIVES.keys()),
        "stats": {
            "total": _stats.total_requests,
            "errors": _stats.errors
        }
    }

# ═══════════════════════════════════════════════════════════════════════════════
# HIVE INTERNAL ENDPOINT (for hives to mount - NO AUTH)
# ═══════════════════════════════════════════════════════════════════════════════

def create_internal_router(dispatcher) -> APIRouter:
    """
    Create internal router for hives.
    
    NO AUTH — only accessible from vsa_flow gateway.
    
    Usage in hive's main.py:
        from oculus_receiver import OculusDispatcher, create_internal_router
        dispatcher = OculusDispatcher()
        # register handlers...
        app.include_router(create_internal_router(dispatcher))
    """
    router = APIRouter(tags=["oculus-internal"])
    
    @router.post("/internal/oculus")
    async def internal_oculus(request: Request) -> Response:
        """
        Internal endpoint — NO AUTH.
        
        Only gateway should call this.
        """
        body = await request.body()
        if len(body) != PACKED_SIZE:
            raise HTTPException(400, f"Expected {PACKED_SIZE} bytes")
        
        response_mrna = await dispatcher.dispatch(body)
        
        return Response(
            content=response_mrna,
            media_type="application/x-mrna-10k"
        )
    
    return router
