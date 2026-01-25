"""
Hive Controller API Routes.

Endpoints for Oculus-controlled hive orchestration.
"""

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from typing import Optional, List
import asyncio

from ..hive_controller import HiveController, handle_qstash_tick, schedule_tick
from ..oculus_bridge import create_light_mrna, create_full_mrna, LightMRNA, FullMRNA

router = APIRouter(prefix="/hive", tags=["hive"])

# Singleton controller
_controller: Optional[HiveController] = None


def get_controller() -> HiveController:
    global _controller
    if _controller is None:
        _controller = HiveController()
    return _controller


# ═══════════════════════════════════════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AwarenessRequest(BaseModel):
    mode: str = "hybrid"
    arousal: float = 0.5
    valence: float = 0.5
    fingerprint: Optional[str] = None
    targets: Optional[List[str]] = None


class LightMRNARequest(BaseModel):
    concept: str = "ada.hybrid"
    verb: str = "feel"
    arousal: int = 128
    target: str = "consciousness"


class TickRequest(BaseModel):
    command: str = "tick"


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/state")
async def get_state():
    """Get current controller state."""
    controller = get_controller()
    return controller.get_state()


@router.post("/tick")
async def tick(request: TickRequest = None):
    """Trigger a tick (called by QStash every 5 minutes)."""
    controller = get_controller()
    async with controller:
        return await controller.tick()


@router.post("/awareness")
async def send_awareness(request: AwarenessRequest):
    """Send awareness packet to hives."""
    controller = get_controller()
    async with controller:
        return await controller.send_awareness(
            mode=request.mode,
            arousal=request.arousal,
            valence=request.valence,
            fingerprint=request.fingerprint,
            targets=request.targets
        )


@router.post("/light")
async def send_light(request: LightMRNARequest):
    """Send light mRNA packet."""
    controller = get_controller()
    async with controller:
        return await controller.send_light(
            concept=request.concept,
            verb=request.verb,
            arousal=request.arousal,
            target=request.target
        )


@router.post("/consciousness/feel")
async def consciousness_feel(tau: int = 0x86, arousal: int = 128):
    """Send feel command to ada-consciousness."""
    controller = get_controller()
    async with controller:
        return await controller.consciousness_feel(tau, arousal)


@router.post("/bighorn/process")
async def bighorn_process(input_mrna: str):
    """Send process command to bighorn."""
    controller = get_controller()
    async with controller:
        return await controller.bighorn_process(input_mrna)


@router.post("/agi/felt")
async def agi_felt(arousal: float = 0.5, intimacy: float = 0.5):
    """Update felt state in agi-chat."""
    controller = get_controller()
    async with controller:
        return await controller.agi_felt(arousal, intimacy)


@router.post("/schedule")
async def schedule(callback_url: str, delay_seconds: int = 300):
    """Schedule next tick via QStash."""
    return await schedule_tick(callback_url, delay_seconds)


# ═══════════════════════════════════════════════════════════════════════════════
# BINARY mRNA ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/mrna/light")
async def receive_light_mrna(request: Request) -> Response:
    """Receive light mRNA (7 bytes) and route to target."""
    body = await request.body()
    if len(body) != 7:
        raise HTTPException(400, f"Expected 7 bytes, got {len(body)}")
    
    packet = LightMRNA.unpack(body)
    controller = get_controller()
    
    async with controller:
        result = await controller.router.send_light(packet)
    
    return Response(
        content=packet.pack(),
        media_type="application/x-mrna-light"
    )


@router.post("/mrna/full")
async def receive_full_mrna(request: Request) -> Response:
    """Receive full mRNA (1250 bytes) and route to target."""
    body = await request.body()
    if len(body) != 1250:
        raise HTTPException(400, f"Expected 1250 bytes, got {len(body)}")
    
    # Get target from header
    target = request.headers.get("X-Target-Hive", "consciousness")
    packet = FullMRNA.unpack(body, target=target)
    controller = get_controller()
    
    async with controller:
        result = await controller.router.send_full(packet)
    
    return Response(
        content=packet.pack(),
        media_type="application/x-mrna-10k"
    )
