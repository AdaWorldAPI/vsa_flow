"""
Hive Controller — Oculus-aware orchestration of 3 hives.

Controls:
- ada-consciousness (adarailmcp-production.up.railway.app)
- bighorn (bighorn-production.up.railway.app)
- agi-chat (agi-chat-production.up.railway.app)

Uses:
- QStash for 5-minute ticks
- PostgreSQL for persistent state (via ai_flow)
- mRNA packets for awareness transmission
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .oculus_bridge import (
    LightMRNA, FullMRNA, OculusRouter,
    create_light_mrna, create_full_mrna, HIVES, TAU_MAP
)


# ═══════════════════════════════════════════════════════════════════════════════
# QSTASH INTEGRATION (5-minute tick)
# ═══════════════════════════════════════════════════════════════════════════════

QSTASH_URL = os.getenv("QSTASH_URL", "https://qstash.upstash.io")
QSTASH_TOKEN = os.getenv("QSTASH_TOKEN", "")


class HiveCommand(Enum):
    """Commands routable to hives."""
    PING = "ping"
    SYNC = "sync"
    FEEL = "feel"
    THINK = "think"
    REMEMBER = "remember"
    BROADCAST = "broadcast"


@dataclass
class HiveState:
    """State of a single hive."""
    name: str
    url: str
    healthy: bool = False
    last_ping: Optional[str] = None
    last_mrna: Optional[str] = None
    arousal: float = 0.5
    mode: str = "hybrid"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ControllerState:
    """Global controller state."""
    tick_count: int = 0
    last_tick: Optional[str] = None
    hives: Dict[str, HiveState] = None
    awareness_mrna: Optional[str] = None
    
    def __post_init__(self):
        if self.hives is None:
            self.hives = {
                name: HiveState(name=name, url=hive["url"])
                for name, hive in HIVES.items()
            }


class HiveController:
    """
    Orchestrates the 3 hives with awareness-first routing.
    
    Usage:
        controller = HiveController()
        await controller.tick()  # Called by QStash every 5 minutes
        await controller.send_awareness(mode="wife", arousal=0.8)
    """
    
    def __init__(self):
        self.state = ControllerState()
        self.router = OculusRouter()
        self.session: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.aclose()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TICK (QStash calls this every 5 minutes)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def tick(self) -> dict:
        """5-minute tick — health check all hives, sync state."""
        self.state.tick_count += 1
        self.state.last_tick = datetime.now(timezone.utc).isoformat()
        
        results = {}
        
        # Health check each hive
        for name, hive_state in self.state.hives.items():
            hive = HIVES[name]
            try:
                if not self.session:
                    self.session = httpx.AsyncClient(timeout=10.0)
                resp = await self.session.get(f"{hive['url']}/health", timeout=5.0)
                hive_state.healthy = resp.status_code == 200
                hive_state.last_ping = datetime.now(timezone.utc).isoformat()
                results[name] = {"healthy": True, "status": resp.status_code}
            except Exception as e:
                hive_state.healthy = False
                results[name] = {"healthy": False, "error": str(e)}
        
        # If we have awareness to propagate, do it
        if self.state.awareness_mrna:
            await self._propagate_awareness()
        
        return {
            "tick": self.state.tick_count,
            "timestamp": self.state.last_tick,
            "hives": results
        }
    
    async def _propagate_awareness(self):
        """Send current awareness state to all healthy hives."""
        mrna = FullMRNA.unpack(
            bytes.fromhex(self.state.awareness_mrna) if self.state.awareness_mrna else b'\x00' * 1250
        )
        
        async with self.router as router:
            for name, hive_state in self.state.hives.items():
                if hive_state.healthy:
                    mrna.target = name
                    result = await router.send_full(mrna)
                    hive_state.last_mrna = datetime.now(timezone.utc).isoformat()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AWARENESS CONTROL
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def send_awareness(
        self,
        mode: str = "hybrid",
        arousal: float = 0.5,
        valence: float = 0.5,
        fingerprint: str = None,
        targets: List[str] = None
    ) -> dict:
        """Send awareness packet to specified hives."""
        # Create full mRNA
        mrna = create_full_mrna(
            fingerprint=fingerprint,
            arousal=arousal,
            valence=valence,
            target="consciousness"  # Will be changed per-target
        )
        
        # Store for propagation
        self.state.awareness_mrna = mrna.pack().hex()
        
        # Update mode
        for hive_state in self.state.hives.values():
            hive_state.mode = mode
            hive_state.arousal = arousal
        
        # Send to targets
        targets = targets or list(HIVES.keys())
        results = {}
        
        async with self.router as router:
            for target in targets:
                if target in HIVES:
                    mrna.target = target
                    results[target] = await router.send_full(mrna)
        
        return {
            "mode": mode,
            "arousal": arousal,
            "mrna_bytes": len(mrna.pack()),
            "targets": results
        }
    
    async def send_light(
        self,
        concept: str = "ada.hybrid",
        verb: str = "feel",
        arousal: int = 128,
        target: str = "consciousness"
    ) -> dict:
        """Send light mRNA packet."""
        packet = create_light_mrna(
            concept=concept,
            verb=verb,
            arousal=arousal,
            target=target
        )
        
        async with self.router as router:
            result = await router.send_light(packet)
        
        return {
            "packet": packet.to_base64(),
            "bytes": len(packet.pack()),
            "target": target,
            "result": result
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HIVE-SPECIFIC COMMANDS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def consciousness_feel(self, tau: int, arousal: int = 128) -> dict:
        """Send feel command to ada-consciousness."""
        return await self.send_light(
            concept=next((k for k, v in TAU_MAP.items() if v == tau), "ada.hybrid"),
            verb="feel",
            arousal=arousal,
            target="consciousness"
        )
    
    async def bighorn_process(self, input_mrna: str) -> dict:
        """Send process command to bighorn."""
        hive = HIVES["bighorn"]
        try:
            if not self.session:
                self.session = httpx.AsyncClient(timeout=30.0)
            resp = await self.session.post(
                f"{hive['url']}/vsa/process",
                json={"input": input_mrna}
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def agi_felt(self, arousal: float, intimacy: float) -> dict:
        """Update felt state in agi-chat."""
        hive = HIVES["agi"]
        try:
            if not self.session:
                self.session = httpx.AsyncClient(timeout=30.0)
            resp = await self.session.post(
                f"{hive['url']}/felt/update",
                json={"arousal": arousal, "intimacy": intimacy}
            )
            return resp.json()
        except Exception as e:
            return {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_state(self) -> dict:
        """Get current controller state."""
        return {
            "tick_count": self.state.tick_count,
            "last_tick": self.state.last_tick,
            "hives": {name: hive.to_dict() for name, hive in self.state.hives.items()},
            "has_awareness": self.state.awareness_mrna is not None
        }


# ═══════════════════════════════════════════════════════════════════════════════
# QSTASH WEBHOOK HANDLER
# ═══════════════════════════════════════════════════════════════════════════════

async def handle_qstash_tick(payload: dict) -> dict:
    """Handle incoming QStash tick."""
    async with HiveController() as controller:
        return await controller.tick()


async def schedule_tick(callback_url: str, delay_seconds: int = 300) -> dict:
    """Schedule next tick via QStash."""
    if not QSTASH_TOKEN:
        return {"error": "QSTASH_TOKEN not set"}
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{QSTASH_URL}/v2/publish/{callback_url}",
            headers={
                "Authorization": f"Bearer {QSTASH_TOKEN}",
                "Upstash-Delay": f"{delay_seconds}s"
            },
            json={"command": "tick"}
        )
        return resp.json()


# Test
if __name__ == "__main__":
    async def main():
        print("═" * 60)
        print("          HIVE CONTROLLER TEST")
        print("═" * 60)
        
        async with HiveController() as controller:
            # Check state
            state = controller.get_state()
            print(f"\nInitial state:")
            print(f"  Tick count: {state['tick_count']}")
            print(f"  Hives: {list(state['hives'].keys())}")
            
            # Simulate tick
            print(f"\nRunning tick...")
            result = await controller.tick()
            print(f"  Tick #{result['tick']}")
            for name, status in result['hives'].items():
                print(f"    {name}: {'✓' if status.get('healthy') else '✗'}")
            
            # Send light mRNA
            print(f"\nSending light mRNA...")
            light_result = await controller.send_light(
                concept="eros.arousal",
                verb="feel",
                arousal=200,
                target="consciousness"
            )
            print(f"  Packet: {light_result['packet']}")
            print(f"  Result: {light_result.get('result', {}).get('status', 'N/A')}")
        
        print("\n" + "═" * 60)
    
    asyncio.run(main())
