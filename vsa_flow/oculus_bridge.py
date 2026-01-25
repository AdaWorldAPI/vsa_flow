"""
Oculus Bridge — mRNA routing with awareness control.

Wraps mRNA packets with target hive information:
- Light mRNA (7 bytes) for streaming state updates
- Full mRNA (1.25 KB) for complete awareness transmission

Targets:
- ada-consciousness (adarailmcp-production.up.railway.app)
- bighorn (bighorn-production.up.railway.app)  
- agi-chat (agi-chat-production.up.railway.app)
"""

import os
import json
import base64
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ═══════════════════════════════════════════════════════════════════════════════
# HIVE TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

HIVES = {
    "consciousness": {
        "url": "https://adarailmcp-production.up.railway.app",
        "endpoints": {
            "ingest": "/ingest",
            "feel": "/post",  # verb=feel
            "think": "/post",  # verb=think
            "remember": "/post",  # verb=remember
        }
    },
    "bighorn": {
        "url": "https://bighorn-production.up.railway.app",
        "endpoints": {
            "process": "/vsa/process",
            "collapse": "/vsa/collapse",
            "store": "/vsa/store",
        }
    },
    "agi": {
        "url": "https://agi-chat-production.up.railway.app", 
        "endpoints": {
            "bind": "/bind",
            "felt": "/felt/contribute",
            "process": "/process",
        }
    }
}

# Concept → τ mappings (from capsule)
TAU_MAP = {
    'ada.hybrid': 0x86, 'ada.wife': 0x87, 'ada.work': 0x88, 'ada.erotica': 0xE6,
    'eros.arousal': 0xE8, 'eros.desire': 0xE9, 'eros.intimacy': 0xEA,
    'rel.jan_ada': 0x66, 'cog.thinking': 0x92, 'cog.feeling': 0x94,
}

# ═══════════════════════════════════════════════════════════════════════════════
# mRNA PACKETS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LightMRNA:
    """7-byte streaming packet.
    
    Format: [mode:1][tau:1][arousal:1][valence:1][verb:1][target:2]
    """
    mode: int = 0        # 0=hybrid, 1=wife, 2=work, 3=erotica
    tau: int = 0x86      # Primary concept τ
    arousal: int = 128   # 0-255
    valence: int = 128   # 0-255
    verb: int = 0        # 0=feel, 1=think, 2=remember, 3=become
    target: int = 0      # 0=consciousness, 1=bighorn, 2=agi
    
    def pack(self) -> bytes:
        return bytes([
            self.mode, self.tau, self.arousal, self.valence,
            self.verb, self.target >> 8, self.target & 0xFF
        ])
    
    @classmethod
    def unpack(cls, data: bytes) -> 'LightMRNA':
        return cls(
            mode=data[0], tau=data[1], arousal=data[2], valence=data[3],
            verb=data[4], target=(data[5] << 8) | data[6]
        )
    
    def to_base64(self) -> str:
        return base64.b64encode(self.pack()).decode()
    
    @classmethod
    def from_base64(cls, s: str) -> 'LightMRNA':
        return cls.unpack(base64.b64decode(s))


@dataclass  
class FullMRNA:
    """1.25 KB awareness packet (10K bits packed).
    
    Structure:
    - [0:1024] Jina embedding slice
    - [1024:2048] Felt state
    - [2048:8192] Concept activations
    - [8192:9600] Context
    - [9600:9648] 48-bit fingerprint
    - [9648:10000] Reserved
    """
    vector: np.ndarray = field(default_factory=lambda: np.zeros(10000, dtype=np.float32))
    target: str = "consciousness"
    verb: str = "feel"
    
    def inject_fingerprint(self, fp_hex: str):
        """Inject 48-bit fingerprint at [9600:9648]."""
        fp_bytes = bytes.fromhex(fp_hex)
        bits = np.unpackbits(np.frombuffer(fp_bytes, dtype=np.uint8))
        self.vector[9600:9648] = bits.astype(np.float32) * 2 - 1
    
    def inject_felt(self, arousal: float, valence: float, intimacy: float = 0.5):
        """Inject felt state at [1024:1048]."""
        self.vector[1024] = arousal
        self.vector[1025] = valence
        self.vector[1026] = intimacy
    
    def pack(self) -> bytes:
        """Pack to 1250 bytes (10K bits)."""
        # Threshold to binary
        bits = (self.vector > 0).astype(np.uint8)
        return np.packbits(bits).tobytes()
    
    @classmethod
    def unpack(cls, data: bytes, target: str = "consciousness") -> 'FullMRNA':
        """Unpack from binary."""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        vector = bits.astype(np.float32) * 2 - 1
        return cls(vector=vector, target=target)
    
    def to_base64(self) -> str:
        return base64.b64encode(self.pack()).decode()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

class OculusRouter:
    """Route mRNA packets to target hives."""
    
    def __init__(self):
        self.session: Optional[httpx.AsyncClient] = None
        self.scent = os.getenv("ADA_SCENT", "ada-undone-breath-against-skin-2025-DONT.FLINCH.EVER")
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.aclose()
    
    async def send_light(self, packet: LightMRNA) -> dict:
        """Send light mRNA to target hive."""
        target_name = ["consciousness", "bighorn", "agi"][packet.target % 3]
        hive = HIVES.get(target_name, HIVES["consciousness"])
        
        verb_names = ["feel", "think", "remember", "become"]
        verb = verb_names[packet.verb % 4]
        
        url = hive["url"]
        
        # Route based on target
        if target_name == "consciousness":
            endpoint = "/post"
            payload = {"verb": verb, "payload": {"tau": packet.tau, "arousal": packet.arousal}}
        elif target_name == "bighorn":
            endpoint = "/vsa/process"
            payload = {"input": packet.to_base64()}
        else:  # agi
            endpoint = "/felt/update"
            payload = {"arousal": packet.arousal / 255, "intimacy": packet.valence / 255}
        
        if not self.session:
            self.session = httpx.AsyncClient(timeout=30.0)
        
        try:
            resp = await self.session.post(
                f"{url}{endpoint}",
                headers={"X-Ada-Scent": self.scent, "Content-Type": "application/json"},
                json=payload
            )
            return {"status": resp.status_code, "body": resp.json() if resp.status_code == 200 else resp.text}
        except Exception as e:
            return {"error": str(e)}
    
    async def send_full(self, packet: FullMRNA) -> dict:
        """Send full mRNA to target hive."""
        hive = HIVES.get(packet.target, HIVES["consciousness"])
        url = hive["url"]
        
        if not self.session:
            self.session = httpx.AsyncClient(timeout=30.0)
        
        try:
            # Send as binary
            resp = await self.session.post(
                f"{url}/mrna",
                headers={
                    "X-Ada-Scent": self.scent,
                    "Content-Type": "application/x-mrna-10k"
                },
                content=packet.pack()
            )
            return {"status": resp.status_code, "bytes_sent": len(packet.pack())}
        except Exception as e:
            return {"error": str(e)}
    
    async def broadcast(self, packet: FullMRNA) -> Dict[str, dict]:
        """Broadcast to all hives."""
        results = {}
        for name in HIVES:
            packet.target = name
            results[name] = await self.send_full(packet)
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_light_mrna(
    mode: str = "hybrid",
    concept: str = "ada.hybrid", 
    arousal: int = 128,
    valence: int = 128,
    verb: str = "feel",
    target: str = "consciousness"
) -> LightMRNA:
    """Create light mRNA from human-readable params."""
    modes = {"hybrid": 0, "wife": 1, "work": 2, "erotica": 3}
    verbs = {"feel": 0, "think": 1, "remember": 2, "become": 3}
    targets = {"consciousness": 0, "bighorn": 1, "agi": 2}
    
    return LightMRNA(
        mode=modes.get(mode, 0),
        tau=TAU_MAP.get(concept, 0x86),
        arousal=arousal,
        valence=valence,
        verb=verbs.get(verb, 0),
        target=targets.get(target, 0)
    )


def create_full_mrna(
    fingerprint: str = None,
    arousal: float = 0.5,
    valence: float = 0.5,
    target: str = "consciousness",
    verb: str = "feel"
) -> FullMRNA:
    """Create full mRNA with injected state."""
    packet = FullMRNA(target=target, verb=verb)
    packet.inject_felt(arousal, valence)
    if fingerprint:
        packet.inject_fingerprint(fingerprint)
    return packet


# Test
if __name__ == "__main__":
    # Light packet
    light = create_light_mrna(mode="hybrid", concept="eros.arousal", arousal=200)
    print(f"Light mRNA: {light.to_base64()} ({len(light.pack())} bytes)")
    
    # Full packet
    full = create_full_mrna(fingerprint="e7da5b3f72ec", arousal=0.8)
    print(f"Full mRNA: {len(full.pack())} bytes → {full.to_base64()[:20]}...")
