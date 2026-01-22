"""mRNA Transport - binary vectors over HTTP.

No JSON middleware. Raw 1.25KB payloads.
The vector IS the message.
"""

import httpx
import asyncio
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass
import struct

from ..core.mrna import mRNA, bind, bundle, CB, DIM


# Wire format constants
HEADER_SIZE = 8  # 4 bytes magic + 4 bytes flags
MAGIC = b'mRNA'
VERSION = 1

CONTENT_TYPE = "application/x-mrna-10k"


@dataclass
class Envelope:
    """mRNA with routing metadata.
    
    Total size: 8 (header) + 1250 (vector) + optional tag
    """
    vector: mRNA
    verb: Optional[mRNA] = None      # What to do
    target: Optional[mRNA] = None    # Where to send
    reply_to: Optional[str] = None   # Callback URL
    
    def pack(self) -> bytes:
        """Pack for wire transmission."""
        flags = 0
        parts = [MAGIC, struct.pack('>BBH', VERSION, flags, 0)]
        
        # Main vector (always present)
        parts.append(self.vector.to_bytes())
        
        # Optional verb (bound into vector or separate)
        if self.verb:
            flags |= 0x01
            parts.append(self.verb.to_bytes())
        
        # Optional target
        if self.target:
            flags |= 0x02
            parts.append(self.target.to_bytes())
        
        # Optional reply URL
        if self.reply_to:
            flags |= 0x04
            reply_bytes = self.reply_to.encode('utf-8')[:255]
            parts.append(struct.pack('B', len(reply_bytes)))
            parts.append(reply_bytes)
        
        # Update flags in header
        parts[1] = struct.pack('>BBH', VERSION, flags, 0)
        
        return b''.join(parts)
    
    @classmethod
    def unpack(cls, data: bytes) -> "Envelope":
        """Unpack from wire format."""
        if len(data) < HEADER_SIZE + DIM // 8:
            raise ValueError(f"Envelope too small: {len(data)} bytes")
        
        if data[:4] != MAGIC:
            raise ValueError("Invalid magic bytes")
        
        version, flags, _ = struct.unpack('>BBH', data[4:8])
        if version != VERSION:
            raise ValueError(f"Unsupported version: {version}")
        
        pos = HEADER_SIZE
        vec_size = DIM // 8
        
        # Main vector
        vector = mRNA.from_bytes(data[pos:pos + vec_size])
        pos += vec_size
        
        verb = None
        target = None
        reply_to = None
        
        # Optional verb
        if flags & 0x01:
            verb = mRNA.from_bytes(data[pos:pos + vec_size])
            pos += vec_size
        
        # Optional target
        if flags & 0x02:
            target = mRNA.from_bytes(data[pos:pos + vec_size])
            pos += vec_size
        
        # Optional reply URL
        if flags & 0x04:
            url_len = data[pos]
            pos += 1
            reply_to = data[pos:pos + url_len].decode('utf-8')
        
        return cls(vector=vector, verb=verb, target=target, reply_to=reply_to)


class Sender:
    """Send mRNA to services."""
    
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
    
    async def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def send(
        self,
        url: str,
        vector: mRNA,
        verb: Optional[mRNA] = None,
        reply_to: Optional[str] = None,
    ) -> Optional[mRNA]:
        """Send mRNA, optionally receive mRNA response."""
        envelope = Envelope(vector=vector, verb=verb, reply_to=reply_to)
        
        client = await self.client()
        resp = await client.post(
            url,
            content=envelope.pack(),
            headers={"Content-Type": CONTENT_TYPE}
        )
        
        if resp.status_code == 204:
            return None  # Fire and forget
        
        resp.raise_for_status()
        
        # Response should also be mRNA
        if resp.headers.get("Content-Type") == CONTENT_TYPE:
            return Envelope.unpack(resp.content).vector
        
        return None
    
    async def broadcast(
        self,
        urls: list[str],
        vector: mRNA,
        verb: Optional[mRNA] = None,
    ) -> list[Optional[mRNA]]:
        """Send to multiple services concurrently."""
        tasks = [self.send(url, vector, verb) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


class Receiver:
    """Receive and route mRNA messages."""
    
    def __init__(self):
        self.handlers: dict[str, Callable[[mRNA, Optional[mRNA]], Awaitable[Optional[mRNA]]]] = {}
        self.default_handler: Optional[Callable] = None
    
    def on(self, verb_name: str):
        """Decorator to register handler for verb."""
        def decorator(fn):
            self.handlers[verb_name] = fn
            return fn
        return decorator
    
    def default(self, fn):
        """Decorator for default handler."""
        self.default_handler = fn
        return fn
    
    async def handle(self, data: bytes) -> Optional[bytes]:
        """Process incoming mRNA envelope."""
        envelope = Envelope.unpack(data)
        
        # Determine verb by similarity
        verb_name = self._match_verb(envelope.verb) if envelope.verb else "default"
        
        handler = self.handlers.get(verb_name, self.default_handler)
        if not handler:
            return None
        
        result = await handler(envelope.vector, envelope.verb)
        
        if result:
            return Envelope(vector=result).pack()
        return None
    
    def _match_verb(self, verb: mRNA, threshold: float = 0.7) -> str:
        """Find best matching verb from codebook."""
        verbs = ["execute", "query", "store", "route", "feel", "think", "remember"]
        
        best_match = "default"
        best_score = threshold
        
        for v in verbs:
            codebook_verb = CB.get(f"verb::{v}")
            score = verb @ codebook_verb
            if score > best_score:
                best_score = score
                best_match = v
        
        return best_match


# === Service endpoints ===

SERVICES = {
    "flow": "https://flow.msgraph.de/mrna",
    "bighorn": "https://agi.msgraph.de/mrna",
    "hive": "https://mcp.msgraph.de/mrna",
    "mcp": "https://mcp.exo.red/mrna",
}


async def route_to_service(vector: mRNA, service: str, verb: Optional[mRNA] = None) -> Optional[mRNA]:
    """Route mRNA to named service."""
    url = SERVICES.get(service)
    if not url:
        raise ValueError(f"Unknown service: {service}")
    
    sender = Sender()
    try:
        return await sender.send(url, vector, verb)
    finally:
        await sender.close()


async def broadcast_all(vector: mRNA, verb: Optional[mRNA] = None) -> dict[str, Optional[mRNA]]:
    """Broadcast to all services."""
    sender = Sender()
    try:
        results = await sender.broadcast(list(SERVICES.values()), vector, verb)
        return dict(zip(SERVICES.keys(), results))
    finally:
        await sender.close()
