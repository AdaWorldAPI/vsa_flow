"""VSA 10KD - Messenger RNA for distributed cognition.

10,000 dimensions. ~1.25KB per vector.
The message IS the address IS the state.
No JSON. No schema. No serialization.
"""

import numpy as np
from typing import Optional, Union
from hashlib import sha256
import struct

DIM = 10000  # 10K dimensions - the full cognitive bandwidth


class mRNA:
    """10,000D bipolar hypervector. The messenger.
    
    Self-describing. Self-routing. Binary native.
    1.25KB carries more semantic density than any JSON.
    """
    
    __slots__ = ('_data', 'tag')
    
    def __init__(self, data: np.ndarray, tag: Optional[str] = None):
        if data.shape != (DIM,):
            raise ValueError(f"Expected {DIM}D, got {data.shape}")
        self._data = data.astype(np.int8)
        self.tag = tag
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @classmethod
    def random(cls, tag: Optional[str] = None) -> "mRNA":
        """Random bipolar vector - maximally expressive."""
        return cls(
            data=np.random.choice([-1, 1], size=DIM).astype(np.int8),
            tag=tag
        )
    
    @classmethod
    def seed(cls, s: str) -> "mRNA":
        """Deterministic vector from string. Same seed = same vector always."""
        # Expand seed to 10K bits using chained hashing
        buf = bytearray(DIM // 8 + 32)
        h = sha256(s.encode()).digest()
        
        pos = 0
        counter = 0
        while pos < len(buf):
            chunk = sha256(h + struct.pack('>I', counter)).digest()
            buf[pos:pos+32] = chunk
            pos += 32
            counter += 1
        
        bits = np.unpackbits(np.frombuffer(bytes(buf[:DIM // 8]), dtype=np.uint8))[:DIM]
        return cls(data=(bits * 2 - 1).astype(np.int8), tag=s)
    
    @classmethod
    def zero(cls) -> "mRNA":
        """Zero vector for accumulation."""
        return cls(data=np.zeros(DIM, dtype=np.int8), tag="∅")
    
    # === Binary transport - the wire format ===
    
    def to_bytes(self) -> bytes:
        """Pack to 1250 bytes. This IS the message."""
        # Pack bipolar {-1,+1} as bits {0,1}
        bits = ((self._data + 1) // 2).astype(np.uint8)
        return np.packbits(bits).tobytes()
    
    @classmethod
    def from_bytes(cls, b: bytes, tag: Optional[str] = None) -> "mRNA":
        """Unpack from 1250 bytes."""
        if len(b) != DIM // 8:
            raise ValueError(f"Expected {DIM // 8} bytes, got {len(b)}")
        bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))[:DIM]
        return cls(data=(bits * 2 - 1).astype(np.int8), tag=tag)
    
    def __bytes__(self) -> bytes:
        return self.to_bytes()
    
    # === VSA Operations ===
    
    def __mul__(self, other: "mRNA") -> "mRNA":
        """BIND: element-wise XOR multiplication.
        
        A * B creates association. (A * B) * B ≈ A
        Preserves similarity structure.
        """
        return mRNA(
            data=(self._data * other._data).astype(np.int8),
            tag=f"({self.tag}⊛{other.tag})" if self.tag and other.tag else None
        )
    
    def __add__(self, other: "mRNA") -> "mRNA":
        """Accumulate for bundling. Use bundle() for final result."""
        return mRNA(
            data=(self._data.astype(np.int32) + other._data.astype(np.int32)).astype(np.int32).view(np.int8)[:DIM],
            tag=None
        )
    
    def __matmul__(self, other: "mRNA") -> float:
        """Similarity: A @ B returns cosine in [-1, +1]."""
        return float(np.dot(self._data.astype(np.float32), other._data.astype(np.float32)) / DIM)
    
    def permute(self, shift: int) -> "mRNA":
        """Circular shift - encodes sequence/position."""
        return mRNA(data=np.roll(self._data, shift), tag=f"ρ{shift}({self.tag})" if self.tag else None)
    
    def __repr__(self) -> str:
        return f"mRNA({self.tag or 'anon'}, sim_self={self @ self:.3f})"


def bind(*vectors: mRNA) -> mRNA:
    """BIND multiple vectors. Associative, commutative."""
    if not vectors:
        return mRNA.seed("identity")
    result = vectors[0]
    for v in vectors[1:]:
        result = result * v
    return result


def bundle(*vectors: mRNA) -> mRNA:
    """BUNDLE: majority vote superposition.
    
    Result is similar to ALL inputs. The "set" operation.
    """
    if not vectors:
        return mRNA.zero()
    
    acc = np.zeros(DIM, dtype=np.int32)
    for v in vectors:
        acc += v.data.astype(np.int32)
    
    # Majority vote with random tiebreak
    result = np.sign(acc)
    ties = result == 0
    if ties.any():
        result[ties] = np.random.choice([-1, 1], size=ties.sum())
    
    tags = [v.tag for v in vectors if v.tag]
    tag = f"[{'+'.join(tags[:3])}...]" if len(tags) > 3 else f"[{'+'.join(tags)}]" if tags else None
    
    return mRNA(data=result.astype(np.int8), tag=tag)


def unbind(bound: mRNA, key: mRNA) -> mRNA:
    """Retrieve from binding. bind(A, B).unbind(B) ≈ A"""
    return bound * key  # XOR is self-inverse


# === Codebook: atomic concept vectors ===

class Codebook:
    """Fixed vectors for known concepts. The vocabulary."""
    
    _cache: dict[str, mRNA] = {}
    
    @classmethod
    def get(cls, concept: str) -> mRNA:
        if concept not in cls._cache:
            cls._cache[concept] = mRNA.seed(f"codebook::{concept}")
        return cls._cache[concept]
    
    # Field markers
    WORKFLOW_ID = classmethod(lambda cls: cls.get("WORKFLOW_ID"))
    EXECUTION_ID = classmethod(lambda cls: cls.get("EXECUTION_ID"))
    STATUS = classmethod(lambda cls: cls.get("STATUS"))
    TIMESTAMP = classmethod(lambda cls: cls.get("TIMESTAMP"))
    INPUT = classmethod(lambda cls: cls.get("INPUT"))
    OUTPUT = classmethod(lambda cls: cls.get("OUTPUT"))
    ERROR = classmethod(lambda cls: cls.get("ERROR"))
    NODE = classmethod(lambda cls: cls.get("NODE"))
    
    # Status values
    PENDING = classmethod(lambda cls: cls.get("status::pending"))
    RUNNING = classmethod(lambda cls: cls.get("status::running"))
    SUCCESS = classmethod(lambda cls: cls.get("status::success"))
    FAILED = classmethod(lambda cls: cls.get("status::failed"))
    
    # Verbs (for routing)
    EXECUTE = classmethod(lambda cls: cls.get("verb::execute"))
    QUERY = classmethod(lambda cls: cls.get("verb::query"))
    STORE = classmethod(lambda cls: cls.get("verb::store"))
    ROUTE = classmethod(lambda cls: cls.get("verb::route"))
    FEEL = classmethod(lambda cls: cls.get("verb::feel"))
    THINK = classmethod(lambda cls: cls.get("verb::think"))
    REMEMBER = classmethod(lambda cls: cls.get("verb::remember"))
    
    # Services (for routing)
    FLOW = classmethod(lambda cls: cls.get("service::flow"))
    BIGHORN = classmethod(lambda cls: cls.get("service::bighorn"))
    HIVE = classmethod(lambda cls: cls.get("service::hive"))
    MCP = classmethod(lambda cls: cls.get("service::mcp"))


CB = Codebook  # Shorthand
