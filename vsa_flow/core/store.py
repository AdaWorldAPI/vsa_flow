"""mRNA Store - vector space as database.

place() is write. nearby() is query.
No SQL. No JSON. No serialization.
"""

import asyncio
import httpx
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import os

from .mrna import mRNA, DIM


@dataclass 
class Hit:
    """Search result."""
    id: str
    score: float
    vector: Optional[mRNA] = None
    meta: Optional[dict] = None


class Store:
    """mRNA vector space.
    
    Backed by Upstash Vector for persistence.
    Local cache for speed.
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        namespace: str = "flow"
    ):
        self.url = url or os.getenv("UPSTASH_VECTOR_URL")
        self.token = token or os.getenv("UPSTASH_VECTOR_TOKEN")
        self.namespace = namespace
        
        # Local cache
        self._local: dict[str, mRNA] = {}
        self._meta: dict[str, dict] = {}
        
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def place(
        self,
        vector: mRNA,
        id: str,
        meta: Optional[dict] = None,
    ) -> None:
        """Place vector in the space.
        
        That's it. No schema. No JSON encoding.
        """
        # Local first
        self._local[id] = vector
        if meta:
            self._meta[id] = meta
        
        # Persist to Upstash
        if self.url and self.token:
            await self._upsert(vector, id, meta)
    
    async def _upsert(self, vector: mRNA, id: str, meta: Optional[dict]) -> None:
        """Persist to Upstash Vector."""
        client = await self._http()
        
        # Convert bipolar to float (Upstash expects float32)
        vec_list = vector.data.astype(float).tolist()
        
        payload = [{
            "id": f"{self.namespace}:{id}",
            "vector": vec_list,
            "metadata": meta or {}
        }]
        
        try:
            resp = await client.post(
                f"{self.url}/upsert",
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload
            )
            resp.raise_for_status()
        except Exception as e:
            print(f"Upstash upsert failed: {e}")
    
    async def nearby(
        self,
        probe: mRNA,
        k: int = 10,
        threshold: float = 0.1,
        include_vectors: bool = False,
    ) -> List[Hit]:
        """Find similar vectors.
        
        This is the query. Similarity IS the filter.
        """
        results = []
        
        # Local search
        for id, vec in self._local.items():
            score = float(probe @ vec)
            if score >= threshold:
                results.append(Hit(
                    id=id,
                    score=score,
                    vector=vec if include_vectors else None,
                    meta=self._meta.get(id)
                ))
        
        # Upstash search
        if self.url and self.token:
            remote = await self._query(probe, k * 2, include_vectors)
            
            # Merge, dedup
            local_ids = {r.id for r in results}
            for hit in remote:
                if hit.id not in local_ids and hit.score >= threshold:
                    results.append(hit)
        
        # Sort by score, take top k
        results.sort(key=lambda h: h.score, reverse=True)
        return results[:k]
    
    async def _query(self, probe: mRNA, k: int, include_vectors: bool) -> List[Hit]:
        """Query Upstash Vector."""
        client = await self._http()
        
        vec_list = probe.data.astype(float).tolist()
        
        try:
            resp = await client.post(
                f"{self.url}/query",
                headers={"Authorization": f"Bearer {self.token}"},
                json={
                    "vector": vec_list,
                    "topK": k,
                    "includeMetadata": True,
                    "includeVectors": include_vectors,
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for item in data.get("result", []):
                vec = None
                if include_vectors and "vector" in item:
                    arr = np.array(item["vector"])
                    arr = np.sign(arr).astype(np.int8)
                    arr[arr == 0] = 1
                    vec = mRNA(data=arr)
                
                # Strip namespace prefix
                id = item["id"]
                if id.startswith(f"{self.namespace}:"):
                    id = id[len(self.namespace) + 1:]
                
                results.append(Hit(
                    id=id,
                    score=item.get("score", 0),
                    vector=vec,
                    meta=item.get("metadata")
                ))
            
            return results
            
        except Exception as e:
            print(f"Upstash query failed: {e}")
            return []
    
    async def get(self, id: str) -> Optional[mRNA]:
        """Direct fetch by ID."""
        if id in self._local:
            return self._local[id]
        
        if self.url and self.token:
            return await self._fetch(id)
        
        return None
    
    async def _fetch(self, id: str) -> Optional[mRNA]:
        """Fetch from Upstash."""
        client = await self._http()
        
        try:
            resp = await client.post(
                f"{self.url}/fetch",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"ids": [f"{self.namespace}:{id}"], "includeVectors": True}
            )
            resp.raise_for_status()
            data = resp.json()
            
            results = data.get("result", [])
            if results and results[0]:
                arr = np.array(results[0]["vector"])
                arr = np.sign(arr).astype(np.int8)
                arr[arr == 0] = 1
                return mRNA(data=arr)
            
        except Exception:
            pass
        
        return None
    
    async def delete(self, id: str) -> None:
        """Remove from space."""
        self._local.pop(id, None)
        self._meta.pop(id, None)
        
        if self.url and self.token:
            client = await self._http()
            try:
                await client.post(
                    f"{self.url}/delete",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json={"ids": [f"{self.namespace}:{id}"]}
                )
            except Exception:
                pass
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def stats(self) -> dict:
        return {
            "dimension": DIM,
            "bytes_per_vector": DIM // 8,
            "local_count": len(self._local),
            "local_bytes": len(self._local) * (DIM // 8),
            "upstash_configured": bool(self.url),
        }
