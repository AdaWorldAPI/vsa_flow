"""mRNA API - binary vectors over HTTP.

POST /mrna — receive mRNA, route by verb
POST /execute — execute workflow, return mRNA
GET /query — similarity search, return mRNA list
"""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Optional
from uuid import UUID, uuid4
import os

from ..core.mrna import mRNA, bind, bundle, CB
from ..core.execution import Execution, probe_workflow, probe_status, probe_composite
from ..core.store import Store
from ..core.encode import encode_uuid
from ..transport.wire import Envelope, CONTENT_TYPE, Receiver


# Global store
store: Optional[Store] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown."""
    global store
    store = Store(
        url=os.getenv("UPSTASH_VECTOR_URL"),
        token=os.getenv("UPSTASH_VECTOR_TOKEN"),
        namespace="flow"
    )
    yield
    await store.close()


app = FastAPI(
    title="vsa_flow",
    description="10KD mRNA workflow execution. No JSON.",
    version="1.0.0",
    lifespan=lifespan
)


# === mRNA receiver ===

receiver = Receiver()


@receiver.on("execute")
async def handle_execute(vector: mRNA, verb: mRNA) -> Optional[mRNA]:
    """Execute workflow from mRNA."""
    # Extract workflow ID from vector by probing
    # (In practice, would decode or use metadata)
    exec = Execution(
        workflow_id=uuid4(),  # Would extract from vector
        status="running"
    )
    
    # Simulate execution
    exec.complete({"result": "ok"})
    
    # Store result
    await store.place(exec.vector, str(exec.execution_id), {
        "workflow_id": str(exec.workflow_id),
        "status": exec.status
    })
    
    return exec.vector


@receiver.on("query")
async def handle_query(vector: mRNA, verb: mRNA) -> Optional[mRNA]:
    """Query by similarity."""
    results = await store.nearby(vector, k=5)
    
    if not results:
        return None
    
    # Return best match
    best = results[0]
    return best.vector


@receiver.default
async def handle_default(vector: mRNA, verb: mRNA) -> Optional[mRNA]:
    """Default: store the vector."""
    id = str(uuid4())
    await store.place(vector, id)
    return vector


@app.post("/mrna")
async def mrna_endpoint(request: Request) -> Response:
    """Native mRNA endpoint.
    
    Accepts: application/x-mrna-10k (binary)
    Returns: application/x-mrna-10k (binary) or 204
    """
    content_type = request.headers.get("content-type", "")
    
    if content_type != CONTENT_TYPE:
        raise HTTPException(400, f"Expected {CONTENT_TYPE}")
    
    body = await request.body()
    result = await receiver.handle(body)
    
    if result:
        return Response(content=result, media_type=CONTENT_TYPE)
    
    return Response(status_code=204)


# === JSON fallback endpoints (for debugging/compatibility) ===

@app.post("/execute")
async def execute_workflow(
    workflow_id: str,
    input_data: dict = {}
) -> dict:
    """Execute workflow, return execution ID + mRNA bytes."""
    wf_uuid = UUID(workflow_id)
    
    exec = Execution(
        workflow_id=wf_uuid,
        input_data=input_data
    ).start()
    
    # Simulate execution
    exec.complete({"result": "executed"})
    
    # Store as mRNA
    await store.place(exec.vector, str(exec.execution_id), {
        "workflow_id": workflow_id,
        "status": exec.status
    })
    
    return {
        "execution_id": str(exec.execution_id),
        "status": exec.status,
        "vector_bytes": len(exec.to_bytes()),
        "vector_hex": exec.to_bytes()[:32].hex() + "..."
    }


@app.get("/executions")
async def list_executions(
    workflow_id: Optional[str] = None,
    status: Optional[str] = None,
    k: int = 20
) -> dict:
    """Query executions by similarity."""
    # Build probe
    probe = probe_composite(
        workflow_id=UUID(workflow_id) if workflow_id else None,
        status=status
    )
    
    results = await store.nearby(probe, k=k)
    
    return {
        "count": len(results),
        "results": [
            {
                "id": hit.id,
                "score": round(hit.score, 4),
                "meta": hit.meta
            }
            for hit in results
        ]
    }


@app.get("/execution/{execution_id}")
async def get_execution(execution_id: str) -> dict:
    """Get single execution."""
    vec = await store.get(execution_id)
    
    if not vec:
        raise HTTPException(404, "Not found")
    
    # Probe for status
    success_score = vec @ probe_status("success")
    error_score = vec @ probe_status("error")
    running_score = vec @ probe_status("running")
    
    likely_status = "unknown"
    if success_score > 0.5:
        likely_status = "success"
    elif error_score > 0.5:
        likely_status = "error"
    elif running_score > 0.5:
        likely_status = "running"
    
    return {
        "id": execution_id,
        "likely_status": likely_status,
        "status_scores": {
            "success": round(success_score, 4),
            "error": round(error_score, 4),
            "running": round(running_score, 4),
        },
        "vector_bytes": DIM // 8
    }


@app.get("/health")
async def health() -> dict:
    """Health check."""
    return {
        "status": "ok",
        "store": store.stats() if store else None,
        "dimension": 10000,
        "wire_format": "mRNA-10K",
        "content_type": CONTENT_TYPE
    }


@app.get("/")
async def root() -> dict:
    """Info."""
    return {
        "service": "vsa_flow",
        "version": "1.0.0",
        "description": "10KD mRNA workflow execution",
        "endpoints": {
            "/mrna": "Binary mRNA endpoint (POST)",
            "/execute": "Execute workflow (POST, JSON)",
            "/executions": "Query executions (GET)",
            "/health": "Health check"
        }
    }


# Import DIM for health endpoint
from ..core.mrna import DIM
