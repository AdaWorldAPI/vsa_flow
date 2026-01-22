# vsa_flow

**10,000D mRNA workflow execution.**

No JSON. No Redis. No datetime serialization bugs.  
The vector IS the state.

## The Problem

```python
# Old way - ai_flow
json.dumps([nr.model_dump() for nr in execution.node_results])
# → "Object of type datetime is not JSON serializable"
# → 900MB memory leak
# → Crashed
```

## The Solution

```python
# New way - vsa_flow
exec.to_bytes()
# → 1250 bytes
# → Done
```

## How It Works

Execution state is encoded as a 10,000-dimensional bipolar hypervector.

```python
from vsa_flow import Execution, Store

# Create execution
exec = Execution(
    workflow_id=uuid4(),
    input_data={"key": "value"}
).start()

# Execute
exec.complete({"result": "done"})

# Store - no serialization
await store.place(exec.vector, str(exec.execution_id))

# Query - similarity search
results = await store.nearby(probe_status("success"), k=10)
```

## Wire Format

```
Content-Type: application/x-mrna-10k

[4 bytes magic: "mRNA"]
[4 bytes header: version, flags]
[1250 bytes: 10KD bipolar vector]
[optional: verb vector, target vector, reply URL]
```

Total: ~1.25KB per message. Self-describing. Self-routing.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mrna` | POST | Native binary mRNA |
| `/execute` | POST | Execute workflow (JSON fallback) |
| `/executions` | GET | Query by similarity |
| `/health` | GET | Health check |

## Why 10,000 Dimensions?

- **Capacity**: Can encode ~10^3000 distinct concepts
- **Robustness**: 30% noise tolerance
- **Composition**: Bind/bundle operations preserve similarity
- **Size**: 1.25KB - smaller than most JSON payloads

## No More

- ❌ `json.dumps()` / `json.loads()`
- ❌ `model_dump()` / `model_validate()`
- ❌ Datetime serialization
- ❌ Schema migrations
- ❌ Redis TTL management
- ❌ SQL queries

## Just

- ✅ `vector.to_bytes()`
- ✅ `store.place(vector, id)`
- ✅ `store.nearby(probe, k=10)`

## Architecture

```
┌─────────────────────────────────────────────┐
│                  vsa_flow                    │
├─────────────────────────────────────────────┤
│  mRNA (10KD)                                │
│  ├── Execution state                        │
│  ├── Workflow ID                            │
│  ├── Status                                 │
│  ├── Timestamps (phase-encoded)             │
│  ├── Input/Output (structure signature)     │
│  └── Error (if any)                         │
├─────────────────────────────────────────────┤
│  Store                                      │
│  ├── place(vector, id) → write              │
│  ├── nearby(probe, k) → similarity search   │
│  └── get(id) → direct fetch                 │
├─────────────────────────────────────────────┤
│  Transport                                  │
│  ├── Binary wire format                     │
│  ├── Verb-based routing                     │
│  └── Service mesh ready                     │
└─────────────────────────────────────────────┘
```

## Environment Variables

```bash
UPSTASH_VECTOR_URL=https://...
UPSTASH_VECTOR_TOKEN=...
PORT=8000
```

## License

MIT
