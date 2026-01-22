"""Execution as mRNA.

No ExecutionResult class. No model_dump(). No JSON.
The execution IS the vector.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from .mrna import mRNA, bind, bundle, CB
from .encode import encode, encode_uuid, encode_timestamp, encode_dict


class Execution:
    """Execution state as 10KD mRNA.
    
    Not a data class. Not Pydantic. The vector IS the state.
    To "serialize": .to_bytes() → 1250 bytes
    To "query": similarity against probes
    """
    
    def __init__(
        self,
        workflow_id: UUID,
        execution_id: Optional[UUID] = None,
        status: str = "pending",
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
        input_data: Optional[dict] = None,
        output_data: Optional[dict] = None,
        error: Optional[str] = None,
        node_count: int = 0,
    ):
        self.execution_id = execution_id or uuid4()
        self.workflow_id = workflow_id
        self.status = status
        self.started_at = started_at or datetime.now(timezone.utc)
        self.finished_at = finished_at
        self.input_data = input_data or {}
        self.output_data = output_data
        self.error = error
        self.node_count = node_count
        
        self._vector: Optional[mRNA] = None
    
    @property
    def vector(self) -> mRNA:
        """The execution as 10KD mRNA. Computed once, cached."""
        if self._vector is None:
            self._vector = self._encode()
        return self._vector
    
    def _encode(self) -> mRNA:
        """Encode full execution state to 10KD."""
        components = []
        
        # Execution ID
        components.append(bind(
            CB.EXECUTION_ID(),
            encode_uuid(self.execution_id)
        ))
        
        # Workflow ID
        components.append(bind(
            CB.WORKFLOW_ID(),
            encode_uuid(self.workflow_id)
        ))
        
        # Status
        components.append(bind(
            CB.STATUS(),
            CB.get(f"status::{self.status}")
        ))
        
        # Started timestamp
        components.append(bind(
            CB.get("field::started_at"),
            encode_timestamp(self.started_at)
        ))
        
        # Finished timestamp (if complete)
        if self.finished_at:
            components.append(bind(
                CB.get("field::finished_at"),
                encode_timestamp(self.finished_at)
            ))
        
        # Input data signature
        if self.input_data:
            components.append(bind(
                CB.INPUT(),
                encode_dict(self.input_data, "input")
            ))
        
        # Output data signature
        if self.output_data:
            components.append(bind(
                CB.OUTPUT(),
                encode_dict(self.output_data, "output")
            ))
        
        # Error signature
        if self.error:
            components.append(bind(
                CB.ERROR(),
                mRNA.seed(f"error::{self.error[:100]}")
            ))
        
        # Node count
        if self.node_count > 0:
            components.append(bind(
                CB.NODE(),
                mRNA.seed(f"count::{self.node_count}")
            ))
        
        # Service marker (this came from flow)
        components.append(CB.FLOW())
        
        return bundle(*components)
    
    def to_bytes(self) -> bytes:
        """1250 bytes. The entire execution state."""
        return self.vector.to_bytes()
    
    def invalidate(self):
        """Clear cached vector (call after mutation)."""
        self._vector = None
    
    # === State transitions ===
    
    def start(self) -> "Execution":
        """Mark as running."""
        self.status = "running"
        self.started_at = datetime.now(timezone.utc)
        self.invalidate()
        return self
    
    def complete(self, output: Optional[dict] = None) -> "Execution":
        """Mark as successful."""
        self.status = "success"
        self.finished_at = datetime.now(timezone.utc)
        if output:
            self.output_data = output
        self.invalidate()
        return self
    
    def fail(self, error: str) -> "Execution":
        """Mark as failed."""
        self.status = "error"
        self.finished_at = datetime.now(timezone.utc)
        self.error = error
        self.invalidate()
        return self
    
    def add_node(self) -> "Execution":
        """Increment node count."""
        self.node_count += 1
        self.invalidate()
        return self
    
    # === Similarity queries ===
    
    def matches_workflow(self, workflow_id: UUID) -> float:
        """How similar is this to the workflow probe?"""
        probe = bind(CB.WORKFLOW_ID(), encode_uuid(workflow_id))
        return self.vector @ probe
    
    def matches_status(self, status: str) -> float:
        """How similar is this to the status probe?"""
        probe = bind(CB.STATUS(), CB.get(f"status::{status}"))
        return self.vector @ probe
    
    def is_success(self) -> bool:
        """Quick check: is this a success?"""
        return self.matches_status("success") > 0.5
    
    def is_error(self) -> bool:
        """Quick check: is this an error?"""
        return self.matches_status("error") > 0.5


# === Probes for querying ===

def probe_workflow(workflow_id: UUID) -> mRNA:
    """Find executions for a workflow."""
    return bind(CB.WORKFLOW_ID(), encode_uuid(workflow_id))


def probe_status(status: str) -> mRNA:
    """Find executions with status."""
    return bind(CB.STATUS(), CB.get(f"status::{status}"))


def probe_success() -> mRNA:
    """Find successful executions."""
    return probe_status("success")


def probe_error() -> mRNA:
    """Find failed executions."""
    return probe_status("error")


def probe_running() -> mRNA:
    """Find running executions."""
    return probe_status("running")


def probe_recent(since: datetime) -> mRNA:
    """Find recent executions (approximate via time similarity)."""
    return bind(CB.get("field::started_at"), encode_timestamp(since))


def probe_composite(workflow_id: Optional[UUID] = None, status: Optional[str] = None) -> mRNA:
    """Composite probe - AND of conditions."""
    parts = []
    
    if workflow_id:
        parts.append(probe_workflow(workflow_id))
    
    if status:
        parts.append(probe_status(status))
    
    if not parts:
        return CB.EXECUTION_ID()  # Match any execution
    
    if len(parts) == 1:
        return parts[0]
    
    return bundle(*parts)
