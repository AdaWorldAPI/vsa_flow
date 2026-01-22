"""Encode anything to 10KD mRNA.

No JSON. The data becomes the vector.
"""

from datetime import datetime, timezone
from typing import Any, Union
from uuid import UUID
import struct

from .mrna import mRNA, bind, bundle, CB, DIM


def encode_uuid(uid: Union[UUID, str]) -> mRNA:
    """UUID → deterministic 10KD vector."""
    if isinstance(uid, str):
        uid = UUID(uid)
    return mRNA.seed(f"uuid::{uid}")


def encode_timestamp(dt: datetime) -> mRNA:
    """Timestamp → phase-encoded vector.
    
    Similar times produce similar vectors.
    Multi-scale: minute, hour, day, month, year.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    ts = int(dt.timestamp())
    
    # Temporal components
    minute = dt.minute
    hour = dt.hour
    day = dt.day
    month = dt.month
    year = dt.year
    day_of_week = dt.weekday()
    
    # Base vectors for each scale
    minute_base = mRNA.seed("time::minute")
    hour_base = mRNA.seed("time::hour")
    day_base = mRNA.seed("time::day")
    month_base = mRNA.seed("time::month")
    year_base = mRNA.seed(f"time::year::{year}")
    dow_base = mRNA.seed("time::dow")
    
    # Permute by position in cycle (scaled to use more of 10K dims)
    minute_vec = minute_base.permute(minute * 166)      # 0-59 → 0-9834
    hour_vec = hour_base.permute(hour * 416)            # 0-23 → 0-9584
    day_vec = day_base.permute(day * 322)               # 1-31 → 322-9982
    month_vec = month_base.permute(month * 833)         # 1-12 → 833-9996
    dow_vec = dow_base.permute(day_of_week * 1428)      # 0-6 → 0-8568
    
    # Bundle all scales - result is similar to all components
    return bundle(minute_vec, hour_vec, day_vec, month_vec, year_base, dow_vec)


def encode_string(s: str, max_len: int = 200) -> mRNA:
    """String → deterministic vector.
    
    Short strings get exact encoding.
    Long strings get truncated hash.
    """
    s = s[:max_len]
    return mRNA.seed(f"str::{s}")


def encode_number(n: Union[int, float], precision: int = 2) -> mRNA:
    """Number → vector with similarity for nearby values.
    
    Numbers within same bucket have high similarity.
    """
    if isinstance(n, float):
        # Round to precision
        n = round(n, precision)
    
    # Encode as bucket + residual
    bucket = int(n) // 10
    residual = int(n) % 10
    
    bucket_vec = mRNA.seed(f"num::bucket::{bucket}")
    residual_vec = mRNA.seed("num::residual").permute(residual * 1000)
    
    return bind(bucket_vec, residual_vec)


def encode_bool(b: bool) -> mRNA:
    """Bool → orthogonal vectors for true/false."""
    return mRNA.seed(f"bool::{b}")


def encode_none() -> mRNA:
    """None → dedicated vector."""
    return mRNA.seed("none::null")


def encode_list(items: list, max_items: int = 20) -> mRNA:
    """List → bundled positional encodings.
    
    Order is preserved via permutation.
    """
    if not items:
        return mRNA.seed("list::empty")
    
    encoded = []
    pos_base = mRNA.seed("list::position")
    
    for i, item in enumerate(items[:max_items]):
        item_vec = encode(item)
        pos_vec = pos_base.permute(i * 500)  # Position marker
        encoded.append(bind(pos_vec, item_vec))
    
    return bundle(*encoded)


def encode_dict(d: dict, prefix: str = "", max_keys: int = 30) -> mRNA:
    """Dict → bundled key-value bindings.
    
    Keys bind to values. Result is queryable.
    """
    if not d:
        return mRNA.seed(f"dict::empty::{prefix}")
    
    pairs = []
    for i, (k, v) in enumerate(list(d.items())[:max_keys]):
        key_vec = mRNA.seed(f"key::{prefix}::{k}")
        val_vec = encode(v)
        pairs.append(bind(key_vec, val_vec))
    
    return bundle(*pairs)


def encode(value: Any) -> mRNA:
    """Universal encoder. Anything → 10KD mRNA."""
    if value is None:
        return encode_none()
    elif isinstance(value, bool):
        return encode_bool(value)
    elif isinstance(value, (int, float)):
        return encode_number(value)
    elif isinstance(value, str):
        return encode_string(value)
    elif isinstance(value, UUID):
        return encode_uuid(value)
    elif isinstance(value, datetime):
        return encode_timestamp(value)
    elif isinstance(value, (list, tuple)):
        return encode_list(list(value))
    elif isinstance(value, dict):
        return encode_dict(value)
    elif isinstance(value, bytes):
        return mRNA.seed(f"bytes::{value[:50].hex()}")
    elif isinstance(value, mRNA):
        return value
    else:
        # Fallback: type + repr
        return mRNA.seed(f"obj::{type(value).__name__}::{repr(value)[:100]}")


# === Query probes ===

def probe_field(field: str, value: Any) -> mRNA:
    """Create probe to find vectors with field=value."""
    field_marker = CB.get(f"field::{field}")
    value_vec = encode(value)
    return bind(field_marker, value_vec)


def probe_status(status: str) -> mRNA:
    """Probe for execution status."""
    return bind(CB.STATUS(), CB.get(f"status::{status}"))


def probe_workflow(workflow_id: UUID) -> mRNA:
    """Probe for workflow ID."""
    return bind(CB.WORKFLOW_ID(), encode_uuid(workflow_id))


def probe_time_range(start: datetime, end: datetime) -> mRNA:
    """Probe for time range (similarity-based, approximate)."""
    # Bundle start and end - result is similar to times in between
    return bundle(
        bind(CB.TIMESTAMP(), encode_timestamp(start)),
        bind(CB.TIMESTAMP(), encode_timestamp(end))
    )


def probe_verb(verb: str) -> mRNA:
    """Probe for action verb."""
    return CB.get(f"verb::{verb}")


def probe_service(service: str) -> mRNA:
    """Probe for target service."""
    return CB.get(f"service::{service}")
