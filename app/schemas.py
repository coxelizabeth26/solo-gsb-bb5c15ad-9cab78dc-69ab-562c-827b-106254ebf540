"""Pydantic request/response models."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Registration ceiling for a dataset's epsilon cap.  Budget transfers may
# raise a target dataset's cap only up to this same limit.
MAX_EPSILON_CAP = 1_000.0

# Dataset identifiers supplied inside JSON payloads follow the same rules as
# the path parameters in main.py.
DATASET_ID_PATTERN = r"^[A-Za-z0-9._\-]+$"


def _finite(name: str):
    @field_validator(name)
    @classmethod
    def _check(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"{name} must be a finite number")
        return v

    return _check


class DatasetRegister(BaseModel):
    epsilon_cap: float = Field(
        ...,
        gt=0.0,
        le=MAX_EPSILON_CAP,
        examples=[1.0],
        description="Total epsilon budget allowed for this dataset.",
    )
    delta_cap: float = Field(
        ...,
        ge=1e-20,
        lt=1.0,
        examples=[1e-6],
        description="Total delta budget; also the target delta used when the "
        "composed RDP curve is converted back to (epsilon, delta)-DP.",
    )
    description: Optional[str] = Field(default=None, max_length=2000)

    _eps_finite = _finite("epsilon_cap")
    _delta_finite = _finite("delta_cap")


class AnalysisRequest(BaseModel):
    request_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        examples=["req-20260923-0001"],
        description="Client-generated idempotency key; unique per dataset.",
    )
    query_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=["gaussian_mean"],
        description="Free-form label of the analysis, e.g. "
        "'gaussian_mean', 'histogram', 'sql_count'.",
    )
    epsilon: float = Field(..., gt=0.0, le=1_000.0, examples=[0.1])
    delta: float = Field(
        ...,
        ge=0.0,
        lt=1.0,
        examples=[1e-7],
        description="Per-request delta; use 0 for pure epsilon-DP.",
    )

    _eps_finite = _finite("epsilon")
    _delta_finite = _finite("delta")


# Longest accepted reservation time-to-live: 90 days.  Prevents a typo from
# locking budget essentially forever; a confirmed reservation consumes the
# budget permanently regardless of the TTL.
MAX_TTL_SECONDS = 90 * 24 * 3600


class ReservationCreate(BaseModel):
    reservation_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        examples=["res-20260923-0001"],
        description="Client-generated idempotency key; unique per dataset.",
    )
    epsilon: float = Field(..., gt=0.0, le=1_000.0, examples=[0.1])
    delta: float = Field(
        ...,
        ge=0.0,
        lt=1.0,
        examples=[1e-7],
        description="Reserved per-analysis delta; use 0 for pure epsilon-DP.",
    )
    ttl_seconds: int = Field(
        ...,
        ge=1,
        le=MAX_TTL_SECONDS,
        examples=[600],
        description="Reservation lifetime in seconds.  An unconfirmed "
        "reservation releases its budget automatically once it has expired; "
        "replays return the stored TTL.",
    )

    _eps_finite = _finite("epsilon")
    _delta_finite = _finite("delta")


class ReservationConfirm(BaseModel):
    request_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        examples=["req-20260923-0001"],
        description="Idempotency key of the analysis request the reservation "
        "is turned into; unique per dataset.",
    )
    query_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=["gaussian_mean"],
        description="Free-form analysis label recorded on the analysis "
        "request.",
    )


class BudgetView(BaseModel):
    epsilon_cap: float
    delta_cap: float
    # Consumed by confirmed / directly granted analysis requests.
    epsilon_consumed: float
    # Additional epsilon projected to be locked by active (non-expired)
    # reservations, counted on the same composed RDP curve.
    epsilon_reserved: float
    # cap - (consumed RDP-projected together with active reservations).
    epsilon_remaining: float
    rdp_best_order: float
    granted_requests: int
    active_reservations: int
    # Composed RDP curve of granted requests *and* active reservations.
    rdp_curve: dict[str, float]


class DatasetInfo(BaseModel):
    dataset_id: str
    description: Optional[str]
    created_at: datetime
    budget: BudgetView


class Verdict(BaseModel):
    """Result of an analysis submission.

    A refusal is returned with HTTP 200 and ``allowed = false``: the request
    was processed correctly, it simply does not fit in the remaining budget.
    """

    dataset_id: str
    request_id: str
    query_type: str
    epsilon: float
    delta: float
    allowed: bool
    status: Literal["granted", "rejected"]
    reason: Optional[str] = None
    epsilon_before: float
    epsilon_after: float
    epsilon_remaining: float
    rdp_best_order: float
    idempotent_replay: bool = False
    created_at: datetime


class RequestRecord(BaseModel):
    dataset_id: str
    request_id: str
    query_type: str
    epsilon: float
    delta: float
    status: Literal["granted", "rejected"]
    reason: Optional[str]
    epsilon_before: float
    epsilon_after: float
    created_at: datetime


class RequestList(BaseModel):
    dataset_id: str
    count: int
    requests: list[RequestRecord]


ReservationStatus = Literal[
    "granted", "rejected", "confirmed", "released", "expired"
]


class ReservationView(BaseModel):
    """One reservation with its current lifecycle state.

    ``allowed`` is the immutable admission decision taken at creation;
    ``status`` is the live state (``granted`` reservations turn into
    ``confirmed``, ``released`` or ``expired`` over time).
    """

    dataset_id: str
    reservation_id: str
    epsilon: float
    delta: float
    ttl_seconds: int
    allowed: bool
    status: ReservationStatus
    reason: Optional[str] = None
    epsilon_before: float
    epsilon_after: float
    epsilon_remaining: float
    rdp_best_order: float
    idempotent_replay: bool = False
    created_at: datetime
    expires_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None
    confirmed_request_id: Optional[str] = None
    released_at: Optional[datetime] = None


class ReservationList(BaseModel):
    dataset_id: str
    count: int
    reservations: list[ReservationView]


class ForecastSnapshot(BaseModel):
    """Budget state for one set of still-valid reservations.

    All epsilon values use the same composed-RDP accounting as
    :class:`BudgetView`: granted requests are composed together with the
    reservations that are still locked at that instant.
    """

    epsilon_consumed: float
    epsilon_reserved: float
    epsilon_remaining: float
    rdp_best_order: float
    active_reservations: int


class ReleaseForecastEvent(ForecastSnapshot):
    """One simulated expiry instant.

    All reservations sharing the same ``expires_at`` are removed together at
    this instant; the budget fields describe the state *after* their removal.
    """

    expires_at: datetime
    released_count: int
    released_reservation_ids: list[str]


class ReleaseForecast(BaseModel):
    """Read-only simulation of TTL-driven budget releases up to a deadline."""

    dataset_id: str
    generated_at: datetime
    deadline: datetime
    # State computed from reservations that are currently valid (now), before
    # any simulated expiry.
    initial: ForecastSnapshot
    # One entry per distinct expires_at <= deadline, chronological.  Reservations
    # still valid at the deadline never appear here.
    events: list[ReleaseForecastEvent]
    # State with every expires_at <= deadline removed; reservations expiring
    # strictly after the deadline (or never) keep counting.
    at_deadline: ForecastSnapshot


# ---------------------------------------------------------- budget transfers --


class TransferCreate(BaseModel):
    """Move ``epsilon_amount`` of epsilon cap between two datasets.

    Only the ``epsilon_cap`` values change: ``delta_cap`` and every recorded
    analysis request / reservation stay exactly as they are.
    """

    transfer_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        examples=["xfer-20260923-0001"],
        description="Client-generated idempotency key; globally unique.",
    )
    source_dataset_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        pattern=DATASET_ID_PATTERN,
        examples=["sales-2026"],
        description="Dataset whose epsilon_cap is decreased.",
    )
    target_dataset_id: str = Field(
        ...,
        min_length=1,
        max_length=200,
        pattern=DATASET_ID_PATTERN,
        examples=["sales-2027"],
        description="Dataset whose epsilon_cap is increased.",
    )
    epsilon_amount: float = Field(
        ...,
        gt=0.0,
        examples=[0.25],
        description="Positive amount of epsilon cap moved from source to "
        "target.",
    )

    _amount_finite = _finite("epsilon_amount")

    @model_validator(mode="after")
    def _distinct_datasets(self) -> "TransferCreate":
        if self.source_dataset_id == self.target_dataset_id:
            raise ValueError("source and target datasets must differ")
        return self


class TransferView(BaseModel):
    """Record of an applied budget transfer.

    The ``*_committed`` fields capture each side's occupied epsilon (granted
    requests composed with non-expired reservations under the dataset's own
    delta_cap) at the instant the transfer was validated.
    """

    transfer_id: str
    source_dataset_id: str
    target_dataset_id: str
    epsilon_amount: float
    source_epsilon_cap_before: float
    source_epsilon_cap_after: float
    target_epsilon_cap_before: float
    target_epsilon_cap_after: float
    source_committed: float
    target_committed: float
    idempotent_replay: bool = False
    created_at: datetime


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
