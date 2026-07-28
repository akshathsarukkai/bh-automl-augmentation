"""Repository-global, crash-safe registry for outer-test evaluation units."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from bh_augmentation.utils.corrected_runs import stable_hash

EVALUATION_REGISTRY_SCHEMA_VERSION = "bh-outer-evaluation-registry-v1"

#: Canonical registry location, expressed relative to the repository root.
REPOSITORY_RELATIVE_REGISTRY_DIRECTORY = Path(
    "results/autonomous_execution/evaluation_registry"
)
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_STATUSES = {
    "reserved",
    "prediction_complete",
    "metrics_complete",
    "failed_or_uncertain",
}
_RECORD_FIELDS = {
    "schema_version",
    "registry_key",
    "identity",
    "status",
    "reserved_at",
    "updated_at",
    "prediction_hash",
    "metrics_hash",
    "metrics_payload",
    "failure_reason",
    "record_payload_hash",
}


class EvaluationRegistryError(ValueError):
    """Base class for outer-evaluation registry failures."""


class EvaluationAlreadyClaimedError(EvaluationRegistryError):
    """Raised when an evaluation identity already has a durable record."""


class EvaluationTransitionError(EvaluationRegistryError):
    """Raised when a registry state transition is invalid."""


class EvaluationRegistryCorruptionError(EvaluationRegistryError):
    """Raised when a persisted registry record fails validation."""


def repository_root() -> Path:
    """Return the checkout root that anchors the repository-global registry.

    The registry is only repository-global if its location does not depend on
    the working directory a runner happens to be launched from.  The root is
    resolved from the installed package location by walking up to the checkout
    that contains ``pyproject.toml`` and ``src/bh_augmentation``.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "bh_augmentation"
        ).is_dir():
            return candidate
    return Path.cwd().resolve()


def default_evaluation_registry_root() -> Path:
    """Return the absolute, working-directory-independent registry root.

    There is deliberately no environment-variable override: an override would
    itself be a way to bypass an existing outer-test claim.
    """
    return repository_root() / REPOSITORY_RELATIVE_REGISTRY_DIRECTORY


@dataclass(frozen=True, slots=True)
class EvaluationIdentity:
    """Artifact-location-independent identity of one outer evaluation unit."""

    dataset_hash: str
    split_or_search_manifest_hash: str
    frozen_policy_hash: str
    evaluation_unit: str

    def __post_init__(self) -> None:
        _require_hash(self.dataset_hash, "dataset_hash")
        _require_hash(
            self.split_or_search_manifest_hash,
            "split_or_search_manifest_hash",
        )
        _require_hash(self.frozen_policy_hash, "frozen_policy_hash")
        _require_text(self.evaluation_unit, "evaluation_unit")

    @property
    def registry_key(self) -> str:
        """Return the stable repository-global key for this identity."""
        return stable_hash(
            {
                "schema_version": EVALUATION_REGISTRY_SCHEMA_VERSION,
                "identity": self.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize the exact identity schema."""
        return {
            "dataset_hash": self.dataset_hash,
            "split_or_search_manifest_hash": self.split_or_search_manifest_hash,
            "frozen_policy_hash": self.frozen_policy_hash,
            "evaluation_unit": self.evaluation_unit,
        }

    @classmethod
    def from_dict(cls, value: Any) -> EvaluationIdentity:
        """Parse a strict evaluation identity."""
        fields = {
            "dataset_hash",
            "split_or_search_manifest_hash",
            "frozen_policy_hash",
            "evaluation_unit",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise EvaluationRegistryCorruptionError(
                "Evaluation registry identity schema mismatch."
            )
        try:
            return cls(**value)
        except (TypeError, ValueError) as exc:
            raise EvaluationRegistryCorruptionError(
                "Evaluation registry identity is invalid."
            ) from exc


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One validated durable state record for an evaluation identity."""

    identity: EvaluationIdentity
    status: str
    reserved_at: str
    updated_at: str
    prediction_hash: str | None = None
    metrics_hash: str | None = None
    metrics_payload: list[dict[str, Any]] | None = None
    failure_reason: str | None = None

    @property
    def registry_key(self) -> str:
        """Return this record's stable identity key."""
        return self.identity.registry_key

    @property
    def is_complete(self) -> bool:
        """Return whether predictions and metrics completed durably."""
        return self.status == "metrics_complete"

    def to_document(self) -> dict[str, Any]:
        """Serialize and hash the complete persisted record."""
        payload = self._payload()
        return {**payload, "record_payload_hash": stable_hash(payload)}

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_REGISTRY_SCHEMA_VERSION,
            "registry_key": self.registry_key,
            "identity": self.identity.to_dict(),
            "status": self.status,
            "reserved_at": self.reserved_at,
            "updated_at": self.updated_at,
            "prediction_hash": self.prediction_hash,
            "metrics_hash": self.metrics_hash,
            "metrics_payload": self.metrics_payload,
            "failure_reason": self.failure_reason,
        }


class EvaluationRegistry:
    """Manage durable outer-evaluation claims under a configurable root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, identity: EvaluationIdentity) -> Path:
        """Return the identity-addressed record path."""
        key = identity.registry_key
        return self.root / key[:2] / f"{key}.json"

    def inspect(
        self,
        identity: EvaluationIdentity,
        *,
        stale_after: timedelta | None = None,
        now: datetime | None = None,
    ) -> EvaluationRecord | None:
        """Inspect one unit and permanently classify stale work as uncertain."""
        path = self.path_for(identity)
        if not path.exists():
            return None
        record = self._load(path, expected_identity=identity)
        if stale_after is None or record.status not in {
            "reserved",
            "prediction_complete",
        }:
            return record
        threshold = _require_stale_after(stale_after)
        current = _normalize_now(now)
        with _record_lock(path):
            record = self._load(path, expected_identity=identity)
            if record.status not in {"reserved", "prediction_complete"}:
                return record
            updated_at = _parse_timestamp(record.updated_at)
            if current - updated_at < threshold:
                return record
            stale = replace(
                record,
                status="failed_or_uncertain",
                updated_at=_timestamp(current),
                failure_reason=(
                    f"Evaluation remained {record.status!r} longer than "
                    f"{threshold.total_seconds():g} seconds; outer-test access is "
                    "uncertain and this identity must never be reevaluated."
                ),
            )
            self._write_replacement(path, stale)
            return self._load(path, expected_identity=identity)

    def reserve(
        self,
        identity: EvaluationIdentity,
        *,
        now: datetime | None = None,
    ) -> EvaluationRecord:
        """Exclusively reserve one unit immediately before test materialization."""
        current = _normalize_now(now)
        timestamp = _timestamp(current)
        record = EvaluationRecord(
            identity=identity,
            status="reserved",
            reserved_at=timestamp,
            updated_at=timestamp,
        )
        _validate_record(record)
        path = self.path_for(identity)
        try:
            self._publish_exclusive(path, record)
        except FileExistsError as exc:
            existing = self._load(path, expected_identity=identity)
            raise EvaluationAlreadyClaimedError(
                "Outer evaluation identity already has a durable "
                f"{existing.status!r} record: {identity.evaluation_unit}."
            ) from exc
        return self._load(path, expected_identity=identity)

    def mark_prediction_complete(
        self,
        identity: EvaluationIdentity,
        *,
        prediction_hash: str,
        now: datetime | None = None,
    ) -> EvaluationRecord:
        """Atomically mark the single outer prediction batch complete."""
        _require_hash(prediction_hash, "prediction_hash")
        return self._transition(
            identity,
            expected_status="reserved",
            status="prediction_complete",
            now=now,
            prediction_hash=prediction_hash,
        )

    def mark_metrics_complete(
        self,
        identity: EvaluationIdentity,
        *,
        metrics_payload: list[dict[str, Any]],
        metrics_hash: str | None = None,
        now: datetime | None = None,
    ) -> EvaluationRecord:
        """Atomically store completed outer metrics and their canonical hash."""
        normalized_payload = _normalize_metrics_payload(metrics_payload)
        computed_hash = stable_hash(normalized_payload)
        if metrics_hash is not None:
            _require_hash(metrics_hash, "metrics_hash")
            if metrics_hash != computed_hash:
                raise ValueError(
                    "metrics_hash does not match the canonical metrics_payload."
                )
        return self._transition(
            identity,
            expected_status="prediction_complete",
            status="metrics_complete",
            now=now,
            metrics_hash=computed_hash,
            metrics_payload=normalized_payload,
        )

    def mark_failed_or_uncertain(
        self,
        identity: EvaluationIdentity,
        *,
        reason: str,
        now: datetime | None = None,
    ) -> EvaluationRecord:
        """Permanently fail a nonterminal identity when test access is uncertain."""
        _require_text(reason, "reason")
        path = self.path_for(identity)
        with _record_lock(path):
            record = self._load(path, expected_identity=identity)
            if record.status not in {"reserved", "prediction_complete"}:
                raise EvaluationTransitionError(
                    "Only an in-progress evaluation can become failed_or_uncertain; "
                    f"observed={record.status!r}."
                )
            failed = replace(
                record,
                status="failed_or_uncertain",
                updated_at=_timestamp(_normalize_now(now)),
                failure_reason=reason,
            )
            self._write_replacement(path, failed)
            return self._load(path, expected_identity=identity)

    def records(
        self,
        *,
        stale_after: timedelta | None = None,
        now: datetime | None = None,
    ) -> tuple[EvaluationRecord, ...]:
        """Discover all validated records, optionally classifying stale work."""
        if not self.root.exists():
            return ()
        records: list[EvaluationRecord] = []
        for path in sorted(self.root.glob("*/*.json")):
            record = self._load(path)
            if path != self.path_for(record.identity):
                raise EvaluationRegistryCorruptionError(
                    f"Evaluation registry record is stored at an invalid path: {path}."
                )
            inspected = self.inspect(
                record.identity,
                stale_after=stale_after,
                now=now,
            )
            if inspected is None:
                raise EvaluationRegistryCorruptionError(
                    "Evaluation registry record disappeared during discovery."
                )
            records.append(inspected)
        return tuple(sorted(records, key=lambda item: item.registry_key))

    def completed_records(self) -> tuple[EvaluationRecord, ...]:
        """Discover completed units that final-evaluation resume must skip."""
        return tuple(record for record in self.records() if record.is_complete)

    def _transition(
        self,
        identity: EvaluationIdentity,
        *,
        expected_status: str,
        status: str,
        now: datetime | None,
        **updates: Any,
    ) -> EvaluationRecord:
        path = self.path_for(identity)
        with _record_lock(path):
            record = self._load(path, expected_identity=identity)
            if record.status != expected_status:
                raise EvaluationTransitionError(
                    f"Evaluation transition requires status {expected_status!r}; "
                    f"observed={record.status!r}."
                )
            updated = replace(
                record,
                status=status,
                updated_at=_timestamp(_normalize_now(now)),
                **updates,
            )
            self._write_replacement(path, updated)
            return self._load(path, expected_identity=identity)

    def _load(
        self,
        path: Path,
        *,
        expected_identity: EvaluationIdentity | None = None,
    ) -> EvaluationRecord:
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationRegistryCorruptionError(
                f"Cannot read evaluation registry record: {path}."
            ) from exc
        if not isinstance(document, dict) or set(document) != _RECORD_FIELDS:
            raise EvaluationRegistryCorruptionError(
                f"Evaluation registry record schema mismatch: {path}."
            )
        recorded_hash = document.pop("record_payload_hash")
        if recorded_hash != stable_hash(document):
            raise EvaluationRegistryCorruptionError(
                f"Evaluation registry record payload hash mismatch: {path}."
            )
        if document["schema_version"] != EVALUATION_REGISTRY_SCHEMA_VERSION:
            raise EvaluationRegistryCorruptionError(
                f"Evaluation registry schema version mismatch: {path}."
            )
        identity = EvaluationIdentity.from_dict(document["identity"])
        if expected_identity is not None and identity != expected_identity:
            raise EvaluationRegistryCorruptionError(
                "Evaluation registry identity does not match the requested unit."
            )
        if document["registry_key"] != identity.registry_key:
            raise EvaluationRegistryCorruptionError(
                f"Evaluation registry key mismatch: {path}."
            )
        record = EvaluationRecord(
            identity=identity,
            status=document["status"],
            reserved_at=document["reserved_at"],
            updated_at=document["updated_at"],
            prediction_hash=document["prediction_hash"],
            metrics_hash=document["metrics_hash"],
            metrics_payload=document["metrics_payload"],
            failure_reason=document["failure_reason"],
        )
        try:
            _validate_record(record)
        except ValueError as exc:
            raise EvaluationRegistryCorruptionError(
                f"Evaluation registry record state is invalid: {path}."
            ) from exc
        return record

    @staticmethod
    def _publish_exclusive(path: Path, record: EvaluationRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp"
        try:
            _write_new_file(temporary, record.to_document())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_replacement(path: Path, record: EvaluationRecord) -> None:
        _validate_record(record)
        temporary = path.parent / f".{path.stem}.{uuid.uuid4().hex}.tmp"
        try:
            _write_new_file(temporary, record.to_document())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _validate_record(record: EvaluationRecord) -> None:
    if record.status not in _STATUSES:
        raise ValueError(f"Unsupported evaluation registry status: {record.status!r}.")
    reserved_at = _parse_timestamp(record.reserved_at)
    updated_at = _parse_timestamp(record.updated_at)
    if updated_at < reserved_at:
        raise ValueError("Evaluation registry updated_at precedes reserved_at.")
    if record.prediction_hash is not None:
        _require_hash(record.prediction_hash, "prediction_hash")
    if record.metrics_hash is not None:
        _require_hash(record.metrics_hash, "metrics_hash")
    if record.metrics_payload is not None:
        normalized_payload = _normalize_metrics_payload(record.metrics_payload)
        if normalized_payload != record.metrics_payload:
            raise ValueError("Evaluation registry metrics payload is not canonical.")
    if record.status == "reserved" and any(
        value is not None
        for value in (
            record.prediction_hash,
            record.metrics_hash,
            record.metrics_payload,
            record.failure_reason,
        )
    ):
        raise ValueError("Reserved evaluation record contains completion fields.")
    if record.status == "prediction_complete" and (
        record.prediction_hash is None
        or record.metrics_hash is not None
        or record.metrics_payload is not None
        or record.failure_reason is not None
    ):
        raise ValueError("Prediction-complete evaluation record is inconsistent.")
    if record.status == "metrics_complete" and (
        record.prediction_hash is None
        or record.metrics_hash is None
        or record.metrics_payload is None
        or record.failure_reason is not None
    ):
        raise ValueError("Metrics-complete evaluation record is inconsistent.")
    if record.status == "metrics_complete" and record.metrics_hash != stable_hash(
        record.metrics_payload
    ):
        raise ValueError("Metrics-complete payload does not match metrics_hash.")
    if record.status == "failed_or_uncertain" and (
        not isinstance(record.failure_reason, str)
        or not record.failure_reason.strip()
    ):
        raise ValueError("Failed evaluation record requires a reason.")
    if record.status == "failed_or_uncertain" and (
        record.metrics_hash is not None or record.metrics_payload is not None
    ):
        raise ValueError("Failed evaluation record cannot contain completed metrics.")


def _write_new_file(path: Path, document: dict[str, Any]) -> None:
    with path.open("x") as handle:
        handle.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _record_lock(path: Path) -> Iterator[None]:
    """Serialize state transitions for one identity across local processes."""
    lock_path = path.with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest.")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _normalize_now(value: datetime | None) -> datetime:
    result = datetime.now(timezone.utc) if value is None else value
    if not isinstance(result, datetime) or result.tzinfo is None:
        raise ValueError("Registry timestamps require timezone-aware datetime values.")
    return result.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("Registry timestamp must be a UTC ISO-8601 string ending in Z.")
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("Registry timestamp is not valid ISO-8601.") from exc
    if not math.isfinite(result.timestamp()):
        raise ValueError("Registry timestamp is not finite.")
    return result


def _require_stale_after(value: timedelta) -> timedelta:
    if not isinstance(value, timedelta):
        raise ValueError("stale_after must be a datetime.timedelta.")
    seconds = value.total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("stale_after must be positive and finite.")
    return value


def _normalize_metrics_payload(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("metrics_payload must be a non-empty list of mapping rows.")
    if any(not isinstance(row, dict) or not row for row in value):
        raise ValueError(
            "metrics_payload must contain only non-empty mapping rows."
        )
    if any(
        not isinstance(key, str) or not key.strip()
        for row in value
        for key in row
    ):
        raise ValueError("metrics_payload row keys must be non-empty strings.")
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "metrics_payload must be finite and JSON serializable."
        ) from exc


def hash_metrics_payload(metrics_payload: list[dict[str, Any]]) -> str:
    """Return the canonical hash stored for a strict metrics payload."""
    return stable_hash(_normalize_metrics_payload(metrics_payload))
