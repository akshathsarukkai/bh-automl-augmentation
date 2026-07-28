"""Read-only inspection of the repository-global outer-test claim registry.

This tool can list and show durable evaluation claims. It deliberately provides
no way to delete, release, reset, or otherwise weaken a claim: an outer-test
identity that has been reserved must never be silently re-evaluated.

Examples
--------
    python -B scripts/inspect_evaluation_registry.py list
    python -B scripts/inspect_evaluation_registry.py list --status metrics_complete
    python -B scripts/inspect_evaluation_registry.py show --registry-key <key>
    python -B scripts/inspect_evaluation_registry.py list --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bh_augmentation.evaluation.evaluation_registry import (  # noqa: E402
    EvaluationRecord,
    EvaluationRegistry,
    EvaluationRegistryError,
    default_evaluation_registry_root,
)

STATUSES = (
    "reserved",
    "prediction_complete",
    "metrics_complete",
    "failed_or_uncertain",
)


def _record_summary(record: EvaluationRecord) -> dict[str, Any]:
    return {
        "registry_key": record.registry_key,
        "status": record.status,
        "evaluation_unit": record.identity.evaluation_unit,
        "dataset_hash": record.identity.dataset_hash,
        "split_or_search_manifest_hash": record.identity.split_or_search_manifest_hash,
        "frozen_policy_hash": record.identity.frozen_policy_hash,
        "reserved_at": record.reserved_at,
        "updated_at": record.updated_at,
        "prediction_hash": record.prediction_hash,
        "metrics_hash": record.metrics_hash,
        "failure_reason": record.failure_reason,
        "is_complete": record.is_complete,
    }


def _load_records(root: Path) -> tuple[EvaluationRecord, ...]:
    if not root.exists():
        return ()
    # `stale_after` is deliberately not passed: this tool never mutates state.
    return EvaluationRegistry(root).records()


def _print_table(summaries: Sequence[dict[str, Any]]) -> None:
    if not summaries:
        print("No evaluation claims found.")
        return
    header = f"{'STATUS':<20} {'KEY':<16} {'UPDATED':<22} EVALUATION UNIT"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary['status']:<20} "
            f"{summary['registry_key'][:16]:<16} "
            f"{summary['updated_at']:<22} "
            f"{summary['evaluation_unit']}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only registry inspector."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect outer-test evaluation claims. This tool is read-only and "
            "provides no way to delete or release a claim."
        )
    )
    parser.add_argument(
        "--registry-directory",
        default=None,
        help="Registry root (defaults to the repository-global registry).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List evaluation claims.")
    list_parser.add_argument("--status", choices=STATUSES, default=None)
    list_parser.add_argument("--evaluation-unit", default=None)
    list_parser.add_argument("--json", action="store_true", dest="as_json")

    show_parser = subparsers.add_parser("show", help="Show one claim in full.")
    show_parser.add_argument("--registry-key", required=True)

    args = parser.parse_args(argv)
    root = (
        Path(args.registry_directory)
        if args.registry_directory is not None
        else default_evaluation_registry_root()
    )
    try:
        records = _load_records(root)
    except EvaluationRegistryError as exc:
        print(f"Registry is corrupt and must be investigated by hand: {exc}", file=sys.stderr)
        return 2

    if args.command == "show":
        matches = [
            record
            for record in records
            if record.registry_key == args.registry_key
            or record.registry_key.startswith(args.registry_key)
        ]
        if len(matches) != 1:
            print(
                f"Expected exactly one claim matching {args.registry_key!r}; "
                f"found {len(matches)}.",
                file=sys.stderr,
            )
            return 1
        document = matches[0].to_document()
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0

    summaries = [_record_summary(record) for record in records]
    if args.status is not None:
        summaries = [item for item in summaries if item["status"] == args.status]
    if args.evaluation_unit is not None:
        summaries = [
            item for item in summaries if item["evaluation_unit"] == args.evaluation_unit
        ]
    if args.as_json:
        print(
            json.dumps(
                {
                    "registry_directory": str(root),
                    "record_count": len(summaries),
                    "records": summaries,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(f"Registry: {root}")
        _print_table(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
