"""Smoke validation-only policy search followed by one frozen outer evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.final_evaluation import run_final_evaluation_command
from bh_augmentation.policy_search import run_policy_search_command


def run_smoke(
    output_directory: str | Path,
    *,
    config_path: str | Path = "configs/corrected_policy_search_tiny.yaml",
) -> dict[str, Any]:
    """Run both protocol stages and validate their data-access attestations."""
    root = Path(output_directory)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 5 smoke output: {root}")
    root.mkdir(parents=True)
    search_paths = run_policy_search_command(
        config_path,
        output_directory=root / "search",
    )
    final_paths = run_final_evaluation_command(
        config_path,
        frozen_policy_path=search_paths["frozen_policy"],
        search_manifest_path=search_paths["search_manifest"],
        output_directory=root / "final",
    )

    search_metrics = pd.read_csv(search_paths["search_metrics"])
    final_metrics = pd.read_csv(final_paths["final_test_metrics"])
    search_manifest = json.loads(search_paths["search_manifest"].read_text())
    final_manifest = json.loads(
        final_paths["final_evaluation_manifest"].read_text()
    )
    if not search_metrics["split"].eq("valid").all():
        raise AssertionError("Policy search emitted a non-validation metric.")
    search_payload = search_manifest["payload"]
    if (
        search_payload["outer_test_labels_accessed"] is not False
        or search_payload["outer_test_predictions_generated"] is not False
        or search_payload["outer_test_metric_evaluations"] != 0
        or search_payload["test_evaluated"] is not False
    ):
        raise AssertionError("Policy search manifest reports forbidden test access.")
    if not final_metrics["split"].eq("test").all():
        raise AssertionError("Final evaluation emitted a non-test metric.")
    final_payload = final_manifest["payload"]
    if (
        final_payload["test_evaluated"] is not True
        or final_payload["outer_test_prediction_batches"] != 1
        or set(final_payload["per_unit_test_evaluation_counts"].values()) != {1}
    ):
        raise AssertionError("Final evaluation did not evaluate each test unit once.")

    output_paths = [
        search_paths["frozen_policy"],
        search_paths["search_metrics"],
        search_paths["search_manifest"],
        final_paths["final_test_metrics"],
        final_paths["final_evaluation_manifest"],
    ]
    output_hashes = {str(path): _sha256_file(path) for path in output_paths}
    manifest = {
        "status": "passed",
        "config_path": str(config_path),
        "search_metric_rows": len(search_metrics),
        "selected_policy_count": int(
            search_metrics.groupby("policy_id")["selected_policy"].first().sum()
        ),
        "search_test_evaluated": search_payload["test_evaluated"],
        "search_outer_test_metric_evaluations": search_payload[
            "outer_test_metric_evaluations"
        ],
        "final_metric_rows": len(final_metrics),
        "final_test_evaluated": final_payload["test_evaluated"],
        "outer_test_prediction_batches": final_payload[
            "outer_test_prediction_batches"
        ],
        "evaluation_unit": final_payload["evaluation_unit"],
        "frozen_policy_hash": final_payload["frozen_policy_hash"],
        "output_hashes": output_hashes,
    }
    (root / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory")
    parser.add_argument(
        "--config",
        default="configs/corrected_policy_search_tiny.yaml",
    )
    args = parser.parse_args()
    manifest = run_smoke(args.output_directory, config_path=args.config)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
