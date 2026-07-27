"""Smoke validation-only joint AE search and one frozen outer evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bh_augmentation.ae_final_evaluation import (
    run_ae_final_evaluation_command,
)
from bh_augmentation.ae_policy_search import run_ae_policy_search_command


def run_smoke(
    output_directory: str | Path,
    *,
    config_path: str | Path = (
        "configs/condition_transfer_supervised_ae_protocol_tiny.yaml"
    ),
) -> dict[str, Any]:
    """Exercise search, frozen refit, and exactly-once test evaluation."""
    root = Path(output_directory)
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite Phase 6 smoke output: {root}")
    root.mkdir(parents=True)
    search_paths = run_ae_policy_search_command(
        config_path,
        output_directory=root / "search",
    )
    final_paths = run_ae_final_evaluation_command(
        config_path,
        frozen_policy_path=search_paths["frozen_policy"],
        search_manifest_path=search_paths["search_manifest"],
        output_directory=root / "final",
    )

    search_metrics = pd.read_csv(search_paths["search_metrics"])
    search_manifest = json.loads(search_paths["search_manifest"].read_text())
    final_metrics = pd.read_csv(final_paths["final_test_metrics"])
    final_manifest = json.loads(
        final_paths["final_evaluation_manifest"].read_text()
    )
    output_claim = json.loads(final_paths["evaluation_claim"].read_text())
    global_claim = json.loads(
        final_paths["global_evaluation_claim"].read_text()
    )
    search_payload = search_manifest["payload"]
    final_payload = final_manifest["payload"]
    inner_split = search_payload["inner_split"]
    if (
        not search_metrics["split"].eq("valid").all()
        or not search_metrics["inner_split_hash"].eq(
            inner_split["split_hash"]
        ).all()
        or search_payload["test_evaluated"] is not False
        or search_payload["outer_test_metric_evaluations"] != 0
    ):
        raise AssertionError("Joint AE search violated its validation-only contract.")
    if (
        not final_metrics["split"].eq("test").all()
        or final_payload["test_evaluated"] is not True
        or final_payload["outer_test_attempts"] != 1
        or final_payload["outer_test_prediction_batches"] != 1
        or final_payload["leakage_audit"]["leakage_detected"]
        or final_payload["n_measured_refit"]
        != inner_split["n_eligible_outer_train"]
    ):
        raise AssertionError("Frozen AE final evaluation violated its refit contract.")
    for claim in (output_claim, global_claim):
        if (
            claim["status"] != "complete"
            or claim["outer_test_attempts"] != 1
            or claim["outer_test_prediction_batches"] != 1
        ):
            raise AssertionError("Exactly-once evaluation claim is incomplete.")

    scientific_paths = [
        *(
            path
            for name, path in search_paths.items()
            if name != "directory"
        ),
        *(
            path
            for name, path in final_paths.items()
            if name != "directory"
        ),
    ]
    unique_paths = sorted(set(scientific_paths), key=str)
    output_hashes = {
        str(path): _sha256_file(path)
        for path in unique_paths
    }
    manifest = {
        "status": "passed",
        "config_path": str(config_path),
        "inner_split_hash": inner_split["split_hash"],
        "n_eligible_outer_train": inner_split["n_eligible_outer_train"],
        "n_ae_fit": inner_split["n_ae_train"],
        "n_ae_internal_validation": inner_split["n_internal_validation"],
        "joint_candidate_count": len(search_payload["candidate_policy_hashes"]),
        "feasible_candidate_count": len(search_payload["feasible_policy_hashes"]),
        "selected_policy_count": int(
            search_metrics.groupby("policy_id")["selected_policy"].first().sum()
        ),
        "selected_transfer_method": final_payload[
            "selected_transfer_method"
        ],
        "n_synthetic_refit": final_payload["n_synthetic_refit"],
        "search_test_evaluated": search_payload["test_evaluated"],
        "final_test_evaluated": final_payload["test_evaluated"],
        "outer_test_attempts": final_payload["outer_test_attempts"],
        "outer_test_prediction_batches": final_payload[
            "outer_test_prediction_batches"
        ],
        "leakage_detected": final_payload["leakage_audit"][
            "leakage_detected"
        ],
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
        default="configs/condition_transfer_supervised_ae_protocol_tiny.yaml",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_smoke(args.output_directory, config_path=args.config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
