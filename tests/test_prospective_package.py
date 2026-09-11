"""Gates for the Phase 18 prospective-experiment package."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from bh_augmentation.data.reaction_roles import CANONICAL_ROLE_NAMES
from bh_augmentation.prospective_package import (
    CANDIDATE_CLASS_DISCOVERY,
    CONTROL_CLASS_MODEL_HIGH,
    CONTROL_CLASS_MODEL_LOW,
    CONTROL_CLASS_REPLICATE,
    PHASE12_SURVIVING_UNCERTAINTY_METHODS,
    PROSPECTIVE_DISCLAIMER,
    PROSPECTIVE_PACKAGE_SCHEMA_VERSION,
    build_prospective_package,
    summarize_prospective_package,
    validate_prospective_package,
)
from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash
from canonical_artifacts import require_canonical_artifacts

pytest.importorskip("rdkit")

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/corrected_prospective_package_phase18.yaml"

_FORBIDDEN_CLAIM_TERMS = (
    "safe to run",
    "readily accessible",
    "commercially available",
    "will succeed",
    "guaranteed",
    "prospectively validated",
)


@pytest.fixture(scope="module")
def package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    require_canonical_artifacts()
    output = tmp_path_factory.mktemp("phase18-package") / "corrected-package"
    build_prospective_package(CONFIG, output_directory=output)
    return output


def _config(tmp_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text())
    config["output"]["directory"] = str(tmp_path / "corrected-unused")
    return config


def test_config_uses_only_phase12_surviving_uncertainty() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    assert config["model"]["uncertainty_method"] in PHASE12_SURVIVING_UNCERTAINTY_METHODS
    assert config["features"]["kind"] == "bh_role_separated"
    assert config["features"]["fingerprint_backend"] == "rdkit"


def test_unsupported_uncertainty_method_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["model"]["uncertainty_method"] = "distance_aware_ridge"
    with pytest.raises(ValueError, match="survived Phase 12"):
        build_prospective_package(config, output_directory=tmp_path / "corrected-x")


def test_manifest_and_documents_carry_the_required_statement(package: Path) -> None:
    manifest = validate_prospective_package(package)
    readme = (package / "README.md").read_text()
    protocol = (package / "experimental_protocol.md").read_text()
    rationale = json.loads((package / "uncertainty_rationale.json").read_text())

    assert manifest["schema_version"] == PROSPECTIVE_PACKAGE_SCHEMA_VERSION
    assert manifest["prospective_disclaimer"] == PROSPECTIVE_DISCLAIMER
    assert manifest["prospective_validation_performed"] is False
    assert manifest["supervised_autoencoder_used"] is False
    assert manifest["synthesis_feasibility_claimed"] is False
    assert manifest["safety_claimed"] is False
    assert manifest["accessibility_claimed"] is False
    assert PROSPECTIVE_DISCLAIMER in readme
    assert PROSPECTIVE_DISCLAIMER in protocol
    assert rationale["prospective_disclaimer"] == PROSPECTIVE_DISCLAIMER


def test_documents_make_no_feasibility_or_safety_claim(package: Path) -> None:
    text = " ".join(
        (package / name).read_text().lower()
        for name in ("README.md", "experimental_protocol.md")
    )
    for term in _FORBIDDEN_CLAIM_TERMS:
        assert term not in text, term
    protocol = (package / "experimental_protocol.md").read_text().lower()
    assert "unknown" in protocol
    assert "temperature" in protocol


def test_every_candidate_declares_all_seven_canonical_roles(package: Path) -> None:
    roles = pd.read_csv(package / "candidate_roles.csv")
    candidates = pd.read_csv(package / "candidates.csv")

    for role in CANONICAL_ROLE_NAMES:
        assert f"canonical_{role}_smiles" in roles.columns
        assert f"recorded_{role}_smiles" in roles.columns
        assert roles[f"canonical_{role}_smiles"].notna().all()
        assert roles[f"canonical_{role}_smiles"].astype(str).str.len().gt(0).all()
    assert set(roles["candidate_id"]) == set(candidates["candidate_id"])
    assert roles["canonical_reaction_key"].is_unique
    assert roles["canonicalization_version"].nunique() == 1


def test_discovery_candidates_are_unmeasured_recombinations(package: Path) -> None:
    candidates = pd.read_csv(package / "candidates.csv")
    roles = pd.read_csv(package / "candidate_roles.csv")
    measured = pd.read_csv(
        ROOT / "data/processed/bh_canonical_roles_v1.csv",
        usecols=["canonical_reaction_key"],
    )
    discovery_ids = set(
        candidates.loc[
            candidates["candidate_class"].eq(CANDIDATE_CLASS_DISCOVERY), "candidate_id"
        ]
    )
    discovery_roles = roles.loc[roles["candidate_id"].isin(discovery_ids)]
    measured_keys = set(measured["canonical_reaction_key"].astype(str))

    assert discovery_ids
    assert not set(discovery_roles["canonical_reaction_key"]) & measured_keys
    discovery = candidates.loc[candidates["candidate_id"].isin(discovery_ids)]
    assert discovery["measured_in_canonical_dataset"].eq(False).all()
    assert discovery["introduces_new_molecule"].eq(False).all()
    assert discovery["changed_role_count"].ge(1).all()
    assert discovery["practical_accessibility"].eq("unknown").all()


def test_candidate_role_values_all_occur_in_measured_data(package: Path) -> None:
    roles = pd.read_csv(package / "candidate_roles.csv")
    measured = pd.read_csv(ROOT / "data/processed/bh_canonical_roles_v1.csv")
    for role in CANONICAL_ROLE_NAMES:
        observed = set(measured[f"canonical_{role}_smiles"].astype(str))
        assert set(roles[f"canonical_{role}_smiles"].astype(str)) <= observed


def test_provenance_names_measured_donor_rows_for_every_role(package: Path) -> None:
    provenance = pd.read_csv(package / "candidate_provenance.csv")
    candidates = pd.read_csv(package / "candidates.csv")
    measured = pd.read_csv(
        ROOT / "data/processed/bh_canonical_roles_v1.csv", usecols=["source_row_id"]
    )
    measured_ids = set(measured["source_row_id"].astype(str))
    discovery = candidates.loc[candidates["candidate_class"].eq(CANDIDATE_CLASS_DISCOVERY)]
    discovery_provenance = provenance.loc[
        provenance["candidate_id"].isin(set(discovery["candidate_id"]))
    ]

    assert set(discovery_provenance["role"]) == set(CANONICAL_ROLE_NAMES)
    assert len(discovery_provenance) == len(discovery) * len(CANONICAL_ROLE_NAMES)
    assert discovery_provenance["role_value_observed_in_measured_data"].all()
    assert discovery_provenance["measured_rows_with_this_role_value"].ge(1).all()
    assert set(discovery_provenance["exemplar_donor_source_row_id"]) <= measured_ids
    assert set(discovery_provenance["block_donor_source_row_id"]) <= measured_ids
    assert set(provenance["candidate_id"]) == set(candidates["candidate_id"])


def test_controls_are_measured_anchors_held_out_of_the_fit(package: Path) -> None:
    controls = pd.read_csv(package / "controls.csv")
    manifest = json.loads((package / "manifest.json").read_text())

    assert controls["measured_in_canonical_dataset"].all()
    assert controls["held_out_of_model_fit"].all()
    assert controls["measured_yield"].notna().all()
    assert set(controls["control_class"]) == {
        CONTROL_CLASS_REPLICATE,
        CONTROL_CLASS_MODEL_HIGH,
        CONTROL_CLASS_MODEL_LOW,
    }
    replicates = controls.loc[controls["control_class"].eq(CONTROL_CLASS_REPLICATE)]
    assert replicates["measured_yield"].max() - replicates["measured_yield"].min() > 50.0
    high = controls.loc[controls["control_class"].eq(CONTROL_CLASS_MODEL_HIGH)]
    low = controls.loc[controls["control_class"].eq(CONTROL_CLASS_MODEL_LOW)]
    assert high["predicted_yield"].min() > low["predicted_yield"].max()
    assert manifest["control_count"] == len(controls)
    assert controls["candidate_id"].is_unique


def test_predictions_carry_calibrated_intervals_and_support(package: Path) -> None:
    predictions = pd.read_csv(package / "candidate_predictions.csv")
    support = pd.read_csv(package / "support_distances.csv")
    candidates = pd.read_csv(package / "candidates.csv")
    rationale = json.loads((package / "uncertainty_rationale.json").read_text())

    assert (predictions["interval_lower"] <= predictions["predicted_yield"]).all()
    assert (predictions["predicted_yield"] <= predictions["interval_upper"]).all()
    assert predictions["interval_half_width"].gt(0).all()
    assert predictions["nominal_coverage"].eq(rationale["nominal_coverage"]).all()
    assert set(predictions["candidate_id"]) == set(candidates["candidate_id"])
    assert set(support["candidate_id"]) == set(candidates["candidate_id"])
    assert support["support_distance"].between(0.0, 1.0).all()
    assert 0.0 <= rationale["control_empirical_coverage"] <= 1.0
    assert rationale["uncertainty_method"] in PHASE12_SURVIVING_UNCERTAINTY_METHODS
    assert rationale["known_limitations"]


def test_diversity_rationale_covers_every_discovery_candidate(package: Path) -> None:
    diversity = pd.read_csv(package / "diversity_rationale.csv")
    candidates = pd.read_csv(package / "candidates.csv")
    discovery = candidates.loc[candidates["candidate_class"].eq(CANDIDATE_CLASS_DISCOVERY)]

    assert set(diversity["candidate_id"]) == set(discovery["candidate_id"])
    assert diversity["measured_rows_sharing_substrate"].ge(1).all()
    assert diversity["measured_rows_sharing_condition_block"].ge(1).all()
    assert diversity["novelty_basis"].notna().all()
    assert diversity["diversity_rationale"].notna().all()


def test_every_informed_policy_ranks_every_discovery_candidate(package: Path) -> None:
    candidates = pd.read_csv(package / "candidates.csv")
    discovery = candidates.loc[candidates["candidate_class"].eq(CANDIDATE_CLASS_DISCOVERY)]
    rank_columns = [name for name in candidates.columns if name.startswith("rank_")]

    assert len(rank_columns) == 6
    for column in rank_columns:
        ranks = discovery[column].astype(int).tolist()
        assert sorted(ranks) == list(range(1, len(discovery) + 1))
    controls = candidates.loc[candidates["candidate_class"].ne(CANDIDATE_CLASS_DISCOVERY)]
    assert controls[rank_columns].isna().all().all()


def test_manifest_hashes_validate_and_summary_reads(package: Path) -> None:
    manifest = json.loads((package / "manifest.json").read_text())
    for name, digest in manifest["output_hashes"].items():
        assert sha256_file(package / name) == digest
    claimed = manifest.pop("manifest_hash")
    assert claimed == stable_hash(manifest)
    summary = summarize_prospective_package(package)
    assert summary["prospective_disclaimer"] == PROSPECTIVE_DISCLAIMER
    assert summary["discovery_candidate_count"] >= 1


def test_validator_rejects_tampered_candidate_table(package: Path, tmp_path: Path) -> None:
    root = tmp_path / "tampered"
    shutil.copytree(package, root)
    path = root / "candidates.csv"
    rows = pd.read_csv(path, float_precision="round_trip")
    rows.loc[0, "predicted_yield"] = 99.0
    rows.to_csv(path, index=False)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_hashes"]["candidates.csv"] = sha256_file(path)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="replay mismatch"):
        validate_prospective_package(root)


def test_validator_rejects_removed_disclaimer(package: Path, tmp_path: Path) -> None:
    root = tmp_path / "no-disclaimer"
    shutil.copytree(package, root)
    path = root / "README.md"
    path.write_text((path.read_text()).replace(PROSPECTIVE_DISCLAIMER, "looks great"))
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_hashes"]["README.md"] = sha256_file(path)
    manifest.pop("manifest_hash")
    manifest["manifest_hash"] = stable_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="required statement"):
        validate_prospective_package(root)


def test_builder_refuses_to_overwrite_existing_package(package: Path) -> None:
    with pytest.raises(FileExistsError, match="overwrite"):
        build_prospective_package(CONFIG, output_directory=package)


def test_two_builds_are_byte_identical(tmp_path: Path) -> None:
    require_canonical_artifacts()
    first = tmp_path / "corrected-pkg-a"
    second = tmp_path / "corrected-pkg-b"
    config = _config(tmp_path)
    build_prospective_package(config, output_directory=first)
    build_prospective_package(config, output_directory=second)
    for name in (
        "package_plan.json",
        "README.md",
        "candidate_roles.csv",
        "candidates.csv",
        "candidate_predictions.csv",
        "candidate_provenance.csv",
        "controls.csv",
        "support_distances.csv",
        "diversity_rationale.csv",
        "uncertainty_rationale.json",
        "experimental_protocol.md",
    ):
        assert sha256_file(first / name) == sha256_file(second / name), name
