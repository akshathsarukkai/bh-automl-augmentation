"""Reviewable prospective-experiment package for the Phase 18 recommendation work.

The package proposes reactions that are *not* present in the measured canonical
dataset, together with controls, provenance, calibrated predictions and an
experimental protocol draft.

Scope and honesty contracts enforced here:

* Candidates are produced only by recombining canonical seven-role values that
  already occur in the measured dataset. No molecule is invented, no chemical
  eligibility rule is relaxed, and fingerprint equality is never treated as
  chemical identity: identity is the RDKit canonical seven-role reaction key.
* The predictive model and its uncertainty are exactly the machinery that
  survived Phase 12 and was used in the Phase 18 simulation. The supervised
  autoencoder is not used.
* Prospective laboratory validation has not been performed. The package makes no
  claim about synthesis feasibility, safety, accessibility, cost, or likelihood
  of laboratory success. Where practical accessibility is unknown, it is
  reported as unknown.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from bh_augmentation.data.canonicalize_roles import (
    CANONICALIZATION_VERSION,
    build_canonical_reaction_identity,
)
from bh_augmentation.data.reaction_roles import CANONICAL_ROLE_NAMES
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata

# Shared deterministic helpers and the Phase 18 acquisition rules are reused
# directly so the package cannot drift from the validated simulation.
from bh_augmentation.recommendation_simulation import (
    ACQUISITION_STRATEGIES,
    INFORMED_STRATEGIES,
    PHASE12_SURVIVING_UNCERTAINTY_METHODS,
    AcquisitionContext,
    _dependency_versions,
    _estimator_params,
    _exact_keys,
    _finite_or_none,
    _frame_records,
    _git_commit,
    _git_dirty,
    _int,
    _load_config,
    _mapping,
    _nonnegative_float,
    _open_unit,
    _positive_int,
    _records_close,
    _stable_uniform,
    _write_json,
)
from bh_augmentation.results.status import assert_result_directory_allowed
from bh_augmentation.uncertainty.calibration import interval_calibration_metrics
from bh_augmentation.uncertainty.estimators import (
    UncertaintyEstimatorConfig,
    fit_uncertainty_estimator,
)
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    resolve_corrected_feature_config,
    sha256_file,
    stable_hash,
)

PROSPECTIVE_PACKAGE_SCHEMA_VERSION = "bh-prospective-package-v1"

#: Required verbatim statement. It appears in the README and in the manifest.
PROSPECTIVE_DISCLAIMER = "Prospective laboratory validation has not been performed."

CANDIDATE_CLASS_DISCOVERY = "unmeasured_role_recombination"
CONTROL_CLASS_REPLICATE = "measured_replicate_anchor"
CONTROL_CLASS_MODEL_HIGH = "model_high_prediction_anchor"
CONTROL_CLASS_MODEL_LOW = "model_low_prediction_anchor"

_SUBSTRATE_ROLES = ("reactant_1", "reactant_2", "product")
_CONDITION_ROLES = ("catalyst", "ligand", "base", "solvent_or_additive")

_OUTPUTS = (
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
)
_PLAN_FIELDS = {
    "schema_version",
    "status",
    "prospective_disclaimer",
    "dataset_hash",
    "canonicalization_version",
    "feature_contract",
    "feature_metadata_hash",
    "candidate_construction",
    "measured_row_count",
    "candidate_space",
    "partition",
    "resolved_scientific_config",
    "config_hash",
    "supervised_autoencoder_used",
    "prospective_validation_performed",
    "synthesis_feasibility_claimed",
    "safety_claimed",
    "accessibility_claimed",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "status",
    "prospective_disclaimer",
    "git_commit",
    "git_dirty_at_execution",
    "dataset_hash",
    "feature_metadata_hash",
    "config_hash",
    "plan_hash",
    "resolved_scientific_config",
    "dependency_versions",
    "command",
    "measured_row_count",
    "discovery_candidate_count",
    "control_count",
    "candidate_row_count",
    "provenance_row_count",
    "diversity_rationale_row_count",
    "support_distance_row_count",
    "uncertainty_method",
    "surviving_uncertainty_methods",
    "nominal_coverage",
    "control_empirical_coverage",
    "estimator_audit_hash",
    "candidate_prediction_hash",
    "recommendation_simulation_directory",
    "recommendation_simulation_manifest_hash",
    "supervised_autoencoder_used",
    "prospective_validation_performed",
    "synthesis_feasibility_claimed",
    "safety_claimed",
    "accessibility_claimed",
    "output_hashes",
}

CANDIDATE_CONSTRUCTION = {
    "method": "canonical_seven_role_recombination_of_measured_role_values",
    "identity": "rdkit_canonical_seven_role_reaction_key",
    "fingerprint_equality_treated_as_identity": False,
    "new_molecules_introduced": False,
    "role_values_restricted_to_measured_inventory": True,
    "candidate_must_be_absent_from_measured_dataset": True,
    "eligibility_rules_relaxed": False,
}


def build_prospective_package(
    config: str | Path | Mapping[str, Any],
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Build the reviewable prospective package and write validated artifacts."""
    contract = _resolve_contract(_load_config(config))
    prepared = _prepare(contract)
    plan = _build_plan(contract, prepared)

    output = Path(
        output_directory if output_directory is not None else contract["output_directory"]
    )
    assert_result_directory_allowed(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prospective package: {output}")
    output.mkdir(parents=True, exist_ok=False)

    tables = _build_tables(contract, prepared, plan)
    _write_json(output / "package_plan.json", plan)
    (output / "README.md").write_text(_readme(contract, prepared, plan, tables))
    (output / "experimental_protocol.md").write_text(_protocol(contract, prepared, tables))
    _write_json(output / "uncertainty_rationale.json", tables["uncertainty_rationale"])
    for name in (
        "candidate_roles",
        "candidates",
        "candidate_predictions",
        "candidate_provenance",
        "controls",
        "support_distances",
        "diversity_rationale",
    ):
        tables[name].to_csv(output / f"{name}.csv", index=False)

    manifest = _build_manifest(contract, prepared, plan, tables, config, output)
    _write_json(output / "manifest.json", manifest)
    validate_prospective_package(output)
    return {
        **{name.split(".")[0]: output / name for name in _OUTPUTS},
        "output_directory": output,
        "manifest": output / "manifest.json",
    }


def validate_prospective_package(directory: str | Path) -> dict[str, Any]:
    """Independently rebuild the package from its saved plan and compare it."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    claimed_manifest_hash = manifest.pop("manifest_hash", None)
    if (
        claimed_manifest_hash != stable_hash(manifest)
        or set(manifest) != _MANIFEST_FIELDS
        or manifest.get("schema_version") != PROSPECTIVE_PACKAGE_SCHEMA_VERSION
        or manifest.get("status") != "prospective_package_complete"
        or manifest.get("prospective_disclaimer") != PROSPECTIVE_DISCLAIMER
        or manifest.get("supervised_autoencoder_used") is not False
        or manifest.get("prospective_validation_performed") is not False
        or manifest.get("synthesis_feasibility_claimed") is not False
        or manifest.get("safety_claimed") is not False
        or manifest.get("accessibility_claimed") is not False
        or manifest.get("uncertainty_method") not in PHASE12_SURVIVING_UNCERTAINTY_METHODS
        or manifest.get("surviving_uncertainty_methods")
        != list(PHASE12_SURVIVING_UNCERTAINTY_METHODS)
    ):
        raise ValueError("Invalid prospective package completion manifest.")
    manifest["manifest_hash"] = claimed_manifest_hash
    if set(manifest.get("output_hashes", {})) != set(_OUTPUTS):
        raise ValueError("Prospective package output hash coverage mismatch.")
    for name, digest in manifest["output_hashes"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Prospective package output hash mismatch: {name}.")

    plan = json.loads((root / "package_plan.json").read_text())
    claimed_plan_hash = plan.pop("plan_hash", None)
    if (
        claimed_plan_hash != stable_hash(plan)
        or set(plan) != _PLAN_FIELDS
        or plan.get("schema_version") != PROSPECTIVE_PACKAGE_SCHEMA_VERSION
        or plan.get("prospective_disclaimer") != PROSPECTIVE_DISCLAIMER
        or plan.get("candidate_construction") != CANDIDATE_CONSTRUCTION
        or plan.get("supervised_autoencoder_used") is not False
        or plan.get("prospective_validation_performed") is not False
        or plan.get("config_hash") != stable_hash(plan.get("resolved_scientific_config"))
    ):
        raise ValueError("Invalid prospective package plan.")
    plan["plan_hash"] = claimed_plan_hash
    if manifest["plan_hash"] != claimed_plan_hash or manifest["config_hash"] != plan[
        "config_hash"
    ]:
        raise ValueError("Prospective package manifest-plan mismatch.")

    contract = _contract_from_scientific(plan["resolved_scientific_config"], root)
    prepared = _prepare(contract)
    replay_plan = _build_plan(contract, prepared)
    if replay_plan["plan_hash"] != claimed_plan_hash:
        raise ValueError("Prospective package plan replay mismatch.")
    tables = _build_tables(contract, prepared, replay_plan)
    for name in (
        "candidate_roles",
        "candidates",
        "candidate_predictions",
        "candidate_provenance",
        "controls",
        "support_distances",
        "diversity_rationale",
    ):
        observed = pd.read_csv(root / f"{name}.csv", float_precision="round_trip")
        if not _records_close(_frame_records(tables[name]), _frame_records(observed)):
            raise ValueError(f"Prospective package replay mismatch: {name}.csv.")
    if json.loads((root / "uncertainty_rationale.json").read_text()) != tables[
        "uncertainty_rationale"
    ]:
        raise ValueError("Prospective package replay mismatch: uncertainty_rationale.json.")

    readme = (root / "README.md").read_text()
    protocol = (root / "experimental_protocol.md").read_text()
    if PROSPECTIVE_DISCLAIMER not in readme or PROSPECTIVE_DISCLAIMER not in protocol:
        raise ValueError("Prospective package documents omit the required statement.")
    _assert_candidate_contracts(tables, manifest)
    return manifest


def _assert_candidate_contracts(
    tables: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    """Fail unless identity, novelty and control contracts hold in the artifacts."""
    roles = tables["candidate_roles"]
    candidates = tables["candidates"]
    discovery = candidates.loc[candidates["candidate_class"].eq(CANDIDATE_CLASS_DISCOVERY)]
    controls = tables["controls"]
    missing_roles = [
        f"canonical_{role}_smiles"
        for role in CANONICAL_ROLE_NAMES
        if f"canonical_{role}_smiles" not in roles.columns
    ]
    if missing_roles:
        raise ValueError(f"Candidate roles omit canonical role columns: {missing_roles}.")
    if roles["canonical_reaction_key"].isna().any() or roles["canonical_reaction_key"].eq(
        ""
    ).any():
        raise ValueError("Every candidate requires a canonical seven-role reaction key.")
    if roles["canonical_reaction_key"].duplicated().any():
        raise ValueError("Candidate rows repeat a canonical seven-role reaction.")
    if not discovery["measured_in_canonical_dataset"].eq(False).all():
        raise ValueError("A discovery candidate is already present in the measured data.")
    if not discovery["introduces_new_molecule"].eq(False).all():
        raise ValueError("A discovery candidate introduces a molecule outside the data.")
    if not discovery["changed_role_count"].ge(1).all():
        raise ValueError("A discovery candidate does not change any role.")
    if controls.empty or not controls["measured_in_canonical_dataset"].all():
        raise ValueError("Controls must be already-measured anchors.")
    if set(controls["control_class"]) - {
        CONTROL_CLASS_REPLICATE,
        CONTROL_CLASS_MODEL_HIGH,
        CONTROL_CLASS_MODEL_LOW,
    }:
        raise ValueError("Unknown control class in the prospective package.")
    if int(manifest["discovery_candidate_count"]) != len(discovery):
        raise ValueError("Manifest discovery candidate count mismatch.")
    if int(manifest["control_count"]) != len(controls):
        raise ValueError("Manifest control count mismatch.")
    if not candidates["practical_accessibility"].eq("unknown").all():
        raise ValueError("Practical accessibility must be reported as unknown.")


def _prepare(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Load measured rows, enumerate the candidate space and build features."""
    dataset_path = Path(contract["dataset_path"])
    measured = pd.read_csv(dataset_path)
    measured["source_row_id"] = measured["source_row_id"].astype(str)
    if not bool(measured["all_required_roles_parse_valid"].all()):
        raise ValueError("Prospective packaging requires fully role-valid measured rows.")
    if measured["canonical_reaction_key"].duplicated().any():
        raise ValueError("Measured canonical reactions must be unique.")
    dataset_hash = sha256_file(dataset_path)

    recovered_of = _recovered_lookup(measured)
    substrates = (
        measured[
            [
                "canonical_substrate_key",
                *[f"canonical_{role}_smiles" for role in _SUBSTRATE_ROLES],
            ]
        ]
        .drop_duplicates()
        .sort_values("canonical_substrate_key", kind="stable")
        .reset_index(drop=True)
    )
    conditions = (
        measured[
            [
                "canonical_condition_key",
                *[f"canonical_{role}_smiles" for role in _CONDITION_ROLES],
            ]
        ]
        .drop_duplicates()
        .sort_values("canonical_condition_key", kind="stable")
        .reset_index(drop=True)
    )
    measured_keys = set(measured["canonical_reaction_key"].astype(str))
    candidate_rows: list[dict[str, Any]] = []
    for _, substrate in substrates.iterrows():
        for _, condition in conditions.iterrows():
            row = {
                f"canonical_{role}_smiles": str(substrate[f"canonical_{role}_smiles"])
                for role in _SUBSTRATE_ROLES
            }
            row.update(
                {
                    f"canonical_{role}_smiles": str(condition[f"canonical_{role}_smiles"])
                    for role in _CONDITION_ROLES
                }
            )
            row.update(
                {
                    f"{role}_parse_valid": True
                    for role in CANONICAL_ROLE_NAMES
                }
            )
            row["all_required_roles_parse_valid"] = True
            identity = build_canonical_reaction_identity(row)
            if identity["key"] is None:
                raise ValueError("Candidate construction produced an invalid identity.")
            if str(identity["key"]) in measured_keys:
                continue
            row["canonical_reaction_key"] = str(identity["key"])
            row["canonical_reaction_hash"] = str(identity["hash"])
            row["canonical_substrate_key"] = str(substrate["canonical_substrate_key"])
            row["canonical_condition_key"] = str(condition["canonical_condition_key"])
            for role in CANONICAL_ROLE_NAMES:
                row[f"recovered_{role}_smiles"] = recovered_of[role][
                    row[f"canonical_{role}_smiles"]
                ]
            candidate_rows.append(row)
    candidates = pd.DataFrame(candidate_rows).sort_values(
        "canonical_reaction_hash", kind="stable"
    ).reset_index(drop=True)
    if candidates.empty:
        raise ValueError(
            "The measured dataset already covers every eligible role recombination; "
            "no prospective candidate can be supported without inventing chemistry."
        )
    candidates.insert(
        0,
        "candidate_id",
        [f"cand-{value[:16]}" for value in candidates["canonical_reaction_hash"]],
    )

    combined = pd.concat(
        [
            candidates.assign(yield_placeholder=0.0),
            measured.assign(yield_placeholder=0.0),
        ],
        ignore_index=True,
    )
    combined["yield"] = 0.0
    features, _, feature_names, feature_metadata = build_feature_matrix_with_metadata(
        combined, contract["feature_config"]
    )
    features = np.asarray(features, dtype=np.float32)
    if not np.isin(features, (0.0, 1.0)).all():
        raise ValueError("Prospective packaging requires binary fingerprint features.")
    partition = _partition(measured, contract)
    return {
        "dataset_path": dataset_path,
        "dataset_hash": dataset_hash,
        "measured": measured,
        "candidates": candidates,
        "features": features,
        "candidate_features": features[: len(candidates)],
        "measured_features": features[len(candidates) :],
        "feature_contract": feature_contract_record(feature_metadata, feature_names),
        "substrate_count": len(substrates),
        "condition_count": len(conditions),
        "partition": partition,
        "recovered_of": recovered_of,
    }


def _recovered_lookup(measured: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Map each canonical role value to the exact recorded role string."""
    lookup: dict[str, dict[str, str]] = {}
    for role in CANONICAL_ROLE_NAMES:
        pairs = measured[
            [f"canonical_{role}_smiles", f"recovered_{role}_smiles"]
        ].drop_duplicates()
        counts = pairs.groupby(f"canonical_{role}_smiles").size()
        if int(counts.max()) != 1:
            raise ValueError(
                f"Canonical {role} values do not map to a unique recorded string."
            )
        lookup[role] = {
            str(canonical): str(recorded)
            for canonical, recorded in pairs.itertuples(index=False)
        }
    return lookup


def _partition(measured: pd.DataFrame, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Split measured rows into model fit, conformal calibration and control pool."""
    salt = f"prospective-{contract['base_seed']}"
    calibration: list[str] = []
    control_pool: list[str] = []
    fit: list[str] = []
    upper = contract["calibration_fraction"] + contract["control_pool_fraction"]
    for source_id in measured["source_row_id"]:
        value = _stable_uniform(salt, source_id)
        if value < contract["calibration_fraction"]:
            calibration.append(source_id)
        elif value < upper:
            control_pool.append(source_id)
        else:
            fit.append(source_id)
    required = (
        contract["measured_replicate_anchors"]
        + contract["model_high_anchors"]
        + contract["model_low_anchors"]
    )
    if len(calibration) < 50 or len(control_pool) < required or len(fit) < 100:
        raise ValueError(
            "Measured partition is too small for a reviewable package "
            f"(fit={len(fit)}, calibration={len(calibration)}, controls={len(control_pool)})."
        )
    return {
        "fit_source_ids": tuple(sorted(fit)),
        "calibration_source_ids": tuple(sorted(calibration)),
        "control_pool_source_ids": tuple(sorted(control_pool)),
    }


def _build_plan(contract: Mapping[str, Any], prepared: Mapping[str, Any]) -> dict[str, Any]:
    scientific = _scientific_config(contract)
    partition = prepared["partition"]
    plan = {
        "schema_version": PROSPECTIVE_PACKAGE_SCHEMA_VERSION,
        "status": "frozen_before_model_fitting",
        "prospective_disclaimer": PROSPECTIVE_DISCLAIMER,
        "dataset_hash": prepared["dataset_hash"],
        "canonicalization_version": CANONICALIZATION_VERSION,
        "feature_contract": prepared["feature_contract"],
        "feature_metadata_hash": prepared["feature_contract"]["feature_metadata_hash"],
        "candidate_construction": CANDIDATE_CONSTRUCTION,
        "measured_row_count": len(prepared["measured"]),
        "candidate_space": {
            "measured_substrate_groups": prepared["substrate_count"],
            "measured_condition_groups": prepared["condition_count"],
            "complete_recombination_grid": prepared["substrate_count"]
            * prepared["condition_count"],
            "already_measured": len(prepared["measured"]),
            "unmeasured_candidates": len(prepared["candidates"]),
            "candidate_id_hash": stable_hash(
                list(prepared["candidates"]["canonical_reaction_hash"])
            ),
        },
        "partition": {
            "fit_count": len(partition["fit_source_ids"]),
            "calibration_count": len(partition["calibration_source_ids"]),
            "control_pool_count": len(partition["control_pool_source_ids"]),
            "fit_source_id_hash": stable_hash(list(partition["fit_source_ids"])),
            "calibration_source_id_hash": stable_hash(
                list(partition["calibration_source_ids"])
            ),
            "control_pool_source_id_hash": stable_hash(
                list(partition["control_pool_source_ids"])
            ),
        },
        "resolved_scientific_config": scientific,
        "config_hash": stable_hash(scientific),
        "supervised_autoencoder_used": False,
        "prospective_validation_performed": False,
        "synthesis_feasibility_claimed": False,
        "safety_claimed": False,
        "accessibility_claimed": False,
    }
    plan["plan_hash"] = stable_hash(plan)
    return plan


def _build_tables(
    contract: Mapping[str, Any], prepared: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Fit the surviving estimator once and derive every reviewable table."""
    measured = prepared["measured"]
    position_of = {
        source_id: index for index, source_id in enumerate(measured["source_row_id"])
    }
    yields = measured["yield"].to_numpy(dtype=float)
    partition = prepared["partition"]
    fit_ids = partition["fit_source_ids"]
    calibration_ids = partition["calibration_source_ids"]
    control_ids = partition["control_pool_source_ids"]
    measured_features = prepared["measured_features"]

    estimator_config = UncertaintyEstimatorConfig(
        method=contract["uncertainty_method"],
        coverage=contract["coverage"],
        random_state=contract["base_seed"],
        **contract["estimator_params"],
    )
    estimator = fit_uncertainty_estimator(
        measured_features[[position_of[value] for value in fit_ids]],
        np.asarray([yields[position_of[value]] for value in fit_ids], dtype=float),
        train_source_ids=fit_ids,
        calibration_features=measured_features[
            [position_of[value] for value in calibration_ids]
        ],
        calibration_labels=np.asarray(
            [yields[position_of[value]] for value in calibration_ids], dtype=float
        ),
        calibration_source_ids=calibration_ids,
        config=estimator_config,
    )
    candidates = prepared["candidates"]
    candidate_ids = tuple(candidates["candidate_id"])
    candidate_prediction = estimator.predict(
        prepared["candidate_features"], source_ids=candidate_ids
    )
    control_prediction = estimator.predict(
        measured_features[[position_of[value] for value in control_ids]],
        source_ids=control_ids,
    )

    fit_features = measured_features[[position_of[value] for value in fit_ids]]
    candidate_similarity = _cosine_similarity(prepared["candidate_features"], fit_features)
    control_similarity = _cosine_similarity(
        measured_features[[position_of[value] for value in control_ids]], fit_features
    )
    candidate_support = 1.0 - candidate_similarity.max(axis=1)
    control_support = 1.0 - control_similarity.max(axis=1)
    nearest_fit = [fit_ids[int(index)] for index in candidate_similarity.argmax(axis=1)]

    ranking = _policy_rankings(
        contract=contract,
        prepared=prepared,
        fit_ids=fit_ids,
        fit_features=fit_features,
        yields=yields,
        position_of=position_of,
        candidate_prediction=candidate_prediction,
        candidate_support_similarity=candidate_similarity.max(axis=1),
    )
    controls = _select_controls(
        contract=contract,
        measured=measured,
        position_of=position_of,
        yields=yields,
        control_ids=control_ids,
        prediction=control_prediction,
        support=control_support,
    )
    control_metrics = interval_calibration_metrics(
        np.asarray([yields[position_of[value]] for value in control_ids], dtype=float),
        np.asarray(control_prediction.point, dtype=float),
        np.asarray(control_prediction.interval_lower, dtype=float),
        np.asarray(control_prediction.interval_upper, dtype=float),
        np.asarray(control_prediction.calibrated_uncertainty, dtype=float),
        nominal_coverage=contract["coverage"],
    )

    roles = _candidate_roles(candidates, controls, measured)
    candidate_table = _candidate_table(
        candidates,
        controls,
        candidate_prediction,
        candidate_support,
        ranking,
        _donor_conditions(measured),
    )
    predictions = _prediction_table(
        candidates,
        controls,
        candidate_prediction,
        estimator.audit.audit_hash,
        contract,
    )
    provenance = _provenance_table(candidates, controls, measured)
    support = _support_table(
        candidates, controls, candidate_support, nearest_fit, len(fit_ids)
    )
    diversity = _diversity_table(candidates, measured, prepared, candidate_support)
    rationale = _uncertainty_rationale(
        contract=contract,
        estimator_audit_hash=estimator.audit.audit_hash,
        candidate_prediction_hash=candidate_prediction.prediction_hash,
        plan=plan,
        control_metrics=control_metrics,
        candidate_uncertainty=np.asarray(
            candidate_prediction.calibrated_uncertainty, dtype=float
        ),
    )
    return {
        "candidate_roles": roles,
        "candidates": candidate_table,
        "candidate_predictions": predictions,
        "candidate_provenance": provenance,
        "controls": controls,
        "support_distances": support,
        "diversity_rationale": diversity,
        "uncertainty_rationale": rationale,
        "estimator_audit_hash": estimator.audit.audit_hash,
        "candidate_prediction_hash": candidate_prediction.prediction_hash,
        "control_metrics": control_metrics,
        "ranking": ranking,
    }


def _donor_conditions(measured: pd.DataFrame) -> dict[str, dict[str, str]]:
    """Return each substrate group's lexicographically first measured condition."""
    ordered = measured.sort_values("source_row_id", kind="stable")
    donors = ordered.drop_duplicates(subset="canonical_substrate_key", keep="first")
    return {
        str(row["canonical_substrate_key"]): {
            role: str(row[f"canonical_{role}_smiles"]) for role in _CONDITION_ROLES
        }
        for _, row in donors.iterrows()
    }


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return exact cosine similarity between binary fingerprint blocks."""
    left_norm = np.sqrt(np.maximum(left.sum(axis=1, dtype=np.float64), 1.0e-12))
    right_norm = np.sqrt(np.maximum(right.sum(axis=1, dtype=np.float64), 1.0e-12))
    products = np.asarray(left @ right.T, dtype=np.float64)
    return np.clip(products / np.outer(left_norm, right_norm), 0.0, 1.0)


def _policy_rankings(
    *,
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
    fit_ids: Sequence[str],
    fit_features: np.ndarray,
    yields: np.ndarray,
    position_of: Mapping[str, int],
    candidate_prediction: Any,
    candidate_support_similarity: np.ndarray,
) -> dict[str, list[str]]:
    """Rank candidates under every Phase 18 informed policy.

    Each policy is applied by repeated single-item selection under its own rule,
    with the support similarity updated after each pick, which reproduces the
    batch behaviour used in the validated simulation.
    """
    candidates = prepared["candidates"]
    candidate_ids = list(candidates["candidate_id"])
    labeled_values = {value: float(yields[position_of[value]]) for value in fit_ids}
    pool_features = np.vstack([prepared["candidate_features"], fit_features])
    similarity = _cosine_similarity(pool_features, pool_features)
    params = {
        "ucb_beta": contract["ucb_beta"],
        "expected_improvement_xi": contract["expected_improvement_xi"],
        "diversity_penalty": contract["diversity_penalty"],
        "support_distance_yield_weight": contract["support_distance_yield_weight"],
        "coverage_z": float(norm.ppf(0.5 * (1.0 + contract["coverage"]))),
    }
    point = np.asarray(candidate_prediction.point, dtype=np.float64)
    uncertainty = np.asarray(candidate_prediction.calibrated_uncertainty, dtype=np.float64)
    rankings: dict[str, list[str]] = {}
    for strategy in INFORMED_STRATEGIES:
        remaining = list(range(len(candidate_ids)))
        nearest = np.asarray(candidate_support_similarity, dtype=np.float64).copy()
        generator = np.random.default_rng(
            int(contract["base_seed"]) + int(stable_hash([strategy])[:8], 16) % 1_000_000
        )
        order: list[str] = []
        while remaining:
            context = AcquisitionContext(
                evaluation_unit="prospective-package",
                strategy=strategy,
                replicate=0,
                round_index=len(order) + 1,
                batch_size=1,
                candidate_source_ids=tuple(candidate_ids[index] for index in remaining),
                candidate_pool_positions=np.asarray(remaining, dtype=np.int64),
                labeled_source_ids=tuple(fit_ids),
                labeled_values=labeled_values,
                similarity=similarity,
                nearest_similarity=np.asarray(
                    [nearest[index] for index in remaining], dtype=np.float64
                ),
                point=point[remaining],
                calibrated_uncertainty=uncertainty[remaining],
                rng=generator,
                params=params,
            )
            selection = ACQUISITION_STRATEGIES[strategy](context)
            chosen = remaining[int(selection.positions[0])]
            order.append(candidate_ids[chosen])
            nearest = np.maximum(nearest, similarity[: len(candidate_ids), chosen])
            remaining.remove(chosen)
        rankings[strategy] = order
    return rankings


def _select_controls(
    *,
    contract: Mapping[str, Any],
    measured: pd.DataFrame,
    position_of: Mapping[str, int],
    yields: np.ndarray,
    control_ids: Sequence[str],
    prediction: Any,
    support: np.ndarray,
) -> pd.DataFrame:
    """Choose already-measured anchors that expose systematic bias if re-run."""
    frame = pd.DataFrame(
        {
            "source_row_id": list(control_ids),
            "measured_yield": [float(yields[position_of[value]]) for value in control_ids],
            "predicted_yield": np.asarray(prediction.point, dtype=float),
            "interval_lower": np.asarray(prediction.interval_lower, dtype=float),
            "interval_upper": np.asarray(prediction.interval_upper, dtype=float),
            "calibrated_uncertainty": np.asarray(
                prediction.calibrated_uncertainty, dtype=float
            ),
            "support_distance": np.asarray(support, dtype=float),
        }
    )
    by_measured = frame.sort_values(["measured_yield", "source_row_id"], kind="stable")
    replicate_count = contract["measured_replicate_anchors"]
    stride = np.linspace(0, len(by_measured) - 1, replicate_count)
    replicate_ids = [
        str(by_measured.iloc[int(round(position))]["source_row_id"]) for position in stride
    ]
    by_prediction = frame.sort_values(
        ["predicted_yield", "source_row_id"], ascending=[False, True], kind="stable"
    )
    high_ids = [
        value
        for value in by_prediction["source_row_id"]
        if value not in set(replicate_ids)
    ][: contract["model_high_anchors"]]
    low_ids = [
        value
        for value in reversed(list(by_prediction["source_row_id"]))
        if value not in set(replicate_ids) | set(high_ids)
    ][: contract["model_low_anchors"]]
    classes = {value: CONTROL_CLASS_REPLICATE for value in replicate_ids}
    classes.update({value: CONTROL_CLASS_MODEL_HIGH for value in high_ids})
    classes.update({value: CONTROL_CLASS_MODEL_LOW for value in low_ids})
    selected = frame.loc[frame["source_row_id"].isin(classes)].copy()
    selected["control_class"] = selected["source_row_id"].map(classes)
    selected["control_purpose"] = selected["control_class"].map(
        {
            CONTROL_CLASS_REPLICATE: (
                "Re-run a measured reaction spanning the recorded yield range to detect "
                "systematic offset between this laboratory and the source dataset."
            ),
            CONTROL_CLASS_MODEL_HIGH: (
                "Re-run a measured reaction the model scores highest to detect optimistic "
                "bias at the top of the predicted range."
            ),
            CONTROL_CLASS_MODEL_LOW: (
                "Re-run a measured reaction the model scores lowest to detect pessimistic "
                "bias at the bottom of the predicted range."
            ),
        }
    )
    selected["measured_in_canonical_dataset"] = True
    selected["measured_within_interval"] = (
        selected["measured_yield"].between(
            selected["interval_lower"], selected["interval_upper"]
        )
    )
    selected["held_out_of_model_fit"] = True
    selected["candidate_id"] = [
        f"ctrl-{value[:16]}" for value in selected["source_row_id"]
    ]
    keys = measured.set_index("source_row_id")
    selected["canonical_reaction_key"] = selected["source_row_id"].map(
        keys["canonical_reaction_key"]
    )
    selected["canonical_reaction_hash"] = selected["source_row_id"].map(
        keys["canonical_reaction_hash"]
    )
    ordered = selected.sort_values(
        ["control_class", "source_row_id"], kind="stable"
    ).reset_index(drop=True)
    return ordered[
        [
            "candidate_id",
            "control_class",
            "control_purpose",
            "source_row_id",
            "canonical_reaction_key",
            "canonical_reaction_hash",
            "measured_yield",
            "predicted_yield",
            "interval_lower",
            "interval_upper",
            "calibrated_uncertainty",
            "measured_within_interval",
            "support_distance",
            "measured_in_canonical_dataset",
            "held_out_of_model_fit",
        ]
    ]


def _candidate_roles(
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    measured: pd.DataFrame,
) -> pd.DataFrame:
    """Write all seven canonical roles explicitly for every proposed reaction."""
    rows: list[dict[str, Any]] = []
    for record in candidates.to_dict("records"):
        row = {
            "candidate_id": record["candidate_id"],
            "candidate_class": CANDIDATE_CLASS_DISCOVERY,
            "canonical_reaction_key": record["canonical_reaction_key"],
            "canonical_reaction_hash": record["canonical_reaction_hash"],
            "canonicalization_version": CANONICALIZATION_VERSION,
        }
        for role in CANONICAL_ROLE_NAMES:
            row[f"canonical_{role}_smiles"] = record[f"canonical_{role}_smiles"]
            row[f"recorded_{role}_smiles"] = record[f"recovered_{role}_smiles"]
        rows.append(row)
    measured_rows = measured.set_index("source_row_id")
    for record in controls.to_dict("records"):
        source = measured_rows.loc[record["source_row_id"]]
        row = {
            "candidate_id": record["candidate_id"],
            "candidate_class": record["control_class"],
            "canonical_reaction_key": str(source["canonical_reaction_key"]),
            "canonical_reaction_hash": str(source["canonical_reaction_hash"]),
            "canonicalization_version": CANONICALIZATION_VERSION,
        }
        for role in CANONICAL_ROLE_NAMES:
            row[f"canonical_{role}_smiles"] = str(source[f"canonical_{role}_smiles"])
            row[f"recorded_{role}_smiles"] = str(source[f"recovered_{role}_smiles"])
        rows.append(row)
    return pd.DataFrame(rows)


def _changed_role_count(
    record: Mapping[str, Any], donor_conditions: Mapping[str, Mapping[str, str]]
) -> int:
    """Count condition roles the candidate changes relative to its substrate donor.

    The substrate donor is the lexicographically first measured row sharing the
    candidate's canonical substrate key. A value of zero would mean the candidate
    reproduces a measured reaction and is rejected upstream.
    """
    donor = donor_conditions[str(record["canonical_substrate_key"])]
    return sum(
        1
        for role in _CONDITION_ROLES
        if str(record[f"canonical_{role}_smiles"]) != str(donor[role])
    )


def _candidate_table(
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    prediction: Any,
    support: np.ndarray,
    ranking: Mapping[str, Sequence[str]],
    donor_conditions: Mapping[str, Mapping[str, str]],
) -> pd.DataFrame:
    """Return the reviewer-facing table of proposed experiments."""
    rank_of = {
        strategy: {value: index + 1 for index, value in enumerate(order)}
        for strategy, order in ranking.items()
    }
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(candidates.to_dict("records")):
        row = {
            "candidate_id": record["candidate_id"],
            "candidate_class": CANDIDATE_CLASS_DISCOVERY,
            "canonical_reaction_key": record["canonical_reaction_key"],
            "canonical_reaction_hash": record["canonical_reaction_hash"],
            "canonical_substrate_key": record["canonical_substrate_key"],
            "canonical_condition_key": record["canonical_condition_key"],
            "measured_in_canonical_dataset": False,
            "introduces_new_molecule": False,
            "changed_role_count": _changed_role_count(record, donor_conditions),
            "predicted_yield": float(prediction.point[index]),
            "interval_lower": float(prediction.interval_lower[index]),
            "interval_upper": float(prediction.interval_upper[index]),
            "calibrated_uncertainty": float(prediction.calibrated_uncertainty[index]),
            "support_distance": float(support[index]),
            "measured_yield": None,
            "practical_accessibility": "unknown",
            "prospective_validation_performed": False,
        }
        for strategy, mapping in rank_of.items():
            row[f"rank_{strategy}"] = int(mapping[record["candidate_id"]])
        row["consensus_rank_sum"] = int(
            sum(mapping[record["candidate_id"]] for mapping in rank_of.values())
        )
        rows.append(row)
    for record in controls.to_dict("records"):
        row = {
            "candidate_id": record["candidate_id"],
            "candidate_class": record["control_class"],
            "canonical_reaction_key": record["canonical_reaction_key"],
            "canonical_reaction_hash": record["canonical_reaction_hash"],
            "canonical_substrate_key": None,
            "canonical_condition_key": None,
            "measured_in_canonical_dataset": True,
            "introduces_new_molecule": False,
            "changed_role_count": 0,
            "predicted_yield": float(record["predicted_yield"]),
            "interval_lower": float(record["interval_lower"]),
            "interval_upper": float(record["interval_upper"]),
            "calibrated_uncertainty": float(record["calibrated_uncertainty"]),
            "support_distance": float(record["support_distance"]),
            "measured_yield": float(record["measured_yield"]),
            "practical_accessibility": "unknown",
            "prospective_validation_performed": False,
        }
        for strategy in rank_of:
            row[f"rank_{strategy}"] = None
        row["consensus_rank_sum"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _prediction_table(
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    prediction: Any,
    estimator_audit_hash: str,
    contract: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(candidates.to_dict("records")):
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "candidate_class": CANDIDATE_CLASS_DISCOVERY,
                "uncertainty_method": contract["uncertainty_method"],
                "nominal_coverage": float(contract["coverage"]),
                "predicted_yield": float(prediction.point[index]),
                "interval_lower": float(prediction.interval_lower[index]),
                "interval_upper": float(prediction.interval_upper[index]),
                "interval_half_width": float(prediction.calibrated_uncertainty[index]),
                "interval_lower_clipped_to_recorded_range": float(
                    min(max(prediction.interval_lower[index], 0.0), 100.0)
                ),
                "interval_upper_clipped_to_recorded_range": float(
                    min(max(prediction.interval_upper[index], 0.0), 100.0)
                ),
                "interval_exceeds_recorded_yield_range": bool(
                    prediction.interval_lower[index] < 0.0
                    or prediction.interval_upper[index] > 100.0
                ),
                "measured_yield": None,
                "estimator_audit_hash": estimator_audit_hash,
                "prediction_hash": prediction.prediction_hash,
            }
        )
    for record in controls.to_dict("records"):
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "candidate_class": record["control_class"],
                "uncertainty_method": contract["uncertainty_method"],
                "nominal_coverage": float(contract["coverage"]),
                "predicted_yield": float(record["predicted_yield"]),
                "interval_lower": float(record["interval_lower"]),
                "interval_upper": float(record["interval_upper"]),
                "interval_half_width": float(record["calibrated_uncertainty"]),
                "interval_lower_clipped_to_recorded_range": float(
                    min(max(record["interval_lower"], 0.0), 100.0)
                ),
                "interval_upper_clipped_to_recorded_range": float(
                    min(max(record["interval_upper"], 0.0), 100.0)
                ),
                "interval_exceeds_recorded_yield_range": bool(
                    record["interval_lower"] < 0.0 or record["interval_upper"] > 100.0
                ),
                "measured_yield": float(record["measured_yield"]),
                "estimator_audit_hash": estimator_audit_hash,
                "prediction_hash": prediction.prediction_hash,
            }
        )
    return pd.DataFrame(rows)


def _provenance_table(
    candidates: pd.DataFrame, controls: pd.DataFrame, measured: pd.DataFrame
) -> pd.DataFrame:
    """Record which measured rows supply every role value of every candidate."""
    substrate_groups = measured.groupby("canonical_substrate_key")["source_row_id"]
    condition_groups = measured.groupby("canonical_condition_key")["source_row_id"]
    role_groups = {
        role: measured.groupby(f"canonical_{role}_smiles")["source_row_id"]
        for role in CANONICAL_ROLE_NAMES
    }
    rows: list[dict[str, Any]] = []
    for record in candidates.to_dict("records"):
        substrate_rows = sorted(substrate_groups.get_group(record["canonical_substrate_key"]))
        condition_rows = sorted(condition_groups.get_group(record["canonical_condition_key"]))
        for role in CANONICAL_ROLE_NAMES:
            donors = sorted(role_groups[role].get_group(record[f"canonical_{role}_smiles"]))
            group = "substrate" if role in _SUBSTRATE_ROLES else "condition"
            block_rows = substrate_rows if group == "substrate" else condition_rows
            rows.append(
                {
                    "candidate_id": record["candidate_id"],
                    "candidate_class": CANDIDATE_CLASS_DISCOVERY,
                    "role": role,
                    "role_group": group,
                    "canonical_role_smiles": record[f"canonical_{role}_smiles"],
                    "recorded_role_smiles": record[f"recovered_{role}_smiles"],
                    "measured_rows_with_this_role_value": len(donors),
                    "exemplar_donor_source_row_id": donors[0],
                    "block_donor_source_row_id": block_rows[0],
                    "measured_rows_in_donor_block": len(block_rows),
                    "donor_source_row_id_hash": stable_hash(donors),
                    "role_value_observed_in_measured_data": True,
                }
            )
    for record in controls.to_dict("records"):
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "candidate_class": record["control_class"],
                "role": "all_seven_roles",
                "role_group": "measured_row",
                "canonical_role_smiles": record["canonical_reaction_key"],
                "recorded_role_smiles": record["canonical_reaction_key"],
                "measured_rows_with_this_role_value": 1,
                "exemplar_donor_source_row_id": record["source_row_id"],
                "block_donor_source_row_id": record["source_row_id"],
                "measured_rows_in_donor_block": 1,
                "donor_source_row_id_hash": stable_hash([record["source_row_id"]]),
                "role_value_observed_in_measured_data": True,
            }
        )
    return pd.DataFrame(rows)


def _support_table(
    candidates: pd.DataFrame,
    controls: pd.DataFrame,
    candidate_support: np.ndarray,
    nearest_fit: Sequence[str],
    fit_count: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(candidates.to_dict("records")):
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "candidate_class": CANDIDATE_CLASS_DISCOVERY,
                "support_distance": float(candidate_support[index]),
                "nearest_support_similarity": float(1.0 - candidate_support[index]),
                "nearest_support_source_row_id": str(nearest_fit[index]),
                "support_row_count": int(fit_count),
                "support_definition": (
                    "one minus the maximum cosine similarity between the candidate "
                    "role-separated fingerprint and any reaction used to fit the model"
                ),
            }
        )
    for record in controls.to_dict("records"):
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "candidate_class": record["control_class"],
                "support_distance": float(record["support_distance"]),
                "nearest_support_similarity": float(1.0 - record["support_distance"]),
                "nearest_support_source_row_id": None,
                "support_row_count": int(fit_count),
                "support_definition": (
                    "one minus the maximum cosine similarity between the control "
                    "role-separated fingerprint and any reaction used to fit the model"
                ),
            }
        )
    return pd.DataFrame(rows)


def _diversity_table(
    candidates: pd.DataFrame,
    measured: pd.DataFrame,
    prepared: Mapping[str, Any],
    candidate_support: np.ndarray,
) -> pd.DataFrame:
    """Explain what each candidate adds relative to the measured inventory."""
    pairwise = _cosine_similarity(
        prepared["candidate_features"], prepared["candidate_features"]
    )
    np.fill_diagonal(pairwise, 0.0)
    substrate_counts = measured["canonical_substrate_key"].value_counts()
    condition_counts = measured["canonical_condition_key"].value_counts()
    role_counts = {
        role: measured[f"canonical_{role}_smiles"].value_counts()
        for role in _CONDITION_ROLES
    }
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(candidates.to_dict("records")):
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "measured_rows_sharing_substrate": int(
                    substrate_counts[record["canonical_substrate_key"]]
                ),
                "measured_rows_sharing_condition_block": int(
                    condition_counts[record["canonical_condition_key"]]
                ),
                **{
                    f"measured_rows_sharing_{role}": int(
                        role_counts[role][record[f"canonical_{role}_smiles"]]
                    )
                    for role in _CONDITION_ROLES
                },
                "support_distance": float(candidate_support[index]),
                "max_similarity_to_other_candidates": float(pairwise[index].max())
                if len(candidates) > 1
                else 0.0,
                "novelty_basis": (
                    "the exact seven-role canonical reaction is absent from the measured "
                    "dataset while every individual role value is measured many times"
                ),
                "diversity_rationale": (
                    "Selecting this reaction extends the measured design to an untested "
                    "substrate/condition cell without introducing any molecule outside "
                    "the measured inventory."
                ),
            }
        )
    return pd.DataFrame(rows)


def _uncertainty_rationale(
    *,
    contract: Mapping[str, Any],
    estimator_audit_hash: str,
    candidate_prediction_hash: str,
    plan: Mapping[str, Any],
    control_metrics: Any,
    candidate_uncertainty: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": PROSPECTIVE_PACKAGE_SCHEMA_VERSION,
        "prospective_disclaimer": PROSPECTIVE_DISCLAIMER,
        "uncertainty_method": contract["uncertainty_method"],
        "surviving_uncertainty_methods": list(PHASE12_SURVIVING_UNCERTAINTY_METHODS),
        "why_this_estimator": (
            "Phase 12 froze thirty calibrated uncertainty policies. Only "
            "bootstrap_extra_trees and heterogeneous_disagreement passed calibrated "
            "selection, and bootstrap_extra_trees won twenty-six of thirty. Phase 18 "
            "used the same estimator, so its intervals are the only calibrated "
            "intervals this repository has evidence for."
        ),
        "interval_construction": (
            "Split-conformal scaled-residual intervals. The estimator is fitted on the "
            "model-fit partition only; the conformal quantile is taken on a disjoint "
            "calibration partition of measured rows."
        ),
        "nominal_coverage": float(contract["coverage"]),
        "fit_row_count": int(plan["partition"]["fit_count"]),
        "calibration_row_count": int(plan["partition"]["calibration_count"]),
        "control_row_count": int(control_metrics.sample_count),
        "control_empirical_coverage": float(control_metrics.empirical_coverage),
        "control_calibration_error": float(control_metrics.calibration_error),
        "control_mean_interval_width": float(control_metrics.mean_interval_width),
        "control_uncertainty_error_spearman": (
            None
            if control_metrics.uncertainty_error_spearman is None
            else float(control_metrics.uncertainty_error_spearman)
        ),
        "candidate_mean_interval_half_width": float(candidate_uncertainty.mean()),
        "candidate_max_interval_half_width": float(candidate_uncertainty.max()),
        "known_limitations": [
            "Coverage is marginal over the calibration partition, not conditional on a "
            "single candidate.",
            "Phase 18 observed empirical coverage below the nominal level on batches "
            "chosen by exploitative acquisition rules, because a selected batch is not "
            "an exchangeable sample.",
            "Intervals describe recorded yield in the source dataset's measurement "
            "protocol. They do not describe another laboratory's measurement process.",
            "Conformal intervals are reported unclipped so the coverage guarantee is "
            "preserved; a bound outside 0-100 percent is a statement about the interval, "
            "not a claim that such a yield is attainable. Clipped bounds are reported "
            "separately in candidate_predictions.csv.",
            "Practical accessibility of every proposed reaction is unknown and is not "
            "modelled here.",
        ],
        "estimator_audit_hash": estimator_audit_hash,
        "candidate_prediction_hash": candidate_prediction_hash,
        "recommendation_simulation_directory": contract["simulation_directory"],
        "recommendation_simulation_manifest_hash": contract["simulation_manifest_hash"],
    }


def _readme(
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
    plan: Mapping[str, Any],
    tables: Mapping[str, Any],
) -> str:
    space = plan["candidate_space"]
    candidates = tables["candidates"]
    discovery = candidates.loc[candidates["candidate_class"].eq(CANDIDATE_CLASS_DISCOVERY)]
    controls = tables["controls"]
    lines = [
        "# Phase 18 prospective-experiment package",
        "",
        f"**{PROSPECTIVE_DISCLAIMER}**",
        "",
        "This package proposes experiments for review. It is a retrospective-model "
        "proposal only. Nothing in it has been run in a laboratory.",
        "",
        "## What this package does and does not claim",
        "",
        "- It does **not** claim synthesis feasibility, safety, accessibility, cost, or "
        "likelihood of laboratory success for any proposed reaction.",
        "- Practical accessibility of every proposal is reported as `unknown`.",
        "- Predicted yields describe the recorded yield of the source dataset's "
        "measurement protocol, not a yield another laboratory would obtain.",
        f"- {PROSPECTIVE_DISCLAIMER}",
        "",
        "## Candidate space",
        "",
        f"- Measured reactions available: {space['already_measured']}",
        f"- Measured substrate groups: {space['measured_substrate_groups']}",
        f"- Measured condition groups: {space['measured_condition_groups']}",
        f"- Complete recombination grid: {space['complete_recombination_grid']}",
        f"- **Unmeasured candidates that survive the eligibility contract: "
        f"{space['unmeasured_candidates']}**",
        "",
        "Candidates are formed only by combining canonical seven-role values that already "
        "occur in the measured dataset, and only where the exact seven-role canonical "
        "reaction is absent from that dataset. No molecule is invented and no eligibility "
        "rule is relaxed. Identity is the RDKit canonical seven-role reaction key; "
        "fingerprint equality is never treated as chemical identity.",
        "",
        "This dataset is a near-complete factorial design, so the eligible unmeasured "
        "space is very small. That is a property of the data, not a modelling choice. "
        "Producing a longer candidate list would require inventing chemistry outside the "
        "measured inventory, which this package deliberately does not do.",
        "",
        "## Contents",
        "",
        "| File | Purpose |",
        "| --- | --- |",
        "| `candidates.csv` | Proposed experiments and controls with predictions, "
        "intervals, support distance and per-policy ranks |",
        "| `candidate_roles.csv` | All seven canonical roles, explicit, for every row |",
        "| `candidate_provenance.csv` | Measured rows supplying every role value |",
        "| `candidate_predictions.csv` | Calibrated predictions and interval bounds |",
        "| `controls.csv` | Already-measured anchors held out of model fitting |",
        "| `support_distances.csv` | Distance from the model's support |",
        "| `diversity_rationale.csv` | What each candidate adds to the measured design |",
        "| `uncertainty_rationale.json` | Estimator choice, calibration and limitations |",
        "| `experimental_protocol.md` | Protocol draft and what the data cannot supply |",
        "| `package_plan.json`, `manifest.json` | Frozen plan and hash-verifiable manifest |",
        "",
        "## Proposed experiments",
        "",
        f"- Discovery candidates: {len(discovery)}",
        f"- Controls (already measured, held out of model fitting): {len(controls)}",
        f"  - replicate anchors spanning the recorded yield range: "
        f"{int((controls['control_class'] == CONTROL_CLASS_REPLICATE).sum())}",
        f"  - highest-predicted anchors: "
        f"{int((controls['control_class'] == CONTROL_CLASS_MODEL_HIGH).sum())}",
        f"  - lowest-predicted anchors: "
        f"{int((controls['control_class'] == CONTROL_CLASS_MODEL_LOW).sum())}",
        "",
        "Controls exist so a reviewer can detect systematic bias. If the re-measured "
        "yields of the replicate anchors are offset from the recorded yields, the "
        "predictions for the discovery candidates inherit that offset.",
        "",
        "## Model and uncertainty",
        "",
        f"- Estimator: `{contract['uncertainty_method']}` (mean of a bootstrap "
        "extremely-randomised-tree ensemble for the point prediction).",
        "- Chosen because Phase 12 calibrated selection retained only "
        "`bootstrap_extra_trees` and `heterogeneous_disagreement`, and "
        "`bootstrap_extra_trees` won 26 of 30 frozen policies. Phase 18 used the same "
        "estimator, so its intervals are the only calibrated intervals with supporting "
        "evidence in this repository.",
        "- The supervised autoencoder is **not** used. Its Phase 14 status is undecided.",
        f"- Nominal interval coverage: {contract['coverage']}. Observed coverage on the "
        f"held-out controls: "
        f"{tables['uncertainty_rationale']['control_empirical_coverage']:.3f}.",
        "",
        "## Provenance",
        "",
        f"- Dataset hash: `{prepared['dataset_hash']}`",
        f"- Plan hash: `{plan['plan_hash']}`",
        f"- Estimator audit hash: `{tables['estimator_audit_hash']}`",
        f"- Retrospective evidence: `{contract['simulation_directory']}` "
        f"(manifest hash `{contract['simulation_manifest_hash']}`)",
        "",
        f"**{PROSPECTIVE_DISCLAIMER}**",
        "",
    ]
    return "\n".join(lines)


def _protocol(
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
    tables: Mapping[str, Any],
) -> str:
    candidates = tables["candidates"]
    discovery = candidates.loc[candidates["candidate_class"].eq(CANDIDATE_CLASS_DISCOVERY)]
    lines = [
        "# Experimental protocol draft",
        "",
        f"**{PROSPECTIVE_DISCLAIMER}**",
        "",
        "This is a draft for review by a chemist. It records only what the canonical "
        "dataset actually supports. It is not an authorisation to run anything, and it "
        "makes no safety, feasibility, accessibility or cost claim.",
        "",
        "## Reaction definition",
        "",
        "Each proposed experiment is defined by seven explicit canonical roles: "
        + ", ".join(f"`{role}`" for role in CANONICAL_ROLE_NAMES)
        + ". The exact strings are in `candidate_roles.csv`; `canonical_*` columns are "
        "RDKit canonical SMILES and `recorded_*` columns are the strings as recorded in "
        "the source dataset.",
        "",
        "## Suggested run order",
        "",
        "1. Run the control anchors in `controls.csv` first. Compare the re-measured "
        "yields with the `measured_yield` column to establish whether this laboratory "
        "reproduces the source dataset's recorded values.",
        "2. Only if the controls reproduce within an acceptable offset, run the "
        f"{len(discovery)} discovery candidates in `candidates.csv`.",
        "3. Record every outcome, including failures, and retain them.",
        "",
        "## What the canonical dataset does NOT supply",
        "",
        "The following are **not** recoverable from `"
        f"{Path(contract['dataset_path']).name}` and must be taken from the original "
        "publication or decided by the reviewing chemist:",
        "",
        "- reaction temperature (recorded as `UNKNOWN` / `NOT_RECOVERABLE` for every row)",
        "- reaction time",
        "- stoichiometry, loadings and concentrations",
        "- solvent volume and reaction scale",
        "- atmosphere, vessel, and work-up",
        "- analytical method used to determine yield",
        "",
        "No value for any of the above is invented here. Because they are unknown, the "
        "predictions in this package are conditional on the source dataset's own "
        "unstated protocol.",
        "",
        "## Measurement comparability",
        "",
        f"- Recorded yields in the source data span "
        f"{prepared['measured']['yield'].min():.2f} to "
        f"{prepared['measured']['yield'].max():.2f}.",
        "- Predictions are for the same recorded quantity. A different analytical method "
        "will not be comparable.",
        "",
        "## Accessibility",
        "",
        "Every role value used by every candidate already appears in the measured "
        "dataset, so no molecule outside that inventory is required. Whether any of "
        "those materials is practically available to the reviewing laboratory is "
        "**unknown** and is not assessed here.",
        "",
        f"**{PROSPECTIVE_DISCLAIMER}**",
        "",
    ]
    return "\n".join(lines)


def _build_manifest(
    contract: Mapping[str, Any],
    prepared: Mapping[str, Any],
    plan: Mapping[str, Any],
    tables: Mapping[str, Any],
    config: Any,
    output: Path,
) -> dict[str, Any]:
    candidates = tables["candidates"]
    discovery = candidates.loc[candidates["candidate_class"].eq(CANDIDATE_CLASS_DISCOVERY)]
    manifest = {
        "schema_version": PROSPECTIVE_PACKAGE_SCHEMA_VERSION,
        "status": "prospective_package_complete",
        "prospective_disclaimer": PROSPECTIVE_DISCLAIMER,
        "git_commit": _git_commit(),
        "git_dirty_at_execution": _git_dirty(),
        "dataset_hash": prepared["dataset_hash"],
        "feature_metadata_hash": prepared["feature_contract"]["feature_metadata_hash"],
        "config_hash": plan["config_hash"],
        "plan_hash": plan["plan_hash"],
        "resolved_scientific_config": plan["resolved_scientific_config"],
        "dependency_versions": _dependency_versions(),
        "command": (
            "python -B scripts/build_prospective_package.py "
            f"--config {config} --output-directory {output}"
        ),
        "measured_row_count": len(prepared["measured"]),
        "discovery_candidate_count": len(discovery),
        "control_count": len(tables["controls"]),
        "candidate_row_count": len(candidates),
        "provenance_row_count": len(tables["candidate_provenance"]),
        "diversity_rationale_row_count": len(tables["diversity_rationale"]),
        "support_distance_row_count": len(tables["support_distances"]),
        "uncertainty_method": contract["uncertainty_method"],
        "surviving_uncertainty_methods": list(PHASE12_SURVIVING_UNCERTAINTY_METHODS),
        "nominal_coverage": float(contract["coverage"]),
        "control_empirical_coverage": _finite_or_none(
            float(tables["control_metrics"].empirical_coverage)
        ),
        "estimator_audit_hash": tables["estimator_audit_hash"],
        "candidate_prediction_hash": tables["candidate_prediction_hash"],
        "recommendation_simulation_directory": contract["simulation_directory"],
        "recommendation_simulation_manifest_hash": contract["simulation_manifest_hash"],
        "supervised_autoencoder_used": False,
        "prospective_validation_performed": False,
        "synthesis_feasibility_claimed": False,
        "safety_claimed": False,
        "accessibility_claimed": False,
        "output_hashes": {name: sha256_file(output / name) for name in _OUTPUTS},
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    return manifest


def _scientific_config(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_path": str(contract["dataset_path"]),
        "feature_config": dict(contract["feature_config"]),
        "uncertainty_method": contract["uncertainty_method"],
        "coverage": contract["coverage"],
        "calibration_fraction": contract["calibration_fraction"],
        "control_pool_fraction": contract["control_pool_fraction"],
        "estimator_params": dict(contract["estimator_params"]),
        "measured_replicate_anchors": contract["measured_replicate_anchors"],
        "model_high_anchors": contract["model_high_anchors"],
        "model_low_anchors": contract["model_low_anchors"],
        "ucb_beta": contract["ucb_beta"],
        "expected_improvement_xi": contract["expected_improvement_xi"],
        "diversity_penalty": contract["diversity_penalty"],
        "support_distance_yield_weight": contract["support_distance_yield_weight"],
        "simulation_directory": contract["simulation_directory"],
        "simulation_manifest_hash": contract["simulation_manifest_hash"],
        "base_seed": contract["base_seed"],
    }


def _resolve_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        raw,
        {
            "dataset",
            "features",
            "model",
            "controls",
            "strategies",
            "evidence",
            "base_seed",
            "output",
        },
        "config",
    )
    dataset = _mapping(raw["dataset"], "dataset")
    _exact_keys(dataset, {"path"}, "dataset")
    model = _mapping(raw["model"], "model")
    _exact_keys(
        model,
        {
            "uncertainty_method",
            "coverage",
            "calibration_fraction",
            "control_pool_fraction",
            "estimator_params",
        },
        "model",
    )
    controls = _mapping(raw["controls"], "controls")
    _exact_keys(
        controls,
        {"measured_replicate_anchors", "model_high_anchors", "model_low_anchors"},
        "controls",
    )
    strategies = _mapping(raw["strategies"], "strategies")
    _exact_keys(
        strategies,
        {
            "ucb_beta",
            "expected_improvement_xi",
            "diversity_penalty",
            "support_distance_yield_weight",
        },
        "strategies",
    )
    evidence = _mapping(raw["evidence"], "evidence")
    _exact_keys(
        evidence,
        {"recommendation_simulation_directory", "recommendation_simulation_manifest_hash"},
        "evidence",
    )
    output = _mapping(raw["output"], "output")
    _exact_keys(output, {"directory"}, "output")
    method = str(model["uncertainty_method"])
    if method not in PHASE12_SURVIVING_UNCERTAINTY_METHODS:
        raise ValueError(
            "model.uncertainty_method must be an estimator that survived Phase 12 "
            f"calibrated selection: {list(PHASE12_SURVIVING_UNCERTAINTY_METHODS)}."
        )
    return {
        "dataset_path": Path(str(dataset["path"])),
        "feature_config": resolve_corrected_feature_config(
            _mapping(raw["features"], "features"), required_kind="bh_role_separated"
        ),
        "uncertainty_method": method,
        "coverage": _open_unit(model["coverage"], "model.coverage"),
        "calibration_fraction": _open_unit(
            model["calibration_fraction"], "model.calibration_fraction"
        ),
        "control_pool_fraction": _open_unit(
            model["control_pool_fraction"], "model.control_pool_fraction"
        ),
        "estimator_params": _estimator_params(model["estimator_params"], method),
        "measured_replicate_anchors": _positive_int(
            controls["measured_replicate_anchors"], "controls.measured_replicate_anchors"
        ),
        "model_high_anchors": _positive_int(
            controls["model_high_anchors"], "controls.model_high_anchors"
        ),
        "model_low_anchors": _positive_int(
            controls["model_low_anchors"], "controls.model_low_anchors"
        ),
        "ucb_beta": _nonnegative_float(strategies["ucb_beta"], "strategies.ucb_beta"),
        "expected_improvement_xi": _nonnegative_float(
            strategies["expected_improvement_xi"], "strategies.expected_improvement_xi"
        ),
        "diversity_penalty": _nonnegative_float(
            strategies["diversity_penalty"], "strategies.diversity_penalty"
        ),
        "support_distance_yield_weight": _nonnegative_float(
            strategies["support_distance_yield_weight"],
            "strategies.support_distance_yield_weight",
        ),
        "simulation_directory": str(evidence["recommendation_simulation_directory"]),
        "simulation_manifest_hash": str(
            evidence["recommendation_simulation_manifest_hash"]
        ),
        "base_seed": _int(raw["base_seed"], "base_seed"),
        "output_directory": Path(str(output["directory"])),
    }


def _contract_from_scientific(scientific: Mapping[str, Any], output: Path) -> dict[str, Any]:
    return {
        "dataset_path": Path(scientific["dataset_path"]),
        "feature_config": dict(scientific["feature_config"]),
        "uncertainty_method": str(scientific["uncertainty_method"]),
        "coverage": float(scientific["coverage"]),
        "calibration_fraction": float(scientific["calibration_fraction"]),
        "control_pool_fraction": float(scientific["control_pool_fraction"]),
        "estimator_params": dict(scientific["estimator_params"]),
        "measured_replicate_anchors": int(scientific["measured_replicate_anchors"]),
        "model_high_anchors": int(scientific["model_high_anchors"]),
        "model_low_anchors": int(scientific["model_low_anchors"]),
        "ucb_beta": float(scientific["ucb_beta"]),
        "expected_improvement_xi": float(scientific["expected_improvement_xi"]),
        "diversity_penalty": float(scientific["diversity_penalty"]),
        "support_distance_yield_weight": float(scientific["support_distance_yield_weight"]),
        "simulation_directory": str(scientific["simulation_directory"]),
        "simulation_manifest_hash": str(scientific["simulation_manifest_hash"]),
        "base_seed": int(scientific["base_seed"]),
        "output_directory": output,
    }


def summarize_prospective_package(directory: str | Path) -> dict[str, Any]:
    """Return a short reviewer-facing summary of an existing package."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    candidates = pd.read_csv(root / "candidates.csv", float_precision="round_trip")
    discovery = candidates.loc[candidates["candidate_class"].eq(CANDIDATE_CLASS_DISCOVERY)]
    return {
        "manifest_hash": manifest["manifest_hash"],
        "prospective_disclaimer": manifest["prospective_disclaimer"],
        "discovery_candidate_count": int(manifest["discovery_candidate_count"]),
        "control_count": int(manifest["control_count"]),
        "predicted_yield_range": [
            _finite_or_none(float(discovery["predicted_yield"].min())),
            _finite_or_none(float(discovery["predicted_yield"].max())),
        ],
        "support_distance_range": [
            _finite_or_none(float(discovery["support_distance"].min())),
            _finite_or_none(float(discovery["support_distance"].max())),
        ],
        "control_empirical_coverage": manifest["control_empirical_coverage"],
    }


