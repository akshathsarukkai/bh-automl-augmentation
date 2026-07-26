"""Configuration and construction tests for corrected condition transfer."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from bh_augmentation.augmentation.condition_transfer import (
    build_anonymous_condition_transfer_roles,
)
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    build_role_transferred_reaction_smiles,
)
from bh_augmentation.corrected_condition_transfer import select_corrected_policy_rows
from bh_augmentation.data.reaction_roles import ReactionRoles, reaction_roles_dataframe
from bh_augmentation.features.featurize import DEPRECATED_FEATURE_ALIASES
from bh_augmentation.run_role_aware_condition_transfer import (
    _effective_policy_mode,
    _iter_role_transfer_policies,
)

CORRECTED_CONFIGS = [
    "configs/corrected_representation_baselines_xgboost.yaml",
    "configs/corrected_representation_baselines_tiny.yaml",
    "configs/corrected_anonymous_condition_transfer_xgboost.yaml",
    "configs/corrected_anonymous_condition_transfer_tiny.yaml",
    "configs/corrected_role_aware_condition_transfer_xgboost.yaml",
    "configs/corrected_role_aware_condition_transfer_tiny.yaml",
    "configs/condition_transfer_supervised_ae_xgboost.yaml",
    "configs/condition_transfer_supervised_ae_tiny.yaml",
]

CANONICAL_DATASET_PATH = "data/processed/bh_canonical_roles_v1.csv"
CANONICAL_SPLIT_DIRECTORY = "results/corrected_canonical_splits"


@pytest.mark.parametrize("path", CORRECTED_CONFIGS)
def test_corrected_configs_never_use_legacy_feature_aliases(path: str) -> None:
    config = yaml.safe_load(Path(path).read_text())
    feature_values = [config["features"].get("kind", "")]
    feature_values.extend(config["features"].get("representations", []))
    assert not set(feature_values) & set(DEPRECATED_FEATURE_ALIASES)
    assert config["features"]["fingerprint_backend"] == "rdkit"


@pytest.mark.parametrize("path", CORRECTED_CONFIGS)
def test_corrected_configs_use_saved_canonical_assignments(path: str) -> None:
    config = yaml.safe_load(Path(path).read_text())
    split_manifest = yaml.safe_load(
        Path(CANONICAL_SPLIT_DIRECTORY, "split_manifest.json").read_text()
    )

    assert config["corrected_revalidation"]["enabled"] is True
    assert config["dataset"]["path"] == CANONICAL_DATASET_PATH
    assert config["splits"] == {
        "method": "canonical_saved",
        "directory": CANONICAL_SPLIT_DIRECTORY,
    }
    assert set(config["seeds"]) <= set(split_manifest["seeds"])
    assert set(config["low_data"]["train_fractions"]) <= set(
        split_manifest["train_fractions"]
    )


@pytest.mark.parametrize(
    "path",
    [
        "configs/corrected_anonymous_condition_transfer_xgboost.yaml",
        "configs/corrected_anonymous_condition_transfer_tiny.yaml",
        "configs/corrected_role_aware_condition_transfer_xgboost.yaml",
        "configs/corrected_role_aware_condition_transfer_tiny.yaml",
    ],
)
def test_corrected_transfer_configs_use_role_separated_rdkit(path: str) -> None:
    config = yaml.safe_load(Path(path).read_text())
    assert config["features"]["kind"] == "bh_role_separated"
    assert config["features"]["fingerprint_backend"] == "rdkit"


@pytest.mark.parametrize(
    ("path", "section"),
    [
        (
            "configs/corrected_anonymous_condition_transfer_xgboost.yaml",
            "condition_transfer",
        ),
        (
            "configs/corrected_anonymous_condition_transfer_tiny.yaml",
            "condition_transfer",
        ),
        (
            "configs/corrected_role_aware_condition_transfer_xgboost.yaml",
            "role_aware_condition_transfer",
        ),
        (
            "configs/corrected_role_aware_condition_transfer_tiny.yaml",
            "role_aware_condition_transfer",
        ),
        (
            "configs/condition_transfer_supervised_ae_xgboost.yaml",
            "condition_transfer",
        ),
        (
            "configs/condition_transfer_supervised_ae_tiny.yaml",
            "condition_transfer",
        ),
    ],
)
def test_corrected_transfer_configs_declare_strict_role_change_and_reject_fallback(
    path: str,
    section: str,
) -> None:
    config = yaml.safe_load(Path(path).read_text())

    assert config[section]["role_change_requirement"] == "all"
    assert config[section]["fallback_policy"] == "reject"


@pytest.mark.parametrize(
    "path",
    [
        "configs/condition_transfer_supervised_ae_xgboost.yaml",
        "configs/condition_transfer_supervised_ae_tiny.yaml",
    ],
)
def test_corrected_hybrid_configs_disable_random_full_data_reference(path: str) -> None:
    config = yaml.safe_load(Path(path).read_text())

    assert config["evaluate_full_data_reference"] is False
    assert config["features"]["kind"] == "bh_role_separated"
    assert config["features"]["fingerprint_backend"] == "rdkit"
    assert config["condition_transfer"]["donor_similarity_backend"] == "rdkit"
    assert config["output"]["directory"].startswith(
        "results/corrected_condition_transfer_supervised_ae_"
    )
    for key, value in config["output"].items():
        if key.endswith("_path"):
            assert str(value).startswith(config["output"]["directory"] + "/")


def test_role_aware_policy_factory_propagates_strict_role_change_config() -> None:
    config = {
        "role_aware_condition_transfer": {
            "role_transfer_modes": ["ligand_base"],
            "donor_strategies": ["random"],
            "label_strategies": ["source_label"],
            "synthetic_multipliers": [0.5],
            "role_change_requirement": "all",
            "fallback_policy": "reject",
        }
    }
    role_counts = pd.DataFrame(
        [
            {"role": role, "n_unique": 2, "invariant": False}
            for role in ["catalyst", "ligand", "base", "solvent_or_additive"]
        ]
    )

    policy = _iter_role_transfer_policies(config, seed=7, role_value_counts=role_counts)[
        0
    ]

    assert policy.role_change_requirement == "all"
    assert policy.fallback_policy == "reject"


def test_anonymous_transfer_preserves_substrates_product_and_replaces_conditions() -> None:
    source, donor = _frames()
    roles = build_anonymous_condition_transfer_roles(source.iloc[0], donor.iloc[0])

    assert (roles.reactant_1, roles.reactant_2, roles.product) == (
        "Brc1ccccc1",
        "CN",
        "c1ccccc1NC",
    )
    assert (roles.catalyst, roles.ligand, roles.base, roles.solvent_or_additive) == (
        "[Ni]",
        "P(CC)(CC)CC",
        "K2CO3",
        "CCCO",
    )


def test_role_aware_transfer_changes_only_requested_roles() -> None:
    source, donor = _frames()
    result = build_role_transferred_reaction_smiles(
        source.iloc[0], donor.iloc[0], "ligand_base"
    )

    assert result["changed_ligand"] and result["changed_base"]
    assert not result["changed_catalyst"]
    assert not result["changed_solvent_or_additive"]


def test_zero_synthetic_policy_is_not_selected_and_test_metric_is_ignored() -> None:
    rows = pd.DataFrame(
        [
            _selection_row("zero", "valid", 1.0, 0),
            _selection_row("valid_best", "valid", 2.0, 3),
            _selection_row("test_best", "valid", 3.0, 3),
            _selection_row("valid_best", "test", 99.0, 3),
            _selection_row("test_best", "test", 0.1, 3),
        ]
    )

    selected = select_corrected_policy_rows(rows)

    assert selected["policy_id"].tolist() == ["valid_best"]


def test_invariant_only_role_request_is_excluded() -> None:
    effective, excluded = _effective_policy_mode(
        "catalyst_only", {"catalyst"}, exclude_invariant_roles=True
    )
    assert effective is None
    assert excluded == {"catalyst"}


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = reaction_roles_dataframe(
        [ReactionRoles("Brc1ccccc1", "CN", "[Pd]", "P(C)(C)C", "K3PO4", "CCO", "c1ccccc1NC")],
        yields=[50.0],
    )
    donor = reaction_roles_dataframe(
        [ReactionRoles("Clc1ccccc1", "CCN", "[Ni]", "P(CC)(CC)CC", "K2CO3", "CCCO", "c1ccccc1NCC")],
        yields=[80.0],
    )
    return source, donor


def _selection_row(policy: str, split: str, value: float, n_synthetic: int) -> dict[str, object]:
    return {
        "seed": 0,
        "train_fraction": 0.1,
        "model": "xgboost",
        "policy_id": policy,
        "split": split,
        "metric": "rmse",
        "value": value,
        "n_synthetic_train": n_synthetic,
    }
