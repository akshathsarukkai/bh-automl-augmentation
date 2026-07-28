"""Rejection tests for the shared strict configuration validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from bh_augmentation.results.status import InvalidResultError
from bh_augmentation.utils.strict_config import (
    OutputDirectoryReuseError,
    StrictConfigError,
    assert_reconstruction_vocabulary_matches_contract,
    load_strict_config,
    require_choice,
    require_dimensions,
    require_exact_keys,
    require_feature_kind,
    require_feature_reconstruction_combination,
    require_fresh_output_directory,
    require_hash_matches,
    require_int,
    require_mapping,
    require_reconstruction_objective,
    require_sha256,
    require_unique_fractions,
    require_unique_ints,
    require_weight,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64


def test_strict_config_errors_remain_value_errors() -> None:
    assert issubclass(StrictConfigError, ValueError)
    assert issubclass(OutputDirectoryReuseError, ValueError)
    assert issubclass(OutputDirectoryReuseError, FileExistsError)


# --- unknown / misspelled / missing fields ---------------------------------


def test_unknown_config_field_is_rejected_with_suggestion() -> None:
    with pytest.raises(StrictConfigError) as excinfo:
        require_exact_keys({"dataset": {}, "featurs": {}}, {"dataset", "features"}, "config")
    message = str(excinfo.value)
    assert "unknown=['featurs']" in message
    assert "'featurs' -> 'features'" in message


def test_missing_config_field_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match=r"missing=\['features'\]"):
        require_exact_keys({"dataset": {}}, {"dataset", "features"}, "config")


def test_non_mapping_section_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="must be a mapping"):
        require_mapping([1, 2, 3], "splits")


def test_non_mapping_yaml_document_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- one\n- two\n")
    with pytest.raises(StrictConfigError, match="must contain a YAML mapping"):
        load_strict_config(path)


# --- malformed / inconsistent hashes ---------------------------------------


@pytest.mark.parametrize("value", ["", "abc", "A" * 64, "z" * 64, 12345, None, "a" * 63])
def test_malformed_hash_is_rejected(value: object) -> None:
    with pytest.raises(StrictConfigError, match="SHA-256 digest"):
        require_sha256(value, "dataset_hash")


def test_inconsistent_hash_pair_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="is inconsistent"):
        require_hash_matches(_HASH_A, _HASH_B, "dataset_hash")
    assert require_hash_matches(_HASH_A, _HASH_A, "dataset_hash") == _HASH_A


# --- feature kind and reconstruction objective combinations -----------------


def test_unsupported_feature_kind_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="unsupported feature kind"):
        require_feature_kind("bh_role_seperated")


def test_deprecated_feature_alias_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="legacy alias"):
        require_feature_kind("role_separated_conditions")


def test_removed_feature_kind_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="removed feature kind"):
        require_feature_kind("fp_concat")


def test_unsupported_reconstruction_objective_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="reconstruction.objective must be one of"):
        require_reconstruction_objective("huber")


def test_incompatible_feature_kind_and_reconstruction_objective_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="is unsupported for features.kind"):
        require_feature_reconstruction_combination("bh_role_separated", "count_aware_mse")
    with pytest.raises(StrictConfigError, match="is unsupported for features.kind"):
        require_feature_reconstruction_combination(
            "bh_role_separated_delta", "binary_cross_entropy"
        )


def test_supported_feature_kind_and_objective_combination_is_accepted() -> None:
    resolved = require_feature_reconstruction_combination(
        "bh_role_separated", "binary_cross_entropy"
    )
    assert resolved == {
        "feature_kind": "bh_role_separated",
        "value_domain": "binary_bit",
        "objective": "binary_cross_entropy",
        "decoder_mode": "bernoulli_logits",
    }


def test_decoder_mode_inconsistent_with_objective_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="requires decoder mode"):
        require_feature_reconstruction_combination(
            "bh_role_separated", "binary_cross_entropy", decoder_mode="identity"
        )
    with pytest.raises(StrictConfigError, match="decoder_mode must be one of"):
        require_feature_reconstruction_combination(
            "bh_role_separated", "mse", decoder_mode="softmax"
        )


# --- weights ----------------------------------------------------------------


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="must be non-negative"):
        require_weight(-0.25, "loss.reconstruction_weight")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weight_is_rejected(value: float) -> None:
    with pytest.raises(StrictConfigError, match="must be finite"):
        require_weight(value, "loss.reconstruction_weight")


def test_weight_outside_declared_choice_set_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="declared weights"):
        require_weight(0.3, "loss.reconstruction_weight", choices=[0.0, 0.5, 1.0])
    assert require_weight(0.5, "loss.reconstruction_weight", choices=[0.0, 0.5, 1.0]) == 0.5


def test_zero_weight_is_rejected_when_strictly_positive_is_required() -> None:
    with pytest.raises(StrictConfigError, match="must be strictly positive"):
        require_weight(0.0, "loss.supervised_weight", allow_zero=False)


def test_non_numeric_weight_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="must be a finite number"):
        require_weight("0.5", "loss.reconstruction_weight")


# --- dimensions -------------------------------------------------------------


@pytest.mark.parametrize("latent", [0, -4])
def test_non_positive_dimension_is_rejected(latent: int) -> None:
    with pytest.raises(StrictConfigError, match="must be positive"):
        require_dimensions(input_width=64, latent_width=latent)


@pytest.mark.parametrize("latent", [8.0, "8", True, None])
def test_non_integer_dimension_is_rejected(latent: object) -> None:
    with pytest.raises(StrictConfigError, match="must be an integer"):
        require_dimensions(input_width=64, latent_width=latent)


def test_latent_dimension_above_input_width_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="must not exceed"):
        require_dimensions(input_width=32, latent_width=64)
    assert require_dimensions(input_width=32, latent_width=32) == (32, 32)


def test_non_integer_scalar_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="must be an integer"):
        require_int(3.0, "base_seed")


# --- lists and choice sets --------------------------------------------------


def test_duplicate_seed_values_are_rejected() -> None:
    with pytest.raises(StrictConfigError, match="must contain unique values"):
        require_unique_ints([0, 1, 1], "splits.random_seeds")


def test_out_of_range_train_fraction_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match=r"must contain values in \(0, 1\]"):
        require_unique_fractions([0.2, 1.5], "splits.random_fractions")
    with pytest.raises(StrictConfigError, match="must be a non-empty list"):
        require_unique_fractions([], "splits.random_fractions")


def test_value_outside_choice_set_is_rejected() -> None:
    with pytest.raises(StrictConfigError, match="must be one of"):
        require_choice("mape", ["rmse", "mae", "r2"], "selection_metric")


# --- output directory reuse -------------------------------------------------


def test_existing_output_directory_is_rejected(tmp_path: Path) -> None:
    existing = tmp_path / "corrected-existing"
    existing.mkdir()
    with pytest.raises(OutputDirectoryReuseError, match="Refusing to reuse or overwrite"):
        require_fresh_output_directory(existing)
    with pytest.raises(FileExistsError):
        require_fresh_output_directory(existing)


def test_existing_empty_output_directory_is_still_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "corrected-empty"
    empty.mkdir()
    assert not any(empty.iterdir())
    with pytest.raises(OutputDirectoryReuseError):
        require_fresh_output_directory(empty)


def test_existing_output_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "corrected-output"
    path.write_text("stale\n")
    with pytest.raises(OutputDirectoryReuseError):
        require_fresh_output_directory(path)


def test_fresh_output_directory_is_accepted_without_creating_it(tmp_path: Path) -> None:
    target = tmp_path / "corrected-fresh"
    assert require_fresh_output_directory(target) == target
    assert not target.exists()


def test_invalid_result_family_output_directory_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "role_aware_condition_transfer_v2"
    with pytest.raises(InvalidResultError):
        require_fresh_output_directory(target)


def test_local_reconstruction_vocabulary_matches_the_matrix_level_contract() -> None:
    pytest.importorskip("bh_augmentation.features.reconstruction_contract")
    assert_reconstruction_vocabulary_matches_contract()


def test_strict_config_imports_without_the_reconstruction_contract_module() -> None:
    """Config validation must not depend on any model implementation module."""
    source = Path("src/bh_augmentation/utils/strict_config.py").read_text()
    header = source.split("STRICT_CONFIG_SCHEMA_VERSION", 1)[0]
    assert "reconstruction_contract" not in header
