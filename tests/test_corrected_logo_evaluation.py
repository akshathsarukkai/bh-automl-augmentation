from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import bh_augmentation.logo_evaluation as logo_evaluation
from bh_augmentation.logo_evaluation import run_corrected_logo
from bh_augmentation.results.status import InvalidResultError
from bh_augmentation.utils.corrected_runs import stable_hash
from corrected_test_utils import corrected_dataset, write_corrected_config


def _logo_config(tmp_path: Path, *, output_name: str = "corrected_logo") -> tuple[Path, dict]:
    base_path = write_corrected_config(
        tmp_path,
        kind="anonymous",
        output_name="unused_corrected_output",
    )
    config = yaml.safe_load(base_path.read_text())
    config["splits"] = {
        "method": "leave_one_group_out",
        "canonical_dependency_directory": config["splits"]["directory"],
        "targets": ["product_key", "reactant_key"],
    }
    config["features"]["kind"] = "bh_role_separated"
    config["models"] = [{"name": "ridge", "params": {"alpha": 1.0}}]
    config["metrics"] = ["rmse", "mae"]
    config["output"] = {"directory": str(tmp_path / output_name)}
    config.pop("condition_transfer")
    config_path = tmp_path / f"{output_name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config_path, config


def test_corrected_logo_runs_both_canonical_targets_and_writes_complete_audits(
    tmp_path: Path,
) -> None:
    config_path, config = _logo_config(tmp_path)

    paths = run_corrected_logo(config_path)

    assert paths["output_directory"] == Path(config["output"]["directory"])
    completion = json.loads(paths["completion_manifest"].read_text())
    assert completion["status"] == "complete"
    assert completion["expected_targets"] == ["product_key", "reactant_key"]
    assert completion["observed_targets"] == ["product_key", "reactant_key"]
    completion_payload = dict(completion)
    completion_hash = completion_payload.pop("completion_payload_hash")
    assert stable_hash(completion_payload) == completion_hash
    for target, canonical_column in (
        ("product_key", "canonical_product_key"),
        ("reactant_key", "canonical_substrate_key"),
    ):
        directory = paths[f"{target}_directory"]
        expected_names = {
            "fold_assignments.csv",
            "group_sizes.csv",
            "overlap_audit.csv",
            "per_fold_metrics.csv",
            "summary_metrics.csv",
            "manifest.json",
        }
        assert {path.name for path in directory.iterdir()} == expected_names
        assignments = pd.read_csv(directory / "fold_assignments.csv")
        group_sizes = pd.read_csv(directory / "group_sizes.csv")
        overlap = pd.read_csv(directory / "overlap_audit.csv")
        metrics = pd.read_csv(directory / "per_fold_metrics.csv")
        manifest = json.loads((directory / "manifest.json").read_text())

        heldout = assignments.loc[assignments["outer_split"].eq("test")]
        observed_test_groups = heldout.groupby("fold_index")["fold_group"].first().tolist()
        assert observed_test_groups == manifest["heldout_groups"]
        assert len(observed_test_groups) == assignments["group_value"].nunique()
        assert group_sizes["fold_group"].tolist() == observed_test_groups
        assert overlap["all_overlaps_zero"].all()
        assert (overlap.filter(like="overlap_count") == 0).all().all()
        assert assignments["split_method"].eq("leave_one_group_out").all()
        assert assignments["group_column"].eq(canonical_column).all()
        assert metrics["test_evaluation_count"].eq(1).all()
        assert not metrics["test_used_for_policy_selection"].any()
        assert manifest["split_method"] == "leave_one_group_out"
        assert manifest["group_column"] == canonical_column
        assert manifest["expected_group_count"] == manifest["observed_fold_count"]
        assert manifest["all_train_test_overlaps_zero"]
        assert manifest["test_evaluation_count_per_fold_method"] == 1
        assert not manifest["test_used_for_policy_selection"]
        assert manifest["canonical_split_dependency"]["aggregate_split_hash"]
        assert manifest["canonical_split_dependency"]["canonicalization_version"]
        assert manifest["canonical_split_dependency"]["split_schema_version"]
        assert manifest["feature_metadata_hash"]
        assert set(manifest["output_hashes"]) == expected_names - {"manifest.json"}
        assert len(metrics) == len(observed_test_groups) * 2
        completion_target = completion["target_outputs"][target]
        assert completion_target["fold_count"] == len(observed_test_groups)
        assert (
            completion_target["manifest_hash"]
            == hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        )
        for name, expected_hash in completion_target["artifact_hashes"].items():
            assert (
                hashlib.sha256((directory / name).read_bytes()).hexdigest()
                == expected_hash
            )


def test_corrected_logo_rejects_tampered_saved_split_dependency(tmp_path: Path) -> None:
    config_path, config = _logo_config(tmp_path, output_name="corrected_logo_tampered")
    split_path = (
        Path(config["splits"]["canonical_dependency_directory"])
        / "outer_split_assignments.csv"
    )
    assignments = pd.read_csv(split_path)
    assignments.loc[0, "outer_split"] = (
        "test" if assignments.loc[0, "outer_split"] != "test" else "train"
    )
    assignments.to_csv(split_path, index=False)

    with pytest.raises(ValueError, match="split hash mismatch|assignment"):
        run_corrected_logo(config_path)

    assert not Path(config["output"]["directory"]).exists()


def test_corrected_logo_rejects_random_split_mislabeled_as_logo(tmp_path: Path) -> None:
    config_path, config = _logo_config(tmp_path, output_name="corrected_logo_random")
    config["splits"]["method"] = "random"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="leave_one_group_out"):
        run_corrected_logo(config_path)

    assert not Path(config["output"]["directory"]).exists()


def test_corrected_logo_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    config_path, config = _logo_config(tmp_path, output_name="corrected_logo_existing")
    output = Path(config["output"]["directory"])
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_corrected_logo(config_path)


def test_corrected_logo_accepts_a_fresh_output_directory_override(tmp_path: Path) -> None:
    config_path, _ = _logo_config(tmp_path, output_name="corrected_logo_config_default")
    override = (
        tmp_path
        / "results"
        / "autonomous_execution"
        / "phase_07"
        / "corrected_logo_override"
    )

    paths = run_corrected_logo(config_path, output_directory=override)

    assert paths["output_directory"] == override
    assert json.loads(paths["product_key_manifest"].read_text())[
        "output_directory"
    ] == str(override)


def test_corrected_logo_rejects_output_under_invalid_historical_logo_family(
    tmp_path: Path,
) -> None:
    config_path, _ = _logo_config(tmp_path, output_name="corrected_logo_safe_default")
    invalid = tmp_path / "results" / "stress" / "logo_product" / "new_run"

    with pytest.raises(InvalidResultError, match="historical_logo_product"):
        run_corrected_logo(config_path, output_directory=invalid)

    assert not invalid.exists()


def test_corrected_logo_uses_distinct_estimator_returned_by_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _logo_config(tmp_path, output_name="corrected_logo_returned_model")
    returned_estimators: list[object] = []

    def fake_get_model(*args: object, **kwargs: object) -> object:
        return object()

    def fake_train_model(
        estimator: object,
        X_train: object,
        y_train: object,
    ) -> object:
        fitted = object()
        assert estimator is not fitted
        returned_estimators.append(fitted)
        return fitted

    def fake_predict_model(estimator: object, X_test: object) -> object:
        assert estimator in returned_estimators
        return pd.Series(0.0, index=range(len(X_test))).to_numpy()

    monkeypatch.setattr(logo_evaluation, "get_model", fake_get_model)
    monkeypatch.setattr(logo_evaluation, "train_model", fake_train_model)
    monkeypatch.setattr(logo_evaluation, "predict_model", fake_predict_model)

    paths = run_corrected_logo(config_path)

    assert paths["completion_manifest"].exists()
    assert returned_estimators


def test_second_target_failure_leaves_no_root_completion_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, config = _logo_config(
        tmp_path,
        output_name="corrected_logo_partial_failure",
    )
    original_evaluate = logo_evaluation._evaluate_target
    calls = 0

    def fail_second_target(*args: object, **kwargs: object) -> dict[str, Path]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("deliberate second-target failure")
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(logo_evaluation, "_evaluate_target", fail_second_target)

    with pytest.raises(RuntimeError, match="second-target failure"):
        run_corrected_logo(config_path)

    output = Path(config["output"]["directory"])
    assert (output / "product_key" / "manifest.json").exists()
    assert not (output / "completion_manifest.json").exists()


def test_corrected_logo_rejects_undefined_singleton_r2_with_context(
    tmp_path: Path,
) -> None:
    source = corrected_dataset()
    singleton_product = "COC"
    source.loc[0, "product_key"] = singleton_product
    source.loc[0, "recovered_product_smiles"] = singleton_product
    role_columns = [
        "recovered_reactant_1_smiles",
        "recovered_reactant_2_smiles",
        "recovered_catalyst_smiles",
        "recovered_ligand_smiles",
        "recovered_base_smiles",
        "recovered_solvent_or_additive_smiles",
    ]
    source.loc[0, "reaction_smiles"] = (
        ".".join(str(source.loc[0, column]) for column in role_columns)
        + f">>{singleton_product}"
    )
    source.to_csv(tmp_path / "corrected_source_dataset.csv", index=False)
    config_path, config = _logo_config(tmp_path, output_name="corrected_logo_singleton_r2")
    config["metrics"] = ["r2"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(
        ValueError,
        match=(
            "Non-finite requested LOGO metric.*target='product_key'.*"
            "model='ridge'.*metric='r2'"
        ),
    ):
        run_corrected_logo(config_path)

    assert not (
        Path(config["output"]["directory"]) / "completion_manifest.json"
    ).exists()


def test_corrected_logo_rejects_undefined_constant_prediction_spearman(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, config = _logo_config(
        tmp_path,
        output_name="corrected_logo_constant_spearman",
    )
    config["metrics"] = ["spearman"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    def constant_prediction(estimator: object, X_test: object) -> object:
        return pd.Series(0.0, index=range(len(X_test))).to_numpy()

    monkeypatch.setattr(logo_evaluation, "predict_model", constant_prediction)

    with pytest.raises(
        ValueError,
        match=(
            "Non-finite requested LOGO metric.*target='product_key'.*"
            "model='ridge'.*metric='spearman'"
        ),
    ):
        run_corrected_logo(config_path)

    assert not (
        Path(config["output"]["directory"]) / "completion_manifest.json"
    ).exists()


@pytest.mark.parametrize(
    ("corruption", "expected"),
    [
        ("length", "identical lengths"),
        ("feature_nan", "feature matrix contains non-finite"),
        ("label_nan", "label vector contains non-finite"),
    ],
)
def test_corrected_logo_rejects_invalid_feature_or_label_data_before_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
    expected: str,
) -> None:
    config_path, config = _logo_config(
        tmp_path,
        output_name=f"corrected_logo_invalid_{corruption}",
    )
    original_builder = logo_evaluation.build_feature_matrix_with_metadata

    def corrupt_builder(*args: object, **kwargs: object) -> tuple:
        X, y, names, metadata = original_builder(*args, **kwargs)
        if corruption == "length":
            X = X[:-1]
        elif corruption == "feature_nan":
            X[0, 0] = float("nan")
        else:
            y[0] = float("nan")
        return X, y, names, metadata

    monkeypatch.setattr(
        logo_evaluation,
        "build_feature_matrix_with_metadata",
        corrupt_builder,
    )

    with pytest.raises(ValueError, match=expected):
        run_corrected_logo(config_path)

    assert not Path(config["output"]["directory"]).exists()
