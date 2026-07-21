#!/usr/bin/env python
"""Build a deterministic, no-training report for the corrected feature contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from bh_augmentation.augmentation.condition_transfer import build_anonymous_condition_transfer_roles
from bh_augmentation.augmentation.role_aware_condition_transfer import (
    ROLE_TRANSFER_MODES,
    build_role_transferred_reaction_smiles,
)
from bh_augmentation.data.reaction_roles import (
    CANONICAL_ROLE_NAMES,
    ReactionRoles,
    reaction_roles_dataframe,
    reaction_roles_from_row,
)
from bh_augmentation.features.compatibility import block_slice
from bh_augmentation.features.featurize import build_feature_matrix_with_metadata
from bh_augmentation.utils.corrected_runs import (
    feature_contract_record,
    prepare_fresh_output_directory,
    write_json,
)

REPRESENTATIONS = (
    "reaction_section_concat",
    "reaction_section_concat_delta",
    "bh_role_separated",
    "bh_role_separated_delta",
)
STACK_SITES = (
    "run_condition_transfer.py: corrected anonymous real/synthetic stack",
    "run_role_aware_condition_transfer.py: corrected role-aware real/synthetic stack",
    "corrected_condition_transfer.py: teacher and downstream feature compatibility",
    "run_condition_transfer_supervised_ae.py: hybrid real/synthetic stack",
    "run_supervised_ae_latent_interpolation.py: latent coordinate stack",
    "utility_guided_feature_gan.py: generated coordinate stacks",
)


def build_report(output_directory: str | Path) -> dict[str, Path]:
    """Reproduce identity and locality assertions without fitting a model."""
    output = prepare_fresh_output_directory(output_directory)
    paths = {
        "directory": output,
        "json": output / "feature_contract.json",
        "csv": output / "feature_contract.csv",
        "identity": output / "identity_equivalence_summary.csv",
        "locality": output / "role_locality_summary.csv",
        "readme": output / "README.txt",
    }
    source = reaction_roles_dataframe([_source()], yields=[50.0])
    donor = reaction_roles_dataframe([_donor()], yields=[70.0])

    contracts: list[dict[str, object]] = []
    contract_json: dict[str, object] = {
        "canonical_role_order": list(CANONICAL_ROLE_NAMES),
        "canonical_representation_names": list(REPRESENTATIONS),
        "fingerprint_backend": "rdkit",
        "stack_sites_protected": list(STACK_SITES),
        "representations": {},
    }
    for kind in REPRESENTATIONS:
        config = {
            "kind": kind,
            "n_bits": 2048,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        }
        X, _, names, metadata = build_feature_matrix_with_metadata(source, config)
        record = feature_contract_record(metadata, names)
        contract_json["representations"][kind] = record
        contracts.append(
            {
                "representation_kind": kind,
                "total_width": X.shape[1],
                "n_bits": metadata.n_bits,
                "radius": metadata.radius,
                "fingerprint_backend": metadata.fingerprint_backend,
                "feature_name_hash": record["feature_name_hash"],
                "feature_metadata_hash": record["feature_metadata_hash"],
                "role_ordering": "|".join(metadata.role_ordering),
            }
        )

    identity_rows = _identity_rows(source)
    locality_rows = _locality_rows(source, donor)
    if not all(row["bit_identical"] for row in identity_rows):
        raise AssertionError("Corrected feature-contract identity equivalence failed.")
    if not all(row["only_requested_blocks_changed"] for row in locality_rows):
        raise AssertionError("Corrected feature-contract role locality failed.")

    write_json(paths["json"], contract_json)
    pd.DataFrame(contracts).to_csv(paths["csv"], index=False)
    pd.DataFrame(identity_rows).to_csv(paths["identity"], index=False)
    pd.DataFrame(locality_rows).to_csv(paths["locality"], index=False)
    paths["readme"].write_text(
        "Corrected Buchwald-Hartwig feature-contract verification\n\n"
        "No model training was performed. Identity equivalence reconstructs one measured "
        "reaction through anonymous and role-aware synthetic constructors. Role locality "
        "checks each named role block against the canonical metadata slices.\n\n"
        "Protected stack sites:\n- "
        + "\n- ".join(STACK_SITES)
        + "\n"
    )
    return paths


def _identity_rows(source: pd.DataFrame) -> list[dict[str, object]]:
    anonymous_roles = build_anonymous_condition_transfer_roles(source.iloc[0], source.iloc[0])
    role_record = build_role_transferred_reaction_smiles(
        source.iloc[0], source.iloc[0], "all_transferable_roles"
    )
    reconstructions = {
        "anonymous_condition_transfer": reaction_roles_dataframe(
            [anonymous_roles], yields=[50.0]
        ),
        "role_aware_condition_transfer": reaction_roles_dataframe(
            [reaction_roles_from_row(role_record)], yields=[50.0]
        ),
    }
    rows: list[dict[str, object]] = []
    for kind in ("bh_role_separated", "bh_role_separated_delta"):
        config = {
            "kind": kind,
            "n_bits": 2048,
            "radius": 2,
            "fingerprint_backend": "rdkit",
        }
        X_real, _, names, metadata = build_feature_matrix_with_metadata(source, config)
        for path, frame in reconstructions.items():
            X_other, _, other_names, other_metadata = build_feature_matrix_with_metadata(
                frame, config
            )
            rows.append(
                {
                    "representation_kind": kind,
                    "synthetic_path": path,
                    "bit_identical": bool(np.array_equal(X_real, X_other)),
                    "feature_names_identical": names == other_names,
                    "metadata_identical": metadata == other_metadata,
                }
            )
    return rows


def _locality_rows(source: pd.DataFrame, donor: pd.DataFrame) -> list[dict[str, object]]:
    modes = [
        "catalyst_only",
        "ligand_only",
        "base_only",
        "solvent_or_additive_only",
        "ligand_base",
        "all_transferable_roles",
    ]
    config = {
        "kind": "bh_role_separated",
        "n_bits": 2048,
        "radius": 2,
        "fingerprint_backend": "rdkit",
    }
    X_source, _, names, metadata = build_feature_matrix_with_metadata(source, config)
    rows: list[dict[str, object]] = []
    for mode in modes:
        record = build_role_transferred_reaction_smiles(source.iloc[0], donor.iloc[0], mode)
        frame = reaction_roles_dataframe([reaction_roles_from_row(record)], yields=[60.0])
        X_other, _, other_names, other_metadata = build_feature_matrix_with_metadata(
            frame, config
        )
        observed = {
            role
            for role in CANONICAL_ROLE_NAMES
            if not np.array_equal(
                X_source[:, block_slice(metadata, role)],
                X_other[:, block_slice(metadata, role)],
            )
        }
        requested = set(ROLE_TRANSFER_MODES[mode])
        rows.append(
            {
                "role_transfer_mode": mode,
                "requested_roles": "|".join(ROLE_TRANSFER_MODES[mode]),
                "observed_changed_roles": "|".join(
                    role for role in CANONICAL_ROLE_NAMES if role in observed
                ),
                "feature_names_identical": names == other_names,
                "metadata_identical": metadata == other_metadata,
                "only_requested_blocks_changed": (
                    observed == requested
                    and names == other_names
                    and metadata == other_metadata
                ),
            }
        )
    return rows


def _source() -> ReactionRoles:
    return ReactionRoles(
        "c1ccccc1Br", "CN", "[Pd]", "P(C)(C)C", "N1CCCCC1", "CCO", "c1ccccc1NC"
    )


def _donor() -> ReactionRoles:
    return ReactionRoles(
        "CCBr", "CCN", "[Ni]", "P(CC)(CC)CC", "CN(C)C(=N)N(C)C", "CCCO", "CCNC"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/corrected_feature_contract_report",
    )
    args = parser.parse_args()
    paths = build_report(args.output_dir)
    print(f"Saved corrected feature-contract report to {paths['directory']}")


if __name__ == "__main__":
    main()
