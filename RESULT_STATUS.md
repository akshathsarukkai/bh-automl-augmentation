# Result Status

This registry freezes the interpretation of existing experiment outputs. Code
repairs do not retroactively validate metrics produced by an invalid feature
path. Corrected experiments must use fresh output directories after review.

| Result family | Status |
| --- | --- |
| Role-aware condition transfer v2 | **Invalid** |
| Any table containing role-aware v2 | **Invalid** |
| Historical `results/stress/logo_product/` outputs | **Invalid** |
| Historical `results/stress/logo_reactant/` outputs | **Invalid** |
| Historical baseline `stress_logo_*_metrics.csv` files | **Invalid** |
| Fresh corrected Phase 7 LOGO outputs | Eligible after split-contract validation |
| Anonymous condition transfer | Development evidence |
| Anonymous transfer + supervised AE | Development evidence |
| Random-split baselines | Sanity checks |
| GAN experiments | Experimental only |
| Latent interpolation | Experimental only |
| Corrected representation baselines | Corrected revalidation |
| Corrected anonymous condition transfer | Corrected revalidation |
| Corrected role-aware condition transfer | Corrected revalidation |
| Corrected anonymous-versus-role-aware comparison | Corrected revalidation |
| Batch 2 corrected random-split outputs | Development evidence |
| Canonical seven-role data audit | Data-quality evidence |
| Canonical grouped split assignments | Required for future benchmark evidence |

## Why role-aware v2 is invalid

Measured role-aware rows and synthetic role-aware rows were featurized with
inconsistent semantic block meanings. Measured rows used three reaction
sections while synthetic rows were manually assembled from individually named
chemical roles. Equal vector widths therefore did not guarantee equal feature
semantics. Existing role-aware v2 metrics and tables containing those metrics
must not be used in benchmark claims.

The old output directories are retained for historical inspection. Loading
them requires an explicit invalid-result override. Alias support for old feature
configuration names does not make these outputs valid.

## Representation terminology

`reaction_section_concat` means reactant section + agent section + product
section. It does not identify individual chemical roles.

`bh_role_separated` means reactant 1 + reactant 2 + catalyst + ligand + base +
solvent/additive + product in that exact order.

Equal matrix widths are insufficient for compatibility. Feature names,
fingerprint settings, role order, backend, and exact block slices must also
match before measured and synthetic matrices can be stacked.

Corrected revalidation outputs use fresh directories containing `corrected`,
a `run_manifest.json` with `historical_results_loaded=false`, and row-level
`result_status=corrected_revalidation`. A corrected name alone is insufficient:
comparison scripts validate manifests, feature hashes, split hashes, dataset
hashes, and matched real-only metrics.

## Why historical LOGO outputs are invalid

The retained product and reactant outputs under `results/stress/logo_product/`
and `results/stress/logo_reactant/`, together with the corresponding
`results/baseline/stress_logo_product_metrics.csv` and
`stress_logo_reactant_metrics.csv` files, do not contain sufficient evidence to
verify leave-one-group-out evaluation. In particular, the baseline tables do
not provide reproducible fold assignments, split hashes, and train/test group
overlap audits. Their labels alone cannot establish that the saved metrics came
from valid LOGO splits.

These artifacts remain on disk for provenance. Default result readers reject
them; `allow_invalid=True` permits explicit historical inspection only. Fresh
corrected Phase 7 directories remain eligible when their manifests establish
the declared held-out group, zero train/test group overlap, complete expected
fold coverage, and assignment hashes.

## Canonical identity and future benchmarks

Batch 2 corrected the measured/synthetic feature-semantics mismatch. Its
random-split outputs remain valid development evidence, but they are not final
benchmark evidence because canonical molecular duplicates and experimental
replicates had not yet been audited when those splits were constructed.

Batch 3 preserves Batch 2 outputs and introduces versioned RDKit canonical
identities plus group-safe split assignments. Future corrected benchmark runs
should use the canonical grouped assignments so rows representing the same
seven-role canonical reaction cannot cross train, validation, and test.

Canonicalization completing successfully does not establish that the dataset is
clean. The duplicate, replicate, yield-conflict, invalid-structure, and
fingerprint-collision reports require separate scientific review. Batch 3 does
not generate new model results.
