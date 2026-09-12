# Reproducibility

This document describes how a scientific result in this repository is bound to
the exact code, data, split, features, configuration, policy, evaluation claim,
and outputs that produced it — and records an **actually executed** clean-checkout
reproduction, including the parts that could *not* be reproduced and why.

## 1. What a result is traceable to

Every completed scientific bundle carries a hash-addressed manifest. The shared
contract is `bh-scientific-run-manifest-v1`, built by
`bh_augmentation.utils.scientific_manifest.build_scientific_manifest` and written
as `scientific_manifest.json`. It records:

| Field | Binds the result to |
| --- | --- |
| `git_commit`, `git_dirty_at_execution` | the exact code |
| `dataset_path`, `dataset_hash` | the exact input data |
| `split_hash` | the exact train/validation/test assignment |
| `feature_metadata_hash` | the exact representation contract |
| `config_hash`, `resolved_scientific_config` | the exact resolved configuration |
| `plan_hash` | the plan frozen before any outcome was loaded |
| `row_counts` | the size of every emitted table |
| `output_hashes` | the SHA-256 of every output file |
| `dependency_versions` | numpy / pandas / scikit-learn / scipy / torch / rdkit / xgboost / PyYAML / python |
| `platform_record` | platform, system, release, machine, processor, logical and affinity CPU counts, python implementation/version/compiler, byte order, torch version + `num_threads` + `num_interop_threads` + determinism flag, and every set BLAS/OpenMP variable (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `KMP_*`, `MKL_THREADING_LAYER`, `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`) |
| `manifest_hash` | the manifest's own contents |
| `command` | the command that produced the bundle |

Outer-test **evaluation claims** live outside the bundle, in the repository-global
`EvaluationRegistry` (`results/autonomous_execution/evaluation_registry`). A claim
is keyed only by `(dataset_hash, split_or_search_manifest_hash, frozen_policy_hash,
evaluation_unit)` — never by an artifact path — so moving, copying, or deleting a
result directory cannot release it.

### Verifying a bundle

```bash
python -B - <<'PY'
from bh_augmentation.utils.scientific_manifest import verify_manifest
print(verify_manifest("results/<some-corrected-bundle>").to_dict())
PY
```

`verify_manifest` recomputes the manifest hash from the manifest's own stored
fields and re-hashes every declared output. It also understands the pre-existing
runner manifests (`manifest.json`) that already carry `output_hashes` and
`manifest_hash`, so one entry point covers every hash-addressed bundle here.

Mutating a single byte of any declared output makes verification fail. CI proves
this on every push via `scripts/assert_manifest_tamper_evident.py`.

**Limit of the guarantee.** The manifest hash chain proves integrity *relative to
the manifest*. Someone who rewrites an output, its `output_hashes` entry, and the
`manifest_hash` together produces a self-consistent bundle. Tamper-evidence against
that requires an externally recorded hash: `results/autonomous_execution/state.json`
records per-artifact SHA-256 values for passed phases, and
`autonomous_execution.verify_passed_phases` reopens a phase whose artifact hashes
have changed. Keep that ledger, not the bundle, as the root of trust.

## 2. Building a local reproducibility bundle

```bash
python -B scripts/build_reproducibility_bundle.py \
    --result-directory results/<some-corrected-bundle> \
    --bundle-directory /tmp/my-bundle \
    --config configs/<the-config-used>.yaml
```

The bundle contains `run_manifest.json`, the copied config, `environment.json`
(dependency + hardware record as recorded at execution *and* as observed now),
`git.json`, `input_hashes.json` (every input hashed, with anything absent from the
checkout explicitly flagged as "must be supplied by a human"), `REPRODUCE.md` with
the exact commands, and a self-hashing `bundle_manifest.json`.

The tool publishes nothing, performs no network access, and refuses to write into
an existing path. It also refuses to bundle a result whose outputs no longer hash
to their recorded values.

## 3. Clean-checkout reproduction — what was actually done

Performed on 2026-07-28 at commit `851d64e` on macOS (Darwin 25.5.0, arm64),
CPython 3.12, torch 2.10.0.

Three independent clean checkouts were made **outside** the working tree:

```bash
git worktree add --detach /tmp/scratch/checkout_a HEAD   # checkout A
git archive HEAD | tar -x -C /tmp/scratch/checkout_b     # checkout B (no .git)
git worktree add --detach /tmp/scratch/checkout_c HEAD   # checkout C
```

Nothing was installed globally. The bounded workflow was launched with
`python -B scripts/run_production_path_smoke.py --output-directory .bh_repro_v2 --rows 300`
from inside each checkout; the runner script prepends its own checkout's `src/` to
`sys.path`, and `bh_augmentation.__file__` was confirmed to resolve inside the
checkout rather than the ambient editable install.

The workflow is the real production path, not a mock:

1. `bh_augmentation.data.audit_canonical_dataset.run_canonical_data_audit` —
   RDKit canonicalization of the first 300 rows of the committed
   `data/processed/bh_clean_stress.csv`, producing a canonical seven-role dataset.
2. `bh_augmentation.data.canonical_splits.run_canonical_grouped_splits` —
   group-safe outer split assignments plus cumulative low-data subsets.
3. `bh_augmentation.low_complexity_benchmark.run_low_complexity_benchmark` — the
   Phase 13 benchmark over all six low-complexity families at widths 8 and 16,
   with within-family validation selection, frozen policies, and a single outer-test
   evaluation per unit and method.
4. `validate_low_complexity_benchmark` (full replay) and `verify_manifest`.

Each run took ~57 s.

### What was actually observed

**Checkouts A and C (two independent `git worktree` checkouts) — data preparation:
byte-identical.**

| Artifact | Result |
| --- | --- |
| `corrected_canonical_roles_smoke.csv` | MATCH |
| `corrected_canonical_splits/split_manifest.json` | MATCH |
| `corrected_canonical_splits/outer_split_assignments.csv` | MATCH |
| `corrected_canonical_splits/low_data_subset_assignments.csv` | MATCH |
| `corrected_canonical_splits/split_overlap_audit.csv` | MATCH |
| `corrected_canonical_splits/split_summary.csv` | MATCH |

**Benchmark bundle: 12 of 15 files byte-identical, and every scientific value
identical.**

| Artifact | Result |
| --- | --- |
| `benchmark_plan.json` | MATCH |
| `candidate_policies.csv` | MATCH |
| `split_units.csv` | MATCH |
| `search_predictions.csv` | MATCH |
| `search_metrics.csv` | MATCH |
| `search_fit_audit.csv` | MATCH |
| `frozen_method_policies.json` | MATCH |
| `evaluation_claims.csv` | MATCH |
| `refit_audit.csv` | MATCH |
| `final_predictions.csv` | MATCH |
| `final_test_metrics.csv` | MATCH |
| `summary.csv` | MATCH |
| `resource_metrics.csv` | DIFFER (timings/memory only, see below) |
| `manifest.json` | DIFFER (inherits the `resource_metrics.csv` hash) |
| `scientific_manifest.json` | DIFFER (`created_at` timestamp + inherited hashes) |

Scientific identity hashes all matched exactly across the two checkouts:
`plan_hash`, `config_hash`, `dataset_hash`, `canonical_split_hash`,
`feature_metadata_hash`, `frozen_document_hash`.

`resource_metrics.csv` differs in exactly six measured-cost columns —
`elapsed_seconds`, `training_time_seconds`, `rss_baseline_bytes`,
`rss_peak_bytes`, `rss_increment_bytes`, `peak_memory_bytes` — while its other 16
columns are identical. These are wall-clock and memory measurements and are not
expected to be reproducible; `manifest.json` differs only because it hashes that
file, and `scientific_manifest.json` differs only by its `created_at` timestamp
and the hashes it inherits. **No predicted value, metric, split, policy, or claim
differed.**

A local reproducibility bundle was also built inside checkout A against its own
result directory: 14 outputs verified, `missing_input_count = 0`.

### Two real defects this exercise found and fixed

1. **`git archive` exports cannot run the scientific runners.** Checkout B
   completed the canonical audit and the split builder — its canonical dataset,
   `outer_split_assignments.csv`, and `low_data_subset_assignments.csv` were
   byte-identical to A and C — and then failed at the benchmark with
   `ValueError: Scientific runs require an available Git commit`, because a
   `git archive` tarball has no `.git`. This is correct, intended behaviour — a
   run with no resolvable commit has no provenance — but it means **reproduction
   requires `git worktree` or `git clone`, not `git archive`**. The CI smoke job
   is configured accordingly. Checkout B was not re-run after the path fix
   below, so the A/C comparison table above is the authoritative one.
2. **Absolute paths leaked into the hash chain.** In the first attempt the split
   manifest recorded the absolute source path, which differed per checkout and
   propagated into `config_hash` and `plan_hash`, and from there into nearly every
   emitted CSV — 12 of 15 files differed for a purely cosmetic reason. The smoke
   runner now passes the source dataset as a checkout-relative path
   (`checkout_relative_source()`), after which everything above matched. Anyone
   comparing bundles across machines should be aware that absolute paths recorded
   in a config make bundles non-comparable by construction.

A third, unrelated defect surfaced in the clean checkout and was fixed:
`utils/strict_config.py` originally imported `features/reconstruction_contract.py`
at module scope. That module is uncommitted Phase 14 work, so the shared config
validator could not be imported from a clean checkout at all. The reconstruction
vocabulary is now declared locally in `strict_config.py`, with
`assert_reconstruction_vocabulary_matches_contract()` (and a covering test)
failing loudly if the two definitions ever drift apart.

### Honest caveat about this reproduction

The Phase 17 changes described here are **not yet committed**. Checkouts A and C
were therefore created from `HEAD` (`851d64e`) and the eight uncommitted Phase 17
files were copied in explicitly:

```
src/bh_augmentation/utils/strict_config.py
src/bh_augmentation/utils/scientific_manifest.py
src/bh_augmentation/low_complexity_benchmark.py
src/bh_augmentation/evaluation/evaluation_registry.py
src/bh_augmentation/nested_ood_final.py
scripts/run_production_path_smoke.py
scripts/build_reproducibility_bundle.py
scripts/inspect_evaluation_registry.py
```

Everything else — all data, all configs, and every other module — came from git.
Once Phase 17 is committed, the same procedure runs with no copying at all.

## 4. What CANNOT be reproduced from a clean checkout

The reproduction above deliberately used a workflow whose inputs are committed.
The **production** Phase 10–18 bundles cannot be regenerated from a clean checkout
as-is, because their two canonical inputs are gitignored and are absent from a
fresh worktree (verified: both paths do not exist in checkouts A, B, and C):

| Missing input | Why | What a human must do |
| --- | --- | --- |
| `data/processed/bh_canonical_roles_v1.csv` | `.gitignore` excludes `data/processed/*`; the 12 MB canonical dataset is a generated artifact | Regenerate with `python -m bh_augmentation.data.audit_canonical_dataset --config configs/canonical_data_audit.yaml`. Its source, `data/processed/bh_clean_stress.csv`, **is** committed. Then confirm the result hashes to `df61cb657747e7f7715ff93060b869550a154a435f6e325d84448477f161d365` (the value pinned in `results/autonomous_execution/state.json`). |
| `results/corrected_canonical_splits/` | `.gitignore` excludes `results/` | After the canonical dataset exists, regenerate with `python -m bh_augmentation.data.canonical_splits --config configs/canonical_grouped_splits.yaml`, then confirm `split_manifest.json` hashes to `70b11ef0d5ecf3517f92b47113132f081e94236911b821519a1dd9a40a6977bd`, `outer_split_assignments.csv` to `1bf2cd66417bc2607c6363bf1f3dccfa1c58a599c448df52b43065c1546b34c8`, and `low_data_subset_assignments.csv` to `1d2220f6c8ea1c01d882fafaf3c65d52422ba70786b253cf00b835922d60bd61`. |

Both regeneration steps are the same production runners the reproduction above
exercised, just at full size instead of 300 rows, so the path is real rather than
hypothetical. **This was not executed at full scale here** — regenerating the full
canonical dataset and the five-seed split set is a long job and the machine was
already running a production benchmark. The claim made in this document is
therefore precisely: *the bounded workflow reproduces byte-identically from a
clean checkout*, and *the full-scale inputs are regenerable by a documented
command whose expected hashes are pinned*, not that the full-scale regeneration
was observed.

Two further items a clean checkout cannot supply:

- **No licensed external dataset.** Phase 16's Suzuki–Miyaura validation is
  externally blocked: no licensed dataset is present locally and acquisition needs
  a human with network access. See `docs/EXTERNAL_DATASETS.md`.
- **Bit-identical results across different hardware are not claimed.** Different
  CPUs, BLAS/OpenMP thread counts, or dependency versions can change
  floating-point reduction order. That is exactly why `platform_record` and
  `dependency_versions` are in the manifest: divergence can be attributed rather
  than guessed at. A concrete instance: two executions of the development
  candidate-scope reanalysis on this machine agree on every candidate count and
  on the oracle metrics to three decimals, but their validation RMSEs differ by
  up to 0.6 because, through a config-forwarding defect since fixed, XGBoost ran
  with its default thread count. Under the confirmatory protocol every policy
  pins `n_jobs: 1`.

## 4a. What IS committed: the two confirmatory bundles

The two confirmatory results are the exception to the gitignored `results/`
tree. Their small, re-derivable artifacts are force-tracked so a clone can
recompute each verdict without any gitignored input:

| Result | Committed files |
| --- | --- |
| Phase 15 primary confirmation | `results/autonomous_execution/phase_15/corrected-20260729-13fbfde-phase15-confirmation-v1/` — `final_test_metrics.csv`, `pool_summary.csv`, `summary.csv`, `manifest.json`, `confirmatory_analysis.json`, `confirmatory_report.md`, `confirmatory_analysis_manifest.json` |
| Exploratory fraction sweep (development evidence) | `results/corrected_exploratory_fraction_sweep_f0p01/` and `..._f0p10/` (same per-unit layout plus `search/pool_accounting.json` sidecars, registry snapshots, file indexes), `results/corrected_exploratory_fraction_sweep_analysis_v1/`, and `results/corrected_exploratory_fraction_sweep_pool_accounting_v1/` (all but its 39 MB `candidate_audit.csv`) |
| Observed-only condition transfer | `results/corrected_candidate_scope_reanalysis_20260903T194209Z/confirmatory/<family>/seed_<n>/` (`search/` frozen policy, manifest, metrics or `degenerate_unit.json`; `final/` metrics, claim, manifest), `results/corrected_candidate_scope_reanalysis_20260903T194209Z/summary/` (registry snapshots before and after, `run_file_index.json`, `observed_only_confirmatory_analysis/`), `results/corrected_candidate_scope_reanalysis_20260903T194209Z/candidate_scope_audit/`, and the treatment-size accounting under `results/corrected_observed_only_transfer_pool_accounting_v1/` (all but its 18 MB `candidate_audit.csv`, whose hash is recorded in that bundle's `manifest.json`) |

```bash
python -m pytest tests/test_committed_confirmatory_artifacts.py -q
```

re-derives both verdicts (and the exploratory sweep's analyses) from those
files, compares the recomputed analysis hashes with the committed ones, and
verifies the analysis manifests. One caveat for anyone *regenerating* candidate
pools rather than re-deriving from committed files: after every run recorded
here, `_cosine_similarity_matrix` gained float64 evaluation and rounding to ten
decimals so that donor ties break identically across platforms (a
permutation-stability test failed on Linux CI while passing on macOS). Pools
regenerated with the new code can differ from the recorded ones only where two
donors were tied to within 1e-10, and the committed pool accounting is the
record of what the runs actually used. The
per-unit files are the bytes the protocol wrote; `run_file_index.json` lists
the SHA-256 of every file the observed-only run produced, including the ones
too large to commit.

## 5. Continuous integration

`.github/workflows/tests.yml` runs three jobs:

- **lint** — `python -B -m ruff check .`
- **tests** — the full `pytest` suite
- **production_path_smoke** — the bounded real production path above on committed
  data only, followed by a reproducibility-bundle build and a tamper-evidence
  assertion that mutating one output byte is detected.

The smoke job exists because unit tests cannot catch an end-to-end break in a
scientific runner. It needs no gitignored artifact and no network access.

## 6. Inspecting outer-test claims

```bash
python -B scripts/inspect_evaluation_registry.py list
python -B scripts/inspect_evaluation_registry.py list --status metrics_complete --json
python -B scripts/inspect_evaluation_registry.py show --registry-key <key>
```

This CLI is read-only by construction: it exposes only `list` and `show`, and the
registry class itself offers no delete, release, reset, or unclaim operation. An
outer-test identity that has been reserved must never be silently re-evaluated.
