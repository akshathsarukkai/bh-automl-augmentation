# Autonomous Execution Blockers

External empirical validation and prospective laboratory validation were
anticipated to require blocker entries if the necessary local data or laboratory
results proved unavailable. Phase 16 has now confirmed the first of these.

## Phase 16 — external empirical validation

    implementation passed
    external empirical validation blocked

Recorded 2026-07-28 at commit a173ea12a52f78c126d461ea4e188e876dbb9b58.

No Suzuki-Miyaura dataset exists anywhere in this repository, and nothing here
downloads one. The only reaction data present is Buchwald-Hartwig, derived from
the Therapeutics Data Commons `Yields(name="Buchwald-Hartwig")` export.

The Phase 16 implementation is complete and tested against a clearly labelled
synthetic fixture. No empirical Suzuki result is claimed, produced, or implied.

Unblocking requires a human with network access to obtain a real dataset under
its own license and record complete provenance. `docs/EXTERNAL_DATASETS.md`
lists candidate datasets, acquisition steps, the required column schema, the
mandatory provenance fields, and the exact commands to run once the file is in
place. Every URL, DOI and license term in that document is marked
verify-before-use and must be confirmed against the primary source; a config
that records an unverified license is worse than one recording
"verify before use".

## Candidate-scope reanalysis — RDKit canonicalization pin

    implementation passed
    production execution requires a matching RDKit

Recorded 2026-08-25.

The canonical dataset stores `canonical_reaction_key` values produced by
`rdkit==2023.9.6` (see the `dependency_versions` block of
`results/corrected_canonical_data_audit/run_manifest.json`). Candidate
identities are computed live from roles. Under a newer RDKit these two
vocabularies stop intersecting — for example the Pd catalyst canonicalizes as
`O=S(=O)(O[Pd]1Nc2ccccc2-c2ccccc21)C(F)(F)F` under 2023.09.6 and
`O=S(=O)([O][Pd]1[NH]c2ccccc2-c2cccc[c]21)C(F)(F)F` under 2026.03.5, and **all
3,955 stored keys disagree**.

That is not a cosmetic mismatch. An eligibility gate comparing two disjoint
identity vocabularies rejects nothing, including candidates identical to
reactions the simulated learner has observed, and no metric shows a symptom.
`assert_stored_identities_match_roles` now makes this a hard failure, and
`scripts/run_lowdata_candidate_scope_reanalysis.sh` checks it in preflight
before doing any work.

Unblocking requires only an environment pinned to `rdkit==2023.9.6`; no data,
license or network access is needed. `.github/workflows/tests.yml` now pins
`rdkit==2023.9.6` in both the unit-test and production-smoke jobs, so CI
exercises the real identity contract instead of erroring at fixture setup
whenever the published wheel moves ahead. Note that this makes CI slower: the
Phase 14/15 benchmark module only runs at all under the pin, and it is
genuinely expensive (see the next entry). The same mismatch is why
`tests/test_redesigned_ae_benchmark.py` and
`tests/test_hidden_measured_calibration.py` error on a newer RDKit — a
pre-existing condition, not a regression. Both now pass under the pin.

## Phase 14/15 benchmark tests are slow, not hung

    resolved -- no defect
    recorded so the runtime is not mistaken for a hang again

Recorded 2026-08-25, corrected the same day.

An earlier entry here claimed `tests/test_redesigned_ae_benchmark.py`
deadlocked. **That was wrong.** The module is simply slow:
`test_plan_and_manifest_record_the_search_budget` alone takes **9m16s** and
passes. The misdiagnosis came from sampling the parent process, which sits at
0% CPU blocked in `select.poll` inside `multiprocessing.Process.join` while a
spawned child does the work at ~98% CPU. A short kill window looked exactly
like a deadlock.

`run_isolated_fit` already bounds every child with `process.join(timeout)`
followed by terminate/kill, so there is no unbounded wait to fix. The cost is
inherent: each isolated fit spawns a fresh interpreter that re-imports torch,
RDKit and scikit-learn.

The module was only reached at all once RDKit was pinned; before that its tests
errored at fixture setup, which is why the runtime had never been observed. No
code change was made, and nothing is excluded from the test suite.

## External reaction-family candidate scope

    semantics corrected
    empirical run still blocked

Recorded 2026-08-25.

`run_external_family_validation.py` previously decided candidate eligibility
against `usable` — train, validation and test together — while restricting
*generation* to training rows. The eligibility rule now follows the declared
`candidate_scope.mode`, defaulting to `observed_only_low_data`. Code, config
schema, documentation and tests are updated. **No empirical run was produced**:
the Suzuki-Miyaura blocker above is unchanged, and no external result is claimed
in either direction.
