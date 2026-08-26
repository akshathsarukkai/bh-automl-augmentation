# Autonomous completion report

> **Addendum, 2026-08-25 — candidate-scope audit.** A repository-wide audit of
> candidate-eligibility semantics ran after this report was written. It does not
> change the Phase 15 confirmatory verdict, which was already scoped to the rows
> each phase could observe. It does narrow one argument used throughout this
> report: the "exactly five eligible unmeasured reactions" figure is a statement
> about **global discovery headroom** and must not be used to explain low-data
> augmentation nulls, because a low-data learner has observed only a small
> fraction of the matrix and its transfer headroom is correspondingly large. It
> also withdraws the Phase 11 chemical-augmentation conclusions as uninformative:
> under the executed global eligibility rule those arms carried 0.4 to 1.2
> effective added rows against a nominal budget of 159. See `RESULT_STATUS.md`
> § *Candidate-scope semantics* and `docs/CANDIDATE_SCOPE.md`. No historical
> number in this report is edited.

All 18 roadmap phases are passed and checkpointed locally. Nothing has been
pushed.

Branch `main`, 14 commits ahead of the handoff checkpoint
`58613dd0907602ce295958ca81aa70b9ed618d54`. Remote `origin` exists and has not
been contacted.

---

## 1. The headline scientific result

**The project's central thesis does not hold, and neither does its headline
method.**

Two independent findings, both obtained under protocols frozen before the
results were visible:

1. **The supervised autoencoder adds no reproducible value** (Phase 14). It is
   classified as a **secondary ablation** and was excluded from confirmation.
2. **Condition-transfer augmentation produces no practically meaningful change**
   in low-data yield prediction (Phase 15). The preregistered verdict is
   **null**.

Both are negative or null results. Both are reported as the finding rather than
re-cut in search of something positive.

A third, structural finding constrains how much any augmentation method could
ever have achieved here: **the canonical dataset is a near-complete factorial.**
15 canonical substrate groups times 264 canonical condition groups is 3960
cells, of which 3955 are measured. Exactly **five** eligible unmeasured
reactions exist. This was verified independently and explains the recurring
zero-accepted-candidate null results recorded in Phases 1 through 4, which had
previously read like implementation failures.

---

## 2. Completed phases and their commits

| Phase | Name | Commit |
| --- | --- | --- |
| 1 | Canonicalize and deduplicate synthetic candidates | `5268df37a39ba443fdf26ee15dc0a09d3bcb59ae` |
| 2 | Enforce exact role-change semantics | `4c9f9c7497d0cc19927a3689841de91b5525d0ca` |
| 3 | Uniform similarity, source caps, deterministic ranking | `1bbfa0ebd08f514e2e1844687f944f3c46352adb` |
| 4 | Canonical split integration | `e601b1445cf08394d4ec2408c82ec3ac10bd3f04` |
| 5 | Separate policy search from final evaluation | `0e3013887c9701fece8e052ecb1f3f6624f4367a` |
| 6 | Repair AE internal validation and joint selection | `fc5b59aaae917e20594dff27abf7f48d58644248` |
| 7 | Rebuild product and reactant LOGO runners | `f4134eff438f3465656aef4dbcea30ec202deba2` |
| 8 | Nested group-aware OOD validation | `fbbe2bf0811f6e8bd58b7e443ea68a52f2aef517` |
| 9 | Scaffold, cluster, similarity, condition OOD splits | `596e65801414c5e77fc63cd59b048bd9e1a12a83` |
| 10 | No-augmentation representation baselines | `5c2262fef524ae633e00baf9bb224eb4c0800490` |
| 11 | Simple augmentation controls | `a39a3ebfc391b2d5f9bdf50979ccc5b5bc9ce51e` |
| 12 | Pseudo-label accuracy and uncertainty calibration | `3fff5baf284d130b09557064f65a0f99b68a01ca` |
| 13 | Low-complexity representation baselines | `58613dd0907602ce295958ca81aa70b9ed618d54` |
| **14** | **Reassess the supervised autoencoder** | `6a618ed139c8d6870592208b0c8046f7624ad65c` |
| **15** | **Primary confirmatory experiment** | preregistered `13fbfdef7ac0cfd05dda2c1c33bb6a59b4d64fff`, executed `253e70e601e2fac7cf3846fa37935c4fdfdd61a3` |
| **16** | **External typed-dataset adapter** | `a173ea12a52f78c126d461ea4e188e876dbb9b58` |
| **17** | **Reproducibility and tamper-evidence** | `07479aeb9d0973c78a840de8a960c8685c3e2ee7` |
| **18** | **Recommendation simulation and prospective package** | `085649cb428a2ad4999d1a5a09a5e01de4e7fa24` |

Phases 16, 17 and 18 were committed before Phase 14 finished. Each is
independent of the Phase 14 outcome, Phase 16's empirical arm is externally
blocked, and none asserts any Phase 14 result. The roadmap's ordering rule
permits this and each commit records the justification.

---

## 3. Authoritative result directories

| Phase | Directory |
| --- | --- |
| 13 | `results/autonomous_execution/phase_13/corrected-20260726-3fff5ba-phase13-production-v1` |
| 14 | `results/autonomous_execution/phase_14/corrected-20260728-851d64e-phase14-production-v3` |
| 15 | `results/autonomous_execution/phase_15/corrected-20260729-13fbfde-phase15-confirmation-v1` |
| 15 | `results/corrected_canonical_splits_phase15` (fresh seeds 5-14) |
| 18 | `results/autonomous_execution/phase_18/corrected-20260727-phase18-production-v1` |
| 18 | `results/autonomous_execution/phase_18/corrected-20260728-phase18-prospective-package-v1` |

Retained but **invalid**, preserved under the no-deletion policy and marked in
place: the two pre-hardening Phase 14 smoke bundles, the interrupted
`phase14-production-v1`, and the session-killed `phase14-production-v2`.

---

## 4. Scientific findings

### Phase 14 — the supervised autoencoder fails

Decided on placement-validation data only, against a criterion fixed in config
before any outer-test outcome existed, frozen before the first outer-test read.
All five criteria failed.

| Fraction | Mean delta | Median | Margin wins | Practical losses |
| --- | --- | --- | --- | --- |
| 0.2 | -0.999 | -1.099 | 0 of 5 | 4 of 5 |
| 1.0 | -0.453 | -0.237 | 0 of 5 | 1 of 5 |

Delta is best-control RMSE minus AE RMSE, so negative means the AE is worse.
Pooled seed-cluster bootstrap: mean -0.726, 95% CI [-1.101, -0.395]. The AE lost
to the best simple control in 9 of 10 evaluation units.

**The result is conservative because the protocol favoured the AE.** Per-unit
search budgets were AE 11, matched direct MLP 10, transfer controls 6 each,
truncated SVD 6, linear autoencoder 6, direct XGBoost 3. Selection is a minimum
over inner-validation RMSE, so the largest budget benefits most from winner's
curse. The AE had it and still lost.

**Why it fails, from diagnostics beyond aggregate loss.** Aggregate
reconstruction MSE looks excellent (0.0034 at fraction 0.2) but is dominated by
5.6 million zero bits against 65 thousand positive bits. On the chemically
meaningful set bits, reconstruction is 13 to 32 times worse: positive-bit MSE
0.0799 against zero-bit MSE 0.0025. The decoder is largely learning to emit
zeros. An aggregate-loss-only view would have made this representation look
healthy.

Validation selection was also unstable — six different AE candidates won across
ten units. And the AE is the most expensive family: 136.7 s and 1.24 GB peak RSS
per final fit, against 57.3 s and 0.52 GB for the matched direct MLP.

### Phase 15 — the primary confirmatory experiment is null

Preregistered before any confirmatory result existed; the document was not
modified afterwards and no amendment was required.

| Quantity | Value |
| --- | --- |
| Mean paired effect | -0.6609 RMSE (negative means augmentation is worse) |
| 95% bootstrap CI | [-0.8446, -0.4778] |
| Units improved / worsened | 0 of 10 / 10 of 10 |
| Wilcoxon two-sided p (secondary) | 0.001953 |
| Included / degenerate / failed | 10 / 0 / 0 |

**Verdict: null**, not negative and not positive. The interval excludes zero and
the Wilcoxon p is 0.002, so a results-first reading would report significant
harm. But the entire interval lies inside the plus or minus 1.0 RMSE
practical-equivalence band fixed in advance, which the preregistration defines as
null: a practically meaningful benefit and a practically meaningful harm are both
excluded. This is exactly the case the preregistration was written to adjudicate.

The null is not an artifact of the treatment failing to apply. Every unit had 335
to 362 accepted synthetic rows against 633 measured rows, roughly 55%
augmentation.

**A protocol fact.** The confirmation is directionally worse than Phase 14's
near-tie of -0.006 because Phase 14 gave the augmented arm 6 candidate policies
against the comparator's 3, including a synthetic-downweighting knob with no
real-only analogue. The preregistration requires equal budgets, so under a
genuinely matched comparison the augmented arm loses on all ten seeds. **Part of
what looked like augmentation parity in development evidence was the augmented
arm being allowed to search harder.**

### Phase 18 — informed acquisition works, on the right metrics

Over a 320-experiment budget against a complete 256-replicate random
distribution, diversity-aware and greedy acquisition returned about 4.3 times
the high-yield hits (229.8 and 222.2 against 51.6), about 2.2 times the mean
acquired yield (73.6 and 72.3 against 33.2), and recovered roughly 80% of the
true top-50 against 13% for random — at the 100th percentile of the random
distribution in every unit, and the 0th percentile for cumulative regret.

### Phase 13 — representation baselines

Direct MLP and bottleneck MLP were best (13.596 and 13.647 mean RMSE at 20%
training; 10.196 and 9.977 at full). SVD, linear autoencoder, PLS-plus-Ridge and
sparse selection were all worse.

---

## 5. Null and negative findings, retained

- The supervised autoencoder adds no reproducible value (Phase 14).
- Condition-transfer augmentation produces no practically meaningful change
  (Phase 15).
- `best_yield_discovered` is a weak discriminator: 99.9 informed against 98.4
  random, only the 76th to 93rd percentile. Random with 320 draws from a dense
  grid already finds near-maximal yield. **This metric must not be quoted as
  evidence on this dataset.**
- `unique_substrate_keys` is uninformative — only 15 substrate pairs exist and
  every strategy saturates.
- **Condition diversity is a negative result.** Every informed strategy explored
  fewer unique condition blocks than random (139-197 against 207), at the 0th
  percentile in all units. Exploiting predicted yield narrows the condition
  space.
- Conformal coverage degrades on acquisition-selected batches: 0.82 empirical
  against 0.90 nominal, because a selected batch is not exchangeable.
- Truncated SVD and the linear autoencoder are far worse than every other family
  in both Phases 13 and 14.
- The eligible prospective candidate space is exactly five reactions. The list
  was not padded.
- Strict anonymous full-condition transfer has no eligible donor on the dense
  canonical subset while catalyst is invariant (Phases 1-4).

---

## 6. Confirmatory conclusion

**One** confirmatory result exists. Everything else in this repository is
development evidence, experimental, invalidated, or externally blocked.

> Anonymous condition-transfer augmentation does not produce a practically
> meaningful change in low-data Buchwald-Hartwig yield prediction relative to a
> matched real-only XGBoost baseline, on canonical grouped random splits at
> training fraction 0.2, across ten fresh evaluation units.

Scope limits that must travel with this claim: training fraction 0.2 only;
canonical grouped random splits only; one dataset; **not** OOD evidence; and
fraction 1.0 was preregistered as secondary and descriptive only and was not
run.

---

## 7. External data status

```
implementation passed
external empirical validation blocked
```

No Suzuki-Miyaura dataset exists anywhere in this repository and nothing here
downloads one. The only reaction data present is Buchwald-Hartwig from the TDC
export. Every Suzuki artifact is exercised against a clearly labelled synthetic
fixture whose yields are a deterministic function of the row index. **No
empirical Suzuki result is claimed, produced, or implied.**

Unblocking requires a human with network access to obtain a real dataset under
its own license and record complete provenance. `docs/EXTERNAL_DATASETS.md` lists
candidate datasets, acquisition steps, the required column schema, the mandatory
provenance fields, and the exact commands. Every URL, DOI and license term there
is marked verify-before-use.

---

## 8. Prospective validation status

```
Prospective laboratory validation has not been performed.
```

The Phase 18 package is a proposal for review, not a validated result. It
contains 5 discovery candidates and 24 held-out measured controls, with seven
explicit canonical roles per candidate, per-role provenance, calibrated intervals
reported unclipped to preserve the coverage guarantee, support distances,
diversity and uncertainty rationales, and a protocol draft that lists what the
canonical dataset **cannot** supply — temperature, time, stoichiometry,
concentration, scale, atmosphere, work-up, analytical method — rather than
inventing values.

Every candidate's practical accessibility is recorded as `unknown`. No claim is
made about synthesis feasibility, safety, cost, or likelihood of laboratory
success. Wet-lab execution is external to this repository.

---

## 9. Audit results

| Check | Result |
| --- | --- |
| All completed phases have commits | 18 of 18 |
| Ledgers agree with git history | 18 of 18, `phase_passed` events agree |
| Required hashes validate | `verify_passed_phases` reports zero issues; all 5 dependency hashes match |
| Production bundle artifact hashes | 74 artifacts across 5 bundles, 0 mismatches |
| Invalid historical results excluded | role-aware v2 and historical LOGO remain invalidated in `RESULT_STATUS.md` |
| Scientific runners use valid split protocols | complete group separation verified on both split sets |
| Search cannot access outer-test outcomes | source ordering verified: search, then reservation, then first test read |
| Outer-test claims cannot be reused | 145 claims, all `metrics_complete`, zero duplicate keys, no release path |
| Synthetic candidates obey identity contracts | canonical seven-role identity enforced; fingerprint equality never treated as identity |
| Result directories not overwritten | fresh directories throughout; failed runs retained and marked |
| Confirmatory work follows preregistration | `PRIMARY_EXPERIMENT.md` unmodified since commit; no amendment needed |
| External validation status accurate | blocked, recorded in `blockers.md` |
| Prospective claims limited | disclaimer verbatim in 5 files; all four claim flags false |

Full suite and lint results are recorded in the Phase 15 ledger entry.

---

## 10. Defects found and fixed during this work

These were not cosmetic. Several would have silently corrupted the science.

1. **The ledger's own verifier was reopening five phases.** Phases 9 through 13
   omitted a `status` key inside their test records, so `verify_passed_phases`
   treated all of them as unvalidated. The reported "missing Phase 13 commit" was
   the smaller half of the defect.
2. **Phase 14 was rigged in the AE's favour.** The AE had 11 candidate policies
   against 2 for every control, and only the AE could choose its data protocol,
   so any AE win would have conflated architecture with synthetic-pool access.
3. **The one-outer-test-evaluation lock was defeatable.** The evaluation identity
   embedded the git commit, float placement metrics, and which policy won the
   search, so an unrelated commit or a one-ULP difference minted a fresh claim.
4. **The registry root was working-directory-relative.** Running a scientific
   runner from any other directory created a fresh, empty registry and permitted
   re-evaluating a consumed outer-test identity. Now anchored to the checkout,
   with no environment-variable override, and covered by regression tests over
   every scientific runner.
5. **Transfer pools consumed outer-test row identities**, contradicting the
   runner's own declared training-only protocol.
6. **The retention bootstrap tested against zero rather than the practical
   margin**, making that criterion near-vacuous.
7. **A zero-row transfer pool silently degraded a transfer control into plain
   XGBoost** while still being counted as a transfer arm.
8. **Structural validation ran only after the outer test was consumed.**
9. **A shared config validator could not be imported from a clean checkout at
   all** — caught only by actually performing a clean-checkout reproduction.

---

## 11. Known remaining debt

- `hidden_measured_calibration.py` still carries a duplicated config validator.
  Consolidating it requires re-validating that runner's production bundle.
- The canonical dataset and canonical split directory are gitignored, so
  full-scale Phase 10-18 bundles are **not** directly reproducible from a clean
  checkout. `docs/REPRODUCIBILITY.md` documents the regeneration commands and
  pins expected hashes; that full-scale regeneration was not executed.
- Bit-identical results across different hardware are not claimed.
- Deleting a registry record file from the filesystem does release its claim.
  Not addressable in-process, which is why `state.json` remains the root of
  trust.

---

## 12. Reproduction instructions

Verify the ledger and every passed phase:

```bash
python -B -m bh_augmentation.autonomous_execution --repository .
```

Verify a production bundle by full replay, not just checksums:

```bash
python -B scripts/run_redesigned_ae_benchmark.py \
  --validate results/autonomous_execution/phase_14/corrected-20260728-851d64e-phase14-production-v3
```

Re-derive the confirmatory analysis:

```bash
python -B scripts/run_primary_confirmatory_analysis.py \
  --result-directory results/autonomous_execution/phase_15/corrected-20260729-13fbfde-phase15-confirmation-v1
```

Bounded production path on committed fixtures only:

```bash
python -B scripts/run_production_path_smoke.py --output-directory <fresh> --rows 300
```

Full gate:

```bash
python -B -m ruff check . && python -B -m pytest -q && git diff --check
```

See `docs/REPRODUCIBILITY.md` for the clean-checkout procedure and its limits.

---

## 13. Resume instructions

All 18 phases are passed; there is no roadmap work outstanding. To extend:

1. Read `results/autonomous_execution/state.json` — it is the root of trust.
2. Run the verifier above; it must report zero issues before any new work.
3. Any new outer-test evaluation needs **fresh** split assignments in a new
   immutable directory. Seeds 0-4 and 5-14 are consumed. Follow
   `configs/canonical_grouped_splits_phase15.yaml` as the template.
4. Never re-evaluate a consumed identity. The registry will refuse, and that
   refusal is correct.
5. Any new confirmatory claim requires its own preregistration, committed before
   the run, following `PRIMARY_EXPERIMENT.md`.

---

## 14. Branch and remote

Branch `main`, 14 commits ahead of `58613dd`. Remote `origin` is configured and
has **not** been contacted — nothing was fetched, pushed, or released.

Nothing here has been pushed, and the decision to push is the user's. If and
when that is wanted, the command is:

```bash
git push origin main
```

Review `git log 58613dd..HEAD` first. The push includes negative and null
results as first-class findings; that is intentional and should not be edited
out.
