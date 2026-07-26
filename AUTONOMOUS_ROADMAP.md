# Autonomous Scientific Roadmap

This ledger governs the post–Batch 3 roadmap. Phases are executed in numeric order
unless a phase is externally blocked and a later phase is demonstrably independent.
A phase passes only after focused tests, the full suite, lint, diff validation, a
production-path smoke test, artifact validation, and a selective local checkpoint
commit all succeed.

Canonical inputs:

- `data/processed/bh_canonical_roles_v1.csv`
- `results/corrected_canonical_splits/`

Non-negotiable contracts:

- RDKit canonical seven-role identities and the Batch 3 canonicalization version
- group-safe saved outer assignments and cumulative low-data training subsets
- fixed validation/test assignments across fractions
- training-only augmentation, representation fitting, and policy selection
- no test-label or test-metric access during policy search
- immutable, hash-verifiable, non-overwriting scientific outputs
- paired evaluation with negative and null results retained
- no prospective laboratory claims

## Phase gates

1. Canonicalize and deduplicate synthetic candidates.
2. Enforce exact role-change semantics.
3. Apply uniform similarity, source caps, and deterministic ranking.
4. Integrate canonical nested splits into every corrected runner.
5. Separate policy search from final outer-test evaluation.
6. Repair AE internal validation and joint policy selection.
7. Rebuild product and reactant LOGO runners.
8. Add nested group-aware OOD validation.
9. Add scaffold, cluster, similarity, and condition OOD splits.
10. Re-establish no-augmentation representation baselines.
11. Add simple augmentation controls with matched budgets.
12. Measure pseudo-label accuracy and calibrate uncertainty.
13. Add low-complexity representation-learning baselines.
14. Redesign and reassess the supervised AE.
15. Freeze and execute the primary confirmatory hypothesis.
16. Add an external typed-dataset adapter and Suzuki–Miyaura validation.
17. Complete strict reproducibility and immutable packaging.
18. Rebuild recommendation simulations and prepare a prospective package.

Operational state is authoritative in
`results/autonomous_execution/state.json`. Events are append-only in
`results/autonomous_execution/events.jsonl`. A passed phase is automatically
reopened if its required artifacts or dependency hashes no longer validate.
