# Buchwald-Hartwig Condition Reader

`bh_augmentation.data.bh_condition_reader` is a small, isolated inspector for
the processed Buchwald-Hartwig CSVs used in this repo.

The parser assumes the current processed `reaction_smiles` format has exactly
six dot-separated left-side tokens before `>>`:

```text
reactant_1.reactant_2.catalyst_or_precatalyst.ligand.base.solvent_or_additive>>product
```

It can recover:

- reactant 1 and reactant 2 identities
- catalyst or precatalyst identity
- ligand identity
- base identity
- solvent/additive-like identity
- product identity
- the full condition block

It cannot recover temperature unless the processed CSV already has a non-empty,
non-`UNKNOWN` temperature column. Temperature is not encoded as an explicit field
in the observed six-token `reaction_smiles`, so missing temperature is marked
`NOT_RECOVERABLE` rather than inferred.

These role assignments are dataset-specific positional assumptions for the
processed Buchwald-Hartwig files. They should not be generalized silently to
other reaction datasets or raw TDC exports.

## Role validation and repair

The reader first applies the dataset-specific positional token assignment, then
checks whether the assigned condition roles make chemical sense using
conservative deterministic heuristics.

The validator looks for obvious transition-metal catalyst markers, phosphine or
Buchwald-ligand-like components, common inorganic or organic bases, and
solvent/additive-like neutral molecules. If exactly one condition token can be
assigned to each required role, and that assignment differs from the positional
order, the reader repairs the recovered catalyst, ligand, base, and
solvent/additive fields. Repaired rows are marked with
`role_validation_status = repaired`.

Nucleophile detection is reactant-position-aware. A neutral N-containing
molecule may be a nucleophilic substrate in token 1, but inside the condition
block the reader first looks for catalyst, ligand, and base roles. If those are
uniquely identified and exactly one neutral organic condition token remains, it
is treated as the solvent/additive-like component and the validation notes
record `remaining_condition_token_assigned_to_solvent_or_additive`.

If multiple tokens could plausibly fill the same role, or a role cannot be
identified confidently, the reader refuses to guess. It preserves the original
positional fields and records `role_validation_status = ambiguous` with notes
describing the token classifications. Parse failures are marked
`invalid_or_unresolved`.

These heuristics are intended for inspection before model training, not as a
general chemistry parser. Temperature still cannot be recovered unless it is
explicitly present as a non-empty, non-`UNKNOWN` column in the input CSV.

Preview usage:

```bash
python scripts/inspect_bh_condition_reader.py \
  --input data/processed/bh_clean_stress.csv \
  --nrows 25
```
