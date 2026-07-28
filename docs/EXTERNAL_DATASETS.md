# External typed-reaction datasets (Phase 16)

## Status

```
implementation passed
external empirical validation blocked
```

**Reason for the blocked status:** no licensed Suzuki data present locally; acquisition
requires a human with network access.

The only reaction data in this repository is Buchwald–Hartwig, derived from the
Therapeutics Data Commons `Yields(name="Buchwald-Hartwig")` export
(`data/raw/buchwald_hartwig_tdc.csv`). There is no Suzuki–Miyaura dataset anywhere in
the tree, and nothing in this repository downloads one. Every Suzuki artefact here —
the adapter, the canonicalization, the grouped splits, the typed condition transfer,
and the search / final-evaluation smoke — is exercised against a clearly labelled
**synthetic fixture** (`tests/suzuki_fixture.py`) whose yields are a deterministic
arithmetic function of the row index.

**No empirical Suzuki–Miyaura result is claimed, produced, or implied by this phase.**
The empirical arm remains blocked until a human obtains a real dataset under its own
license.

---

## What this phase actually answers

The scientific question was: *can the canonical typed-reaction, transfer, uncertainty
and evaluation framework be applied to a chemically distinct reaction family without
redesigning the system around that dataset?*

The implementation answer is yes, and the seam is one object.

### Reused unchanged (family-agnostic)

| Capability | Where it lives |
| --- | --- |
| RDKit canonicalization of one molecule (no neutralization, no tautomer handling, stereochemistry preserved) | `bh_augmentation.data.canonicalize_roles.canonicalize_smiles` |
| Stable identity encodings (`stable_json`, `sha256_text`, `source_row_id`) | `bh_augmentation.data.canonicalize_roles` |
| Complete-group train/validation/test separation and nested low-data prefixes | `bh_augmentation.data.canonical_splits.build_grouped_outer_assignments`, `build_nested_low_data_assignments`, `split_assignment_hash` |
| Validation-only policy search, policy freezing, single-shot outer-test evaluation | `bh_augmentation.evaluation.policy_protocol` |
| Regression metrics | `bh_augmentation.evaluation.metrics` |
| Hashing, manifests, feature contracts | `bh_augmentation.utils.corrected_runs` |
| Morgan fingerprints and feature-block metadata | `bh_augmentation.features.featurize`, `bh_augmentation.features.compatibility` |

### Family-specific (declared once, in an adapter)

`bh_augmentation.data.reaction_family.ReactionFamilyAdapter` holds, and holds *only*:
family id and name; the ordered tuple of typed role names; the role → source column
mapping; which roles are substrates, which are conditions, which subset of conditions
is transferable, and which is the product; the family-scoped canonicalization version
and reaction-key schema version; the reaction-SMILES assembly order; a **required**
`DatasetProvenance` record; and a family-specific eligibility hook.

The Buchwald–Hartwig adapter is built *from* the existing constants
(`CANONICAL_ROLE_NAMES`, `CANONICAL_ROLE_COLUMNS`, `CANONICALIZATION_VERSION`,
`REACTION_KEY_SCHEMA_VERSION`), so it is definitionally identical to today's behaviour;
`tests/test_reaction_family_adapter.py` asserts that the adapter reproduces
`canonicalize_reaction_roles_dataframe` key-for-key. The existing BH code path is
untouched.

### Suzuki–Miyaura typed roles

| Role | Class | Meaning |
| --- | --- | --- |
| `organohalide` | substrate | Electrophile consumed by oxidative addition: Ar–I/Br/Cl, heteroaryl and alkenyl halides, and OTf/OTs/OMs pseudohalides. Not called `aryl_halide`, because heteroaryl/alkenyl electrophiles and triflates are routine. |
| `organoboron` | substrate | Nucleophilic transmetalating partner: boronic acid, pinacol/neopentyl boronate, potassium trifluoroborate, MIDA boronate. |
| `catalyst` | condition | Pd (or Ni) source / precatalyst. |
| `ligand` | condition, **transferable** | Supporting phosphine or NHC. |
| `base` | condition, **transferable** | Base that activates boron toward transmetalation. |
| `solvent_or_additive` | condition, **transferable** | Reaction medium plus any additive reported as part of the medium. Public HTE exports usually report a solvent *system* rather than separable solvent/additive fields. |
| `product` | product | Measured coupled product. |

Buchwald–Hartwig terminology is deliberately not reused: a Suzuki coupling has an
organohalide electrophile and an organoboron nucleophile, not an "aryl halide + amine",
and `reactant_1` / `reactant_2` are positional labels that assert nothing chemical.

**Why the boron reagent is a substrate, not a condition.** A condition role describes
*how* two partners are coupled; a substrate role describes *what* is coupled. The
organoboron reagent contributes the carbon fragment that ends up in the product
skeleton, so replacing it changes the product constitution and therefore changes which
reaction is being measured. Typed condition transfer must preserve substrate and
product identity; treating boron as transferable would silently fabricate a different
reaction and attach a donor's yield to it. Boron and base are mechanistically coupled
(base activates boron for transmetalation), but that is a reason to keep base transfer
honest, not a reason to reclassify boron.

`catalyst` is a condition role but is excluded from the *default* transferable set,
because in most Suzuki HTE designs the Pd source is held fixed or is bound to the
ligand as a precatalyst while ligand, base and solvent are varied. Widen it explicitly
via `reaction_family.transferable_roles` if your dataset actually varies it.

---

## Which Suzuki–Miyaura dataset to obtain

> **Verify every URL, DOI and license below against the primary source before use.**
> These are pointers to real, well-known work, but licensing terms for supplementary
> data change and are frequently not stated explicitly. Do not record a license string
> in a config unless you have confirmed it. If you cannot confirm it, record
> `"verify before use"` — that is an accepted value and is far better than a false one.

### Option A (recommended): Perera *et al.* flow Suzuki–Miyaura HTE screen

- **What it is.** A nanomole-scale automated flow screen of Suzuki–Miyaura couplings
  varying ligand, base and solvent across a set of electrophile/boron pairs. This is the
  dataset most commonly referred to in the ML literature as "the Suzuki HTE dataset",
  and its design (a few substrate pairs × a condition grid) is exactly the shape this
  framework expects.
- **Citation.** Perera, D.; Tucker, J. W.; Brahmbhatt, S.; Helal, C. J.; Chong, A.;
  Farrell, W.; Richardson, P.; Sach, N. W. "A platform for automated nanomole-scale
  reaction screening and micromole-scale synthesis in flow." *Science* **2018**, *359*,
  429–434.
- **How to obtain it.** Retrieve the paper's supplementary data from the publisher's
  article page (search the title on the publisher site; do not rely on a URL copied from
  here). Many downstream ML repositories also redistribute a tabular version — if you
  use a redistribution, cite *both* the original paper and the redistribution, and record
  the redistribution's license, not the paper's.
- **License.** Publisher supplementary material; terms vary. **Verify before use.**

### Option B: Open Reaction Database (ORD)

- **What it is.** An open repository of reaction data in a structured schema, including
  Suzuki–Miyaura HTE submissions (the Perera dataset among them).
- **Where.** `https://open-reaction-database.org` and the data repository
  `https://github.com/open-reaction-database/ord-data`. **Verify both before use.**
- **Why you might prefer it.** ORD records already carry structured roles and machine-readable
  provenance, which maps cleanly onto `ReactionFamilyAdapter.role_columns`.
- **License.** ORD states a license on its site and in its repository. **Read it and record
  it verbatim; do not assume CC-BY.**

### Option C: Reizman *et al.* automated Suzuki optimization

- **What it is.** A much smaller closed-loop optimization dataset over Suzuki couplings
  (ligand and condition variation). Useful as a secondary check, too small to be a primary
  external-validation set.
- **Citation.** Reizman, B. J.; Wang, Y.-M.; Buchwald, S. L.; Jensen, K. F.
  "Suzuki–Miyaura cross-coupling optimization enabled by automated feedback."
  *Reaction Chemistry & Engineering* **2016**, *1*, 658–666.
- **License.** Publisher supplementary material. **Verify before use.**

### What is *not* available here

Therapeutics Data Commons, which supplies this repository's Buchwald–Hartwig export,
does not provide a Suzuki–Miyaura set in the `Yields` group as used here. Check the
current TDC dataset list before assuming that is still true.

---

## Required file layout and columns

Put the file at:

```
data/external/suzuki_miyaura/suzuki_miyaura_hte.csv
```

(Or anywhere else, and point `dataset.path` at it. That directory does not exist yet;
create it. It is not tracked and nothing here writes to it.)

The CSV must contain one row per measured reaction, with these columns:

| Column | Required | Contents |
| --- | --- | --- |
| `smiles_organohalide` | yes | SMILES of the electrophile |
| `smiles_organoboron` | yes | SMILES of the boron reagent |
| `smiles_catalyst` | yes | SMILES of the Pd/Ni source |
| `smiles_ligand` | yes | SMILES of the ligand |
| `smiles_base` | yes | SMILES of the base |
| `smiles_solvent_or_additive` | yes | SMILES of the solvent / medium |
| `smiles_product` | yes | SMILES of the coupled product |
| `yield` | yes | Numeric yield (0–100) |
| `reaction_id` | optional | Any stable source identifier, carried through untouched |

If your file uses different column names, do **not** rename the file's columns — map
them in the config instead, via `reaction_family.role_columns` (positionally aligned
with the role order in the table above). That keeps the raw file byte-identical to what
you downloaded, which is what the provenance SHA-256 attests to.

Two naming choices in this schema are deliberate and worth knowing about:

- Columns are `smiles_<role>`, not `<role>_smiles`. `ligand_smiles`, `base_smiles` and
  `additive_smiles` are *deprecated Buchwald–Hartwig component columns*
  (`bh_augmentation.data.clean_data.DEPRECATED_COMPONENT_COLUMNS`), and shipped configs
  are checked for them by `tests/test_canonical_representation.py`. The Suzuki columns
  are unrelated to those legacy BH fields, so they are named so that they cannot be
  confused with them.
- The representation is declared as `features.representation_kind`, not `features.kind`.
  `features.kind` is reserved for the Buchwald–Hartwig representation vocabulary
  enumerated in `tests/test_canonical_representation.py`; a family-scoped representation
  such as `suzuki_miyaura_role_separated` is not a member of it. Config validation
  rejects an external-family config that uses `features.kind`.

Rows are never silently dropped. Every row is canonicalized and audited; rows failing
the family eligibility rule (boron reagent containing no boron, electrophile with no
C–Cl/Br/I bond and no sulfonate-ester pseudohalide, catalyst containing neither Pd nor
Ni, or a product identical to a substrate) are recorded with their reason in
`family_eligibility_audit.csv` and excluded from fitting.

---

## Provenance fields you must fill in

All six are **required**; config validation rejects the file if any is missing or blank
(`tests/test_suzuki_external_config.py` asserts this for each field individually).

| Field | What to put |
| --- | --- |
| `source_name` | The named dataset and its origin, e.g. `"Suzuki-Miyaura flow HTE screen, Perera et al., Science 2018"` |
| `source_url` | The exact URL you downloaded from |
| `license` | The license or terms of use that actually apply. If unconfirmed, write `"verify before use"` |
| `citation` | Full citation of the primary publication, plus the redistribution if you used one |
| `retrieved_date` | ISO-8601 date on which you downloaded the file |
| `raw_file_sha256` | `shasum -a 256 <your file>` — 64 lowercase hex characters |

The runner hashes the file on disk and **refuses to run** if it does not match
`raw_file_sha256`. That binds every result to a specific byte sequence, not to a
filename.

---

## Exact commands, once the data is present

```bash
# 0. Place the file and record its digest.
mkdir -p data/external/suzuki_miyaura
cp /path/to/your/download.csv data/external/suzuki_miyaura/suzuki_miyaura_hte.csv
shasum -a 256 data/external/suzuki_miyaura/suzuki_miyaura_hte.csv

# 1. Paste that digest into configs/suzuki_external_validation.yaml as
#    reaction_family.provenance.raw_file_sha256, and fill in the other five
#    provenance fields. Map role_columns if your column names differ.

# 2. Confirm the config still validates (this does not touch the data).
python -B -m pytest -q tests/test_suzuki_external_config.py

# 3. Run search + final evaluation. Writes to output.directory, which must not
#    already exist or must be empty.
python -B -m bh_augmentation.run_external_family_validation \
    --config configs/suzuki_external_validation.yaml

# 4. Verify the outputs against the manifest.
python -B - <<'PY'
import hashlib, json, pathlib
directory = pathlib.Path("results/external/suzuki_miyaura_external_validation")
manifest = json.loads((directory / "run_manifest.json").read_text())
for name, digest in manifest["output_file_hashes"].items():
    actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    print(f"{'OK ' if actual == digest else 'BAD'} {name}")
PY
```

Outputs written to `output.directory`:

| File | Contents |
| --- | --- |
| `search_metrics.csv` | Validation-only metrics for every policy, per evaluation unit |
| `final_test_metrics.csv` | Outer-test metrics, one evaluation per unit, tagged with the frozen policy hash |
| `frozen_policy__<unit>.json` | Hash-verified frozen policy, written before any test access |
| `split_overlap_audit.csv` | Group-overlap audit for every seed |
| `family_eligibility_audit.csv` | Per-row canonicalization and eligibility outcome |
| `synthetic_candidate_audit.csv` | Every typed-transfer candidate with its accept/reject reason |
| `run_manifest.json` | Config hash, dataset hash, provenance, feature contract, split hashes, frozen policy hashes, and SHA-256 of every other output |

---

## Scientific contracts that still hold for an external family

These are enforced by the same code that enforces them for Buchwald–Hartwig, not by a
parallel implementation:

- **Canonical identity.** Chemical identity is an RDKit canonical reaction key scoped by
  the family key-schema version (`suzuki-seven-role-v1`). Fingerprint or feature-vector
  equality is never treated as chemical identity.
- **Grouped splits.** Complete canonical-reaction groups are assigned to exactly one of
  train / validation / test; replicates of the same canonical reaction can never straddle
  a split. Low-data subsets are cumulative group prefixes.
- **Training-only augmentation and teacher fitting.** Typed condition transfer refuses to
  read any row whose `outer_split` is not `train`, and pseudo-label teachers are fit on
  training rows only.
- **Typed transfer.** Only roles in `transferable_roles` may change. An accepted candidate
  has byte-identical canonical substrate and product keys to its source. A candidate whose
  canonical reaction key already exists in the measured data is rejected as
  `already_measured`.
- **Validation-only selection, frozen before test.** Policies are scored on validation
  only, then frozen into a hash-verified envelope bound to the dataset hash, split hashes,
  canonicalization version, feature-metadata hash and config hash. The test partition is
  touched only after freezing.
- **One test evaluation per unit.** A second evaluation of the same
  `family|seed|train_fraction` unit raises.
- **Immutable, hash-verifiable outputs.** The runner refuses to write into a non-empty
  directory, and the manifest records the SHA-256 of every file it wrote.

---

## Reproducing the local (synthetic) demonstration

```bash
python -B -m pytest -q \
    tests/test_reaction_family_adapter.py \
    tests/test_suzuki_family_pipeline.py \
    tests/test_suzuki_external_validation_smoke.py \
    tests/test_suzuki_external_config.py
```

These run entirely on the synthetic fixture and write only into pytest's `tmp_path`.
They demonstrate that the implementation works. They do **not** constitute empirical
validation on Suzuki–Miyaura chemistry, and must never be reported as if they did.
