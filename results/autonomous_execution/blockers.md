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
