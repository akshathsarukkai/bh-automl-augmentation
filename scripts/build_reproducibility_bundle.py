"""Build a LOCAL reproducibility bundle for one scientific result directory.

The bundle collects everything a human needs to re-derive a result: the run
manifest, the configuration, the resolved dependency and hardware record, the
git commit, the hashes of every input the run consumed, and the exact commands
to reproduce it.

This tool is deliberately inert:

* it never publishes anything and never performs any network access;
* it only writes into the single bundle directory named by the caller;
* it refuses to write into an existing path.

Example
-------
    python -B scripts/build_reproducibility_bundle.py \
        --result-directory results/corrected_low_complexity_benchmark_phase13 \
        --bundle-directory /tmp/phase13-repro-bundle \
        --config configs/corrected_low_complexity_benchmark_phase13.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bh_augmentation.utils.corrected_runs import sha256_file, stable_hash  # noqa: E402
from bh_augmentation.utils.scientific_manifest import (  # noqa: E402
    ManifestError,
    dependency_versions,
    platform_record,
    read_manifest,
    verify_manifest,
)
from bh_augmentation.utils.strict_config import (  # noqa: E402
    require_fresh_output_directory,
)

BUNDLE_SCHEMA_VERSION = "bh-reproducibility-bundle-v1"
BUNDLE_FILES = (
    "run_manifest.json",
    "environment.json",
    "input_hashes.json",
    "git.json",
    "REPRODUCE.md",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPOSITORY_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_record() -> dict[str, Any]:
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--always", "--dirty"),
        "dirty_at_bundle_time": bool(_git("status", "--porcelain")),
        "repository_root": str(REPOSITORY_ROOT),
    }


def _candidate_inputs(
    manifest: Mapping[str, Any],
    extra_inputs: Sequence[str],
) -> list[Path]:
    candidates: list[Path] = []

    def _add(value: Any) -> None:
        if not isinstance(value, str) or not value.strip():
            return
        path = Path(value)
        if not path.is_absolute():
            path = REPOSITORY_ROOT / path
        if path not in candidates:
            candidates.append(path)

    _add(manifest.get("dataset_path"))
    scientific = manifest.get("resolved_scientific_config")
    if isinstance(scientific, Mapping):
        for key in (
            "dataset_path",
            "canonical_split_directory",
            "split_directory",
            "source_path",
        ):
            _add(scientific.get(key))
    for value in extra_inputs:
        _add(value)
    return candidates


def _input_hashes(paths: Sequence[Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if path.is_file():
            records.append(
                {
                    "path": _relative(path),
                    "kind": "file",
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    records.append(
                        {
                            "path": _relative(child),
                            "kind": "file",
                            "sha256": sha256_file(child),
                            "size_bytes": child.stat().st_size,
                        }
                    )
        else:
            records.append(
                {
                    "path": _relative(path),
                    "kind": "missing",
                    "sha256": None,
                    "size_bytes": None,
                    "note": (
                        "Input is not present in this checkout and must be supplied "
                        "by a human before reproduction."
                    ),
                }
            )
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "input_count": len(records),
        "missing_input_count": sum(1 for item in records if item["kind"] == "missing"),
        "inputs": records,
    }


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def _reproduce_document(
    *,
    result_directory: Path,
    manifest: Mapping[str, Any],
    git_record: Mapping[str, Any],
    input_record: Mapping[str, Any],
    config_copy: str | None,
) -> str:
    command = str(manifest.get("command", "")).strip()
    missing = [item for item in input_record["inputs"] if item["kind"] == "missing"]
    lines = [
        "# Reproducing this result",
        "",
        f"- Result directory: `{_relative(result_directory)}`",
        f"- Manifest schema: `{manifest.get('schema_version', 'unknown')}`",
        f"- Manifest hash: `{manifest.get('manifest_hash', 'unknown')}`",
        f"- Recorded git commit: `{manifest.get('git_commit', 'unknown')}`",
        f"- Worktree dirty at execution: `{manifest.get('git_dirty_at_execution', 'unknown')}`",
        f"- Bundle built from commit: `{git_record['commit']}`",
        "",
        "## 1. Check out the exact code",
        "",
        "```bash",
        f"git worktree add /tmp/bh-repro {manifest.get('git_commit', '<commit>')}",
        "cd /tmp/bh-repro",
        "python -m venv .venv && . .venv/bin/activate",
        'python -m pip install -e ".[dev]"',
        "python -m pip install rdkit",
        "```",
        "",
        "## 2. Supply the inputs",
        "",
    ]
    if missing:
        lines.append(
            "The following inputs are not present in the checkout and must be "
            "supplied by a human (they are gitignored generated or licensed data):"
        )
        lines.append("")
        lines.extend(f"- `{item['path']}`" for item in missing)
    else:
        lines.append("Every recorded input is present in this checkout.")
    lines.extend(
        [
            "",
            "Verify each input against `input_hashes.json` in this bundle:",
            "",
            "```bash",
            "python -B - <<'PY'",
            "import json",
            "from bh_augmentation.utils.corrected_runs import sha256_file",
            "record = json.load(open('input_hashes.json'))",
            "for item in record['inputs']:",
            "    if item['kind'] == 'file':",
            "        assert sha256_file(item['path']) == item['sha256'], item['path']",
            "print('inputs verified')",
            "PY",
            "```",
            "",
            "## 3. Re-run the workflow",
            "",
            "```bash",
            command if command else "# No command was recorded in the manifest.",
            "```",
            "",
            "## 4. Verify the outputs",
            "",
            "```bash",
            "python -B - <<'PY'",
            "from bh_augmentation.utils.scientific_manifest import verify_manifest",
            f"print(verify_manifest('{_relative(result_directory)}').to_dict())",
            "PY",
            "```",
            "",
            "A rerun on different hardware, BLAS threading, or dependency versions",
            "may not reproduce outputs bit-for-bit. `environment.json` records the",
            "platform, CPU, thread settings, and dependency versions of the original",
            "run so any divergence can be attributed rather than guessed at.",
            "",
        ]
    )
    if config_copy:
        lines.extend(
            [
                "## Configuration",
                "",
                f"The exact configuration is copied into this bundle as `{config_copy}`.",
                "",
            ]
        )
    return "\n".join(lines)


def build_bundle(
    *,
    result_directory: str | Path,
    bundle_directory: str | Path,
    config_path: str | Path | None = None,
    extra_inputs: Sequence[str] = (),
    verify: bool = True,
) -> Path:
    """Write a self-contained local reproducibility bundle."""
    result = Path(result_directory)
    if not result.is_dir():
        raise ManifestError(f"Result directory does not exist: {result}")
    bundle = require_fresh_output_directory(bundle_directory, name="bundle directory")

    manifest = read_manifest(result)
    verification = verify_manifest(result).to_dict() if verify else None

    bundle.mkdir(parents=True, exist_ok=False)
    (bundle / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    config_copy: str | None = None
    if config_path is not None:
        source = Path(config_path)
        if not source.is_file():
            raise ManifestError(f"Config file does not exist: {source}")
        config_copy = f"config/{source.name}"
        (bundle / "config").mkdir()
        shutil.copy2(source, bundle / "config" / source.name)

    environment = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "recorded_at_execution": {
            "dependency_versions": manifest.get("dependency_versions"),
            "platform_record": manifest.get("platform_record"),
        },
        "observed_at_bundle_time": {
            "dependency_versions": dependency_versions(),
            "platform_record": platform_record(),
        },
    }
    (bundle / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    git_record = _git_record()
    (bundle / "git.json").write_text(json.dumps(git_record, indent=2, sort_keys=True) + "\n")

    input_record = _input_hashes(_candidate_inputs(manifest, extra_inputs))
    (bundle / "input_hashes.json").write_text(
        json.dumps(input_record, indent=2, sort_keys=True) + "\n"
    )

    (bundle / "REPRODUCE.md").write_text(
        _reproduce_document(
            result_directory=result,
            manifest=manifest,
            git_record=git_record,
            input_record=input_record,
            config_copy=config_copy,
        )
    )

    contents = {
        name: sha256_file(bundle / name)
        for name in BUNDLE_FILES
        if (bundle / name).is_file()
    }
    if config_copy:
        contents[config_copy] = sha256_file(bundle / config_copy)
    bundle_manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "published": False,
        "network_access": False,
        "result_directory": _relative(result),
        "result_manifest_hash": manifest.get("manifest_hash"),
        "result_verification": verification,
        "bundle_file_hashes": contents,
        "missing_input_count": input_record["missing_input_count"],
    }
    bundle_manifest["bundle_hash"] = stable_hash(bundle_manifest)
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n"
    )
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local reproducibility bundle builder."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result-directory", required=True)
    parser.add_argument(
        "--bundle-directory",
        required=True,
        help="Fresh path to write the bundle into. Must not already exist.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", action="append", default=[], dest="inputs")
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Bundle a result whose outputs currently fail hash verification.",
    )
    args = parser.parse_args(argv)
    bundle = build_bundle(
        result_directory=args.result_directory,
        bundle_directory=args.bundle_directory,
        config_path=args.config,
        extra_inputs=args.inputs,
        verify=not args.skip_verification,
    )
    print(f"Wrote local reproducibility bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
