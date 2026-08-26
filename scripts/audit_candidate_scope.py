#!/usr/bin/env python
"""Audit every candidate-eligibility call site and verify its declared semantics.

The audit does two things:

1. Walks the source tree and records, for every call that can reach the
   canonical identity gate, how that call supplies its eligibility rule --
   an explicit :class:`CandidateScopePolicy`, the legacy key-set keyword,
   forwarded keyword arguments, or nothing at all.
2. Checks each call site's module against the declared semantics in
   :mod:`bh_augmentation.augmentation.candidate_scope_registry` and fails when
   a module that must decide eligibility from the labeled low-data subset is
   found handing the complete measured universe to a generator instead.

Writing the audit as a script rather than a document is the point: a
markdown table cannot fail a build, and this is exactly the class of defect
that survives review because nothing re-checks it.

Usage::

    python scripts/audit_candidate_scope.py
    python scripts/audit_candidate_scope.py --output-directory results/corrected_candidate_scope_audit
    python scripts/audit_candidate_scope.py --check    # exit nonzero on drift, write nothing
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bh_augmentation.augmentation.candidate_scope import (  # noqa: E402
    CANDIDATE_SCOPE_CLASSIFICATIONS,
    OBSERVED_ONLY_LOW_DATA,
)
from bh_augmentation.augmentation.candidate_scope_registry import (  # noqa: E402
    CANDIDATE_SCOPE_REGISTRY_VERSION,
    declaration_for,
    registry_record,
)

CANDIDATE_SCOPE_AUDIT_SCHEMA_VERSION = "bh-candidate-scope-audit-v1"

#: Functions whose invocation decides, or forwards, candidate eligibility.
_GATE_FUNCTIONS = {
    "audit_candidate_identities": "identity_gate",
    "generate_condition_transfer_examples": "anonymous_generator",
    "generate_role_aware_condition_transfer_examples": "typed_generator",
    "generate_condition_recombined_candidates": "recombination_generator",
    "generate_condition_recombine_augmented_frame": "recombination_generator",
    "build_anonymous_transfer_pool": "scientific_pool",
    "build_strict_context_matched_typed_transfer_pool": "scientific_pool",
    "build_chemical_augmentation_control": "chemical_control",
    "build_condition_transfer_candidates": "external_family_generator",
}
#: Keyword arguments that carry an eligibility rule.
_EXPLICIT_KEYWORDS = {"candidate_scope"}
_LEGACY_KEYWORDS = {
    "measured_identity_keys",
    "measured_keys",
    "global_measured_identity_keys",
    "measured_canonical_keys",
}
#: Expressions that name the complete measured universe.
_GLOBAL_UNIVERSE_HINTS = (
    "saved.canonical",
    "complete_measured_identity_keys",
    "global_measured_keys",
    "global_identity_keys",
)
_SEARCH_ROOTS = ("src", "scripts", "tests")

_SUPPLY_EXPLICIT = "explicit_candidate_scope_policy"
_SUPPLY_LEGACY = "legacy_key_set_keyword"
_SUPPLY_FORWARDED = "forwarded_keyword_arguments"
_SUPPLY_DEFAULT = "generator_default_observed_only"


@dataclass(frozen=True, slots=True)
class CallSite:
    """One located call that can reach the canonical identity gate."""

    file: str
    line: int
    symbol: str
    role: str
    supply: str
    references_global_universe: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize one located call site."""
        return {
            "file": self.file,
            "line": self.line,
            "symbol": self.symbol,
            "role": self.role,
            "eligibility_supply": self.supply,
            "references_global_universe": self.references_global_universe,
        }


def discover_call_sites(root: Path = REPOSITORY_ROOT) -> list[CallSite]:
    """Locate every eligibility-deciding call in the repository."""
    sites: list[CallSite] = []
    for path in _python_files(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        relative = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            symbol = _called_name(node.func)
            role = _GATE_FUNCTIONS.get(symbol or "")
            if role is None:
                continue
            keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
            forwarded = any(keyword.arg is None for keyword in node.keywords)
            if keywords & _EXPLICIT_KEYWORDS:
                supply = _SUPPLY_EXPLICIT
            elif keywords & _LEGACY_KEYWORDS:
                supply = _SUPPLY_LEGACY
            elif forwarded:
                supply = _SUPPLY_FORWARDED
            else:
                supply = _SUPPLY_DEFAULT
            sites.append(
                CallSite(
                    file=relative,
                    line=node.lineno,
                    symbol=symbol or "<unknown>",
                    role=role,
                    supply=supply,
                    references_global_universe=_references_global_universe(node),
                )
            )
    return sorted(sites, key=lambda site: (site.file, site.line))


def build_audit(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Build the complete machine-readable candidate-scope audit."""
    sites = discover_call_sites(root)
    registry = registry_record()
    by_module: dict[str, list[CallSite]] = {}
    for site in sites:
        by_module.setdefault(site.file, []).append(site)

    modules: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    undeclared: list[str] = []
    for module, module_sites in sorted(by_module.items()):
        declaration = declaration_for(module)
        if declaration is None:
            if module.startswith("tests/"):
                classification = "not_applicable"
                mechanism = "test_fixture"
                rationale = (
                    "Test module. Fixtures exercise the gate directly and are "
                    "not a scientific claim."
                )
            else:
                undeclared.append(module)
                classification = "undeclared"
                mechanism = "undeclared"
                rationale = "No entry in candidate_scope_registry."
        else:
            classification = declaration.classification
            mechanism = declaration.mechanism
            rationale = declaration.rationale

        modules.append(
            {
                "module": module,
                "classification": classification,
                "mechanism": mechanism,
                "rationale": rationale,
                "affects": list(declaration.affects) if declaration else [],
                "call_sites": [site.to_dict() for site in module_sites],
            }
        )
        violations.extend(_module_violations(module, classification, module_sites))

    located = {entry["module"] for entry in modules}
    # Representation-level families never call the identity gate -- that is
    # precisely their semantic rule -- so they have no located call site. Fold
    # the declarations back in so the audit reports the complete taxonomy
    # rather than only the modules that happen to invoke a generator.
    for declaration in registry["declarations"]:
        if declaration["module"] in located:
            continue
        modules.append(
            {
                **declaration,
                "call_sites": [],
                "note": (
                    "Declared semantics with no located identity-gate call site."
                ),
            }
        )
    modules.sort(key=lambda entry: entry["module"])

    counts: dict[str, int] = dict.fromkeys(CANDIDATE_SCOPE_CLASSIFICATIONS, 0)
    for entry in modules:
        if entry["classification"] in counts:
            counts[entry["classification"]] += 1

    return {
        "schema_version": CANDIDATE_SCOPE_AUDIT_SCHEMA_VERSION,
        "registry_version": CANDIDATE_SCOPE_REGISTRY_VERSION,
        "classifications": list(CANDIDATE_SCOPE_CLASSIFICATIONS),
        "module_count": len(modules),
        "call_site_count": len(sites),
        "declared_without_call_site_count": sum(
            1 for entry in modules if not entry["call_sites"]
        ),
        "classification_counts": counts,
        "undeclared_modules": sorted(undeclared),
        "violations": violations,
        "modules": modules,
        "declarations": registry["declarations"],
    }


def _module_violations(
    module: str,
    classification: str,
    sites: list[CallSite],
) -> list[dict[str, Any]]:
    """Report a module whose wiring contradicts its declared semantics."""
    problems: list[dict[str, Any]] = []
    if classification == "undeclared":
        problems.append(
            {
                "module": module,
                "kind": "undeclared_module",
                "detail": (
                    "This module reaches the candidate identity gate but declares "
                    "no candidate-scope semantics. Add an entry to "
                    "candidate_scope_registry."
                ),
            }
        )
        return problems
    if classification != OBSERVED_ONLY_LOW_DATA:
        return problems
    for site in sites:
        if site.role in {"identity_gate"}:
            continue
        if site.supply == _SUPPLY_LEGACY and site.references_global_universe:
            problems.append(
                {
                    "module": module,
                    "kind": "global_universe_in_low_data_scope",
                    "detail": (
                        f"{module}:{site.line} calls {site.symbol} with a legacy key "
                        "set derived from the complete measured dataset, but the "
                        "module is declared observed_only_low_data. A low-data "
                        "learner cannot know complete-dataset membership."
                    ),
                    "line": site.line,
                    "symbol": site.symbol,
                }
            )
    return problems


def _references_global_universe(node: ast.Call) -> bool:
    """Whether any argument expression names the complete measured universe."""
    try:
        rendered = ast.unparse(node)
    except Exception:  # pragma: no cover - defensive on exotic nodes
        return False
    return any(hint in rendered for hint in _GLOBAL_UNIVERSE_HINTS)


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _python_files(root: Path) -> Iterator[Path]:
    for search_root in _SEARCH_ROOTS:
        for path in sorted((root / search_root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


def _render_summary(audit: dict[str, Any]) -> str:
    lines = [
        "Candidate-scope call-site audit",
        f"  schema        : {audit['schema_version']}",
        f"  modules       : {audit['module_count']}",
        f"  call sites    : {audit['call_site_count']}",
        "  classification:",
    ]
    for name, count in sorted(audit["classification_counts"].items()):
        lines.append(f"    {name:34s} {count}")
    lines.append(f"  declared-only : {audit['declared_without_call_site_count']}")
    lines.append(f"  undeclared    : {len(audit['undeclared_modules'])}")
    lines.append(f"  violations    : {len(audit['violations'])}")
    for violation in audit["violations"]:
        lines.append(f"    ! {violation['kind']}: {violation['detail']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Build, optionally write, and verify the candidate-scope audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        default="results/corrected_candidate_scope_audit",
        help="Directory to write the machine-readable audit into.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify declarations without writing any artifact.",
    )
    args = parser.parse_args(argv)

    audit = build_audit()
    print(_render_summary(audit))

    if not args.check:
        output = Path(args.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        target = output / "candidate_scope_call_sites.json"
        target.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {target}")

    if audit["violations"]:
        print(
            "\nFAIL: candidate-scope declarations do not match the code.",
            file=sys.stderr,
        )
        return 1
    print("\nOK: every candidate-scope call site matches its declared semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
