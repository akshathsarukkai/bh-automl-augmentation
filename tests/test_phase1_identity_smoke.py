"""Contract tests for the Phase 1 production-path smoke runner."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.run_phase1_identity_smoke import run_smoke


def test_smoke_refuses_to_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_smoke(output)
