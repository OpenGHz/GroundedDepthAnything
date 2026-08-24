from __future__ import annotations

import importlib.util
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "setup-fingerprint.py"
SPEC = importlib.util.spec_from_file_location("gda_setup_fingerprint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
setup_fingerprint_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup_fingerprint_module)

PLATFORM_SCRIPT = PROJECT_ROOT / "scripts" / "check-setup-platform.py"
PLATFORM_SPEC = importlib.util.spec_from_file_location("gda_check_setup_platform", PLATFORM_SCRIPT)
assert PLATFORM_SPEC is not None and PLATFORM_SPEC.loader is not None
setup_platform_module = importlib.util.module_from_spec(PLATFORM_SPEC)
PLATFORM_SPEC.loader.exec_module(setup_platform_module)


def _manifest_fingerprint(path: Path) -> str:
    match = re.search(
        r"^export GDA_SETUP_INPUTS_SHA256=([0-9a-f]{64})$",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert match is not None
    return match.group(1)


def test_platform_manifests_use_current_shared_fingerprint() -> None:
    expected = setup_fingerprint_module.setup_fingerprint(PROJECT_ROOT)
    assert _manifest_fingerprint(PROJECT_ROOT / "scripts/setup-h200.sh") == expected
    assert _manifest_fingerprint(PROJECT_ROOT / "scripts/setup-b300.sh") == expected


def test_checkout_cannot_switch_pixi_platform(tmp_path: Path) -> None:
    marker = tmp_path / ".pixi" / "gda-platform"
    setup_platform_module.ensure_platform("h200", marker)
    setup_platform_module.ensure_platform("h200", marker)
    assert marker.read_text(encoding="utf-8") == "h200\n"

    try:
        setup_platform_module.ensure_platform("b300", marker)
    except RuntimeError as exc:
        assert "separate checkout" in str(exc)
    else:
        raise AssertionError("platform switch must be rejected")
