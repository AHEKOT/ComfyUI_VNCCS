"""Tamper-resistant regression tests for the mandatory VNCCS security scanner."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace

from scripts.security_scan import (
    COMFY_REGISTRY_ISSUE_TYPES,
    SCANNER_RULE_IDS,
    main as scanner_main,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCANNER = PROJECT_ROOT / "scripts" / "security_scan.py"
EXPECTED_SCANNER_SHA256 = "166e20b06e126b6c5dcf0923f2de2a00137e4c97bdd108bd2525cab75a182b20"
EXPECTED_RULE_IDS = {
    "BLACKLISTED_URL",
    "CREDENTIAL_URL",
    "CREDENTIAL_SOURCE",
    "ENVIRONMENT_ACCESS",
    "HF_CREDENTIALS_DISABLED",
    "JS_BIND_SCANNER_TRIGGER",
    "JS_CONNECT_SCANNER_TRIGGER",
    "JS_DYNAMIC_EXECUTION",
    "JS_EXTERNAL_NETWORK",
    "JS_REQUESTS_SCANNER_TRIGGER",
    "PY_COMMAND_EXECUTION",
    "PY_DYNAMIC_EXECUTION",
    "PY_DYNAMIC_IMPORT",
    "PY_NETWORK_CLIENT",
    "PY_PRIVILEGE_ESCALATION",
    "PY_TEMP_DIRECTORY_ACCESS",
    "PY_URL_COMMAND_EXECUTION",
    "REMOVED_BEN2_CODE",
}
EXPECTED_COMFY_REGISTRY_ISSUE_TYPES = {
    "contains_blacklisted_url",
    "python_bytecode_manipulation",
    "python_command_injection_risk",
    "python_dynamic_execution",
    "python_environment_manipulation",
    "python_network_operations",
    "python_privilege_escalation",
    "python_url_command_execution",
}


def _run_scanner(root: Path, *extra_args: str):
    stdout = io.StringIO()
    stderr = io.StringIO()
    returncode = 0
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            returncode = scanner_main(["--root", str(root), *extra_args])
        except SystemExit as exc:
            returncode = int(exc.code or 0)
    return SimpleNamespace(
        returncode=returncode,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
    )


def test_security_scanner_source_is_pinned():
    digest = hashlib.sha256(SCANNER.read_bytes()).hexdigest()
    assert digest == EXPECTED_SCANNER_SHA256, (
        "The security scanner changed. Do not weaken, bypass, baseline, or allowlist findings. "
        "Changing this pinned scanner requires explicit project-owner approval."
    )


def test_security_rule_catalogs_are_complete():
    assert SCANNER_RULE_IDS == EXPECTED_RULE_IDS
    assert COMFY_REGISTRY_ISSUE_TYPES == EXPECTED_COMFY_REGISTRY_ISSUE_TYPES


def test_security_scanner_detects_every_mandatory_rule(tmp_path):
    credential_names = (
        "HF" + "_TOKEN",
        "HUGGING" + "_FACE_HUB_TOKEN",
        "hf" + "_token",
        "civitai" + "_token",
    )
    removed_model = "BE" + "N2"
    url_command = "ff" + "mpeg"
    blacklisted_url = "https://raw.github" + "usercontent.com/example/project/main/file.py"
    credential_url = "https://user" + ":password@example.invalid/file"
    privilege_command = "s" + "udo"
    (tmp_path / "unsafe.py").write_text(
        f"""
import importlib
import os
import requests
import subprocess
import tempfile
from huggingface_hub import hf_hub_download

SECRET_NAMES = {credential_names!r}
MODEL = {removed_model!r}
URL_COMMAND = [{url_command!r}, "-i", "input.webm"]
SOURCE = {blacklisted_url!r}
CREDENTIAL_SOURCE = {credential_url!r}
PRIVILEGE_COMMAND = [{privilege_command!r}, "tool"]

os.environ.get("SECRET")
os.getenv("SECRET")
os.system("id")
tempfile.gettempdir()
eval("1 + 1")
exec("value = 1")
compile("value = 1", "fixture", "exec")
importlib.import_module("example")
requests.get("https://example.invalid")
subprocess.run(["tool"])
hf_hub_download(repo_id="example/project", filename="model.bin")
""",
        encoding="utf-8",
    )
    (tmp_path / "unsafe.js").write_text(
        """
const callback = handler.bind(context);
transport.connect(target);
Requests.get(sequence);
eval("1 + 1");
const generated = new Function("return 1");
fetch("https://example.invalid/data");
""",
        encoding="utf-8",
    )

    result = _run_scanner(tmp_path, "--json")
    assert result.returncode == 1
    findings = json.loads(result.stdout)
    found_rule_ids = {finding["rule_id"] for finding in findings}
    assert found_rule_ids == EXPECTED_RULE_IDS


def test_security_scanner_scans_repository_tests(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "unsafe_test.py").write_text(
        'eval("test code")\n',
        encoding="utf-8",
    )

    result = _run_scanner(tmp_path, "--json")
    assert result.returncode == 1
    findings = json.loads(result.stdout)
    assert any(
        finding["rule_id"] == "PY_DYNAMIC_EXECUTION"
        and finding["path"] == "tests/unsafe_test.py"
        for finding in findings
    )


def test_security_scanner_has_no_allowlist_interface(tmp_path):
    result = _run_scanner(tmp_path, "--allowlist", "anything")
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_security_scanner_blocks_import_alias_bypasses(tmp_path):
    (tmp_path / "aliases.py").write_text(
        """
import importlib as module_loader
import os as operating_system
import tempfile as temporary_files
from http import client
from huggingface_hub import hf_hub_download as download_public_asset
from tempfile import gettempdir as temporary_root
from urllib import request

operating_system.getenv("SECRET")
module_loader.import_module("example")
temporary_files.gettempdir()
temporary_root()
download_public_asset(repo_id="example/project", filename="model.bin")
""",
        encoding="utf-8",
    )

    result = _run_scanner(tmp_path, "--json")
    assert result.returncode == 1
    findings = json.loads(result.stdout)
    found_rule_ids = {finding["rule_id"] for finding in findings}
    assert {
        "ENVIRONMENT_ACCESS",
        "HF_CREDENTIALS_DISABLED",
        "PY_DYNAMIC_IMPORT",
        "PY_NETWORK_CLIENT",
        "PY_TEMP_DIRECTORY_ACCESS",
    } <= found_rule_ids


def test_security_scanner_blocks_all_privilege_escalation_commands(tmp_path):
    commands = (
        "s" + "udo",
        "s" + "u",
        "do" + "as",
        "pk" + "exec",
        "run" + "as",
    )
    for index, command in enumerate(commands):
        (tmp_path / f"privilege_{index}.py").write_text(
            f"COMMAND = [{command!r}, 'tool']\n",
            encoding="utf-8",
        )

    result = _run_scanner(tmp_path, "--json")
    assert result.returncode == 1
    findings = json.loads(result.stdout)
    privilege_findings = [
        finding for finding in findings
        if finding["rule_id"] == "PY_PRIVILEGE_ESCALATION"
    ]
    assert len(privilege_findings) == len(commands)


def test_repository_passes_mandatory_security_scan():
    result = _run_scanner(PROJECT_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ci_runs_security_scan_before_pytest():
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    scan_command = "python scripts/security_scan.py"
    test_command = "pytest tests/ -v --tb=short"
    assert scan_command in workflow
    assert workflow.index(scan_command) < workflow.index(test_command)
    assert "branches-ignore" not in workflow


def test_security_files_require_project_owner_review():
    codeowners = (PROJECT_ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    assert "/scripts/security_scan.py @AHEKOT" in codeowners
    assert "/tests/test_security_scan.py @AHEKOT" in codeowners
    assert "/.github/workflows/tests.yml @AHEKOT" in codeowners
