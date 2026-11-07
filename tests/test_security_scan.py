"""Tamper-resistant regression tests for the mandatory VNCCS security scanner."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from zipfile import ZipFile

from scripts.security_scan import (
    COMFY_REGISTRY_ISSUE_TYPES,
    SCANNER_RULE_IDS,
    main as scanner_main,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCANNER = PROJECT_ROOT / "scripts" / "security_scan.py"
EXPECTED_SCANNER_SHA256 = "6d6cdf182d1ea39e6e6002895262ff105ed1f6dd32571c8476acd635e96033d6"
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
    "PY_UNSAFE_DESERIALIZATION",
    "PY_URL_COMMAND_EXECUTION",
    "RUNTIME_CODE_INSTALL",
    "REMOVED_BEN2_CODE",
    "UNGUARDED_SIDE_EFFECT",
    "ZIP_TOO_LARGE",
}
EXPECTED_COMFY_REGISTRY_ISSUE_TYPES = {
    "code-install",
    "contains_blacklisted_url",
    "insecure-deserialization",
    "python_bytecode_manipulation",
    "python_command_injection_risk",
    "python_dynamic_execution",
    "python_environment_manipulation",
    "python_network_operations",
    "python_privilege_escalation",
    "python_url_command_execution",
    "unauthenticated-side-effect",
    "zip_too_large",
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


def _write_sized_zip(path, size):
    # Sparse padding keeps the real archive boundary test small on disk.
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("asset.bin", b"test")
    with path.open("wb") as handle:
        handle.seek(size - payload.tell())
        handle.write(payload.getvalue())


def test_registry_zip_size_boundary(tmp_path):
    archive = tmp_path / "node.zip"
    limit = 100 * 1024 * 1024
    assert _run_scanner(tmp_path, "--archive", str(archive)).returncode != 0
    archive.write_bytes(b"not a zip")
    assert _run_scanner(tmp_path, "--archive", str(archive)).returncode != 0
    _write_sized_zip(archive, limit)
    assert _run_scanner(tmp_path, "--archive", str(archive)).returncode == 0
    _write_sized_zip(archive, limit + 1)
    result = _run_scanner(tmp_path, "--json")
    assert result.returncode == 1
    finding, = json.loads(result.stdout)
    assert finding["rule_id"] == "ZIP_TOO_LARGE"
    assert str(limit + 1) in finding["message"]


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
from pip._internal.cli.main import main as pip_main
from torch import load as read_weights

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
pip_main(["install", "example"])
read_weights("cache.pt")

@routes.get("/vnccs/create_costume")
async def create_costume(request):
    pass
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

    _write_sized_zip(tmp_path / "node.zip", 100 * 1024 * 1024 + 1)
    result = _run_scanner(tmp_path, "--json")
    assert result.returncode == 1
    findings = json.loads(result.stdout)
    found_rule_ids = {finding["rule_id"] for finding in findings}
    assert found_rule_ids == EXPECTED_RULE_IDS


def test_security_scanner_distinguishes_registry_policy_cases(tmp_path):
    sources = {
        "unsafe_pickle.py": "from pickle import loads as read\nread(payload)\n",
        "unsafe_numpy.py": "import numpy as np\nnp.load(path, None, True)\n",
        "unsafe_torch.py": "import torch as pt\npt.load(path, weights_only=False)\n",
        "unsafe_get_route.py": '@routes.route("GET", "/vnccs/delete/{id}")\nasync def handler(request): pass\n',
        "unsafe_installer.py": "from ensurepip import bootstrap\nbootstrap()\n",
        "install.py": "print('automatic installer')\n",
        "unsafe_http.py": """
import asyncio

async def handler(request):
    await asyncio.to_thread(output_path.write_bytes, b"data")

routes.head("/vnccs/item")(handler)
""",
        "safe.py": """
from torch import load
import numpy as np
from huggingface_hub import hf_hub_download

load(path, weights_only=True)
np.load(path)
np.load(path, allow_pickle=False)
np.save(path, array, allow_pickle=False)
hf_hub_download(repo_id="example/model", filename="weights.safetensors", token=False)

@routes.get("/vnccs/item")
async def read(request):
    return path.read_bytes()

@routes.post("/vnccs/create")
async def create(request):
    path.write_bytes(b"data")
""",
    }
    for name, source in sources.items():
        (tmp_path / name).write_text(source, encoding="utf-8")

    result = _run_scanner(tmp_path, "--json")
    assert result.returncode == 1
    assert [(finding["path"], finding["rule_id"]) for finding in json.loads(result.stdout)] == [
        ("install.py", "RUNTIME_CODE_INSTALL"),
        ("unsafe_get_route.py", "UNGUARDED_SIDE_EFFECT"),
        ("unsafe_http.py", "UNGUARDED_SIDE_EFFECT"),
        ("unsafe_installer.py", "RUNTIME_CODE_INSTALL"),
        ("unsafe_numpy.py", "PY_UNSAFE_DESERIALIZATION"),
        ("unsafe_pickle.py", "PY_UNSAFE_DESERIALIZATION"),
        ("unsafe_torch.py", "PY_UNSAFE_DESERIALIZATION"),
    ]


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


def test_security_scanner_skips_git_and_comfy_ignored_sources(tmp_path):
    (tmp_path / ".gitignore").write_text(
        "git-only/\ngenerated/*\n!generated/keep.py\n",
        encoding="utf-8",
    )
    (tmp_path / ".comfyignore").write_text("package-only/\n", encoding="utf-8")
    for directory in ("git-only", "package-only", "generated"):
        (tmp_path / directory).mkdir()
    for relative in (
        "git-only/unsafe.py",
        "package-only/unsafe.py",
        "generated/unsafe.py",
        "generated/keep.py",
        "visible.py",
    ):
        (tmp_path / relative).write_text('eval("test code")\n', encoding="utf-8")

    result = _run_scanner(tmp_path, "--json")
    assert result.returncode == 1
    findings = json.loads(result.stdout)
    assert {finding["path"] for finding in findings} == {
        "generated/keep.py",
        "visible.py",
    }


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
    scan_command = "python scripts/security_scan.py --archive node.zip"
    test_command = "pytest tests/ -v --tb=short"
    assert workflow.index("node pack") < workflow.index(scan_command)
    assert workflow.index(scan_command) < workflow.index(test_command)
    assert "branches-ignore" not in workflow


def test_security_files_require_project_owner_review():
    codeowners = (PROJECT_ROOT / ".github" / "CODEOWNERS").read_text(encoding="utf-8")
    assert "/scripts/security_scan.py @AHEKOT" in codeowners
    assert "/tests/test_security_scan.py @AHEKOT" in codeowners
    assert "/.github/workflows/tests.yml @AHEKOT" in codeowners
