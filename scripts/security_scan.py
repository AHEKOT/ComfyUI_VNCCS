#!/usr/bin/env python3
"""Fail-closed source scanner for VNCCS security and Comfy Registry hygiene."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys


SCANNER_RULE_IDS = frozenset({
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
})

COMFY_REGISTRY_ISSUE_TYPES = frozenset({
    "contains_blacklisted_url",
    "python_bytecode_manipulation",
    "python_command_injection_risk",
    "python_dynamic_execution",
    "python_environment_manipulation",
    "python_network_operations",
    "python_privilege_escalation",
    "python_url_command_execution",
})

SOURCE_SUFFIXES = {".py", ".js", ".mjs", ".toml", ".json"}
SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
SCANNER_RELATIVE_PATH = Path("scripts/security_scan.py")

FORBIDDEN_CREDENTIAL_MARKERS = (
    "hf_token",
    "hugging_face_hub_token",
    "civitai_token",
)
BLACKLISTED_URL_MARKERS = ("raw.githubusercontent.com",)
CREDENTIAL_URL_RE = re.compile(
    r"https?://[^/@\s\"'`]+:[^/@\s\"'`]+@",
    flags=re.IGNORECASE,
)
PRIVILEGE_ESCALATION_RE = re.compile(
    r"(?<![\w-])(?:sudo|su|doas|pkexec|runas)(?![\w-])",
    flags=re.IGNORECASE,
)
RAW_CLIENT_MODULES = {
    "ftplib",
    "http.client",
    "requests",
    "smtplib",
    "socket",
    "subprocess",
    "urllib.request",
}
OS_COMMAND_METHODS = {
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "popen",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "system",
}
URL_CAPABLE_COMMAND_RE = re.compile(
    r"(?<![\w-])(?:curl|ffmpeg|ffprobe|imgkit|pandoc|pdfkit|wget|"
    r"wkhtmltopdf|youtube-dl|yt-dlp)(?![\w-])",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    rule_id: str
    path: str
    line: int
    message: str


def _call_name(node: ast.AST) -> tuple[str | None, str | None]:
    if isinstance(node, ast.Name):
        return None, node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return node.value.id, node.attr
    return None, None


def _python_findings(path: Path, relative: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        tree = ast.parse(text, filename=relative)
    except SyntaxError as exc:
        return [Finding("PY_DYNAMIC_EXECUTION", relative, exc.lineno or 1, f"Python syntax error: {exc.msg}")]

    os_aliases = {"os"}
    importlib_aliases = {"importlib"}
    tempfile_aliases = {"tempfile"}
    gettempdir_names = set()
    hf_download_names = {"hf_hub_download"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "os":
                    os_aliases.add(alias.asname or alias.name)
                elif alias.name == "importlib":
                    importlib_aliases.add(alias.asname or alias.name)
                elif alias.name == "tempfile":
                    tempfile_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "tempfile":
                for alias in node.names:
                    if alias.name == "gettempdir":
                        gettempdir_names.add(alias.asname or alias.name)
            if node.module == "huggingface_hub":
                for alias in node.names:
                    if alias.name == "hf_hub_download":
                        hf_download_names.add(alias.asname or alias.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name
                if module in RAW_CLIENT_MODULES or any(module.startswith(f"{name}.") for name in RAW_CLIENT_MODULES):
                    rule = "PY_COMMAND_EXECUTION" if module == "subprocess" else "PY_NETWORK_CLIENT"
                    findings.append(Finding(rule, relative, node.lineno, f"Forbidden raw module import: {module}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in RAW_CLIENT_MODULES or any(module.startswith(f"{name}.") for name in RAW_CLIENT_MODULES):
                rule = "PY_COMMAND_EXECUTION" if module == "subprocess" else "PY_NETWORK_CLIENT"
                findings.append(Finding(rule, relative, node.lineno, f"Forbidden raw module import: {module}"))
            if module == "aiohttp" and any(alias.name == "ClientSession" for alias in node.names):
                findings.append(Finding("PY_NETWORK_CLIENT", relative, node.lineno, "aiohttp ClientSession is forbidden"))
            if module == "urllib" and any(alias.name == "request" for alias in node.names):
                findings.append(Finding("PY_NETWORK_CLIENT", relative, node.lineno, "urllib.request is forbidden"))
            if module == "http" and any(alias.name == "client" for alias in node.names):
                findings.append(Finding("PY_NETWORK_CLIENT", relative, node.lineno, "http.client is forbidden"))
            if module == "os":
                for alias in node.names:
                    if alias.name in {"environ", "getenv"}:
                        findings.append(Finding("ENVIRONMENT_ACCESS", relative, node.lineno, f"os.{alias.name} import is forbidden"))
                    elif alias.name in OS_COMMAND_METHODS:
                        findings.append(Finding("PY_COMMAND_EXECUTION", relative, node.lineno, f"os.{alias.name} import is forbidden"))
            if module == "importlib" and any(alias.name == "import_module" for alias in node.names):
                findings.append(Finding("PY_DYNAMIC_IMPORT", relative, node.lineno, "importlib.import_module import is forbidden"))
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in os_aliases and node.attr == "environ":
                findings.append(Finding("ENVIRONMENT_ACCESS", relative, node.lineno, "os.environ access is forbidden"))
        elif isinstance(node, ast.Call):
            owner, name = _call_name(node.func)
            if owner is None and name in hf_download_names:
                credential_kw = next((item for item in node.keywords if item.arg == "token"), None)
                explicitly_disabled = (
                    credential_kw is not None
                    and isinstance(credential_kw.value, ast.Constant)
                    and credential_kw.value.value is False
                )
                if not explicitly_disabled:
                    findings.append(Finding(
                        "HF_CREDENTIALS_DISABLED",
                        relative,
                        node.lineno,
                        "hf_hub_download() must set token=False",
                    ))
            if (owner is None or owner == "builtins") and name in {"eval", "exec", "compile"}:
                findings.append(Finding("PY_DYNAMIC_EXECUTION", relative, node.lineno, f"Built-in {name}() is forbidden"))
            elif owner in importlib_aliases and name == "import_module":
                findings.append(Finding("PY_DYNAMIC_IMPORT", relative, node.lineno, "importlib.import_module() is forbidden"))
            elif owner in os_aliases and name == "getenv":
                findings.append(Finding("ENVIRONMENT_ACCESS", relative, node.lineno, "os.getenv() is forbidden"))
            elif owner in os_aliases and name in OS_COMMAND_METHODS:
                findings.append(Finding("PY_COMMAND_EXECUTION", relative, node.lineno, f"os.{name}() is forbidden"))
            elif owner == "aiohttp" and name == "ClientSession":
                findings.append(Finding("PY_NETWORK_CLIENT", relative, node.lineno, "aiohttp.ClientSession() is forbidden"))
            elif (owner in tempfile_aliases and name == "gettempdir") or (owner is None and name in gettempdir_names):
                findings.append(Finding(
                    "PY_TEMP_DIRECTORY_ACCESS",
                    relative,
                    node.lineno,
                    "tempfile.gettempdir() triggers the Comfy network rule",
                ))

    for match in URL_CAPABLE_COMMAND_RE.finditer(text):
        findings.append(Finding(
            "PY_URL_COMMAND_EXECUTION",
            relative,
            _line_number(text, match.start()),
            f"URL-capable external command marker is forbidden: {match.group(0)}",
        ))

    return findings


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _text_findings(relative: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    lowered = text.lower()
    for marker in FORBIDDEN_CREDENTIAL_MARKERS:
        for match in re.finditer(re.escape(marker), lowered):
            findings.append(Finding("CREDENTIAL_SOURCE", relative, _line_number(text, match.start()), f"Forbidden credential marker: {marker}"))
    for marker in BLACKLISTED_URL_MARKERS:
        for match in re.finditer(re.escape(marker), lowered):
            findings.append(Finding("BLACKLISTED_URL", relative, _line_number(text, match.start()), f"Blacklisted URL host: {marker}"))
    for match in CREDENTIAL_URL_RE.finditer(text):
        findings.append(Finding(
            "CREDENTIAL_URL",
            relative,
            _line_number(text, match.start()),
            "URL-embedded credentials are forbidden",
        ))
    if relative.lower().endswith(".py"):
        for match in PRIVILEGE_ESCALATION_RE.finditer(text):
            findings.append(Finding(
                "PY_PRIVILEGE_ESCALATION",
                relative,
                _line_number(text, match.start()),
                f"Privilege-escalation command marker is forbidden: {match.group(0)}",
            ))
    for match in re.finditer(r"\bBEN2\b", text, flags=re.IGNORECASE):
        findings.append(Finding("REMOVED_BEN2_CODE", relative, _line_number(text, match.start()), "BEN2 support was removed and must not return"))
    return findings


def _javascript_findings(relative: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    patterns = (
        ("JS_BIND_SCANNER_TRIGGER", re.compile(r"\.bind\s*\("), "Function.bind() triggers the Comfy network rule"),
        ("JS_CONNECT_SCANNER_TRIGGER", re.compile(r"\.connect\s*\("), "connect() triggers the Comfy network rule"),
        ("JS_DYNAMIC_EXECUTION", re.compile(r"(?<![.\w])eval\s*\(|\bnew\s+Function\s*\("), "Dynamic JavaScript execution is forbidden"),
        ("JS_EXTERNAL_NETWORK", re.compile(r"\bfetch\s*\(\s*[`\"']https?://", re.IGNORECASE), "Direct external fetch() is forbidden"),
        (
            "JS_REQUESTS_SCANNER_TRIGGER",
            re.compile(r"\bRequests\.(?:get|delete)\s*\(", re.IGNORECASE),
            "Requests.get/delete triggers the Comfy network rule",
        ),
    )
    for rule_id, pattern, message in patterns:
        for match in pattern.finditer(text):
            findings.append(Finding(rule_id, relative, _line_number(text, match.start()), message))
    return findings


def _iter_source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SOURCE_SUFFIXES:
            continue
        relative_path = path.relative_to(root)
        if relative_path == SCANNER_RELATIVE_PATH or any(part in SKIP_PARTS for part in relative_path.parts):
            continue
        yield path, relative_path.as_posix()


def scan_root(root: Path) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for path, relative in _iter_source_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        findings.extend(_text_findings(relative, text))
        if path.suffix.lower() == ".py":
            findings.extend(_python_findings(path, relative, text))
        elif path.suffix.lower() in {".js", ".mjs"}:
            findings.extend(_javascript_findings(relative, text))
    return sorted(set(findings), key=lambda item: (item.path, item.line, item.rule_id, item.message))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    findings = scan_root(args.root)
    if args.json:
        print(json.dumps([asdict(item) for item in findings], indent=2, sort_keys=True))
    elif findings:
        for finding in findings:
            print(f"{finding.path}:{finding.line}: {finding.rule_id}: {finding.message}")
        print(f"Security scan failed with {len(findings)} finding(s).")
    else:
        print("Security scan passed: no forbidden VNCCS source patterns found.")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
