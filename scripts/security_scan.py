#!/usr/bin/env python3
"""Fail-closed source scanner for VNCCS security and Comfy Registry hygiene."""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import re
import sys
from zipfile import is_zipfile


MAX_REGISTRY_ZIP_BYTES = 100 * 1024 * 1024

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
    "PY_UNSAFE_DESERIALIZATION",
    "PY_URL_COMMAND_EXECUTION",
    "RUNTIME_CODE_INSTALL",
    "REMOVED_BEN2_CODE",
    "UNGUARDED_SIDE_EFFECT",
    "ZIP_TOO_LARGE",
})

COMFY_REGISTRY_ISSUE_TYPES = frozenset({
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
IGNORE_FILES = (".gitignore", ".comfyignore")

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


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted_name(node.value)}.{node.attr}"
    return ""


def _import_names(nodes: list[ast.AST]) -> dict[str, str]:
    names = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                names[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _call_argument(call: ast.Call, name: str, index: int | None = None) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return call.args[index] if index is not None and index < len(call.args) else None


def _unsafe_deserialization(name: str, call: ast.Call, qualify) -> bool:
    if name in {"torch.load", "torch.serialization.load"}:
        weights_only = _call_argument(call, "weights_only")
        pickle_module = _call_argument(call, "pickle_module", 2)
        return (
            not (isinstance(weights_only, ast.Constant) and weights_only.value is True)
            or (pickle_module is not None and not (
                isinstance(pickle_module, ast.Constant) and pickle_module.value is None
            ))
            or any(isinstance(arg, ast.Starred) for arg in call.args)
        )
    if name in {"numpy.load", "numpy.save"}:
        allow_pickle = _call_argument(call, "allow_pickle", 2)
        if allow_pickle is None:
            return any(keyword.arg is None for keyword in call.keywords) or any(
                isinstance(arg, ast.Starred) for arg in call.args
            )
        return not (isinstance(allow_pickle, ast.Constant) and allow_pickle.value is False)
    if name in {"yaml.load", "yaml.load_all"}:
        loader = _call_argument(call, "Loader", 1)
        return qualify(loader) not in {"yaml.SafeLoader", "yaml.CSafeLoader"}
    module, _, method = name.rpartition(".")
    return (
        module in {"pickle", "_pickle", "dill", "cloudpickle", "joblib"}
        and method in {"load", "loads", "Unpickler"}
    ) or name in {
        "pandas.read_pickle", "shelve.open", "yaml.unsafe_load", "yaml.unsafe_load_all",
        "torch.serialization.add_safe_globals", "torch.serialization.safe_globals",
    }


PACKAGE_INSTALL_COMMAND_RE = re.compile(
    r"\b(?:pip(?:3(?:\.\d+)?)?|npm|pnpm|yarn|uv\s+pip)\s+(?:install|add)\b"
)
MUTATING_ROUTE_RE = re.compile(
    r"/(?:create|delete|remove|save|update|publish|reset|clear|cancel)"
    r"(?:[-_][\w-]+)?(?=/|$)"
)
FILE_MUTATION_METHODS = {"write_bytes", "write_text", "unlink", "rmdir"}
FILE_MUTATION_CALLS = {
    "os.remove", "os.unlink", "os.rmdir", "os.rename", "os.replace", "shutil.rmtree",
}


def _literal_fragments(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_literal_fragments(value) for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_fragments(node.left) + _literal_fragments(node.right)
    return "{}"


def _runtime_install(name: str, call: ast.Call) -> bool:
    if name in {"pip.main", "ensurepip.bootstrap"} or name.startswith("pip._internal."):
        return True
    if not (name.startswith("subprocess.") or name in {"os.system", "os.popen"}):
        return False
    command = _call_argument(call, "args", 0)
    if isinstance(command, (ast.List, ast.Tuple)):
        return bool(PACKAGE_INSTALL_COMMAND_RE.search(" ".join(map(_literal_fragments, command.elts))))
    return bool(PACKAGE_INSTALL_COMMAND_RE.search(_literal_fragments(command)))


def _writes_files(handler: ast.AST, qualify) -> bool:
    # shortcut: inspect the handler body. Use call-graph analysis if reads and writes can be distinguished there.
    for call in ast.walk(handler):
        if not isinstance(call, ast.Call):
            continue
        name = qualify(call.func)
        owner, _, method = name.rpartition(".")
        if name in FILE_MUTATION_CALLS or method in FILE_MUTATION_METHODS:
            return True
        if name == "asyncio.to_thread" and call.args:
            target = qualify(call.args[0])
            if target in FILE_MUTATION_CALLS or target.rsplit(".", 1)[-1] in FILE_MUTATION_METHODS:
                return True
        if method == "open":
            mode = _call_argument(call, "mode", 1 if owner in {"", "builtins", "io"} else 0)
            if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
                if any(flag in mode.value for flag in "wax+"):
                    return True
    return False


def _policy_findings(relative: str, tree: ast.Module) -> list[Finding]:
    nodes = list(ast.walk(tree))
    imports = _import_names(nodes)

    def qualify(node: ast.AST) -> str:
        head, sep, tail = _dotted_name(node).partition(".")
        return imports.get(head, head) + sep + tail

    findings = []
    if relative == "install.py":
        findings.append(Finding("RUNTIME_CODE_INSTALL", relative, 1, "Automatic install.py hooks are forbidden"))

    functions = {node.name: node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    handlers = {
        decorator: node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list if isinstance(decorator, ast.Call)
    }
    handlers.update({
        node.func: functions.get(_dotted_name(node.args[0]))
        for node in nodes if isinstance(node, ast.Call) and isinstance(node.func, ast.Call) and node.args
    })
    for call in nodes:
        if not isinstance(call, ast.Call):
            continue
        name = qualify(call.func)
        if _unsafe_deserialization(name, call, qualify):
            findings.append(Finding("PY_UNSAFE_DESERIALIZATION", relative, call.lineno, f"Unsafe deserialization: {name}()"))
        if _runtime_install(name, call):
            findings.append(Finding("RUNTIME_CODE_INSTALL", relative, call.lineno, f"Runtime code installation: {name}()"))

        owner, _, method = name.rpartition(".")
        if owner.split(".")[-1] != "routes" and owner != "aiohttp.web":
            continue
        if method in {"route", "add_route"}:
            verb = _literal_fragments(_call_argument(call, "method", 0)).upper()
            if verb not in {"GET", "HEAD"}:
                continue
            path_index = 1
        elif method in {"get", "head", "add_get", "add_head"}:
            path_index = 0
        else:
            continue
        path = _call_argument(call, "path", path_index)
        handler = handlers.get(call)
        if method.startswith("add_") or owner == "aiohttp.web":
            handler = functions.get(_dotted_name(_call_argument(call, "handler", path_index + 1)))
        if path is not None and (
            MUTATING_ROUTE_RE.search(_literal_fragments(path))
            or (handler is not None and _writes_files(handler, qualify))
        ):
            findings.append(Finding(
                "UNGUARDED_SIDE_EFFECT", relative, call.lineno,
                "GET/HEAD routes must not expose write or delete operations",
            ))
    return findings


def _python_findings(path: Path, relative: str, text: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        tree = ast.parse(text, filename=relative)
    except SyntaxError as exc:
        return [Finding("PY_DYNAMIC_EXECUTION", relative, exc.lineno or 1, f"Python syntax error: {exc.msg}")]

    findings.extend(_policy_findings(relative, tree))

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


def _read_ignore_rules(path: Path):
    if not path.is_file():
        return []
    rules = []
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.rstrip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:]
        elif line.startswith(r"\!") or line.startswith(r"\#"):
            line = line[1:]
        anchored = line.startswith("/")
        directory_only = line.endswith("/")
        line = line.strip("/")
        line = re.sub(r"\\([ #!\\])", r"\1", line)
        if line:
            rules.append((tuple(line.split("/")), negated, directory_only, anchored))
    return rules


def _ignore_glob_matches(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    if not pattern_parts:
        return not path_parts
    if pattern_parts[0] == "**":
        return _ignore_glob_matches(path_parts, pattern_parts[1:]) or (
            bool(path_parts) and _ignore_glob_matches(path_parts[1:], pattern_parts)
        )
    return bool(path_parts) and fnmatchcase(path_parts[0], pattern_parts[0]) and _ignore_glob_matches(
        path_parts[1:], pattern_parts[1:]
    )


def _rules_ignore(relative_path: Path, is_dir: bool, rules) -> bool:
    parts = relative_path.parts
    ignored = False
    for pattern_parts, negated, directory_only, anchored in rules:
        prefixes = [parts[:length] for length in range(1, len(parts) + 1)]
        if directory_only and not is_dir:
            prefixes = prefixes[:-1]
        if len(pattern_parts) == 1 and not anchored:
            matched = any(fnmatchcase(prefix[-1], pattern_parts[0]) for prefix in prefixes)
        else:
            matched = any(_ignore_glob_matches(prefix, pattern_parts) for prefix in prefixes)
        if matched:
            ignored = not negated
    return ignored


def _iter_source_files(root: Path):
    ignore_rule_sets = [_read_ignore_rules(root / name) for name in IGNORE_FILES]
    for current, directories, filenames in os.walk(root):
        current_path = Path(current)
        kept_directories = []
        for name in sorted(directories):
            relative_path = (current_path / name).relative_to(root)
            if any(part in SKIP_PARTS for part in relative_path.parts):
                continue
            if any(_rules_ignore(relative_path, True, rules) for rules in ignore_rule_sets):
                continue
            kept_directories.append(name)
        directories[:] = kept_directories
        for name in sorted(filenames):
            path = current_path / name
            if path.suffix.lower() not in SOURCE_SUFFIXES:
                continue
            relative_path = path.relative_to(root)
            if relative_path == SCANNER_RELATIVE_PATH:
                continue
            if any(_rules_ignore(relative_path, False, rules) for rules in ignore_rule_sets):
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
    parser.add_argument("--archive", type=Path, help="Check the ZIP produced by comfy node pack")
    args = parser.parse_args(argv)

    findings = scan_root(args.root)
    archive = args.archive if args.archive is not None else args.root / "node.zip"
    if args.archive is not None or archive.exists():
        if not archive.is_file() or not is_zipfile(archive):
            parser.error(f"Missing or invalid Registry ZIP: {archive}")
        size = archive.stat().st_size
        if size > MAX_REGISTRY_ZIP_BYTES:
            findings.append(Finding("ZIP_TOO_LARGE", archive.name, 1,
                                    f"Registry ZIP is {size} bytes; limit is {MAX_REGISTRY_ZIP_BYTES} bytes"))
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
