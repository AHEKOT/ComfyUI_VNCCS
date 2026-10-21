"""Qwen runtime installation without downloads, PyTorch, or native libraries."""

from importlib.metadata import PackageNotFoundError
import sys
from types import SimpleNamespace

import pytest

import install as installer


@pytest.mark.parametrize("system,machine,python_version,cuda,release,filename", [
    ("Windows", "AMD64", (3, 12), "13.0", "v0.3.46-cu130-win-20260808",
     "llama_cpp_python-0.3.46+cu130-cp312-cp312-win_amd64.whl"),
    ("Linux", "x86_64", (3, 11), "12.8", "v0.3.46-cu128-linux-20260808",
     "llama_cpp_python-0.3.46+cu128-cp311-cp311-linux_x86_64.whl"),
    ("Linux", "x86_64", (3, 12), "13.0", "v0.4.2-cu130-linux-20261003",
     "llama_cpp_python-0.4.2+cu130-cp312-cp312-linux_x86_64.whl"),
    ("Darwin", "arm64", (3, 13), None, "v0.3.46-Metal-macos-20260808",
     "llama_cpp_python-0.3.46-cp313-cp313-macosx_11_0_arm64.whl"),
])
def test_selects_published_wheel_for_existing_runtime(system, machine, python_version, cuda, release, filename):
    assert installer.qwen_wheel_url(system, machine, python_version, cuda) == (
        f"https://github.com/JamePeng/llama-cpp-python/releases/download/{release}/{filename}"
    )


@pytest.mark.parametrize("system,machine,python_version,cuda", [
    ("Windows", "AMD64", (3, 12), None),
    ("Linux", "x86_64", (3, 12), "12.1"),
    ("Windows", "ARM64", (3, 12), "13.0"),
    ("Linux", "x86_64", (3, 9), "12.8"),
    ("Darwin", "x86_64", (3, 12), None),
])
def test_unsupported_runtime_cannot_select_an_unrelated_backend(system, machine, python_version, cuda):
    with pytest.raises(ValueError):
        installer.qwen_wheel_url(system, machine, python_version, cuda)


@pytest.mark.parametrize("version,source,expected", [
    ("0.3.46", "https://github.com/JamePeng/llama-cpp-python", True),
    ("0.4.2+cu130", "https://github.com/JamePeng/llama-cpp-python", True),
    ("0.3.33", "https://github.com/JamePeng/llama-cpp-python", False),
    ("0.3.46", "https://github.com/abetlen/llama-cpp-python", False),
])
def test_only_compatible_fork_satisfies_the_qwen_requirement(monkeypatch, version, source, expected):
    metadata = SimpleNamespace(get_all=lambda name, default: [f"Homepage, {source}"])
    monkeypatch.setattr(installer, "distribution", lambda name: SimpleNamespace(version=version, metadata=metadata))
    assert installer.qwen_runtime_installed() is expected


def test_missing_distribution_needs_installation(monkeypatch):
    def missing(name):
        raise PackageNotFoundError(name)
    monkeypatch.setattr(installer, "distribution", missing)
    assert installer.qwen_runtime_installed() is False


def test_existing_easy_install_runtime_is_preserved_without_loading_torch(monkeypatch):
    monkeypatch.setattr(installer, "qwen_runtime_installed", lambda: True)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert installer.main() == 0


def test_installer_uses_existing_torch_cuda_and_only_installs_binary_runtime(monkeypatch):
    from pip._internal.cli import main as pip_entry
    commands = []
    monkeypatch.setattr(installer, "qwen_runtime_installed", lambda: False)
    monkeypatch.setattr(installer.platform, "system", lambda: "Windows")
    monkeypatch.setattr(installer.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(installer.sys, "version_info", (3, 12))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="13.0")))
    monkeypatch.setattr(pip_entry, "main", lambda args: commands.append(args) or 0)
    assert installer.main() == 0
    assert commands == [[
        "install", "--only-binary=:all:",
        "https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.46-cu130-win-20260808/"
        "llama_cpp_python-0.3.46+cu130-cp312-cp312-win_amd64.whl",
    ]]


def test_unsupported_runtime_reports_failure_without_running_installer(monkeypatch, capsys):
    monkeypatch.setattr(installer, "qwen_runtime_installed", lambda: False)
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(installer.sys, "version_info", (3, 12))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(version=SimpleNamespace(cuda="12.1")))
    assert installer.main() == 1
    assert "will not replace PyTorch or CUDA" in capsys.readouterr().err
