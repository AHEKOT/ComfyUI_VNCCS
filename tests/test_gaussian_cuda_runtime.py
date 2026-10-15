"""CUDA selection and installation contracts without importing PyTorch or CUDA."""

import ctypes as C
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from nodes import gaussian_cuda_runtime as runtime


@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize("release", ["11.8", "12.1", "12.8", "13.0", "13.4"])
def test_installer_supplies_only_missing_nvrtc_for_the_existing_torch_release(monkeypatch, platform, release):
    calls = []
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(runtime, "torch_cuda_installation", lambda: ("torch/__init__.py", release))
    monkeypatch.setattr(runtime, "distribution", lambda _name: SimpleNamespace(requires=[]))

    def missing(*_args):
        raise runtime.NvrtcUnavailable("missing")

    monkeypatch.setattr(runtime, "load_nvrtc", missing)
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", SimpleNamespace(
        main=lambda arguments: calls.append(arguments) or 0,
    ))
    assert runtime.install_nvrtc() == 0
    assert calls == [[
        "--isolated", "--disable-pip-version-check", "install", "--no-input",
        "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
        f"cuda-toolkit[nvrtc]=={release}.*",
    ]]


def test_installer_preserves_torch_nvrtc_requirement_without_resolving_other_cuda_packages(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(runtime, "torch_cuda_installation", lambda: ("torch/__init__.py", "12.8"))
    monkeypatch.setattr(runtime, "distribution", lambda _name: SimpleNamespace(
        requires=['nvidia-cuda-nvrtc-cu12==12.8.61; platform_system == "Linux"'],
    ))

    def missing(*_args):
        raise runtime.NvrtcUnavailable("missing")

    monkeypatch.setattr(runtime, "load_nvrtc", missing)
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", SimpleNamespace(
        main=lambda arguments: calls.append(arguments) or 1,
    ))
    assert runtime.install_nvrtc() == 1
    assert calls == [[
        "--isolated", "--disable-pip-version-check", "install", "--no-input",
        "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
        "nvidia-cuda-nvrtc-cu12==12.8.61", "--no-deps",
    ]]


@pytest.mark.parametrize("release", [None, "12.8"])
def test_installer_does_not_touch_cpu_or_already_complete_cuda_environments(monkeypatch, release):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(runtime, "torch_cuda_installation", lambda: ("torch/__init__.py", release))
    monkeypatch.setattr(runtime, "load_nvrtc", lambda *_args: (object(), (), "torch/lib/nvrtc"))
    monkeypatch.setitem(sys.modules, "pip._internal.cli.main", SimpleNamespace(
        main=lambda *_args: pytest.fail("No package installation should occur"),
    ))
    assert runtime.install_nvrtc() == 0


def test_install_hook_loads_node_helper_without_registering_nodes_or_importing_torch(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("vnccs_install_hook", root / "install.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    monkeypatch.setattr(sys, "platform", "darwin")
    for name in ("nodes", "torch", "comfy"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, "gaussian_cuda_runtime", raising=False)
    original_path = sys.path[:]
    try:
        assert installer.main() == 0
        helper = sys.modules["gaussian_cuda_runtime"]
        assert Path(helper.__file__).resolve() == root / "nodes" / "gaussian_cuda_runtime.py"
        assert sys.path == original_path
    finally:
        sys.modules.pop("gaussian_cuda_runtime", None)


@pytest.mark.parametrize("cuda,hip", [("11.8", None), ("12.8", None), ("13.0", None), (None, None), (None, "6.4")])
def test_torch_cuda_selection_reads_metadata_without_executing_torch(tmp_path, monkeypatch, cuda, hip):
    root = tmp_path / "torch"
    root.mkdir()
    (root / "version.py").write_text(
        f"raise AssertionError('Must not execute this module')\ncuda: str = {cuda!r}\nhip = {hip!r}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime, "distribution", lambda _name: SimpleNamespace(locate_file=lambda name: tmp_path / name))
    torch_file, selected = runtime.torch_cuda_installation()
    assert torch_file == root / "__init__.py"
    assert selected == cuda


@pytest.mark.parametrize("value", [None, "cuda12", "12.8 --upgrade", "12.8; torch", "../12.8"])
def test_invalid_cuda_metadata_cannot_create_package_arguments(value):
    with pytest.raises(ValueError):
        runtime.nvrtc_requirement(value)


def test_nvrtc_selection_preserves_torch_component_patch_and_rejects_other_releases():
    dependencies = [
        'nvidia-cuda-nvrtc-cu12==12.6.85; platform_system == "Linux"',
        'nvidia-cuda-nvrtc-cu12==12.8.61; platform_system == "Linux"',
        'nvidia-cuda-runtime-cu12==12.8.57; platform_system == "Linux"',
    ]
    assert runtime.nvrtc_requirement("12.8", dependencies) == "nvidia-cuda-nvrtc-cu12==12.8.61"
    assert runtime.nvrtc_requirement("13.0", dependencies) == "cuda-toolkit[nvrtc]==13.0.*"


def test_library_discovery_uses_torch_and_matching_installed_wheel_files(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "python"))
    root = tmp_path / "torch" / "lib"
    root.mkdir(parents=True)
    bundled = root / "libnvrtc.so.12"
    bundled.touch()
    (root / "libnvrtc-builtins.so.12.8").touch()
    matching = tmp_path / "wheel" / "libnvrtc.so.12"
    matching.parent.mkdir()
    matching.touch()
    unrelated = tmp_path / "unrelated" / "libnvrtc.so.13"
    unrelated.parent.mkdir()
    unrelated.touch()

    def package(name, version, path):
        return SimpleNamespace(metadata={"Name": name}, version=version, files=[Path(path.name)],
                               locate_file=lambda _entry: path)

    monkeypatch.setattr(runtime, "distributions", lambda: [
        package("nvidia-cuda-nvrtc-cu12", "12.8.93", matching),
        package("nvidia-cuda-nvrtc", "13.0.88", unrelated),
    ])
    assert runtime._nvrtc_paths(root.parent / "__init__.py", "12.8") == (bundled, matching)


def test_loader_rejects_other_cuda_releases_even_when_the_library_name_is_identical(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    wrong, correct = Path("old/libnvrtc.so.12"), Path("current/libnvrtc.so.12")
    monkeypatch.setattr(runtime, "_nvrtc_paths", lambda *_args: (wrong, correct))
    calls = []

    def library(path):
        calls.append(path)

        def version(major, minor):
            major._obj.value, minor._obj.value = 12, 6 if path == str(wrong) else 8
            return 0

        return SimpleNamespace(nvrtcVersion=version)

    monkeypatch.setattr(runtime.C, "CDLL", library)
    _, handles, selected = runtime.load_nvrtc("torch/__init__.py", "12.8")
    assert selected == correct and handles == () and calls == [str(wrong), str(correct)]
    with pytest.raises(runtime.NvrtcUnavailable, match="PyTorch CUDA 13.0"):
        runtime.load_nvrtc("torch/__init__.py", "13.0")


def test_windows_loader_retains_only_the_matching_compiler_dll_directory(monkeypatch):
    import os

    wrong, correct = Path("old/nvrtc64_120_0.dll"), Path("current/nvrtc64_120_0.dll")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(runtime, "_nvrtc_paths", lambda *_args: (wrong, correct))
    opened, closed = [], []

    def directory(path):
        handle = SimpleNamespace(close=lambda: closed.append(path))
        opened.append((path, handle))
        return handle

    def library(path):
        def version(major, minor):
            major._obj.value, minor._obj.value = 12, 6 if path == str(wrong) else 8
            return 0

        return SimpleNamespace(nvrtcVersion=version)

    monkeypatch.setattr(os, "add_dll_directory", directory, raising=False)
    monkeypatch.setattr(runtime.C, "CDLL", library)
    _, handles, selected = runtime.load_nvrtc("torch/__init__.py", "12.8")
    assert selected == correct
    assert closed == [str(wrong.parent)]
    assert handles == (opened[1][1],)
    assert runtime._nvrtc_filename("nvrtc64_130_0.dll")
    assert not runtime._nvrtc_filename("nvrtc-builtins64_130.dll")
    assert not runtime._nvrtc_filename("nvrtc64_130_0.alt.dll")


@pytest.mark.parametrize("capability,expected", [((8, 6), b"--gpu-architecture=sm_86"),
                                                ((9, 0), b"--gpu-architecture=compute_86")])
def test_compiler_uses_native_cubin_or_forward_compatible_ptx_from_the_same_nvrtc(capability, expected):
    calls = []

    def assign(pointer, value):
        pointer._obj.value = value
        return 0

    def architectures(output):
        output[:] = [75, 80, 86]
        return 0

    def image(_program, output):
        C.memmove(output, b"image\0", 6)
        return 0

    library = SimpleNamespace(
        nvrtcGetErrorString=lambda _status: b"error",
        nvrtcCreateProgram=lambda pointer, *_args: assign(pointer, 1),
        nvrtcCompileProgram=lambda _program, _count, options: calls.append(tuple(options)) or 0,
        nvrtcDestroyProgram=lambda _pointer: calls.append("destroy") or 0,
        nvrtcGetProgramLogSize=lambda _program, size: assign(size, 6),
        nvrtcGetProgramLog=image,
        nvrtcGetNumSupportedArchs=lambda count: assign(count, 3),
        nvrtcGetSupportedArchs=architectures,
        nvrtcGetCUBINSize=lambda _program, size: assign(size, 6),
        nvrtcGetCUBIN=image,
        nvrtcGetPTXSize=lambda _program, size: assign(size, 6),
        nvrtcGetPTX=image,
    )
    assert runtime._nvrtc_image(library, "source", capability).value == b"image"
    assert calls == [(b"--std=c++14", expected), "destroy"]


def test_driver_launch_passes_scalar_addresses_and_torch_stream():
    calls = []

    def function(pointer, _module, name):
        pointer._obj.value = 123
        assert name == b"surface_forward"
        return 0

    def launch(handle, *args):
        stream, parameters, extra = args[-3:]
        calls.append((handle.value, args[:7], stream.value,
                      C.cast(parameters[0], C.POINTER(C.c_void_p)).contents.value,
                      C.cast(parameters[1], C.POINTER(C.c_int)).contents.value, extra))
        return 0

    module = SimpleNamespace(handle=C.c_void_p(1), check=lambda status: None,
                             driver=SimpleNamespace(cuModuleGetFunction=function, cuLaunchKernel=launch))
    kernel = runtime._Kernel(module, "surface_forward")
    kernel((2, 3), (16, 16), (C.c_void_p(9876), C.c_int(17)), stream=456)
    assert calls == [(123, (2, 3, 1, 16, 16, 1, 0), 456, 9876, 17, None)]
