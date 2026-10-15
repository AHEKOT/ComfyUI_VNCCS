"""Use the NVRTC release belonging to ComfyUI's existing PyTorch CUDA build."""

import ast
import ctypes as C
from importlib.metadata import distribution, distributions
from pathlib import Path
import re
import sys


class NvrtcUnavailable(RuntimeError):
    pass


def cuda_release(cuda_version):
    if not isinstance(cuda_version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+", cuda_version):
        raise ValueError("An NVIDIA CUDA build of PyTorch is required")
    return tuple(map(int, cuda_version.split(".")))


def nvrtc_requirement(cuda_version, torch_dependencies=()):
    """Select only NVRTC, with the major/minor already selected by PyTorch."""
    major, minor = cuda_release(cuda_version)
    for dependency in torch_dependencies:
        match = re.fullmatch(
            r"(nvidia-cuda-nvrtc(?:-cu[0-9]+)?)\s*\(?\s*==\s*([0-9]+(?:\.[0-9]+)+)\s*\)?(?:\s*;.*)?",
            dependency,
        )
        if match and tuple(match[2].split(".")[:2]) == (str(major), str(minor)):
            # Preserve PyTorch's own component patch version when it declares one.
            return f"{match[1]}=={match[2]}"
    return f"cuda-toolkit[nvrtc]=={major}.{minor}.*"


def torch_cuda_installation():
    """Read generated PyTorch metadata even when missing CUDA libraries prevent import."""
    package = distribution("torch")
    version_file = Path(package.locate_file("torch/version.py"))
    values = {}
    for statement in ast.parse(version_file.read_text(encoding="utf-8")).body:
        if isinstance(statement, ast.AnnAssign):
            targets, value = [statement.target], statement.value
        elif isinstance(statement, ast.Assign):
            targets, value = statement.targets, statement.value
        else:
            continue
        if isinstance(value, ast.Constant):
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {"cuda", "hip"}:
                    values[target.id] = value.value
    if "cuda" not in values:
        raise RuntimeError("Cannot read the installed PyTorch CUDA release; no CUDA packages were changed")
    cuda_version = None if values.get("hip") is not None else values["cuda"]
    if cuda_version is not None:
        cuda_release(cuda_version)
    return package.locate_file("torch/__init__.py"), cuda_version


def _nvrtc_filename(name):
    if sys.platform == "win32":
        return re.fullmatch(r"nvrtc64_[0-9_]+\.dll", name) is not None
    return re.fullmatch(r"libnvrtc(?:-[A-Za-z0-9]+)?\.so(?:\.[0-9]+)*", name) is not None


def _nvrtc_paths(torch_file, cuda_version):
    release = cuda_release(cuda_version)
    torch_lib = Path(torch_file).resolve().parent / "lib"
    # Search this Python environment, never PATH or a system-wide toolkit.
    roots = (torch_lib, Path(sys.prefix) / "Library" / "bin", Path(sys.prefix) / "lib")
    candidates = []
    for root in roots:
        if root.is_dir():
            candidates.extend(path for path in sorted(root.iterdir()) if _nvrtc_filename(path.name))
    for package in distributions():
        name = re.sub(r"[-_.]+", "-", package.metadata.get("Name", "")).lower()
        if not re.fullmatch(r"nvidia-cuda-nvrtc(?:-cu[0-9]+)?", name):
            continue
        if tuple(package.version.split(".")[:2]) != tuple(map(str, release)):
            continue
        for entry in package.files or ():
            if _nvrtc_filename(Path(entry).name):
                candidates.append(Path(package.locate_file(entry)))
    return tuple(dict.fromkeys(path.resolve() for path in candidates if path.is_file()))


def _bind(library, name, arguments, result=C.c_int):
    function = getattr(library, name)
    function.argtypes, function.restype = arguments, result
    return function


def load_nvrtc(torch_file, cuda_version):
    """Retain DLL search handles for the compiler's lazily loaded builtins."""
    expected = cuda_release(cuda_version)
    errors = []
    for path in _nvrtc_paths(torch_file, cuda_version):
        handles = []
        try:
            if sys.platform == "win32":
                from os import add_dll_directory

                handles.append(add_dll_directory(str(path.parent)))
            library = C.CDLL(str(path))
            version = _bind(library, "nvrtcVersion", [C.POINTER(C.c_int), C.POINTER(C.c_int)])
            major, minor = C.c_int(), C.c_int()
            status = version(C.byref(major), C.byref(minor))
            actual = major.value, minor.value
            if status != 0 or actual != expected:
                raise OSError(f"reported NVRTC {major.value}.{minor.value}, status {status}")
            return library, tuple(handles), path
        except (OSError, AttributeError) as error:
            errors.append(f"{path}: {error}")
            for handle in handles:
                handle.close()
    detail = "; ".join(errors) or "no matching library in the current Python environment"
    raise NvrtcUnavailable(
        f"NVRTC matching PyTorch CUDA {cuda_version} is unavailable: {detail}. "
        "Run VNCCS install.py with ComfyUI's Python to install the matching official NVRTC wheel. "
        "The installer does not change PyTorch or the CUDA runtime."
    )


def _nvrtc_image(library, source, capability):
    pointer = C.c_void_p
    _bind(library, "nvrtcGetErrorString", [C.c_int], C.c_char_p)
    _bind(library, "nvrtcCreateProgram", [C.POINTER(pointer), C.c_char_p, C.c_char_p,
                                         C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)])
    _bind(library, "nvrtcCompileProgram", [pointer, C.c_int, C.POINTER(C.c_char_p)])
    _bind(library, "nvrtcDestroyProgram", [C.POINTER(pointer)])
    _bind(library, "nvrtcGetProgramLogSize", [pointer, C.POINTER(C.c_size_t)])
    _bind(library, "nvrtcGetProgramLog", [pointer, pointer])

    def check(status):
        if status:
            reason = library.nvrtcGetErrorString(status).decode("utf-8", errors="replace")
            raise RuntimeError(f"NVRTC: {reason}")

    architecture = capability[0] * 10 + capability[1]
    supported = None
    if hasattr(library, "nvrtcGetSupportedArchs"):
        _bind(library, "nvrtcGetNumSupportedArchs", [C.POINTER(C.c_int)])
        _bind(library, "nvrtcGetSupportedArchs", [C.POINTER(C.c_int)])
        count = C.c_int()
        check(library.nvrtcGetNumSupportedArchs(C.byref(count)))
        architectures = (C.c_int * count.value)()
        check(library.nvrtcGetSupportedArchs(architectures))
        supported = tuple(architectures)
    cubin = supported is not None and architecture in supported and hasattr(library, "nvrtcGetCUBIN")
    if supported is not None and architecture not in supported:
        compatible = [value for value in supported if value <= architecture]
        if not compatible:
            raise RuntimeError("The current PyTorch NVRTC release does not support this GPU architecture")
        architecture = max(compatible)
    image_type = "CUBIN" if cubin else "PTX"
    get_size = _bind(library, f"nvrtcGet{image_type}Size", [pointer, C.POINTER(C.c_size_t)])
    get_image = _bind(library, f"nvrtcGet{image_type}", [pointer, pointer])
    options = (b"--std=c++14", f"--gpu-architecture={'sm' if cubin else 'compute'}_{architecture}".encode("ascii"))
    program = pointer()
    check(library.nvrtcCreateProgram(C.byref(program), source.encode("utf-8"),
                                    b"vnccs_surface.cu", 0, None, None))
    try:
        status = library.nvrtcCompileProgram(program, len(options), (C.c_char_p * len(options))(*options))
        if status:
            size = C.c_size_t()
            check(library.nvrtcGetProgramLogSize(program, C.byref(size)))
            log = C.create_string_buffer(max(1, size.value))
            check(library.nvrtcGetProgramLog(program, log))
            raise RuntimeError(f"VNCCS surface kernel compilation failed:\n{log.value.decode('utf-8', errors='replace')}")
        size = C.c_size_t()
        check(get_size(program, C.byref(size)))
        image = C.create_string_buffer(size.value)
        check(get_image(program, image))
        return image
    finally:
        library.nvrtcDestroyProgram(C.byref(program))


class _Kernel:
    def __init__(self, module, name):
        self.module, self.name, self.function = module, name, C.c_void_p()
        module.check(module.driver.cuModuleGetFunction(C.byref(self.function), module.handle, name.encode("ascii")))

    def __call__(self, grid, block, arguments, *, stream):
        parameters = (C.c_void_p * len(arguments))(*(C.addressof(value) for value in arguments))
        self.module.check(self.module.driver.cuLaunchKernel(
            self.function, grid[0], grid[1], 1, block[0], block[1], 1, 0,
            C.c_void_p(stream), parameters, None,
        ))


class _Module:
    def __init__(self, source, torch_file, cuda_version, capability):
        self.nvrtc, self.dll_handles, self.nvrtc_path = load_nvrtc(torch_file, cuda_version)
        # The driver belongs to the installed NVIDIA driver, not a CUDA toolkit.
        self.driver = C.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
        pointer = C.c_void_p
        _bind(self.driver, "cuGetErrorString", [C.c_int, C.POINTER(C.c_char_p)])
        _bind(self.driver, "cuInit", [C.c_uint])
        _bind(self.driver, "cuCtxGetCurrent", [C.POINTER(pointer)])
        _bind(self.driver, "cuModuleLoadData", [C.POINTER(pointer), pointer])
        _bind(self.driver, "cuModuleGetFunction", [C.POINTER(pointer), pointer, C.c_char_p])
        _bind(self.driver, "cuLaunchKernel", [pointer, *([C.c_uint] * 7), pointer,
                                              C.POINTER(pointer), C.POINTER(pointer)])
        self.check(self.driver.cuInit(0))
        context = pointer()
        self.check(self.driver.cuCtxGetCurrent(C.byref(context)))
        if not context.value:
            raise RuntimeError("Initialize the current PyTorch CUDA stream before loading surface kernels")
        self.handle = pointer()
        image = _nvrtc_image(self.nvrtc, source, capability)
        self.check(self.driver.cuModuleLoadData(C.byref(self.handle), image))

    def check(self, status):
        if status:
            reason = C.c_char_p()
            self.driver.cuGetErrorString(status, C.byref(reason))
            message = reason.value.decode("utf-8", errors="replace") if reason.value else str(status)
            raise RuntimeError(f"VNCCS CUDA driver: {message}")


def load_surface_kernels(source, names, torch_file, cuda_version, capability):
    """The caller caches these handles in PyTorch's existing device context."""
    module = _Module(source, torch_file, cuda_version, capability)
    return tuple(_Kernel(module, name) for name in names)


def install_nvrtc():
    """Installation-hook entry point; never called by the trainer."""
    if sys.platform not in {"win32", "linux"}:
        return 0
    torch_file, cuda_version = torch_cuda_installation()
    if cuda_version is None:
        print("[VNCCS] NVIDIA Gaussian training is unavailable in this PyTorch build; CUDA packages were not installed.")
        return 0
    try:
        _library, _handles, path = load_nvrtc(torch_file, cuda_version)
    except NvrtcUnavailable:
        requirement = nvrtc_requirement(cuda_version, distribution("torch").requires or ())
        print(f"[VNCCS] Installing only NVRTC for the existing PyTorch CUDA {cuda_version}: {requirement}", flush=True)
        # This standalone installer hands control to pip once, then exits.
        from pip._internal.cli.main import main as pip_main

        arguments = [
            "--isolated", "--disable-pip-version-check", "install", "--no-input",
            "--only-binary=:all:", "--index-url", "https://pypi.org/simple", requirement,
        ]
        if requirement.startswith("nvidia-cuda-nvrtc"):
            arguments.append("--no-deps")
        return pip_main(arguments)
    print(f"[VNCCS] Using existing PyTorch CUDA {cuda_version} NVRTC: {path}")
    return 0
