"""Install the Qwen runtime used by ComfyUI Easy Install."""

from importlib.metadata import PackageNotFoundError, distribution
import platform
import sys


def qwen_runtime_installed():
    try:
        package = distribution("llama-cpp-python")
    except PackageNotFoundError:
        return False
    from pip._vendor.packaging.version import Version

    sources = package.metadata.get_all("Project-URL", [])
    return Version(package.version) >= Version("0.3.46") and any(
        "github.com/JamePeng/llama-cpp-python" in source for source in sources
    )


def qwen_wheel_url(system, machine, python_version, cuda_version):
    """Match the existing interpreter and accelerator without replacing PyTorch."""
    if python_version not in {(3, minor) for minor in range(10, 15)}:
        raise ValueError("Qwen runtime wheels require CPython 3.10 through 3.14.")
    python_tag = f"cp{python_version[0]}{python_version[1]}"
    version, date = "0.3.46", "20260808"
    if system == "Darwin" and machine.lower() == "arm64":
        release = f"v{version}-Metal-macos-{date}"
        filename = f"llama_cpp_python-{version}-{python_tag}-{python_tag}-macosx_11_0_arm64.whl"
    elif system in {"Windows", "Linux"} and machine.lower() in {"amd64", "x86_64"}:
        if cuda_version not in {"12.4", "12.6", "12.8", "13.0", "13.1"}:
            raise ValueError(
                f"No matching Qwen CUDA wheel for PyTorch CUDA {cuda_version!r}. "
                "Use a supported JamePeng llama-cpp-python build; VNCCS will not replace PyTorch or CUDA."
            )
        # The first published Linux CUDA 13.0 build after 0.3.46 is 0.4.2.
        if system == "Linux" and cuda_version == "13.0":
            version, date = "0.4.2", "20261003"
        cuda_tag = "cu" + cuda_version.replace(".", "")
        release_os, wheel_platform = (
            ("win", "win_amd64") if system == "Windows" else ("linux", "linux_x86_64")
        )
        release = f"v{version}-{cuda_tag}-{release_os}-{date}"
        filename = f"llama_cpp_python-{version}+{cuda_tag}-{python_tag}-{python_tag}-{wheel_platform}.whl"
    else:
        raise ValueError(f"No published Qwen runtime wheel for {system} {machine}.")
    return f"https://github.com/JamePeng/llama-cpp-python/releases/download/{release}/{filename}"


def main():
    if qwen_runtime_installed():
        print("[VNCCS] Compatible JamePeng Qwen runtime is already installed.")
        return 0
    try:
        import torch

        url = qwen_wheel_url(
            platform.system(), platform.machine(), sys.version_info[:2], torch.version.cuda
        )
    except (ImportError, ValueError) as error:
        print(f"[VNCCS] Cannot install the Qwen runtime: {error}", file=sys.stderr)
        return 1

    from pip._internal.cli.main import main as pip_main

    print(f"[VNCCS] Installing the matching JamePeng Qwen runtime: {url}")
    return pip_main(["install", "--only-binary=:all:", url])


if __name__ == "__main__":
    raise SystemExit(main())
