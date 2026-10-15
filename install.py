"""ComfyUI installation hook for NVRTC matching the existing PyTorch build."""

import sys
from pathlib import Path


def main():
    # Import only the stdlib helper, without running node registration during installation.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "nodes"))
    try:
        from gaussian_cuda_runtime import install_nvrtc
    finally:
        sys.path.pop(0)
    return install_nvrtc()


if __name__ == "__main__":
    raise SystemExit(main())
