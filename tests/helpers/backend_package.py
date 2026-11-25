"""Load backend services without executing the ComfyUI extension entry point."""
from contextlib import contextmanager
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def service_package(name):
    for suffix, directory in (
        ("", ROOT), (".nodes", ROOT / "nodes"),
    ):
        package = types.ModuleType(name + suffix)
        package.__path__ = [str(directory)]
        sys.modules[package.__name__] = package
    sys.modules[name + ".utils"] = sys.modules["_vnccs.utils"]
    from _vnccs import operation_logger
    sys.modules[name + ".operation_logger"] = operation_logger
    def load(relative):
        full_name = name + "." + relative
        if full_name in sys.modules:
            return sys.modules[full_name]
        path = ROOT.joinpath(*relative.split(".")).with_suffix(".py")
        spec = importlib.util.spec_from_file_location(full_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return module
    return load


@contextmanager
def stub_imports(modules):
    """Restore only stubs; keep imported extension modules and C dependencies loaded."""
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
