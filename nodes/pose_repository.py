"""Import a public pose repository in one shallow Git clone, as Pose Studio does."""

import io
import re
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def checkout_repository(root, repo_id):
    from dulwich import porcelain
    from dulwich.config import ConfigDict

    if not re.fullmatch(r"[\w-][\w.-]*/[\w-][\w.-]*", repo_id, flags=re.ASCII):
        raise ValueError("Invalid Hugging Face repository id")
    root.mkdir(parents=True, exist_ok=True)
    config = ConfigDict()
    config.set((b"http",), b"timeout", b"60")
    with tempfile.TemporaryDirectory(prefix=".checkout-", dir=str(root)) as temporary:
        # Bare import avoids checkout filters, hooks, submodules and user Git configuration.
        with porcelain.clone(f"https://huggingface.co/{repo_id}", str(Path(temporary) / "repository"),
                             bare=True, checkout=False, depth=1, config=config, env={}, errstream=io.BytesIO()) as repository:
            yield repository


def read_repository_file(repository, name, repo_id, root):
    from dulwich.object_store import tree_lookup_path

    if (not isinstance(name, str) or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Unsafe pose repository path")
    mode, key = tree_lookup_path(repository.__getitem__, repository[repository.head()].tree, name.encode("utf-8"))
    if not stat.S_ISREG(mode):
        raise ValueError("Pose repository asset must be a regular file")
    raw = repository[key].data
    if raw.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        from huggingface_hub import hf_hub_download

        cache = root / "hub"
        file = Path(hf_hub_download(repo_id=repo_id, filename=name, revision=repository.head().decode("ascii"),
                                    cache_dir=str(cache), token=False))
        if not file.resolve().is_relative_to(cache.resolve()):
            raise ValueError("Pose cache path escapes the library")
        raw = file.read_bytes()
    return raw
