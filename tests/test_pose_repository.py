import stat

import pytest

from conftest import _preload_node
from pose_repository_fixtures import git_repository

transport = _preload_node("pose_repository")


@pytest.mark.parametrize("name", ["../outside.json", "/outside.json", "a\\b.json", "a//b.json", "a/./b.json", "C:/a.json"])
def test_checkout_rejects_unsafe_paths(tmp_path, git_repository, name):
    with transport.checkout_repository(tmp_path, "owner/poses") as repository:
        with pytest.raises(ValueError, match="Unsafe"):
            transport.read_repository_file(repository, name, "owner/poses", tmp_path)


@pytest.mark.parametrize("mode", [stat.S_IFLNK | 0o777, 0o160000])
def test_checkout_rejects_links_and_submodules(tmp_path, git_repository, mode):
    git_repository.files = {"pose.json": b"{}"}
    git_repository.modes = {"pose.json": mode}
    with transport.checkout_repository(tmp_path, "owner/poses") as repository:
        with pytest.raises(ValueError, match="regular file"):
            transport.read_repository_file(repository, "pose.json", "owner/poses", tmp_path)


def test_failed_checkout_cleans_staging_directory(tmp_path, git_repository):
    with pytest.raises(ValueError, match="Failed import"):
        with transport.checkout_repository(tmp_path, "owner/poses"):
            raise ValueError("Failed import")
    assert list(tmp_path.iterdir()) == []


def test_real_git_clone_and_blob_import(tmp_path, monkeypatch):
    porcelain = pytest.importorskip("dulwich.porcelain")
    from dulwich.repo import Repo

    origin = tmp_path / "origin"
    with Repo.init(str(origin), mkdir=True) as repository:
        (origin / "pose.json").write_bytes(b'{"bones": {}}')
        (origin / "preview.webp").write_bytes(b"fixture preview")
        porcelain.add(repository, paths=["pose.json", "preview.webp"])
        porcelain.commit(repository, message=b"Pose fixture", author=b"Fixture <fixture@example.test>",
                         committer=b"Fixture <fixture@example.test>")
    clone = porcelain.clone
    calls = []
    def local_clone(url, target, **kwargs):
        calls.append(url)
        assert kwargs["bare"] and not kwargs["checkout"] and kwargs["depth"] == 1 and kwargs["env"] == {}
        return clone(str(origin), target, **kwargs)
    monkeypatch.setattr(porcelain, "clone", local_clone)
    root = tmp_path / "library"
    with transport.checkout_repository(root, "owner/poses") as repository:
        assert transport.read_repository_file(repository, "pose.json", "owner/poses", root) == b'{"bones": {}}'
        assert transport.read_repository_file(repository, "preview.webp", "owner/poses", root) == b"fixture preview"
    assert calls == ["https://huggingface.co/owner/poses"]
    assert list(root.iterdir()) == []
