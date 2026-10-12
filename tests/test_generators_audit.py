"""Generator boundary regressions without model downloads or GPU sampling."""

from types import SimpleNamespace
import json

import pytest

torch = pytest.importorskip("torch")

from nodes import character_generator as generator


def test_generator_preserves_zero_guidance():
    pipe = SimpleNamespace(model="model", clip="clip", vae="vae", seed_int=42,
                           sample_steps=20, cfg=0.0, denoise=1.0,
                           sampler_name="euler", scheduler="normal")
    assert generator.VNCCS_CharacterGenerator()._extract_pipe(pipe)["cfg"] == 0.0


@pytest.mark.parametrize("qi2", [False, True])
def test_emotion_explicit_zero_seed_reaches_sampler(monkeypatch, qi2):
    node = generator.VNCCS_EmotionsGenerator()
    node._extract_pipe = lambda pipe: {"model": "model", "clip": "clip", "vae": "vae",
        "seed": 42, "steps": 20, "cfg": 3.0, "sampler": "euler", "scheduler": "normal",
        "model_kind": "qi2" if qi2 else "illustrious"}
    image = torch.zeros(1, 16, 16, 3)
    seeds = []

    def call(name, **kwargs):
        if name == "FaceDetailer":
            seeds.append(kwargs["seed"])
            return image, image, None, torch.ones(1, 16, 16)
        return (name,)

    monkeypatch.setattr(generator, "_call_comfy_node", call)

    def crop(*args):
        seeds.append(args[6]["seed"])
        return image, image, torch.ones(1, 16, 16)

    node._run_qi2_emotion_crop_generation = crop
    node._run_emotion_generation_one(image, None, object(), "Smile", "", "", 0)
    assert seeds == [0]


@pytest.mark.parametrize("mode", ["character", "clone", "clothes"])
@pytest.mark.parametrize("cache_state", ["healthy", "missing", "damaged", "invalid_index", "invalid_stage"])
@pytest.mark.parametrize("index", [0, 2])
def test_single_image_regeneration_preserves_published_sprites(tmp_path, monkeypatch, mode, cache_state, index):
    node = {"character": generator.VNCCS_CharacterGenerator,
            "clone": generator.VNCCS_CharacterCloneGenerator,
            "clothes": generator.VNCCS_ClothesGenerator}[mode]()
    output = tmp_path / "output"
    root = output / "VNCCS" / "Characters" / "Alice"
    sheets = root / "Sheets" / "Naked" / "neutral"
    sheets.mkdir(parents=True)
    monkeypatch.setattr(generator.folder_paths, "get_output_directory", lambda: str(output))
    monkeypatch.setattr(node, "_find_pose_lora", lambda *args: {})
    monkeypatch.setattr(node, "_extract_pipe", lambda *args: {"seed": 0})
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: None)
    for method in ("_run_pose_generation", "_run_upscaler", "_run_source_upscaler"):
        monkeypatch.setattr(node, method, lambda images, *args, **kwargs: images)
    images = torch.stack([torch.full((8, 8, 4), value) for value in (0.1, 0.2, 0.3)])
    state = {"character_name": "Alice", "nsfw_enabled": False, "bg_remove": {"preset": "Native"}}
    arguments = (images, images[:1], object(), "Pose")
    node.process(*arguments, widget_data=json.dumps(state), sheets_path=str(sheets), unique_id="audit")
    sprite_set = "Original" if mode == "clone" else "Naked"
    target = root / "Sprites" / sprite_set / "Neutral"
    before = {path.name: path.read_bytes() for path in target.glob("*.png")}
    assert len(before) == 3
    stage = "original_bg_remove" if mode == "clone" else "bg_remove"
    cache_path = next(root.rglob(f"{stage}.pt"))
    if cache_state == "missing":
        cache_path.unlink()
    elif cache_state == "damaged":
        cache_path.write_bytes(b"damaged stage cache")
    selected = 9 if cache_state == "invalid_index" else index
    state.update(regenerate_from="missing-stage" if cache_state == "invalid_stage" else stage, regenerate_index=selected)
    monkeypatch.setattr(node, "_run_bg_remove", lambda images, *args, **kwargs: torch.full_like(images, 0.9))
    if cache_state != "healthy":
        with pytest.raises(ValueError, match="Run the generator normally"):
            node.process(*arguments, widget_data=json.dumps(state), sheets_path=str(sheets), unique_id="audit")
        assert {path.name: path.read_bytes() for path in target.glob("*.png")} == before
    else:
        result = node.process(*arguments, widget_data=json.dumps(state), sheets_path=str(sheets), unique_id="audit")
        assert result[0].shape[0] == 3
        after = {path.name: path.read_bytes() for path in target.glob("*.png")}
        assert after.keys() == before.keys()
        for position, (name, original) in enumerate(sorted(before.items())):
            assert (after[name] != original) == (position == index)


@pytest.mark.parametrize("stage, index", [("emotion_0001_bg_remove", 1), ("missing-stage", 0)])
def test_invalid_emotion_regeneration_preserves_every_group_before_writes(tmp_path, monkeypatch, stage, index):
    output = tmp_path / "output"
    root = output / "VNCCS" / "Characters" / "Alice"
    directory = root / "Sprites" / "Original" / "smile"
    directory.mkdir(parents=True)
    existing = directory / "sprite_smile_0001.png"
    existing.write_bytes(b"published sprite must remain unchanged")
    monkeypatch.setattr(generator.folder_paths, "get_output_directory", lambda: str(output))
    node = generator.VNCCS_EmotionsGenerator()
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(generator, "_save_run_inputs", lambda *args, **kwargs: pytest.fail("Invalid selection must fail before cache writes"))
    monkeypatch.setattr(node, "_run_emotion_generation_one", lambda *args, **kwargs: pytest.fail("Invalid selection must fail before model execution"))
    tasks = [{"emotion_prompt": "smile", "sprite_output_path": "group-a"},
             {"emotion_prompt": "smile", "sprite_output_path": "group-a"},
             {"emotion_prompt": "angry", "sprite_output_path": "group-b"}]
    images = torch.ones(3, 8, 8, 4)
    state = {"character_name": "Alice", "regenerate_from": stage, "regenerate_index": index}
    with pytest.raises(ValueError, match="Run the generator normally"):
        node.process(images, object(), tasks, widget_data=json.dumps(state), unique_id="audit")
    assert existing.read_bytes() == b"published sprite must remain unchanged"


@pytest.mark.parametrize("character, costume", [("Sheets", "Sheets"), ("Sprites", "Sprites")])
def test_generator_publishes_sources_and_faces_under_characters_named_like_storage_folders(tmp_path, monkeypatch, character, costume):
    base = tmp_path / "Sheets" / "Sprites" / "VNCCS" / "Characters"
    root = base / character
    monkeypatch.setattr(generator, "base_output_dir", lambda: str(base))
    monkeypatch.setattr(generator, "character_dir", lambda name: str(base / name))
    images = torch.ones(1, 8, 8, 4)
    sheets = root / "Sheets" / costume / "neutral" / "sheet_neutral"
    node = generator.VNCCS_EmotionsGenerator()

    saved = node._save_final_sprites(images, str(sheets), character, costume)
    source = root / "Sprites" / costume / "Neutral" / "sprite_pose_0001.png"
    assert saved == [str(source)]
    assert node._source_sprite_hw(str(source), character) == (8, 8)
    original = source.read_bytes()
    prefix = root / "Sprites" / costume / "smile" / "sprite_smile_"
    face_prefix = node._face_prefix_from_sprite_prefix(str(prefix), character)
    saved_face = node._save_rgba_image(images, None, face_prefix, 1, character, "Faces")
    assert saved_face == str(root / "Faces" / costume / "smile" / "face_smile_0001.png")
    assert source.read_bytes() == original
    assert not (base / "cache").exists()
    assert not (base / "Sprites" / costume / "Neutral").exists()


@pytest.mark.parametrize("mode", ["character", "clone", "clothes"])
def test_single_image_resolution_change_never_resizes_other_published_poses(tmp_path, monkeypatch, mode):
    node = {"character": generator.VNCCS_CharacterGenerator,
            "clone": generator.VNCCS_CharacterCloneGenerator,
            "clothes": generator.VNCCS_ClothesGenerator}[mode]()
    output = tmp_path / "output"
    root = output / "VNCCS" / "Characters" / "Alice"
    sheets = root / "Sheets" / "Coat" / "neutral"
    sheets.mkdir(parents=True)
    monkeypatch.setattr(generator.folder_paths, "get_output_directory", lambda: str(output))
    monkeypatch.setattr(node, "_find_pose_lora", lambda *args: {})
    monkeypatch.setattr(node, "_extract_pipe", lambda *args: {"seed": 0})
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: None)
    for method in ("_run_pose_generation", "_run_upscaler", "_run_source_upscaler"):
        monkeypatch.setattr(node, method, lambda images, *args, **kwargs: images)
    images = torch.stack([torch.full((8, 8, 4), value) for value in (0.1, 0.2, 0.3)])
    state = {"character_name": "Alice", "nsfw_enabled": False, "bg_remove": {"preset": "Native"}}
    arguments = (images, images[:1], object(), "Pose")
    node.process(*arguments, widget_data=json.dumps(state), sheets_path=str(sheets), unique_id="audit")
    sprite_set = "Original" if mode == "clone" else "Coat" if mode == "clothes" else "Naked"
    target = root / "Sprites" / sprite_set / "Neutral"
    before = {path.name: path.read_bytes() for path in target.glob("*.png")}
    assert len(before) == 3
    state.update(regenerate_from="original_upscaler" if mode == "clone" else "upscaler", regenerate_index=1)
    monkeypatch.setattr(node, "_run_upscaler", lambda images, *args, **kwargs: torch.ones(1, 16, 16, 4))

    with pytest.raises(ValueError, match="Run the generator normally"):
        node.process(*arguments, widget_data=json.dumps(state), sheets_path=str(sheets), unique_id="audit")
    assert {path.name: path.read_bytes() for path in target.glob("*.png")} == before


@pytest.mark.parametrize("targeted", [False, True])
def test_emotion_background_regeneration_preserves_the_generated_face_crop(tmp_path, monkeypatch, targeted):
    from PIL import Image

    output = tmp_path / "output"
    root = output / "VNCCS" / "Characters" / "Alice"
    monkeypatch.setattr(generator.folder_paths, "get_output_directory", lambda: str(output))
    node = generator.VNCCS_EmotionsGenerator()
    previews = []
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: previews.append(args[3])
                        if len(args) > 3 and torch.is_tensor(args[3]) else None)
    source = torch.full((1, 8, 8, 3), 0.2)
    source_path = root / "Sprites" / "Coat" / "Neutral" / "sprite_pose_0001.png"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), (51, 51, 51)).save(source_path)
    monkeypatch.setattr(node, "_run_emotion_bg_remove", lambda images, *args, **kwargs: images)
    crop = torch.full((1, 4, 4, 3), 0.7)
    monkeypatch.setattr(node, "_run_emotion_generation_one", lambda *args, **kwargs: (source, crop, torch.ones(1, 8, 8)))
    task = {"character": "Alice", "emotion_prompt": "smile", "sprite_index": 1,
            "source_path": str(source_path), "sprite_output_path": str(root / "Sprites" / "Coat" / "smile" / "sprite_smile_")}
    state = {"character_name": "Alice", "emotion_pairs": [{"costume": "Coat", "emotion": "smile"}]}
    arguments = {"unique_id": "audit", "prompt": {"viewer": {"inputs": {"images": ["audit", 1]}}}}
    result = node.process([], object(), [task], widget_data=json.dumps(state), **arguments)
    assert len(result[1]) == 1
    assert torch.equal(result[1][0], crop)
    face = root / "Faces" / "Coat" / "smile" / "face_smile_0001.png"
    before = face.read_bytes()
    with Image.open(face) as opened:
        assert opened.size == (4, 4)

    state["regenerate_from"] = "emotion_0001_bg_remove"
    if targeted:
        state["regenerate_index"] = 0
    monkeypatch.setattr(node, "_run_emotion_generation_one", lambda *args, **kwargs: pytest.fail("Background regeneration must reuse the cached face edit"))
    result = node.process([], object(), [task], widget_data=json.dumps(state), **arguments)
    assert face.read_bytes() == before
    assert len(result[1]) == 1
    assert result[1][0].shape == (1, 4, 4, 4)
    assert torch.allclose(result[1][0][..., :3], crop, atol=1 / 255)
    assert previews and all(image.shape[1:3] == (8, 8) for image in previews)

    face.unlink()
    result = node.process([], object(), [task], widget_data=json.dumps(state), **arguments)
    assert result[1] == []
    assert not face.exists()

    state["regenerate_from"] = "emotion_0001"
    changed_crop = torch.full_like(crop, 0.9)
    monkeypatch.setattr(node, "_run_emotion_generation_one", lambda *args, **kwargs: (source, changed_crop, torch.ones(1, 8, 8)))
    result = node.process([], object(), [task], widget_data=json.dumps(state), **arguments)
    assert torch.equal(result[1][0], changed_crop)
    assert face.read_bytes() != before
    with Image.open(face) as opened:
        assert opened.size == (4, 4)


@pytest.mark.parametrize("seed, expected", [
    (0xFFFFFFFFFFFFFFFF, [0xFFFFFFFFFFFFFFFF, 0, 1]),
    (0xFFFFFFFFFFFFFFFE, [0xFFFFFFFFFFFFFFFE, 0xFFFFFFFFFFFFFFFF, 0]),
    (0, [0, 1, 2]),
])
def test_emotion_tasks_wrap_the_seed_before_sampling(tmp_path, monkeypatch, seed, expected):
    from PIL import Image

    output = tmp_path / "output"
    root = output / "VNCCS" / "Characters" / "Alice"
    monkeypatch.setattr(generator.folder_paths, "get_output_directory", lambda: str(output))
    node = generator.VNCCS_EmotionsGenerator()
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: None)
    sampled_seeds = []

    def sample(image, mask, pipe, emotion, details, negative, task_seed, **kwargs):
        noise = torch.Generator().manual_seed(task_seed)
        sampled_seeds.append(noise.initial_seed())
        result = torch.rand(image.shape, generator=noise)
        return result, result[:, :4, :4], torch.ones(1, 8, 8)

    monkeypatch.setattr(node, "_run_emotion_generation_one", sample)
    monkeypatch.setattr(node, "_run_emotion_bg_remove", lambda images, *args, **kwargs: images)
    prefix = root / "Sprites" / "Coat" / "smile" / "sprite_smile_"
    tasks = [{"character": "Alice", "seed": seed, "emotion_prompt": "smile",
              "sprite_index": index, "sprite_output_path": str(prefix)}
             for index in (1, 2, 3)]
    state = {"character_name": "Alice", "emotion_pairs": [{"costume": "Coat", "emotion": "smile"}]}
    images = torch.ones(3, 8, 8, 3)
    result = node.process(images, object(), tasks, widget_data=json.dumps(state), unique_id="seed-audit",
                          prompt={"viewer": {"inputs": {"images": ["seed-audit", 0]}}})

    assert sampled_seeds == expected
    assert len(result[0]) == 3
    for index, image in enumerate(result[0], start=1):
        path = prefix.parent / f"sprite_smile_{index:04d}.png"
        with Image.open(path) as opened:
            assert opened.size == (8, 8)
        assert torch.isfinite(image).all()
