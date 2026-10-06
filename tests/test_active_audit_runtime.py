"""Active workflow regressions requiring the tensor runtime."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

import utils
from nodes import character_creator_v2 as creator
from nodes import character_cloner as cloner
from nodes import clothes_designer as clothes
from nodes import emotion_generator_v2 as emotions
from nodes import character_generator as generator
from nodes import vnccs_control_center as control
from nodes import qwen_vl as helpers


@pytest.fixture
def responses(monkeypatch):
    from aiohttp import web
    monkeypatch.setattr(web, "json_response", lambda data, status=200: SimpleNamespace(status=status, data=data), raising=False)
    monkeypatch.setattr(web, "Response", lambda status=200, **kwargs: SimpleNamespace(status=status, **kwargs), raising=False)
    monkeypatch.setattr(web, "FileResponse", lambda path: SimpleNamespace(status=200, path=path), raising=False)


@pytest.mark.parametrize("name", ["%2e%2e%2fsecret", "../secret", "..\\secret", "/secret"])
def test_emotion_image_rejects_encoded_and_decoded_traversal(name, responses, monkeypatch, tmp_path):
    monkeypatch.setattr(emotions, "emotion_images_dir", lambda: str(tmp_path))
    result = asyncio.run(emotions.get_emotion_image(SimpleNamespace(rel_url=SimpleNamespace(query={"name": name}))))
    assert result.status == 400


def test_emotion_image_rejects_file_symlink_escape(responses, monkeypatch, tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    (tmp_path / "private.webp").write_bytes(b"private")
    (root / "happy.webp").symlink_to(tmp_path / "private.webp")
    monkeypatch.setattr(emotions, "emotion_images_dir", lambda: str(root))
    result = asyncio.run(emotions.get_emotion_image(SimpleNamespace(rel_url=SimpleNamespace(query={"name": "happy"}))))
    assert result.status == 400


def test_costume_sprite_enumeration_rejects_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    for costume in ("../../outside", "..\\outside", "C:\\outside"):
        with pytest.raises(ValueError):
            emotions.list_costume_sprite_paths("Alice", costume)


def test_missing_profile_fails_before_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    monkeypatch.setattr(emotions, "build_emotion_pipe", lambda *args: pytest.fail("Model loading must not start"))
    with pytest.raises(ValueError, match="Character profile missing"):
        emotions.EmotionGeneratorV2().generate_emotions_v2(character="Alice")


@pytest.mark.parametrize("invalidate_during_model_setup", [False, True])
def test_existing_emotion_instance_uses_replaced_disk_prompts(tmp_path, monkeypatch, invalidate_during_model_setup):
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    utils.save_config("Alice", {"character_info": {"name": "Alice", "age": 20}, "costumes": {}})
    sprite_dir = tmp_path / "characters" / "Alice" / "Sprites" / "Naked" / "Neutral"
    sprite_dir.mkdir(parents=True)
    Image.new("RGBA", (2, 2), "red").save(sprite_dir / "pose.png")
    config = tmp_path / "emotions.json"
    monkeypatch.setattr(emotions, "emotions_config_path", lambda: str(config))
    monkeypatch.setattr(emotions.EmotionGeneratorV2, "SAFE_NAME_MAP", None)
    monkeypatch.setattr(emotions.EmotionGeneratorV2, "EMOTIONS_FINGERPRINT", None)
    def build_pipe(*args):
        if invalidate_during_model_setup:
            # A custom-emotion request may invalidate the shared cache while models load.
            emotions.EmotionGeneratorV2.SAFE_NAME_MAP = None
        return object(), 42
    monkeypatch.setattr(emotions, "build_emotion_pipe", build_pipe)
    def write_prompt(text):
        config.write_text(json.dumps({"Mood": [{"safe_name": "happy", "key": "happy",
                                               "description": text, "natural_prompt": text}]}))
    write_prompt("Old smile")
    node = emotions.EmotionGeneratorV2()
    def prompt():
        tasks = node.generate_emotions_v2(generation_model="QI2", character="Alice",
                                         costumes_data='["Naked"]', emotions_data='["happy"]')[2]
        return json.loads(tasks[0])["emotion_prompt"]
    before = node.IS_CHANGED("Alice", '["Naked"]')
    assert "Old smile" in prompt()
    write_prompt("New smile")
    assert node.IS_CHANGED("Alice", '["Naked"]') != before
    assert "New smile" in prompt()
    assert "Old smile" not in prompt()


@pytest.mark.parametrize("kind", ["creator", "cloner"])
def test_profile_transaction_cannot_restore_a_deleted_costume(tmp_path, monkeypatch, kind):
    import threading
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    utils.save_config("Alice", {"character_info": {"name": "Alice", "hair": "old"}, "costumes": {"Dress": {"top": "silk"}}})
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2), "blue").save(source)
    module = creator if kind == "creator" else cloner
    node = creator.CharacterCreatorV2() if kind == "creator" else cloner.CharacterCloner()
    read, release, deleting, deleted = (threading.Event() for _ in range(4))
    errors = []
    original_load = module.load_config
    def pause_after_read(*args, **kwargs):
        config = original_load(*args, **kwargs)
        read.set()
        if not release.wait(2):
            raise RuntimeError("Profile transaction timed out")
        return config
    monkeypatch.setattr(module, "load_config", pause_after_read)
    monkeypatch.setattr(cloner, "_source_image_path", lambda image: str(source))
    def stop_models(*args):
        raise RuntimeError("Test model boundary reached")
    monkeypatch.setattr(creator, "load_generation_assets", stop_models)
    data = {"character": "Alice", "character_info": {"name": "Alice", "hair": "new"}, "source_images": ["source.png"]}
    def profile():
        try:
            node.process(widget_data=json.dumps(data))
        except Exception as error:
            if kind != "creator" or "Test model boundary reached" not in str(error):
                errors.append(error)
    def delete():
        try:
            deleting.set()
            utils.delete_costume("Alice", "Dress")
            deleted.set()
        except Exception as error:
            errors.append(error)
    first, second = threading.Thread(target=profile), threading.Thread(target=delete)
    first.start()
    try:
        assert read.wait(2)
        second.start()
        assert deleting.wait(2)
        assert not deleted.wait(0.03), "Deletion must wait for the complete profile transaction"
    finally:
        release.set()
        first.join(2)
        if second.ident is not None:
            second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert deleted.is_set()
    config = utils.load_config("Alice")
    assert "Dress" not in config["costumes"]
    assert config["character_info"]["hair"] == "new"


@pytest.mark.parametrize("handler", [
    creator.vnccs_character_wizard, creator.preview_generate,
    cloner.cloner_auto_generate, cloner.cloner_download_model,
    clothes.vnccs_clothes_wizard, clothes.vnccs_save_costume,
    emotions.add_custom_emotion, generator.vnccs_character_generator_regenerate,
    generator.vnccs_character_generator_seedvr_download, control.cc_clothes_preview,
    helpers.qwen_vl_download_model,
    control.cc_add_custom_lora, control.cc_delete_custom_lora,
])
def test_privileged_handlers_reject_before_parsing_body(handler, responses):
    async def forbidden_body():
        pytest.fail("Untrusted request must be rejected before processing")
    result = asyncio.run(handler(SimpleNamespace(headers={"Host": "localhost", "Sec-Fetch-Site": "cross-site"}, json=forbidden_body)))
    assert result.status == 403


@pytest.mark.parametrize("group", utils.MAIN_DIRS)
def test_costume_save_rejects_redirected_storage_before_any_write(tmp_path, monkeypatch, responses, group):
    root, outside = tmp_path / "characters", tmp_path / "outside"
    character = root / "Alice"
    character.mkdir(parents=True)
    outside.mkdir()
    (character / group).symlink_to(outside, target_is_directory=True)
    before = set(root.rglob("*"))
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(root))
    async def body():
        return {"character": "Alice", "costume": "Coat", "info": {"top": "silk"}}
    result = asyncio.run(clothes.vnccs_save_costume(SimpleNamespace(
        headers={"Host": "localhost", "X-VNCCS-CSRF": "1"}, json=body)))
    assert result.status == 400
    assert "outside allowed directory" in result.data["error"]
    assert set(root.rglob("*")) == before
    assert not list(outside.iterdir())


@pytest.mark.parametrize("handler,field", [
    (control.cc_add_custom_lora, "repo_id"), (control.cc_add_custom_lora, "path"),
    (control.cc_delete_custom_lora, "repo_id"), (control.cc_delete_custom_lora, "local_path"),
])
@pytest.mark.parametrize("invalid_object", [False, True])
def test_custom_lora_rejects_invalid_bodies_before_persistence(handler, field, invalid_object, responses, monkeypatch):
    payload = [] if invalid_object else {"repo_id": "vnccs", "path": "Coat.safetensors",
                                       "local_path": "models/loras/Coat.safetensors", field: []}
    monkeypatch.setattr(control, "_save_custom_loras", lambda *args: pytest.fail("Invalid body must not persist"))
    monkeypatch.setattr(control, "_remove_custom_lora", lambda **kwargs: pytest.fail("Invalid body must not persist"))
    async def body():
        return payload
    result = asyncio.run(handler(SimpleNamespace(headers={"Host": "localhost", "X-VNCCS-CSRF": "1"}, json=body)))
    assert result.status == 400


def test_generator_cleanup_warning_keeps_successful_publication(tmp_path, monkeypatch, capsys):
    target = tmp_path / "Alice" / "Sprites" / "Naked" / "Neutral"
    target.mkdir(parents=True)
    Image.new("RGB", (2, 2), "red").save(target / "old.png")
    monkeypatch.setattr(generator, "_character_root_from_sheets_path", lambda *args: str(tmp_path / "Alice"))
    remove = utils.shutil.rmtree
    def deny_backup(path, **kwargs):
        if ".vnccs-rollback-" in str(path):
            raise PermissionError("Backup is locked")
        return remove(path, **kwargs)
    monkeypatch.setattr(utils.shutil, "rmtree", deny_backup)
    paths = generator.VNCCS_CharacterGenerator()._save_final_sprites(torch.ones(1, 2, 2, 3), "", "Alice")
    assert len(paths) == 1
    with Image.open(paths[0]) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
    assert (target / "V1" / "old.png").exists()
    assert "Sprites published; could not remove rollback directory" in capsys.readouterr().out


def test_cloner_grid_limits_apply_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr(cloner, "MAX_GRID_PIXELS", 4)
    path = tmp_path / "large.png"
    Image.new("RGB", (3, 3)).save(path)
    monkeypatch.setattr(cloner, "_source_image_path", lambda image: str(path))
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path / "characters"))
    data = {"character": "Alice", "source_images": ["large.png"]}
    with pytest.raises(ValueError, match="source pixels"):
        cloner.CharacterCloner().process(widget_data=json.dumps(data))
    data["source_images"] = ["large.png"] * (cloner.MAX_SOURCE_IMAGES + 1)
    with pytest.raises(ValueError, match="images"):
        cloner.CharacterCloner().process(widget_data=json.dumps(data))


def test_disk_fingerprints_invalidate_unchanged_widget_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    root = tmp_path / "Alice"
    neutral = root / "Sprites" / "Naked" / "Neutral"
    neutral.mkdir(parents=True)
    config = root / "Alice_config.json"
    config.write_text('{"character_info":{"hair":"blue"},"costumes":{}}')
    Image.new("RGB", (2, 2), "red").save(neutral / "one.png")
    source = tmp_path / "source.png"
    Image.new("RGB", (2, 2), "red").save(source)
    monkeypatch.setattr(cloner, "_source_image_path", lambda image: str(source))
    widget = json.dumps({"character": "Alice", "source_images": ["source.png"]})
    checks = [lambda: creator.CharacterCreatorV2.IS_CHANGED(widget),
              lambda: cloner.CharacterCloner.IS_CHANGED(widget),
              lambda: emotions.EmotionGeneratorV2.IS_CHANGED("Alice", '["Naked"]')]
    before = [check() for check in checks]
    config.write_text('{"character_info":{"hair":"pink"},"costumes":{}}')
    assert all(check() != old for check, old in zip(checks, before))
    for check, path in ((checks[0], neutral / "one.png"), (checks[1], source), (checks[2], neutral / "one.png")):
        before = check()
        Image.new("RGB", (2, 2), "blue" if path == source else "green" if check == checks[2] else "blue").save(path)
        assert check() != before
