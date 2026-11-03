"""Generator input boundaries execute in CI without tensor dependencies."""

import asyncio
import json
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
import os

import pytest
from test_proxy_runtime import generator_module
from test_clothes_audit_paths import clothes_paths


@pytest.mark.parametrize("storage", ["metadata", "sprites"])
def test_costume_existence_check_keeps_legacy_slots_and_rejects_deleted_ones(monkeypatch, tmp_path, storage):
    import utils
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    if storage == "metadata":
        utils.save_costume_info("Alice", "Coat", {"top": "authored coat"})
    else:
        directory = Path(utils.character_dir("Alice")) / "Sprites" / "Coat"
        directory.mkdir(parents=True)
        (directory / "pose.png").write_bytes(b"legacy sprite")
    utils.require_existing_costume("Alice", "Coat")
    utils.delete_costume("Alice", "Coat")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(ValueError, match="no longer exists"):
        utils.require_existing_costume("Alice", "Coat")

    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert "Coat" not in utils.list_costumes("Alice")


def test_deleted_costume_stops_designer_before_loading_the_pipe(clothes_paths):
    import utils
    utils.save_costume_info("Alice", "Coat", {"top": "authored coat"})
    utils.delete_costume("Alice", "Coat")

    class UnloadedPipe:
        @property
        def model(self):
            pytest.fail("Deleted costumes must fail before loading models")

    with pytest.raises(ValueError, match="no longer exists"):
        clothes_paths.ClothesDesigner().process(UnloadedPipe(), json.dumps({
            "character": "Alice", "costume": "Coat", "costume_info": {"top": "old draft"},
        }))
    assert not (Path(utils.character_dir("Alice")) / "cache").exists()


@pytest.mark.parametrize("regenerate", [False, True])
def test_deleted_costume_stops_generator_before_cache_or_model_work(generator_module, monkeypatch, tmp_path, regenerate):
    import utils
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(tmp_path))
    monkeypatch.setattr(generator_module, "base_output_dir", lambda: str(tmp_path))
    utils.save_costume_info("Alice", "Coat", {"top": "authored coat"})
    sheets = utils.sheets_dir("Alice", "Coat")
    utils.delete_costume("Alice", "Coat")
    before = Path(utils.config_path("Alice")).read_bytes()
    node = generator_module.VNCCS_ClothesGenerator()
    monkeypatch.setattr(node, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(generator_module, "_remember_generator_context", lambda *args:
                        pytest.fail("Deleted costumes must fail before cache or model work"))
    state = {"character_name": "Alice", "costume": "Coat"}
    if regenerate:
        state["regenerate_from"] = "pose_generation"

    with pytest.raises(ValueError, match="no longer exists"):
        node.process(None, None, object(), "old draft", widget_data=json.dumps(state), sheets_path=sheets)

    assert "Coat" not in utils.list_costumes("Alice")
    assert Path(utils.config_path("Alice")).read_bytes() == before
    assert not (Path(utils.character_dir("Alice")) / "cache").exists()


@pytest.mark.parametrize("generator_type", ["VNCCS_CharacterGenerator", "VNCCS_CharacterCloneGenerator", "VNCCS_ClothesGenerator"])
@pytest.mark.parametrize("inherit", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_regenerate_shifts_local_seeds_without_changing_authored_settings_or_pipe(
    generator_module, monkeypatch, generator_type, inherit, fail,
):
    from contextlib import nullcontext
    generator = generator_module
    pipe = SimpleNamespace(seed_int=100, seed=101)
    state = {"character_name": "Alice", "pose_sampler": {"inherit_pipe": inherit, "seed": 42},
             "remove_clothes_sampler": {"inherit_pipe": inherit, "seed": 43},
             "upscaler": {"inherit_pipe_seed": inherit, "seed": 44}}
    original = json.dumps(state, sort_keys=True)
    monkeypatch.setattr(generator.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(generator, "_get_generator_context", lambda *args:
                        {"pipe": pipe, "cache_dir": "cache", "generator_type": generator_type})
    monkeypatch.setattr(generator, "_load_run_inputs", lambda *args, **kwargs:
                        {"widget_payload": json.loads(original), "sheets_path": "cached"})
    monkeypatch.setattr(generator.random, "randint", lambda *args: 7)
    monkeypatch.setattr(generator.torch, "inference_mode", nullcontext, raising=False)
    observed = []

    def process(self, *args, **kwargs):
        settings = json.loads(kwargs["widget_data"])
        values = {"seed": pipe.seed_int, "steps": 4, "cfg": 1,
                  "sampler": "euler", "scheduler": "normal"}
        observed.extend([self._sampler_settings(values, settings["pose_sampler"])["seed"],
                         self._sampler_settings(values, settings["remove_clothes_sampler"])["seed"],
                         self._upscaler_seed(settings["upscaler"], pipe.seed_int)])
        if fail:
            raise RuntimeError("sampling failed")

    monkeypatch.setattr(getattr(generator, generator_type), "process", process)
    response = generator._regenerate_response({"unique_id": "audit", "stage": "pose_generation", "widget_data": state})

    assert response.status == (500 if fail else 200)
    assert observed == ([107, 107, 107] if inherit else [49, 50, 51])
    assert json.dumps(state, sort_keys=True) == original
    assert (pipe.seed_int, pipe.seed) == (100, 101)


@pytest.fixture
def emotion_module(generator_module, monkeypatch):
    package = generator_module.__package__
    creator = types.ModuleType(package + ".character_creator_v2")
    creator.ANIMA_DEFAULTS = {}
    creator.ILLUSTRIOUS_DEFAULTS = {}
    for name in ("load_generation_assets", "load_anima_assets", "get_lora_full_path", "apply_generation_lora"):
        setattr(creator, name, lambda *args: pytest.fail("No model preparation may begin"))
    creator.normalize_gen_settings = lambda settings: settings
    creator._generation_file_signatures = lambda settings: {}
    monkeypatch.setitem(sys.modules, creator.__name__, creator)
    monkeypatch.setattr(sys.modules[package + ".vnccs_control_center"], "is_qi2_turbo_model", lambda *args: False, raising=False)
    path = Path(__file__).parents[1] / "nodes" / "emotion_generator_v2.py"
    spec = importlib.util.spec_from_file_location(package + ".emotion_generator_v2", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_emotion_sprite_preview_keeps_http_loop_responsive(emotion_module, monkeypatch):
    import threading
    started, release = threading.Event(), threading.Event()
    calls = []

    def load(character, costume, **kwargs):
        calls.append((character, costume, kwargs))
        started.set()
        assert release.wait(2), "Image loading must not block the HTTP loop"
        return []

    monkeypatch.setattr(emotion_module, "load_costume_sprite_images", load)
    monkeypatch.setattr(emotion_module.web, "Response", lambda **kwargs: SimpleNamespace(**kwargs), raising=False)

    async def run():
        request = SimpleNamespace(rel_url=SimpleNamespace(query={"character": "Alice", "costume": "Coat"}))
        job = asyncio.create_task(emotion_module.get_character_sheet_preview(request))
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(.002)
            assert started.is_set()
            assert not job.done()
            release.set()
            assert (await asyncio.wait_for(job, 2)).status == 404
        finally:
            release.set()
            await job

    asyncio.run(run())
    assert calls == [("Alice", "Coat", {"first_only": True}),
                     ("Alice", "Original", {"first_only": True})]


@pytest.mark.parametrize("settings", ['[]', 'null', '{broken',
    '{"mode_settings":[]}', '{"mode_settings":null}', '{"mode_settings":{"qi2":null}}',
    '{"qi2_cache":[]}', '{"lora_stack":{}}', '{"lora_stack":[null]}',
    '{"mode_settings":{"anima":{"lora_stack":"damaged"}}}', '{"turbo_previous_settings":0}'])
def test_invalid_emotion_settings_fail_before_loading_models(emotion_module, settings):
    with pytest.raises(ValueError):
        emotion_module.build_emotion_pipe("Anima", settings)


@pytest.mark.parametrize("indices", [[1.5], ["1garbage"], [True], [0], [-1], "1", {}, [None]])
def test_invalid_emotion_pose_selection_fails_before_models_and_keeps_authored_data(
    emotion_module, monkeypatch, tmp_path, indices,
):
    monkeypatch.setattr(emotion_module.EmotionGeneratorV2, "_setup_emotions_data", classmethod(lambda cls:
        {"smile": {"description": "soft smile", "natural_prompt": "", "key": "Smile"}}))
    monkeypatch.setattr(emotion_module, "load_character_info", lambda *args: {"name": "Alice"})
    monkeypatch.setattr(emotion_module, "load_costume_info", lambda *args: {})
    monkeypatch.setattr(emotion_module, "build_emotion_pipe", lambda *args:
        pytest.fail("Invalid pose selection must fail before model loading or creating tasks"))
    authored = tmp_path / "sprite_smile_0001.png"
    authored.write_bytes(b"authored pose must not be replaced by a truncated index")
    settings = json.dumps({"selected_pose_indices": indices})
    with pytest.raises(ValueError, match="pose"):
        emotion_module.EmotionGeneratorV2().generate_emotions_v2(
            character="Alice", costumes_data='["Coat"]', emotions_data='["smile"]', generation_settings=settings,
        )
    assert json.loads(settings)["selected_pose_indices"] == indices
    assert authored.read_bytes() == b"authored pose must not be replaced by a truncated index"


@pytest.mark.parametrize("indices,base_count,expected", [(None, 0, list(range(1, 11))), ([], 0, []),
    ([1, 2], 0, [1, 2]), (["2", "1_0"], 0, [2, 10]), ([2.0], 0, [2]),
    ([1, 4, 5], 4, [1, 4]), (None, 4, [1, 2, 3, 4])])
def test_valid_emotion_pose_selection_builds_only_the_authored_tasks(
    emotion_module, monkeypatch, tmp_path, indices, base_count, expected,
):
    monkeypatch.setattr(emotion_module.EmotionGeneratorV2, "_setup_emotions_data", classmethod(lambda cls:
        {"smile": {"description": "soft smile", "natural_prompt": "", "key": "Smile"}}))
    monkeypatch.setattr(emotion_module, "load_character_info", lambda *args: {"name": "Alice"})
    monkeypatch.setattr(emotion_module, "load_costume_info", lambda *args: {})
    monkeypatch.setattr(emotion_module, "build_emotion_pipe", lambda *args: ("selected pipe", 721))
    monkeypatch.setattr(emotion_module, "build_face_details", lambda *args: "")
    monkeypatch.setattr(emotion_module, "character_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(emotion_module, "costume_has_source_sprites", lambda *args: True)
    paths = [str(tmp_path / f"sprite_pose_{index:04d}.png") for index in range(1, 11)]
    monkeypatch.setattr(emotion_module, "list_costume_sprite_paths", lambda *args: paths)
    monkeypatch.setattr(emotion_module, "list_pose_preview_files", lambda *args: paths[:base_count])
    settings = json.dumps({"selected_pose_indices": indices})
    _, pipe, tasks = emotion_module.EmotionGeneratorV2().generate_emotions_v2(
        character="Alice", costumes_data='["Coat"]', emotions_data='["smile"]', generation_settings=settings,
    )
    assert pipe == "selected pipe"
    parsed = [json.loads(task) for task in tasks]
    assert [task["sprite_index"] for task in parsed] == expected
    assert [task["source_path"] for task in parsed] == [paths[index - 1] for index in expected]
    assert all(task["seed"] == 721 for task in parsed)
    assert json.loads(settings)["selected_pose_indices"] == indices


@pytest.mark.parametrize("mode", ["Illustrious", "Anima"])
@pytest.mark.parametrize("authored", [{}, {"steps": 17, "cfg": 0}, {"mode_settings": {"illustrious": {"steps": 18, "cfg": 0}}}])
def test_emotion_pipe_preserves_selected_model_defaults_and_authored_settings(
    emotion_module, monkeypatch, mode, authored,
):
    anima = {"sampler": "er_sde", "scheduler": "simple", "steps": 30, "cfg": 4,
        "dmd_lora_name": "anima-turbo.safetensors", "turbo_enabled": False}
    illustrious = {"sampler": "euler", "scheduler": "normal", "steps": 20, "cfg": 8}
    monkeypatch.setattr(emotion_module, "ANIMA_DEFAULTS", anima)
    monkeypatch.setattr(emotion_module, "ILLUSTRIOUS_DEFAULTS", illustrious)
    def normalize(settings):
        defaults = anima if settings["generation_mode"] == "anima" else illustrious
        profile = settings.get("mode_settings", {}).get(settings["generation_mode"], {})
        return {**defaults, **settings, **profile}
    monkeypatch.setattr(emotion_module, "normalize_gen_settings", normalize)
    monkeypatch.setattr(emotion_module, "load_generation_assets", lambda settings: (None, "model", "clip", "vae"))
    monkeypatch.setattr(emotion_module, "get_lora_full_path", lambda *args: pytest.fail("Anima LoRA must never be applied to Illustrious"))
    raw = json.dumps(authored)
    pipe, _seed = emotion_module.build_emotion_pipe(mode, raw)
    expected = {**(anima if mode == "Anima" else illustrious), **authored,
        **authored.get("mode_settings", {}).get(mode.lower(), {})}
    assert (pipe.sample_steps, pipe.cfg, pipe.sampler_name, pipe.scheduler) == (
        expected["steps"], expected["cfg"], expected["sampler"], expected["scheduler"])
    assert json.loads(raw) == authored


@pytest.mark.parametrize("mode", ["Illustrious", "Anima", "QI2"])
@pytest.mark.parametrize("selection", [["missing-custom"], ["smile", "missing-custom"]])
def test_missing_selected_emotion_blocks_all_tasks_before_models_and_preserves_outputs(
    emotion_module, monkeypatch, tmp_path, mode, selection,
):
    original = tmp_path / "sprite_missing-custom_0001.png"
    original.write_bytes(b"authored expression must remain unchanged")
    catalog = {"smile": {"description": "soft smile", "natural_prompt": "", "key": "Smile"}}
    monkeypatch.setattr(emotion_module.EmotionGeneratorV2, "_setup_emotions_data", classmethod(lambda cls: catalog))
    monkeypatch.setattr(emotion_module, "load_character_info", lambda *args: {"name": "Alice"})
    monkeypatch.setattr(emotion_module, "load_costume_info", lambda *args: {})
    monkeypatch.setattr(emotion_module, "build_emotion_pipe", lambda *args:
        pytest.fail("Missing expressions must fail before model loading or preparing any tasks"))
    with pytest.raises(ValueError, match="missing-custom.*not available"):
        emotion_module.EmotionGeneratorV2().generate_emotions_v2(
            generation_model=mode, character="Alice", costumes_data='["Coat"]', emotions_data=json.dumps(selection),
        )
    assert original.read_bytes() == b"authored expression must remain unchanged"


@pytest.mark.parametrize("generator_type", ["VNCCS_CharacterGenerator", "VNCCS_CharacterCloneGenerator",
    "VNCCS_ClothesGenerator", "VNCCS_EmotionsGenerator"])
@pytest.mark.parametrize("raw", ['{broken', 'null', '[]', '{"bg_remove":null}',
    '{"pose_generation":[]}', '{"ui":0}', '{"emotion_generation":"damaged"}'])
def test_generator_invalid_state_fails_before_progress_cache_or_model_work(generator_module, generator_type, raw):
    node = getattr(generator_module, generator_type)()
    node._begin_progress = lambda *args: pytest.fail("Invalid settings must not begin progress, cache writes, or model work")
    arguments = ([], object(), []) if generator_type == "VNCCS_EmotionsGenerator" else ([], [], object(), "prompt")
    with pytest.raises(ValueError):
        node.process(*arguments, widget_data=raw, unique_id="invalid-state-audit")


@pytest.mark.parametrize("generator_type", ["VNCCS_CharacterGenerator", "VNCCS_CharacterCloneGenerator",
    "VNCCS_ClothesGenerator", "VNCCS_EmotionsGenerator"])
@pytest.mark.parametrize("raw", [None, '', '{}', [None], [''], ['{}'],
    '{"bg_remove":{"preset":"disabled"},"ui":{}}'])
def test_generator_empty_legacy_and_valid_settings_still_reach_execution(generator_module, generator_type, raw):
    class ExecutionReached(Exception):
        pass

    node = getattr(generator_module, generator_type)()
    def begin(*args):
        raise ExecutionReached
    node._begin_progress = begin
    arguments = ([], object(), []) if generator_type == "VNCCS_EmotionsGenerator" else ([], [], object(), "prompt")
    with pytest.raises(ExecutionReached):
        node.process(*arguments, widget_data=raw, unique_id="valid-state-audit")


@pytest.mark.parametrize("catalog", ["default", "user"])
@pytest.mark.parametrize("damaged", ['{broken', '{"Custom": [{}]}'])
def test_damaged_emotion_catalog_blocks_tasks_and_recovers_without_losing_data(
    emotion_module, monkeypatch, tmp_path, catalog, damaged,
):
    defaults = tmp_path / "emotions.json"
    users = tmp_path / "user" / "emotions.json"
    users.parent.mkdir()
    entry = {"key": "Smile", "safe_name": "smile", "description": "soft smile"}
    defaults.write_text(json.dumps({"Happy": [entry]}))
    users.write_text(json.dumps({"Custom": []}))
    monkeypatch.setattr(emotion_module, "emotions_config_path", lambda: str(defaults))
    monkeypatch.setattr(emotion_module, "user_emotions_config_path", lambda: str(users))
    monkeypatch.setattr(emotion_module, "load_character_info", lambda *args: {"name": "Alice"})
    monkeypatch.setattr(emotion_module, "load_costume_info", lambda *args: {})
    monkeypatch.setattr(emotion_module, "build_emotion_pipe", lambda *args:
        pytest.fail("A damaged catalog must fail before model preparation or task creation"))
    node = emotion_module.EmotionGeneratorV2()
    original_map = node.SAFE_NAME_MAP
    path = defaults if catalog == "default" else users
    original = path.read_bytes()
    path.write_text(damaged)

    with pytest.raises(ValueError, match="emotion catalog"):
        node.generate_emotions_v2(character="Alice", costumes_data='["Coat"]', emotions_data='["smile"]')

    assert path.read_text() == damaged
    assert node.SAFE_NAME_MAP is original_map
    assert original_map["smile"]["description"] == "soft smile"
    path.write_bytes(original)
    assert node._setup_emotions_data()["smile"]["description"] == "soft smile"
    assert path.read_bytes() == original


def test_long_custom_emotion_names_save_with_valid_unique_generation_identifiers(
    emotion_module, monkeypatch, tmp_path,
):
    defaults = tmp_path / "emotions.json"
    defaults.write_text("{}")
    monkeypatch.setattr(emotion_module, "emotions_config_path", lambda: str(defaults))
    monkeypatch.setattr(emotion_module.web, "json_response", lambda data, status=200:
                        SimpleNamespace(data=data, status=status), raising=False)
    async def body():
        return {"name": "a" * 121, "description": "soft smile"}
    request = SimpleNamespace(json=body, headers={"X-VNCCS-CSRF": "1"})
    names = []
    for _ in range(2):
        response = asyncio.run(emotion_module.add_custom_emotion(request))
        assert response.status == 200
        name = response.data["emotion"]["safe_name"]
        assert emotion_module.ensure_safe_name(name, "emotion") == name
        assert len(name) == 120
        assert response.data["emotion"]["key"] == "a" * 121
        names.append(name)
    assert names[0] != names[1] and names[1].endswith("-2")
    assert set(emotion_module.EmotionGeneratorV2().SAFE_NAME_MAP) == set(names)
    assert defaults.read_text() == "{}"


def test_custom_emotion_write_and_catalog_read_leave_http_loop_responsive(
    emotion_module, monkeypatch, tmp_path,
):
    import threading
    defaults = tmp_path / "emotions.json"
    defaults.write_text("{}")
    monkeypatch.setattr(emotion_module, "emotions_config_path", lambda: str(defaults))
    monkeypatch.setattr(emotion_module.web, "json_response", lambda data, status=200:
                        SimpleNamespace(data=data, status=status), raising=False)
    started, release = threading.Event(), threading.Event()
    def encode(*args):
        started.set()
        assert release.wait(2)
    monkeypatch.setattr(emotion_module, "save_custom_emotion_image", encode)
    async def body():
        return {"name": "Happy"}
    async def run():
        save = asyncio.create_task(emotion_module.add_custom_emotion(
            SimpleNamespace(json=body, headers={"X-VNCCS-CSRF": "1"})))
        try:
            for _ in range(100):
                await asyncio.sleep(.002)
                if started.is_set():
                    break
            assert started.is_set() and not save.done()
            read = asyncio.create_task(emotion_module.get_emotions_config(None))
            await asyncio.sleep(.01)
            assert not read.done(), "Catalog read must wait for publication without blocking HTTP"
        finally:
            release.set()
        assert (await save).status == 200
        assert (await read).data["Custom"][0]["safe_name"] == "happy"
    asyncio.run(run())


def test_custom_emotion_pixel_limit_preserves_library_and_rejects_before_conversion(
    emotion_module, monkeypatch, tmp_path,
):
    import base64
    import io
    from PIL import Image
    defaults = tmp_path / "emotions.json"
    defaults.write_text("{}")
    users = tmp_path / "user" / "emotions.json"
    users.parent.mkdir()
    users.write_text('{"Custom": []}')
    original = users.read_bytes()
    monkeypatch.setattr(emotion_module, "emotions_config_path", lambda: str(defaults))
    monkeypatch.setattr(emotion_module, "MAX_CUSTOM_EMOTION_PIXELS", 256)
    monkeypatch.setattr(emotion_module.ImageOps, "exif_transpose", lambda *args:
                        pytest.fail("Oversized images must fail before decoding or converting pixels"))
    monkeypatch.setattr(emotion_module.web, "json_response", lambda data, status=200:
                        SimpleNamespace(data=data, status=status), raising=False)
    encoded = io.BytesIO()
    Image.new("RGB", (17, 16), "red").save(encoded, format="PNG")
    async def body():
        return {"name": "Large", "image_data": "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode()}
    response = asyncio.run(emotion_module.add_custom_emotion(
        SimpleNamespace(json=body, headers={"X-VNCCS-CSRF": "1"})))
    assert response.status == 400 and "pixels" in response.data["error"]
    assert users.read_bytes() == original
    assert not list(tmp_path.rglob("*.webp"))


def test_emotion_cache_pruning_preserves_active_inputs_unrelated_files_and_links(
    generator_module, monkeypatch, tmp_path,
):
    monkeypatch.setattr(generator_module.folder_paths, "get_output_directory", lambda: str(tmp_path))
    cache = tmp_path / "VNCCS" / "Characters" / "Alice" / "cache" / "poses" / "audit"
    stage = cache / "_stage_cache"
    stage.mkdir(parents=True)
    kept = ["emotion_0001__item_0001.pt", "emotion_0001_detailer_mask__item_0001.pt",
            "input_images_latest.pt", "inputs.json", "manual.pt"]
    removed = ["emotion_0001__item_0002.pt", "emotion_0001_detailer_mask__item_0002.pt",
               "emotion_0002__item_0001.pt", "emotion_0002_detailer_mask__item_0001.pt"]
    for name in kept + removed:
        (stage / name).write_bytes(name.encode())
    outside = tmp_path / "kept.pt"
    outside.write_bytes(b"outside cache")
    link = stage / "emotion_9999__item_0001.pt"
    link.symlink_to(outside)
    generator_module.VNCCS_EmotionsGenerator()._prune_emotion_cache(str(cache), {"emotion_0001": 1})
    assert all((stage / name).read_bytes() == name.encode() for name in kept)
    assert all(not (stage / name).exists() for name in removed)
    assert link.is_symlink() and outside.read_bytes() == b"outside cache"


def test_custom_emotion_added_during_catalog_read_is_available_on_the_next_request(
    emotion_module, monkeypatch, tmp_path,
):
    defaults = tmp_path / "emotions.json"
    original = '{"Mood": [{"key": "Smile", "safe_name": "smile", "description": "soft smile"}]}'
    defaults.write_text(original)
    monkeypatch.setattr(emotion_module, "emotions_config_path", lambda: str(defaults))
    monkeypatch.setattr(emotion_module.web, "json_response", lambda data, status=200:
        SimpleNamespace(data=data, status=status), raising=False)
    load = emotion_module.load_emotions_data
    add_after_read = True
    reads = 0

    async def body():
        return {"name": "Excited", "description": "wide excited eyes"}

    def read_catalog():
        nonlocal add_after_read, reads
        reads += 1
        data = load()
        if add_after_read:
            add_after_read = False
            response = asyncio.run(emotion_module.add_custom_emotion(
                SimpleNamespace(json=body, headers={"X-VNCCS-CSRF": "1"})))
            assert response.status == 200
        return data

    monkeypatch.setattr(emotion_module, "load_emotions_data", read_catalog)
    cls = emotion_module.EmotionGeneratorV2
    first = cls._setup_emotions_data()
    assert set(first) == {"smile"}
    next_map = cls._setup_emotions_data()
    assert set(next_map) == {"smile", "excited"}
    assert next_map["excited"]["description"] == "wide excited eyes"
    assert defaults.read_text() == original
    assert json.loads((tmp_path / "user" / "emotions.json").read_text())["Custom"][0]["safe_name"] == "excited"
    reads_after_reload = reads
    assert cls._setup_emotions_data() is next_map
    assert reads == reads_after_reload


@pytest.mark.parametrize("mode", ["Illustrious", "Anima", "QI2"])
@pytest.mark.parametrize("seed_mode", ["fixed", "randomize"])
@pytest.mark.parametrize("marker", [None, False, True, "true"])
def test_emotion_execution_seed_reaches_pipe_sampler_and_cache(
    emotion_module, generator_module, monkeypatch, tmp_path, mode, seed_mode, marker,
):
    monkeypatch.setattr(emotion_module, "load_generation_assets", lambda settings: (None, "model", "clip", "vae"))
    monkeypatch.setattr(emotion_module, "load_anima_assets", lambda settings: ("model", "clip", "vae"))
    monkeypatch.setattr(emotion_module, "normalize_gen_settings", lambda settings: dict(settings))
    generated = iter((900, 901))
    monkeypatch.setattr(emotion_module, "generate_seed", lambda _: next(generated))
    settings = {"seed": 721, "seed_mode": seed_mode, "steps": 20, "cfg": 3.0,
        "mode_settings": {mode.lower(): {"seed": 999, "seed_mode": "fixed"}}}
    if marker is not None:
        settings["_vnccs_seed_resolved"] = marker
    raw = json.dumps(settings)
    pipe, actual = emotion_module.build_emotion_pipe(mode, raw)
    expected = 721 if seed_mode == "fixed" or marker is True else 900
    assert actual == pipe.seed_int == expected
    assert json.loads(raw) == settings
    sampled = []

    class SampleReached(Exception):
        pass

    def call(name, **kwargs):
        if name in {"FaceDetailer", "KSampler"}:
            sampled.append(kwargs["seed"])
            raise SampleReached
        return (name,)

    monkeypatch.setattr(generator_module, "_call_comfy_node", call)
    node = generator_module.VNCCS_EmotionsGenerator()
    with pytest.raises(SampleReached):
        if mode == "QI2":
            node._qi2_sample(pipe.model, "positive", "negative", "latent", {
                "seed": actual, "steps": pipe.sample_steps, "cfg": pipe.cfg,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            })
        else:
            node._run_emotion_generation_one("image", None, pipe, "Smile", "", "", actual)
    assert sampled == [expected]
    root = tmp_path / "Characters"
    cache = root / "Alice" / "cache" / "poses" / "seed-audit"
    monkeypatch.setattr(generator_module, "base_output_dir", lambda: str(root))
    generator_module._save_run_inputs(str(cache), emotion_data=[{"character": "Alice", "seed": actual}])
    saved = generator_module._load_run_inputs(str(cache))
    assert saved["emotion_data"][0]["seed"] == expected
    assert "_vnccs_seed_resolved" not in json.dumps(saved)
    _, second = emotion_module.build_emotion_pipe(mode, raw)
    assert second == (901 if seed_mode == "randomize" and marker is not True else expected)


@pytest.mark.parametrize("mode", ["Illustrious", "Anima", "QI2"])
@pytest.mark.parametrize("settings,changed", [
    ({"seed": 721, "seed_mode": "fixed"}, False),
    ({"seed": 721, "seed_mode": "randomize"}, True),
    ({"seed": 721, "seed_mode": "randomize", "_vnccs_seed_resolved": True}, False),
    ({"seed": 721, "seed_mode": "randomize", "_vnccs_seed_resolved": "true"}, True),
    ({"mode_settings": {}}, False),
])
def test_emotion_headless_random_requests_are_not_reused_by_node_cache(emotion_module, mode, settings, changed):
    arguments = {"character": "Alice", "generation_model": mode, "generation_settings": json.dumps(settings)}
    first = emotion_module.EmotionGeneratorV2.IS_CHANGED(**arguments)
    second = emotion_module.EmotionGeneratorV2.IS_CHANGED(**arguments)
    assert (first != second) is changed


@pytest.mark.parametrize("mode", ["Illustrious", "Anima", "QI2"])
def test_legacy_nested_random_emotion_profiles_are_not_reused_by_node_cache(emotion_module, mode):
    arguments = {"character": "Alice", "generation_model": mode, "generation_settings": json.dumps({
        "mode_settings": {mode.lower(): {"seed_mode": "randomize", "seed": 721}},
    })}
    first = emotion_module.EmotionGeneratorV2.IS_CHANGED(**arguments)
    second = emotion_module.EmotionGeneratorV2.IS_CHANGED(**arguments)
    assert first != second


def test_seedvr_uses_installed_models_from_extra_paths(generator_module, monkeypatch, tmp_path):
    import folder_paths
    from nodes import vnccs_control_center as cc
    external = tmp_path / "external"
    for folder, name in (("diffusion_models", generator_module.SEEDVR_MODEL_FILES[0]),
                         ("vae", generator_module.SEEDVR_VAE_FILES[0])):
        path = external / folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"installed weights")
    monkeypatch.setattr(folder_paths, "models_dir", str(tmp_path / "default"))
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda category: [str(external / category)])
    monkeypatch.setattr(folder_paths, "get_full_path", lambda *args: None)
    monkeypatch.setattr(generator_module, "_find_model_on_disk", cc._find_model_on_disk)
    monkeypatch.setattr(generator_module, "_download_seedvr_file", lambda *args: pytest.fail("Installed VAE must not download"))
    catalog = generator_module._seedvr_catalog()
    assert catalog["models"][0]["status"] == catalog["vae"][0]["status"] == "installed"
    name = generator_module.SEEDVR_VAE_FILES[0]
    assert generator_module._ensure_seedvr_vae_model(name) == str(external / "vae" / name)


def test_emotion_shape_lookup_decodes_each_source_only_once(generator_module, monkeypatch, tmp_path):
    from PIL import Image, PngImagePlugin
    path = tmp_path / "pose.png"
    Image.new("RGBA", (10, 20)).save(path)
    monkeypatch.setattr(generator_module, "_safe_existing_character_image_path", lambda *args: str(path))
    node = generator_module.VNCCS_EmotionsGenerator()
    decoded = []
    load = PngImagePlugin.PngImageFile.load
    def observed_load(image, *args, **kwargs):
        if image.tile:
            decoded.append(image.filename)
        return load(image, *args, **kwargs)
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", observed_load)
    monkeypatch.setattr(node, "_emotion_batch_plan", lambda *args: (_ for _ in ()).throw(RuntimeError("shape lookup complete")))
    tasks = [{"character": "Alice", "source_path": str(path), "emotion_prompt": str(index)} for index in range(20)]
    with pytest.raises(RuntimeError, match="shape lookup complete"):
        node.process(None, SimpleNamespace(model_kind="qi2"), tasks,
                     widget_data='{"character_name":"Alice"}', unique_id="shape-audit")
    assert decoded == [str(path)]


@pytest.mark.parametrize("mode", ["Illustrious", "Anima", "QI2"])
def test_emotion_workflow_cache_tracks_selected_asset_files(emotion_module, monkeypatch, tmp_path, mode):
    weights = tmp_path / "selected-model"
    weights.write_bytes(b"first weights")
    observed = []
    def signatures(settings):
        observed.append(settings)
        return {"file": [weights.stat().st_size, weights.stat().st_mtime_ns]}
    monkeypatch.setattr(emotion_module, "_generation_file_signatures", signatures)
    arguments = {"character": "Alice", "generation_model": mode,
                 "generation_settings": '{"seed_mode":"fixed"}'}
    before = emotion_module.EmotionGeneratorV2.IS_CHANGED(**arguments)
    assert emotion_module.EmotionGeneratorV2.IS_CHANGED(**arguments) == before
    weights.write_bytes(b"replacement weights")
    assert emotion_module.EmotionGeneratorV2.IS_CHANGED(**arguments) != before
    assert all(settings["generation_mode"] == mode.lower() for settings in observed)


def test_emotion_model_preparation_waits_for_preview_inference(emotion_module, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from nodes.runtime_cleanup import inference_lock
    entered = threading.Event()
    def model_boundary(*args):
        entered.set()
        raise RuntimeError("model boundary reached")
    monkeypatch.setattr(emotion_module, "load_generation_assets", model_boundary)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with inference_lock:
            future = executor.submit(emotion_module.build_emotion_pipe, "Anima", "{}")
            assert not entered.wait(0.1)
        with pytest.raises(RuntimeError, match="model boundary reached"):
            future.result(timeout=2)
    assert entered.is_set()


@pytest.mark.parametrize("valid", [True, False])
def test_seedvr_download_publishes_valid_weights_to_the_configured_folder(
    generator_module, monkeypatch, tmp_path, valid,
):
    import folder_paths
    import huggingface_hub
    from nodes import vnccs_control_center as cc
    installed = tmp_path / "external" / "diffusion_models"
    monkeypatch.setattr(folder_paths, "get_folder_paths", lambda category: [str(installed)])
    monkeypatch.setattr(generator_module, "_find_model_on_disk", cc._find_model_on_disk)
    name = generator_module.SEEDVR_MODEL_FILES[0]
    target = installed / name
    target.parent.mkdir(parents=True)
    cache = tmp_path / "download.safetensors"
    header = json.dumps({"weight": {"dtype": "U8", "shape": [1024], "data_offsets": [0, 1024]}}).encode()
    contents = len(header).to_bytes(8, "little") + header + b"\0" * 1024
    cache.write_bytes(contents if valid else b"invalid downloaded model")
    calls = []
    def download(**options):
        assert options["token"] is False
        assert options["revision"] == generator_module.SEEDVR_HF_REVISION
        assert options["filename"] == f"diffusion_models/{name}"
        calls.append(options)
        return str(cache)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    if valid:
        assert generator_module._download_seedvr_file("models", name) == str(target)
        assert target.read_bytes() == contents
        before = target.stat().st_mtime_ns
        assert generator_module._download_seedvr_file("models", name) == str(target)
        assert target.read_bytes() == contents and target.stat().st_mtime_ns == before
        assert len(calls) == 1
        assert list(installed.iterdir()) == [target]
    else:
        with pytest.raises(ValueError):
            generator_module._download_seedvr_file("models", name)
        assert list(installed.iterdir()) == []


def test_seedvr_missing_vae_keeps_model_missing_and_downloads_the_complete_bundle(generator_module, monkeypatch):
    name = generator_module.SEEDVR_MODEL_FILES[0]
    monkeypatch.setattr(generator_module, "_find_model_on_disk", lambda path: (path, "diffusion_models" in path))
    assert generator_module._seedvr_catalog()["models"][0]["status"] == "missing"
    calls = []
    monkeypatch.setattr(generator_module, "_download_seedvr_file", lambda category, name: calls.append((category, name)))
    generator_module._seedvr_download_worker("models", name)
    assert calls == [("models", name), ("vae", generator_module.SEEDVR_VAE_FILES[0])]
    assert generator_module._SEEDVR_DOWNLOAD_STATUS[f"models:{name}"]["status"] == "installed"


@pytest.mark.parametrize("route", ["vnccs_character_generator_seedvr_download", "vnccs_character_generator_regenerate"])
@pytest.mark.parametrize("body", [None, [], 12, "model"])
def test_generator_routes_reject_non_object_bodies(monkeypatch, generator_module, route, body):
    async def json_body():
        return body

    monkeypatch.setattr(generator_module.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    request = SimpleNamespace(json=json_body, headers={"X-VNCCS-CSRF": "1"})
    response = asyncio.run(getattr(generator_module, route)(request))
    assert response.status == 400


@pytest.mark.parametrize("settings", [None, [], 12, '{broken', '{}'])
def test_regenerate_rejects_explicit_non_object_settings_before_cache_or_model_work(
    generator_module, monkeypatch, tmp_path, settings,
):
    root = tmp_path / "Alice"
    cache = root / "cache" / "poses" / "audit"
    cache.mkdir(parents=True)
    metadata = cache / "inputs.json"
    metadata.write_text('{"widget_payload": {}}')
    published = root / "sprite_pose_0001.png"
    published.write_bytes(b"published authored pose")
    before = {path: path.read_bytes() for path in (metadata, published)}
    monkeypatch.setattr(generator_module.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(generator_module, "_get_generator_context", lambda *args: {
        "pipe": object(), "cache_dir": str(cache), "generator_type": "VNCCS_CharacterGenerator",
    })
    def load(*args, **kwargs):
        raise ValueError("Malformed settings must fail before accessing the healthy legacy cache")
    monkeypatch.setattr(generator_module, "_load_run_inputs", load)
    response = generator_module._regenerate_response({
        "unique_id": "audit", "stage": "bg_remove", "widget_data": settings,
    })
    assert response.status == 400
    assert "settings must be an object" in response.data["error"]
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.parametrize("settings", [{}, "omitted"])
def test_regenerate_legacy_empty_or_omitted_settings_still_reach_the_live_context(
    generator_module, monkeypatch, settings,
):
    monkeypatch.setattr(generator_module.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    calls = []
    monkeypatch.setattr(generator_module, "_get_generator_context", lambda *args: calls.append(args))
    body = {"unique_id": "audit", "stage": "bg_remove"}
    if settings != "omitted":
        body["widget_data"] = settings
    response = generator_module._regenerate_response(body)
    assert response.status == 409
    assert calls == [("audit", None)]


@pytest.mark.parametrize("index", [-1, "-1", "missing", [], {}, True, 0.5, float("inf")])
def test_invalid_single_image_index_never_falls_back_to_full_regeneration(generator_module, index):
    with pytest.raises(ValueError, match="non-negative integer"):
        generator_module.VNCCS_CharacterGenerator()._regenerate_index({"regenerate_index": index})


@pytest.mark.parametrize("index, expected", [(None, None), (0, 0), ("2", 2)])
def test_regeneration_index_accepts_full_run_and_legacy_integer_indices(generator_module, index, expected):
    assert generator_module.VNCCS_CharacterGenerator()._regenerate_index({"regenerate_index": index}) == expected


def test_unknown_stage_never_falls_back_to_full_regeneration(generator_module):
    node = generator_module.VNCCS_CharacterGenerator()
    order = ("pose_generation", "upscaler", "bg_remove")
    with pytest.raises(ValueError, match="stage is unavailable"):
        node._should_regenerate_stage(order, "missing-stage", "pose_generation")
    assert node._should_regenerate_stage(order, "", "pose_generation")
    assert not node._should_regenerate_stage(order, "upscaler", "pose_generation")


@pytest.mark.parametrize("generator_type", ["VNCCS_CharacterGenerator", "VNCCS_CharacterCloneGenerator", "VNCCS_ClothesGenerator", "VNCCS_EmotionsGenerator"])
def test_changed_character_cannot_regenerate_the_previous_characters_outputs(generator_module, monkeypatch, generator_type):
    monkeypatch.setattr(generator_module.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(generator_module, "_get_generator_context", lambda *args: {"pipe": object(), "cache_dir": "cache", "generator_type": generator_type})
    monkeypatch.setattr(generator_module, "_load_run_inputs", lambda *args, **kwargs: {"widget_payload": {"character_name": "Alice"}, "emotion_data": [{"character": "Alice"}]})
    monkeypatch.setattr(generator_module, "_temporarily_shift_pipe_seed", lambda *args: pytest.fail("Different character must fail before model execution"))
    result = generator_module._regenerate_response({"unique_id": "audit", "stage": "bg_remove", "widget_data": {"character_name": "Bob"}})
    assert result.status == 409
    assert "tasks changed" in result.data["error"]


def test_changed_emotion_pairs_cannot_relabel_and_overwrite_the_previous_tasks(generator_module, monkeypatch):
    monkeypatch.setattr(generator_module.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(generator_module, "_get_generator_context", lambda *args: {"pipe": object(), "cache_dir": "cache", "generator_type": "VNCCS_EmotionsGenerator"})
    before = {"character_name": "Alice", "emotion_pairs": [{"costume": "Coat", "emotion": "smile"}]}
    monkeypatch.setattr(generator_module, "_load_run_inputs", lambda *args, **kwargs: {"widget_payload": before, "emotion_data": [{"costume": "Coat", "emotion": "smile"}]})
    monkeypatch.setattr(generator_module, "_temporarily_shift_pipe_seed", lambda *args: pytest.fail("Changed tasks must fail before model execution"))
    after = {"character_name": "Alice", "emotion_pairs": [{"costume": "Original", "emotion": "angry"}]}
    result = generator_module._regenerate_response({"unique_id": "audit", "stage": "emotion_0001_bg_remove", "widget_data": after})
    assert result.status == 409
    assert after == {"character_name": "Alice", "emotion_pairs": [{"costume": "Original", "emotion": "angry"}]}


@pytest.mark.parametrize("cached_costume", [{"costume": "Coat"}, {"costume_name": "Coat"}, {}])
def test_changed_costume_cannot_regenerate_the_previous_outfit(generator_module, monkeypatch, cached_costume):
    monkeypatch.setattr(generator_module.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(generator_module, "_get_generator_context", lambda *args: {"pipe": object(), "cache_dir": "cache", "generator_type": "VNCCS_ClothesGenerator"})
    monkeypatch.setattr(generator_module, "_load_run_inputs", lambda *args, **kwargs: {
        "widget_payload": {"character_name": "Alice", **cached_costume},
        "sheets_path": "/output/VNCCS/Characters/Alice/Sheets/Coat/neutral",
    })
    monkeypatch.setattr(generator_module, "_temporarily_shift_pipe_seed", lambda *args: pytest.fail("Changed costume must fail before model execution"))
    after = {"character_name": "Alice", "costume": "Dress"}
    result = generator_module._regenerate_response({"unique_id": "audit", "stage": "bg_remove", "widget_data": after})
    assert result.status == 409
    assert "costume" in result.data["error"]
    assert after == {"character_name": "Alice", "costume": "Dress"}


def test_custom_neutral_emotion_never_becomes_a_source_pose_directory(emotion_module, monkeypatch, tmp_path):
    name = emotion_module.make_unique_emotion_safe_name("Neutral", [])
    assert name == "neutral-2"
    assert emotion_module.make_unique_emotion_safe_name("Neutral", [name]) == "neutral-3"
    root = tmp_path / "Alice"
    source = root / "Sprites" / "Coat" / "Neutral" / "sprite_pose_0001.png"
    generated = root / "Sprites" / "Coat" / name / "sprite_neutral_0001.png"
    for path in (source, generated):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"sprite inventory only")
    monkeypatch.setattr(emotion_module, "character_dir", lambda *args: str(root))
    assert emotion_module.list_costume_sprite_paths("Alice", "Coat") == [str(source)]


@pytest.mark.parametrize("archive_name", ["V1", "V2", "V21", "v999"])
def test_emotion_sources_never_include_version_archives(emotion_module, monkeypatch, tmp_path, archive_name):
    root = tmp_path / "Alice"
    neutral = root / "Sprites" / "Coat" / "Neutral"
    current = neutral / "sprite_pose_0001.png"
    archived = neutral / archive_name / "sprite_pose_0001.png"
    for path in (current, archived):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.parent.name.encode())
    monkeypatch.setattr(emotion_module, "character_dir", lambda *args: str(root))

    assert emotion_module.list_costume_sprite_paths("Alice", "Coat") == [str(current)]
    current.unlink()
    assert emotion_module.list_costume_sprite_paths("Alice", "Coat") == []


@pytest.mark.parametrize(("character", "costume"), [("V1", "Coat"), ("Alice", "V1")])
def test_emotion_sources_allow_version_like_character_and_costume_names(
    emotion_module, monkeypatch, tmp_path, character, costume,
):
    root = tmp_path / character
    current = root / "Sprites" / costume / "Neutral" / "sprite_pose_0001.png"
    archived = current.parent / "V2" / current.name
    for path in (current, archived):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.parent.name.encode())
    monkeypatch.setattr(emotion_module, "character_dir", lambda *args: str(root))

    assert emotion_module.list_costume_sprite_paths(character, costume) == [str(current)]


@pytest.mark.parametrize("emotion", ["neutral", "Neutral"])
def test_legacy_reserved_emotion_fails_before_loading_models(emotion_module, monkeypatch, emotion):
    monkeypatch.setattr(emotion_module, "load_character_info", lambda *args: {"name": "Alice"})
    node = emotion_module.EmotionGeneratorV2.__new__(emotion_module.EmotionGeneratorV2)
    with pytest.raises(ValueError, match="reserved for source poses"):
        node.generate_emotions_v2(character="Alice", emotions_data='["' + emotion + '"]')


@pytest.mark.parametrize("link_kind", ["file", "neutral_directory", "costume_directory"])
def test_emotion_inventory_skips_unsafe_entries_and_preserves_valid_sources(
    emotion_module, monkeypatch, tmp_path, link_kind,
):
    import utils
    root = tmp_path / "Characters"
    monkeypatch.setattr(utils, "base_output_dir", lambda: str(root))
    outside = tmp_path / "private"
    outside.mkdir()
    (outside / "private.png").write_bytes(b"private pixels must not be read")
    sprites = root / "Alice" / "Sprites"
    coat = sprites / "Coat"
    neutral = coat / "Neutral"
    if link_kind == "costume_directory":
        sprites.mkdir(parents=True)
        coat.symlink_to(outside, target_is_directory=True)
        valid = sprites / "Original" / "Neutral" / "pose.png"
    elif link_kind == "neutral_directory":
        coat.mkdir(parents=True)
        neutral.symlink_to(outside, target_is_directory=True)
        valid = coat / "pose.png"
    else:
        neutral.mkdir(parents=True)
        (neutral / "private.png").symlink_to(outside / "private.png")
        valid = neutral / "pose.png"
    valid.parent.mkdir(parents=True, exist_ok=True)
    valid.write_bytes(b"valid inventory pixels")
    costume = "Original" if link_kind == "costume_directory" else "Coat"
    assert emotion_module.list_costume_sprite_paths("Alice", costume) == [str(valid)]
    monkeypatch.setattr(emotion_module.web, "json_response", lambda data, status=200:
        SimpleNamespace(data=data, status=status), raising=False)
    response = asyncio.run(emotion_module.get_character_costumes(SimpleNamespace(
        rel_url=SimpleNamespace(query={"character": "Alice"}))))
    assert response.status == 200
    assert response.data == [costume]
    assert (outside / "private.png").read_bytes() == b"private pixels must not be read"


@pytest.mark.parametrize("character, costume", [("Sheets", "Sheets"), ("Sprites", "Sprites"), ("Alice", "Sheets")])
@pytest.mark.parametrize("windows_style", [False, True])
def test_storage_folder_names_remain_valid_character_and_costume_names(generator_module, monkeypatch, tmp_path, character, costume, windows_style):
    base = tmp_path / "Sheets" / "Sprites" / "VNCCS" / "Characters"
    root = base / character
    sheets = root / "Sheets" / costume / "neutral" / "sheet_neutral"
    source = root / "Sprites" / costume / "Neutral" / "sprite_pose_0001.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"published source sprite")
    monkeypatch.setattr(generator_module, "base_output_dir", lambda: str(base))
    monkeypatch.setattr(generator_module, "character_dir", lambda name: str(base / name))
    portable = lambda path: str(path).replace(os.sep, "\\") if windows_style else str(path)

    assert generator_module._character_root_from_sheets_path(portable(sheets), character) == str(root)
    assert generator_module._character_root_from_sheets_path(portable(sheets)) == str(root)
    assert generator_module._costume_name_from_sheets_path(portable(sheets)) == costume
    assert generator_module._safe_existing_character_image_path(portable(source), character) == str(source)
    deprecated = root / "Sheets" / costume / "neutral" / "old.png"
    deprecated.parent.mkdir(parents=True)
    deprecated.write_bytes(b"deprecated sheet")
    assert generator_module._safe_existing_character_image_path(portable(deprecated), character) == ""

    prefix = root / "Sprites" / costume / "smile" / "sprite_smile_"
    node = generator_module.VNCCS_EmotionsGenerator()
    assert node._face_prefix_from_sprite_prefix(portable(prefix), character) == str(root / "Faces" / costume / "smile" / "face_smile_")
    assert node._face_prefix_from_sprite_prefix(str(base / "Other" / "Sprites" / "smile" / "sprite_"), character) == ""
    assert source.read_bytes() == b"published source sprite"


@pytest.mark.parametrize("linked_folder", ["Sprites", "Faces", "cache"])
def test_generator_paths_reject_links_outside_the_character_storage(generator_module, monkeypatch, tmp_path, linked_folder):
    base = tmp_path / "VNCCS" / "Characters"
    root = base / "Alice"
    outside = tmp_path / "outside"
    root.mkdir(parents=True)
    outside.mkdir()
    original = outside / "sprite_pose_0001.png"
    original.write_bytes(b"outside file must remain private and unchanged")
    (root / linked_folder).symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(generator_module, "base_output_dir", lambda: str(base))
    monkeypatch.setattr(generator_module, "character_dir", lambda name: str(base / name))

    if linked_folder == "cache":
        assert generator_module._safe_cache_dir(str(root / "cache")) == ""
        assert generator_module._cache_tensor_path(str(root / "cache"), "input") == ""
    else:
        prefix = root / linked_folder / "sprite_"
        assert generator_module._safe_emotion_output_prefix(str(prefix), "Alice", linked_folder) == ""
        assert generator_module._safe_existing_character_image_path(str(root / linked_folder / original.name), "Alice") == ""
        node = generator_module.VNCCS_EmotionsGenerator()
        if linked_folder == "Sprites":
            assert node._face_prefix_from_sprite_prefix(str(prefix), "Alice") == ""
    assert original.read_bytes() == b"outside file must remain private and unchanged"
    assert list(outside.iterdir()) == [original]


@pytest.mark.parametrize("operation", ["tensor_path", "metadata_read", "metadata_write"])
def test_nested_stage_cache_link_cannot_escape_character_storage(generator_module, monkeypatch, tmp_path, operation):
    base = tmp_path / "VNCCS" / "Characters"
    cache = base / "Alice" / "cache" / "poses" / "audit"
    outside = tmp_path / "outside"
    cache.mkdir(parents=True)
    outside.mkdir()
    metadata = outside / "inputs.json"
    original = '{"items":{"prompt":{"type":"json","value":"private outside content"}}}'
    metadata.write_text(original)
    (cache / "_stage_cache").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(generator_module, "base_output_dir", lambda: str(base))

    if operation == "tensor_path":
        assert generator_module._cache_tensor_path(str(cache), "pose_generation") == ""
    elif operation == "metadata_read":
        assert generator_module._load_run_inputs(str(cache)) == {}
    else:
        with pytest.raises(ValueError, match="outside allowed directory"):
            generator_module._save_run_inputs(str(cache), widget_payload={"character_name": "Alice"})
    assert metadata.read_text() == original
    assert list(outside.iterdir()) == [metadata]


def test_preview_file_link_never_truncates_outside_data(generator_module, monkeypatch, tmp_path):
    base = tmp_path / "VNCCS" / "Characters"
    cache = base / "Alice" / "cache" / "poses" / "audit"
    cache.mkdir(parents=True)
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"outside data must remain unchanged")
    (cache / "pose_generation_01.png").symlink_to(outside)
    monkeypatch.setattr(generator_module, "base_output_dir", lambda: str(base))
    monkeypatch.setattr(generator_module, "_tensor_to_png_data_url", lambda *args, **kwargs: ["safe in-memory preview"])

    class Preview:
        ndim = 4
        def detach(self):
            return self
        cpu = detach
        def clamp(self, *args):
            return self
        def __getitem__(self, index):
            return [SimpleNamespace(numpy=lambda: generator_module.np.ones((2, 2, 3)))]

    image = Preview()
    monkeypatch.setattr(generator_module.torch, "is_tensor", lambda value: value is image)
    assert generator_module._tensor_to_preview_urls(image, "audit", "pose_generation", str(cache)) == ["safe in-memory preview"]
    assert outside.read_bytes() == b"outside data must remain unchanged"
