"""Style thumbnails reuse Creator inference, with square latents and direct WebP output."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from conftest import _preload_node

torch = pytest.importorskip("torch")
creator = _preload_node("character_creator_v2")
library = _preload_node("character_styles")


@pytest.mark.parametrize("mode", ["illustrious", "anima", "qi2"])
@pytest.mark.parametrize("scale", [1024, 1536])
@pytest.mark.parametrize("channels", [3, 4])
def test_real_preview_path_samples_a_square_at_selected_scale_and_writes_only_webp(monkeypatch, tmp_path, mode, scale, channels):
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path / "previews"))
    monkeypatch.setattr(creator, "save_style_preview", library.save_style_preview)
    monkeypatch.setattr(creator, "acquire_preview_assets", lambda settings: ("model", "clip", "vae"))
    monkeypatch.setattr(creator, "get_lora_full_path", lambda name: "/test/lora.safetensors")
    monkeypatch.setitem(creator.PREVIEW_CACHE, "loras", {})
    monkeypatch.setattr(creator.comfy.utils, "load_torch_file", lambda *a, **kw: {"weight": 1}, raising=False)
    adapters = []
    monkeypatch.setattr(creator.comfy.sd, "load_lora_for_models", lambda m, c, weights, strength, clip_strength: adapters.append((strength, clip_strength)) or (m, c), raising=False)
    monkeypatch.setattr(creator, "apply_creator_overhaul", lambda model, clip, *a: (model, clip))
    monkeypatch.setattr(creator, "prepare_qi2_model", lambda model, settings: (model, False))
    encoded = []
    def encode(clip, vae, positive, negative, settings, **kwargs):
        encoded.append((positive, settings, kwargs))
        return "positive", "negative", positive
    monkeypatch.setattr(creator, "encode_generation_conditioning", encode)
    monkeypatch.setattr(creator, "validate_anima_conditioning", lambda *a: None)
    dimensions = []
    monkeypatch.setattr(creator, "create_generation_latent", lambda model, width, height, settings: dimensions.append((width, height)) or "latent")
    sampled = []
    monkeypatch.setattr(creator, "sample_generation_latent", lambda **kwargs: sampled.append(kwargs) or "sampled")
    def decode(*args):
        width, height = dimensions[-1]
        return torch.zeros((1, height, width, channels))
    monkeypatch.setattr(creator, "decode_generation_samples", decode)
    monkeypatch.setattr(creator.web, "json_response", lambda data: SimpleNamespace(data=data, status=200), raising=False)
    def unexpected(*args):
        pytest.fail("Style previews must not write the character PNG cache")
    monkeypatch.setattr(creator, "character_dir", unexpected)
    info = {"style": "anime_style", "hair": "black hair", "eyes": "blue eyes", "sex": "female", "age": 30, "framing": "full_body"}
    settings = {"generation_mode": mode, "target_size": scale, "seed": 123, "steps": 17, "cfg": 3.5,
                "sampler": "euler", "scheduler": "normal", "lora_stack": [{"name": "Mine", "strength": .4}], "turbo_enabled": False}
    response = creator._generate_preview_response({"character_info": info, "gen_settings": settings}, style_preview=info["style"])
    assert response.status == 200
    assert dimensions == [library.square_style_resolution(scale)]
    assert (sampled[0]["seed"], sampled[0]["steps"], sampled[0]["cfg"]) == (123, 17, 3.5)
    assert sampled[0]["gen_settings"]["generation_mode"] == mode
    assert (.4, .4) in adapters
    assert encoded[0][2]["style_reference"] == creator.CHARACTER_STYLE_PROMPTS[info["style"]]
    assert "black hair" in encoded[0][0] and "blue eyes" in encoded[0][0]
    assert "head and shoulders" in encoded[0][0]
    assert "cowboy_shot" not in encoded[0][0] and "standing, full body" not in encoded[0][0]
    preview_info = encoded[0][2]["character_info"]
    assert preview_info["image_type"] == "Portrait" and "framing" not in preview_info
    if mode == "qi2":
        assert "Portrait:" in creator._qi2_character_fields(preview_info)["framing"]
    assert info["framing"] == "full_body" and "image_type" not in info
    normal_prompt = creator.CharacterCreatorV2.construct_prompt(info, mode)[0]
    assert ("head to toe" if mode == "qi2" else "standing, full body") in normal_prompt
    path = Path(library.style_preview_path(info["style"]))
    with Image.open(path) as image:
        assert image.format == "WEBP" and image.size == (1024, 1024)
        assert image.mode == "RGB"
        if channels == 4:
            assert all(abs(a - b) <= 3 for a, b in zip(image.getpixel((0, 0)), (41, 32, 52)))
        else:
            assert image.getpixel((0, 0)) == (0, 0, 0)
    assert list(tmp_path.rglob("*.png")) == []
    assert response.data["image"].startswith("/vnccs/character_styles/preview?")
    assert response.data["saved"] is True and response.data["path"] == str(path)


def test_generation_route_validates_style_and_keeps_tags_settings_and_scoped_events(monkeypatch):
    monkeypatch.setattr(creator.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(creator, "load_user_styles", lambda: [])
    events, calls = [], []
    monkeypatch.setattr(creator.server.PromptServer.instance, "send_sync", lambda name, data: events.append((name, data)), raising=False)
    monkeypatch.setattr(creator, "_generate_preview_response", lambda data, style_preview: calls.append((data, style_preview)) or SimpleNamespace(status=200))
    payload = {"style_id": "anime_style", "node_id": "42", "request_id": "batch-1", "character_info": {"hair": "black hair", "eyes": "blue eyes", "style": "legacy"}, "gen_settings": {"target_size": 1536, "seed": 123, "seed_mode": "randomize"}}
    async def body():
        return payload
    request = SimpleNamespace(json=body, headers={"Host": "localhost", "X-VNCCS-CSRF": "1"})
    assert asyncio.run(creator.generate_style_preview(request)).status == 200
    assert calls[0][0]["character_info"]["hair"] == "black hair"
    assert calls[0][0]["character_info"]["style"] == "anime_style"
    assert calls[0][0]["gen_settings"]["target_size"] == 1536
    assert calls[0][0]["gen_settings"]["seed"] == 123
    assert creator.resolve_generation_seed(calls[0][0]["gen_settings"]) == 123
    assert payload["gen_settings"]["seed_mode"] == "randomize"
    assert payload["character_info"]["style"] == "legacy"
    assert [data["status"] for name, data in events] == ["queued", "running", "done"]
    assert all(name == "vnccs.style_preview.stage" and data["node_id"] == "42" and data["request_id"] == "batch-1" for name, data in events)
    for style_id in ["../outside", "not_a_style", "custom"]:
        payload["style_id"] = style_id
        assert asyncio.run(creator.generate_style_preview(request)).status == 400
    assert len(calls) == 1


def test_preview_route_serves_webp_and_catalog_recovers_generated_images(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path))
    monkeypatch.setattr(creator, "style_preview_path", library.style_preview_path)
    monkeypatch.setattr(creator, "style_preview_url", library.style_preview_url)
    monkeypatch.setattr(creator, "load_user_styles", lambda: [])
    monkeypatch.setattr(creator.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(creator.web, "Response", lambda status: SimpleNamespace(status=status), raising=False)
    monkeypatch.setattr(creator.web, "FileResponse", lambda path, headers: SimpleNamespace(path=path, headers=headers, status=200), raising=False)
    request = SimpleNamespace(rel_url=SimpleNamespace(query={"style": "anime_style"}))
    assert asyncio.run(creator.get_style_preview(request)).status == 404
    library.save_style_preview("anime_style", Image.new("RGB", (64, 64)))
    response = asyncio.run(creator.get_style_preview(request))
    assert response.headers["Content-Type"] == "image/webp"
    assert Path(response.path).is_file()
    catalog = asyncio.run(creator.get_character_styles(None)).data
    style = next(s for group in catalog["groups"] for s in group["styles"] if s["id"] == "anime_style")
    assert style["image"] == library.style_preview_url("anime_style")
    assert creator.CHARACTER_STYLE_CATALOG["groups"] != catalog["groups"]
    request.rel_url.query["style"] = "../escape"
    assert asyncio.run(creator.get_style_preview(request)).status == 400


@pytest.mark.parametrize("style_id", ["custom", "user_" + "a" * 32])
def test_custom_preview_route_forces_seed_zero_and_preserves_main_settings(monkeypatch, style_id):
    monkeypatch.setattr(creator.web, "json_response", lambda data, status=200: SimpleNamespace(data=data, status=status), raising=False)
    monkeypatch.setattr(creator, "load_user_styles", lambda: [{"id": "user_" + "a" * 32, "prompt": "User graphite"}])
    monkeypatch.setattr(creator.server.PromptServer.instance, "send_sync", lambda *args: None, raising=False)
    calls = []
    monkeypatch.setattr(creator, "_generate_preview_response", lambda data, style_preview: calls.append((data, style_preview)) or SimpleNamespace(status=200))
    payload = {"style_id": style_id, "character_info": {"style": "photorealism", "custom_style": "User ink", "framing": "full_body"},
               "gen_settings": {"seed": 567, "seed_mode": "randomize", "target_size": 2048,
                                "mode_settings": {"qi2": {"seed": 999}}, "generation_mode": "qi2"}}
    original = json.dumps(payload, sort_keys=True)
    async def body():
        return payload
    request = SimpleNamespace(json=body, headers={"Host": "localhost", "X-VNCCS-CSRF": "1"})
    assert asyncio.run(creator.generate_style_preview(request)).status == 200
    data, selected = calls[0]
    assert selected == style_id
    assert data["character_info"]["style"] == style_id
    if style_id.startswith("user_"):
        assert data["character_info"]["style_prompt"] == "User graphite"
    assert data["gen_settings"]["seed"] == creator.resolve_generation_seed(data["gen_settings"]) == 0
    assert data["gen_settings"]["seed_mode"] == "fixed"
    assert data["gen_settings"]["mode_settings"] == {}
    assert data["gen_settings"]["target_size"] == 2048
    assert json.dumps(payload, sort_keys=True) == original


def test_old_preview_ids_serve_the_canonical_packaged_file(monkeypatch, tmp_path):
    monkeypatch.setattr(library, "STYLE_PREVIEWS_DIR", str(tmp_path))
    monkeypatch.setattr(creator, "style_preview_path", library.style_preview_path)
    monkeypatch.setattr(creator.web, "FileResponse", lambda path, headers: SimpleNamespace(path=path, headers=headers, status=200), raising=False)
    library.save_style_preview("ghibli_miyazaki", Image.new("RGBA", (32, 32), (120, 80, 160, 100)))
    request = SimpleNamespace(rel_url=SimpleNamespace(query={"style": "clio_ghibli_style"}))
    response = asyncio.run(creator.get_style_preview(request))
    assert response.status == 200
    assert Path(response.path) == tmp_path / "ghibli_miyazaki.webp"
    assert not (tmp_path / "clio_ghibli_style.webp").exists()
