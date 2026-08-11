from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1] / "web" / "vnccs_clothes_designer.js"
).read_text(encoding="utf-8")


def test_clothes_core_card_filters_by_connected_model_kind():
    assert "const getConnectedModelKind = () =>" in SOURCE
    assert "const loraMatchesConnectedKind = (entry) =>" in SOURCE
    assert "entryKind === selectedKind" in SOURCE
    assert "const findConnectedClothesCoreLora = (preferredPath = \"\") =>" in SOURCE


def test_clothes_core_sync_resolves_from_its_connected_control_center():
    assert "setClothesCoreLora();" in SOURCE
    assert "const clothesCore = options.find(isClothesCoreLora);" not in SOURCE
    assert 'if (lower.startsWith("models/loras/"))' in SOURCE
