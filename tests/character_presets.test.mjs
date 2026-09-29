import assert from "node:assert/strict";
import fs from "node:fs";
import test from "node:test";
import { presetGroups, presetSelection } from "../web/character_presets.mjs";

const catalog = JSON.parse(fs.readFileSync(new URL("../character_template/character_presets_v2.json", import.meta.url)));

test("every catalog category belongs to exactly one form field", () => {
    const fields = ["race", "skin_color", "body", "face", "hair", "eyes", "additional_details"];
    const visible = fields.flatMap(field => presetGroups(catalog, field).flatMap(group => group.items));
    assert.equal(visible.length, Object.values(catalog.tags).flat().length);
    assert.equal(new Set(visible).size, visible.length);
    assert.equal(presetGroups(catalog, "race").length, new Set(catalog.tags.races.map(item => item.group)).size);
    assert.deepEqual(presetGroups(catalog, "unknown"), []);
});

test("legacy aliases preselect a single species without losing custom text", () => {
    const groups = presetGroups(catalog, "race");
    const selection = presetSelection("cat_girl, CAT BOY, My Custom Species, no tail", groups);
    const cat = catalog.tags.races.find(item => item.tag === "catfolk");
    assert.equal(selection.has(cat), true);
    assert.equal(selection.value(), "cat_girl, My Custom Species, no tail");
    selection.toggle(cat);
    assert.equal(selection.value(), "My Custom Species, no tail");
    selection.toggle(cat);
    assert.equal(selection.value(), "My Custom Species, no tail, catfolk");
    const restored = presetSelection(JSON.parse(JSON.stringify(selection.value())), groups);
    assert.equal(restored.has(cat), true);
    assert.equal(restored.value(), selection.value());
});

test("breast choices keep the existing tag insertion and legacy spelling", () => {
    const groups = presetGroups(catalog, "body");
    for (const item of catalog.tags.breast_size) {
        const selection = presetSelection("", groups);
        selection.toggle(item);
        assert.equal(selection.value(), item.tag.replaceAll("_", " "));
        const restored = presetSelection(item.tag, groups);
        assert.equal(restored.has(item), true);
        assert.equal(restored.value(), item.tag);
    }
});

test("hybrid races can coexist and descriptions are not serialized into the field", () => {
    const selection = presetSelection("", presetGroups(catalog, "race"));
    for (const key of ["elf", "dragonkin"]) {
        selection.toggle(catalog.tags.races.find(item => item.tag === key));
    }
    assert.equal(selection.value(), "elf, dragonkin");
});

test("hair patterns are selected once and face presets never enter the eye picker", () => {
    const hair = presetGroups(catalog, "hair");
    const selection = presetSelection("drill_hair, drills, Silver Hair", hair);
    assert.equal(selection.value(), "drill_hair, Silver Hair");
    assert.equal(presetGroups(catalog, "eyes").flatMap(group => group.items).some(item => item.tag === "oval face"), false);
});
