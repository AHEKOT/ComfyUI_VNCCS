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

test("legacy default hair selects both traits without rewriting saved text", () => {
    const groups = presetGroups(catalog, "hair");
    const black = catalog.tags.hair_color.find(item => item.tag === "black hair");
    const long = catalog.tags.hair_length.find(item => item.synonyms?.includes("long_hair"));
    const selection = presetSelection("black long hair, My Custom Trait", groups);
    assert.equal(selection.has(black), true);
    assert.equal(selection.has(long), true);
    assert.equal(selection.value(), "black long hair, My Custom Trait");
    selection.toggle(black);
    assert.equal(selection.has(black), false);
    assert.equal(selection.has(long), true);
    assert.equal(selection.value(), `My Custom Trait, ${long.tag}`);
    selection.toggle(long);
    assert.equal(selection.value(), "My Custom Trait");
    selection.toggle(black);
    assert.equal(selection.value(), "My Custom Trait, black hair");
    assert.equal(presetSelection("a black long hair ornament", groups).has(black), false);
});

test("new character defaults and eye color tags resolve to active presets", () => {
    const source = fs.readFileSync(new URL("../web/vnccs_character_creator_v2.js", import.meta.url), "utf8");
    assert.match(source, /checkedJSON\("\/vnccs\/create", \{ method: "POST", body: JSON\.stringify\(\{ name: n, catalog: "creator_v2" \}\)/);
    const defaults = [...source.matchAll(/hair: "([^"]+)", eyes:/g)];
    assert.equal(defaults.length, 2);
    for (const [, hair] of defaults) {
        assert.equal(hair, "black hair, waist-length hair");
        const selection = presetSelection(hair, presetGroups(catalog, "hair"));
        for (const token of hair.split(", ")) {
            const item = [...catalog.tags.hair_color, ...catalog.tags.hair_length].find(item => item.tag === token);
            assert.ok(item, token);
            assert.equal(selection.has(item), true);
        }
    }
    for (const [field, value, category] of [
        ["race", "human", "races"], ["face", "freckles", "face_details"],
        ["body", "medium breasts", "breast_size"], ["eyes", "blue eyes", "eye_color"],
    ]) {
        const selection = presetSelection(value, presetGroups(catalog, field));
        assert.equal(catalog.tags[category].filter(item => selection.has(item)).length, 1);
        assert.equal(selection.value(), value);
    }
    const blue = catalog.tags.eye_color.find(item => item.label === "Blue");
    const eyes = presetSelection("", presetGroups(catalog, "eyes"));
    eyes.toggle(blue);
    assert.equal(eyes.value(), "blue eyes");
});
