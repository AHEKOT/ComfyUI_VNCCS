import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

const web = new URL("../web/", import.meta.url);
const theme = readFileSync(new URL("vnccs_common.js", web), "utf8")
    .match(/export const SAKURA_THEME_CSS = `([\s\S]*?)`;/)[1];
const sheets = readdirSync(web, { recursive: true })
    .filter(name => /\.(?:js|mjs)$/.test(name))
    .map(name => {
        const source = readFileSync(new URL(name, web), "utf8").replaceAll("${SAKURA_THEME_CSS}", theme.trimEnd());
        const css = Array.from(source.matchAll(/(?:const\s+\w*(?:STYLE|CSS)\w*|\w+\.textContent)\s*=\s*`([\s\S]*?)`/g), match => match[1])
            .filter(text => /(?:display|position|font-size|background)\s*:/.test(text)).join("\n");
        const selectors = Array.from(css.replace(/\/\*[\s\S]*?\*\//g, "").matchAll(/([^{}]+)\{/g), match => match[1].trim());
        return { name, source, css, selectors };
    }).filter(sheet => sheet.css);

test("every widget owns its CSS classes and animation names", () => {
    const owners = new Map();
    for (const sheet of sheets) {
        const names = new Set(sheet.selectors.flatMap(selector =>
            Array.from(selector.matchAll(/\.((?:vnccs-|ems-|em-|cd-)[\w-]+)/g), match => match[1])
                .filter(name => !name.startsWith("vnccs-common-"))));
        for (const match of sheet.css.matchAll(/@keyframes\s+([\w-]+)/g)) names.add(`@keyframes ${match[1]}`);
        for (const name of names) {
            assert.ok(!owners.has(name), `${name} is shared by ${owners.get(name)} and ${sheet.name}`);
            owners.set(name, sheet.name);
        }
        if (sheet.name !== "vnccs_common.js") {
            for (const selector of sheet.selectors.filter(text => text.includes(".vnccs-common-"))) {
                assert.match(selector, /^\.(?!vnccs-common-)[\w-]+\s+/, `${sheet.name}: unscoped common component override`);
            }
        }
    }
});

test("widget palettes cannot overwrite the page or another widget", () => {
    for (const sheet of sheets) assert.ok(!/:root\b/.test(sheet.css), `${sheet.name}: global palette`);
    for (const [name, root] of [
        ["vnccs_character_creator_v2.js", "vnccs-creator-container"],
        ["vnccs_clothes_designer.js", "vnccs-clothes-container"],
        ["vnccs_emotion_v2.js", "ems-container"],
    ]) {
        const sheet = sheets.find(item => item.name === name);
        assert.match(sheet.css, new RegExp(`\\.${root}\\s*\\{\\s*--bg-primary:`), name);
        assert.ok(sheet.source.includes(`container.className = "${root}"`), name);
    }
});

test("loading other widgets cannot replace Creator's three columns with Clothes Designer's two", () => {
    const creator = sheets.find(sheet => sheet.name === "vnccs_character_creator_v2.js");
    const clothes = sheets.find(sheet => sheet.name === "vnccs_clothes_designer.js");
    assert.ok(/\.vnccs-creator-top-row\s*\{[^}]*grid-template-columns:\s*30% 35% 35%;/.test(creator.css));
    assert.ok(/\.vnccs-clothes-top-row\s*\{[^}]*grid-template-columns:\s*32% minmax\(0, 68%\);/.test(clothes.css));
    for (const sheet of sheets.filter(item => item !== creator)) {
        assert.ok(sheet.selectors.every(selector => !selector.includes(".vnccs-creator-")), sheet.name);
    }
});

test("Creator hover styling only applies to enabled, unselected segmented buttons", () => {
    const creator = sheets.find(sheet => sheet.name === "vnccs_character_creator_v2.js");
    const hover = creator.selectors.filter(selector => selector.includes(".vnccs-creator-segmented-btn:hover"));
    assert.ok(hover.length);
    for (const selector of hover) {
        assert.ok(selector.includes(":not(.is-active)"), `Selected buttons can match ${selector}`);
        assert.ok(selector.includes(":not(:disabled)"), `Disabled buttons can match ${selector}`);
    }
    assert.match(creator.css, /\.vnccs-creator-segmented-btn\.is-active\s*\{[^}]*background:/);
    assert.ok(creator.selectors.some(selector => selector.includes(".vnccs-creator-segmented-btn:focus-visible")));
});

for (const [name, controls] of [
    ["vnccs_character_creator_v2.js", [
        ["vnccs-creator-segmented-btn", "is-active"],
        ["vnccs-creator-graphic-toggle", "is-active"],
        ["vnccs-creator-tab", "is-active"],
        ["vnccs-creator-seed-dice-btn", "is-active"],
        ["vnccs-creator-tag-chip", "selected"],
        ["vnccs-creator-model-card", "is-selected"],
    ]],
    ["vnccs_character_cloner.js", [
        ["vnccs-cloner-segmented-btn", "is-active"],
        ["vnccs-cloner-graphic-toggle", "is-active"],
        ["vnccs-cloner-tag-chip", "selected"],
    ]],
    ["vnccs_clothes_designer.js", [
        ["vnccs-clothes-segmented-btn", "is-active"],
        ["vnccs-clothes-seed-dice-btn", "is-active"],
        ["cd-tab", "active"],
    ]],
    ["vnccs_emotion_v2.js", [
        ["ems-emotion-item", "selected"],
        ["ems-tab", "active"],
    ]],
    ["vnccs_character_generator.js", [["vnccs-seedvr-card", "is-selected"]]],
    ["vnccs_control_center.js", [
        ["vnccs-cc-btn", "vnccs-cc-btn--active", "vnccs-cc-btn--clip-active", "vnccs-cc-btn--cnet-active"],
        ["vnccs-cc-row", "vnccs-cc-row--model-sel", "vnccs-cc-row--clip-sel", "vnccs-cc-row--cnet-sel"],
        ["vnccs-cc-family-tab", "vnccs-cc-family-tab--active"],
        ["vnccs-cc-model-tab", "vnccs-cc-model-tab--active"],
        ["vnccs-cc-turbo-strip", "is-active"],
        ["vnccs-cc-lora-card", "vnccs-cc-lora-card--active"],
    ]],
]) {
    test(`${name}: hover rules exclude every selected control`, () => {
        const sheet = sheets.find(item => item.name === name);
        for (const [control, ...states] of controls) {
            const hover = sheet.selectors.flatMap(selector => selector.split(","))
                .filter(selector => new RegExp(`\\.${control}(?=[:.\\s]|$)`).test(selector) && selector.includes(":hover"));
            assert.ok(hover.length, `Missing hover rules for ${control}`);
            for (const selector of hover) {
                for (const state of states) {
                    assert.ok(selector.includes(`:not(.${state})`), `${name}: selected ${state} can match ${selector.trim()}`);
                }
            }
        }
    });
}

test("Cloner's selected image cannot acquire a second hover border or glow", () => {
    const cloner = sheets.find(item => item.name === "vnccs_character_cloner.js");
    const hover = cloner.selectors.filter(selector => selector.includes(".vnccs-cloner-thumb:hover"));
    assert.ok(hover.length);
    for (const selector of hover) assert.ok(selector.includes(":where(.vnccs-cloner-thumb-wrap:not(.is-selected)) "), selector);
});

test("active-state exclusions retain the priority of Save and generating-thumbnail styles", () => {
    const controlCenter = sheets.find(item => item.name === "vnccs_control_center.js");
    assert.ok(controlCenter.selectors.some(selector => selector.startsWith(".vnccs-cc-btn:hover:where(")));
    assert.ok(controlCenter.selectors.includes(".vnccs-cc-btn--save:hover"));
    const cloner = sheets.find(item => item.name === "vnccs_character_cloner.js");
    assert.ok(cloner.selectors.includes(".vnccs-cloner-thumb.generating"));
});
