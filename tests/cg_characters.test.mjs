import assert from "node:assert/strict";
import test from "node:test";
import { validateCharacters } from "../web/cg_characters.mjs";
import { readState } from "./helpers/cg_frontend.mjs";

const character = () => ({ id: "a".repeat(32), position: [1, 2, -3], scale: .2,
    pose: { bones: { hand_l: [0, 25, -30] }, bonePositions: { root: [0, 0, 0] }, modelRotation: [0, 45, 0] } });

test("character state preserves independent poses and accepts old empty scenes", () => {
    const characters = [character(), { ...character(), id: "b".repeat(32) }];
    assert.equal(validateCharacters(characters), characters);
    const raw = JSON.stringify({ schema_version: 1, characters: { ["c".repeat(32)]: characters } });
    assert.deepEqual(readState(raw).characters["c".repeat(32)], characters);
    assert.equal(readState('{"schema_version":1}').characters, undefined);
});

test("malformed transforms, duplicate IDs and unsafe pose maps are rejected", () => {
    for (const value of [null, {}, [{ ...character(), mesh: { age: 0 } }], [{ ...character(), light: { intensity: Infinity } }],
        [{ ...character(), source: { sprite: "../secret.png" } }], [character(), character()], Array.from({ length: 17 }, character),
        [{ ...character(), scale: 0 }], [{ ...character(), position: [0, Infinity, 0] }],
        [{ ...character(), pose: { ...character().pose, bones: { hand_l: [NaN, 0, 0] } } }],
        [{ ...character(), pose: { ...character().pose, bonePositions: [] } }]])
        assert.throws(() => validateCharacters(value), /CG character/);
    assert.throws(() => readState(JSON.stringify({ schema_version: 1, characters: { invalid: [] } })), /character scene/);
});
