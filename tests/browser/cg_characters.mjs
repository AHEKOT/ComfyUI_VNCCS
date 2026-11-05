import assert from "node:assert/strict";

const rounded = value => JSON.parse(JSON.stringify(value, (_key, item) => typeof item === "number" ? Math.round(item * 1e9) / 1e9 : item));

export async function verifyCharacters(page, { shot, settle, geometry, scenarios }) {
    assert.deepEqual(await page.evaluate(() => ({ secure: isSecureContext, uuid: typeof crypto.randomUUID,
        randomValues: typeof crypto.getRandomValues })), { secure: false, uuid: "undefined", randomValues: "function" },
    "Exercise ordinary HTTP without the secure-context exception for localhost");
    await page.evaluate(() => setup()); await settle();
    const before = await geometry();
    const nodeSize = await page.evaluate(() => [...widget.node.size]);
    await page.locator("[data-add-character]").click();
    await page.waitForFunction(() => widget.characters?.entries.size === 1 && !widget.busy);
    const initial = await page.evaluate(() => {
        const editor = widget.characters, core = editor.active;
        return { sharedScene: core.scene.parent === widget.viewer.scene, sharedCamera: core.camera === widget.viewer.camera,
            camera: widget.viewer.getCameraState(), pose: editor.snapshot(), bones: core.boneList.map(bone => bone.name) };
    });
    assert.ok(initial.sharedScene && initial.sharedCamera, "The rig shares Factory's world and camera");
    assert.match(initial.pose[0].id, /^[a-f0-9]{32}$/, "Character ID retains the saved-state format on ordinary HTTP");
    assert.ok(initial.bones.some(name => name.includes("thumb")), "The full Studio rig includes finger joints");
    await settle(); await shot("after-character");
    assert.deepEqual(await page.evaluate(() => [...widget.node.size]), nodeSize, "Opening character editing preserves the node size");
    const withCharacter = await geometry();
    assert.equal(await page.locator(".bg-camera-toolbar button").count(), 3, "Only Add Character joins the existing two buttons");
    assert.equal(await page.locator(".bg-viewport-panel select,.bg-character-toolbar").count(), 0, "Character editing adds no extra panel or selectors");
    for (const panel of ["viewport", "frame"]) for (const key of ["x", "y", "width", "height"])
        assert.ok(Math.abs(before[panel][key] - withCharacter[panel][key]) <= 1, `Character controls preserve ${panel} ${key}`);

    // Exercise the Studio IK with actual pointer movement, before release.
    const point = await page.evaluate(() => {
        const core = widget.characters.active, marker = core.jointMarkers.find(item => core.boneList[item.userData.boneIndex].name === "hand_l");
        const world = marker.getWorldPosition(new core.THREE.Vector3()).project(core.camera), bounds = core.canvas.getBoundingClientRect();
        return { x: bounds.x + (world.x + 1) * bounds.width / 2, y: bounds.y + (1 - world.y) * bounds.height / 2 };
    });
    await page.mouse.move(point.x, point.y); await page.mouse.down();
    await page.mouse.move(point.x + 45, point.y - 35, { steps: 8 });
    const during = await page.evaluate(() => ({ pose: widget.characters.snapshot(), camera: widget.viewer.getCameraState(),
        serialized: JSON.parse(widget.node.widgets[0].value).characters[widget.state.model_id] }));
    assert.notDeepEqual(during.pose, initial.pose, "IK changes the pose while the pointer is still down");
    assert.deepEqual(during.camera, initial.camera, "Joint drag does not turn the scene camera");
    assert.deepEqual(during.serialized, during.pose, "The newest drag state is serialized immediately");
    await page.mouse.up(); await settle(); await shot("after-character-pose");
    const posed = await page.evaluate(() => widget.characters.snapshot());
    await page.evaluate(() => widget.history.undo());
    assert.deepEqual(await page.evaluate(() => widget.characters.snapshot()), initial.pose);
    await page.evaluate(() => widget.history.redo());
    assert.deepEqual(await page.evaluate(() => widget.characters.snapshot()), posed);

    const viewport = await page.locator("[data-viewer] canvas").boundingBox();
    await page.mouse.move(viewport.x + viewport.width * .75, viewport.y + viewport.height * .25);
    await page.mouse.down({ button: "right" });
    await page.mouse.move(viewport.x + viewport.width * .75 + 40, viewport.y + viewport.height * .25 + 15, { steps: 6 });
    await page.mouse.up({ button: "right" });
    assert.notDeepEqual(await page.evaluate(() => widget.viewer.getCameraState()), initial.camera, "Scene rotation changes the common camera");
    assert.deepEqual(await page.evaluate(() => widget.characters.snapshot()), posed, "Camera movement leaves the rig's world transform intact");
    await page.evaluate(() => widget.history.undo());

    // Every bone, including fingers, is selectable for the Studio FK gizmo.
    const selection = await page.evaluate(() => {
        const editor = widget.characters, core = editor.active;
        return core.boneList.every(bone => { core.selectBone(bone); return core.transform.object === bone; });
    });
    assert.ok(selection, "Every joint can be attached to the rotation gizmo");
    await page.evaluate(() => widget.characters.active.selectBone(widget.characters.active.bones.hand_l));

    const capture = await page.evaluate(async () => {
        const editor = widget.characters, core = editor.active, original = widget.viewer.capturePreview;
        let helpersHidden = false;
        widget.viewer.capturePreview = async function (options) {
            helpersHidden = core.jointMarkers.every(marker => !marker.visible) && !core.transform.getHelper().visible;
            return await original.call(this, options);
        };
        try {
            const png = await widget.captureScene({ width: 512, height: 288 });
            const image = await createImageBitmap(png), canvas = document.createElement("canvas");
            canvas.width = image.width; canvas.height = image.height; canvas.getContext("2d").drawImage(image, 0, 0); image.close();
            window.characterCapture = canvas.toDataURL();
            return { size: png.size, helpersHidden, restored: core.jointMarkers.some(marker => marker.visible) };
        } finally { widget.viewer.capturePreview = original; }
    });
    assert.ok(capture.size > 1000 && capture.helpersHidden && capture.restored, "Capture includes the scene and restores editor helpers");

    // Reload the independently constructed widget state, never a shipped workflow file.
    const saved = await page.evaluate(() => widget.sync());
    await page.evaluate(raw => setup({ raw }), saved); await settle();
    assert.deepEqual(await page.evaluate(() => widget.characters.snapshot()), posed);
    await page.locator("[data-add-character]").click();
    assert.equal(await page.evaluate(() => widget.characters.entries.size), 1);
    for (const mode of ["pose", "rotate"]) {
        await page.locator(`[data-character-mode="${mode}"]`).click();
        const result = await page.evaluate(mode => {
            const editor = widget.characters;
            if (mode === "pose") editor.active.selectBone(editor.active.bones.thumb_01_l);
            return { mode: editor.active.transform.mode, attached: editor.active.transform.object ===
                (mode === "pose" ? editor.active.bones.thumb_01_l : editor.active.skinnedMesh) };
        }, mode);
        assert.equal(result.mode, "rotate"); assert.ok(result.attached);
        await settle();
        const handle = await page.evaluate(() => {
            const core = widget.characters.active, transform = core.transform, rect = core.canvas.getBoundingClientRect();
            core.scene.updateMatrixWorld(true);
            const pointer = new core.THREE.Vector3();
            for (const mesh of transform._gizmo.picker[transform.mode].children.filter(item => item.name === "X")) {
                const positions = mesh.geometry?.attributes.position; if (!positions) continue;
                const index = mesh.geometry.index, count = index?.count || positions.count;
                for (let i = 0; i < count; i += 3) {
                    pointer.set(0, 0, 0);
                    for (let j = 0; j < 3; j++) pointer.add(new core.THREE.Vector3().fromBufferAttribute(positions, index ? index.getX(i + j) : i + j));
                    pointer.multiplyScalar(1 / 3).applyMatrix4(mesh.matrixWorld).project(core.camera);
                    const inside = [[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2]].every(([x, y]) => {
                        transform.pointerHover({ x: pointer.x + x * 2 / rect.width, y: pointer.y + y * 2 / rect.height });
                        return transform.axis === "X";
                    });
                    if (inside) return { x: rect.x + (pointer.x + 1) * rect.width / 2,
                        y: rect.y + (1 - pointer.y) * rect.height / 2 };
                }
            }
            throw new Error("The character transform handle is not pickable.");
        });
        const beforeTransform = await page.evaluate(() => ({ scene: widget.characters.snapshot(), camera: widget.viewer.getCameraState() }));
        await page.mouse.move(handle.x, handle.y); await page.mouse.down();
        const drag = await page.evaluate(() => ({ axis: widget.characters.active.transform.axis, dragging: widget.characters.active.transform.dragging,
            selected: widget.characters.selected, pointer: widget.characters.pointerId, bone: widget.characters.active.transform.object?.name }));
        assert.ok(drag.dragging, `${mode} starts dragging: ${JSON.stringify({ handle, drag })}`);
        await page.mouse.move(handle.x + 45, handle.y + 25, { steps: 8 });
        const moved = await page.evaluate(() => ({ scene: widget.characters.snapshot(), camera: widget.viewer.getCameraState() }));
        assert.notDeepEqual(moved.scene, beforeTransform.scene, `${mode} updates before release`);
        assert.deepEqual(moved.camera, beforeTransform.camera, `${mode} keeps the camera still`);
        if (mode === "rotate") {
            assert.notDeepEqual(moved.scene[0].pose.modelRotation, beforeTransform.scene[0].pose.modelRotation);
            assert.deepEqual(moved.scene[0].pose.bones, beforeTransform.scene[0].pose.bones);
            assert.deepEqual(moved.scene[0].position, beforeTransform.scene[0].position);
            assert.deepEqual(await page.evaluate(() => JSON.parse(widget.node.widgets[0].value).characters[widget.state.model_id]), moved.scene);
            await shot("after-character-rotate");
        }
        await page.mouse.up();
    }
    await page.waitForFunction(() => !widget.characterPanel.loading);
    const choose = async (key, value) => {
        await page.locator(`[data-character-source="${key}"]`).evaluate((select, value) => {
            select.value = value; select.dispatchEvent(new Event("change", { bubbles: true }));
        }, value);
        await page.waitForFunction(() => !widget.characterPanel.loading);
    };
    await choose("character", "Maya");
    assert.equal(await page.evaluate(() => widget.characters.active.mesh.gender), 0);
    assert.equal(await page.locator(".cg-sprite").count(), 4);
    await choose("costume", "Evening"); await choose("emotion", "Happy");
    assert.equal(await page.evaluate(() => widget.characters.active.source.emotion), "Happy");
    assert.equal(await page.locator('.cg-character-tabs [role="tab"][aria-selected="true"]').count(), 1);
    assert.equal(await page.locator('[data-character-value="weight"],[data-character-value="muscle"],[data-character-value="height"],[data-character-age],[data-character-gender]').count(), 0);
    const lightInput = async (label, value) => page.getByRole("slider", { name: label, exact: true }).evaluate((input, value) => {
        input.value = value; input.dispatchEvent(new Event("input", { bubbles: true })); input.dispatchEvent(new Event("change", { bubbles: true }));
    }, value);
    await lightInput("Intensity", "2.4"); await lightInput("Light height", "35");
    assert.equal(await page.evaluate(() => widget.characters.active.keyLight.intensity), 2.4);
    assert.equal(await page.evaluate(() => widget.characters.active.keyLight.position.y), 35);
    assert.ok(await page.evaluate(() => widget.characters.active.keyLight.layers.mask === 2));
    for (const kind of ["light", "x", "y"]) {
        const radar = page.locator(`[data-character-radar="${kind}"]`);
        assert.equal(await radar.evaluate(canvas => canvas.width), kind === "light" ? 100 : 140);
        const before = await page.evaluate(() => ({ scene: widget.characters.snapshot(), camera: widget.viewer.getCameraState() }));
        const box = await radar.boundingBox();
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.down();
        await page.mouse.move(box.x + box.width * .65, box.y + box.height * .4, { steps: 5 });
        assert.notDeepEqual(await page.evaluate(() => widget.characters.snapshot()), before.scene);
        assert.deepEqual(await page.evaluate(() => widget.viewer.getCameraState()), before.camera);
        await page.mouse.up();
        await page.locator("[data-character-undo]").click();
        assert.deepEqual(rounded(await page.evaluate(() => widget.characters.snapshot())), rounded(before.scene));
    }
    const beforeMove = await page.evaluate(() => widget.characters.snapshot());
    await page.locator('[data-character-radar="x"]').press("ArrowRight");
    assert.notDeepEqual(await page.evaluate(() => widget.characters.snapshot()), beforeMove);
    await page.locator("[data-character-undo]").click();
    assert.deepEqual(rounded(await page.evaluate(() => widget.characters.snapshot())), rounded(beforeMove));
    await page.locator("[data-character-redo]").click();
    await page.locator('[data-character-mode="move"]').click();
    assert.ok(await page.evaluate(() => widget.characters.active.transform.mode === "translate" && widget.characters.active.transform.object === widget.characters.active.skinnedMesh));
    const body = await page.evaluate(() => {
        const core = widget.characters.active, position = core.bones.spine03 || core.bones.spine02 || core.boneList.find(bone => bone.name.includes("spine"));
        const point = position.getWorldPosition(new core.THREE.Vector3()).project(core.camera), rect = core.canvas.getBoundingClientRect();
        return { x: rect.x + (point.x + 1) * rect.width / 2, y: rect.y + (1 - point.y) * rect.height / 2,
            position: core.skinnedMesh.position.toArray(), camera: widget.viewer.getCameraState() };
    });
    await page.mouse.move(body.x, body.y); await page.mouse.down(); await page.mouse.move(body.x + 35, body.y - 20, { steps: 5 }); await page.mouse.up();
    assert.notDeepEqual(await page.evaluate(() => widget.characters.active.skinnedMesh.position.toArray()), body.position);
    assert.deepEqual(await page.evaluate(() => widget.viewer.getCameraState()), body.camera);
    await page.locator('[data-character-mode="pose"]').click();
    await page.locator("[data-character-library]").click();
    await page.locator(".vnccs-ps-library-item").click();
    await page.waitForFunction(() => !widget.characterPanel.library);
    assert.ok(await page.evaluate(() => Math.abs(widget.characters.active.getPose().bones.upperarm_l[2] - 35) < .001));
    const settings = await page.evaluate(() => ({ raw: widget.sync(), snapshot: widget.characters.snapshot() }));
    await page.evaluate(raw => setup({ raw }), settings.raw); await settle();
    assert.deepEqual(rounded(await page.evaluate(() => widget.characters.snapshot())), rounded(settings.snapshot));
    await page.locator("[data-add-character]").click(); await page.waitForFunction(() => !widget.characterPanel.loading);
    await page.locator("[data-character-back]").click();
    assert.ok(await page.locator(".bg-model-source").isVisible());
    await page.locator("[data-add-character]").click(); await page.waitForFunction(() => !widget.characterPanel.loading);
    await shot("after-character-panel");

    for (const width of [1000, 1500, 2102]) {
        await page.evaluate(width => { widget.node.setSize([width, 1100]); widget.node.onResize(); }, width);
        await settle();
        const radars = await page.locator('[data-character-radar]').evaluateAll(canvases => canvases.map(canvas => {
            const rect = canvas.getBoundingClientRect();
            return { y: rect.y + rect.height / 2, size: rect.height / widget.layoutScale };
        }));
        assert.equal(radars.length, 3);
        assert.ok(Math.max(...radars.map(item => item.y)) - Math.min(...radars.map(item => item.y)) < 1, "All three radar axes share one Y coordinate");
        assert.ok(radars.every(item => item.size >= 259.5), `The native radar controls are enlarged: ${JSON.stringify(radars)}`);
        const fit = await page.evaluate(() => {
            const panel = document.querySelector(".cg-character-panel").getBoundingClientRect();
            return [...document.querySelectorAll(".cg-character-panel>*,.cg-character-panel input,.cg-character-panel button,.cg-character-panel canvas")].every(element => {
                const rect = element.getBoundingClientRect();
                return rect.width === 0 || rect.x >= panel.x - 1 && rect.right <= panel.right + 1 && rect.y >= panel.y - 1 && rect.bottom <= panel.bottom + 1;
            });
        });
        assert.ok(fit, `Character controls stay inside the node at width ${width}`);
        await shot(`after-characters-${width}`);
    }
    const legacyCharacters = await page.evaluate(() => {
        const characters = widget.characters.snapshot();
        characters.push({ ...structuredClone(characters[0]), id: "a".repeat(32) });
        widget.characters.restore(characters); widget.sync(); widget.characterPanel.refresh();
        return widget.characters.snapshot();
    });
    await page.getByRole("button", { name: "Delete character", exact: true }).click();
    const deletion = page.getByRole("dialog", { name: "Delete character", exact: true });
    await deletion.getByRole("button", { name: "Cancel", exact: true }).click();
    assert.deepEqual(await page.evaluate(() => widget.characters.snapshot()), legacyCharacters);
    await page.getByRole("button", { name: "Delete character", exact: true }).click();
    await deletion.getByRole("button", { name: "Delete", exact: true }).click();
    assert.deepEqual(await page.evaluate(() => widget.characters.snapshot()), legacyCharacters.slice(1), "Deleting the tab removes only its character");
    await page.evaluate(() => widget.history.undo());
    assert.deepEqual(await page.evaluate(() => widget.characters.snapshot()), legacyCharacters);
    await page.evaluate(() => widget.history.redo());
    await page.waitForFunction(() => !widget.characterPanel.loading);
    await page.getByRole("button", { name: "Delete character", exact: true }).click();
    await deletion.getByRole("button", { name: "Delete", exact: true }).click();
    assert.equal(await page.evaluate(() => widget.characters.entries.size), 0);
    assert.ok(await page.locator(".cg-character-panel").isHidden());
    assert.equal(await page.locator("[data-add-character]").textContent(), "Add Character");
    await page.evaluate(() => widget.history.undo());
    assert.deepEqual(await page.evaluate(() => widget.characters.snapshot()), legacyCharacters.slice(1));
    await page.evaluate(() => {
        const core = widget.characters.active, viewer = widget.viewer;
        window.disposedCharacter = core; window.disposedViewer = viewer;
        widget.node.onRemoved();
    });
    assert.ok(await page.evaluate(() => disposedCharacter._disposed && disposedViewer._disposed));
    scenarios.push({ name: "pose-studio-characters", sharedScene: initial.sharedScene, joints: initial.bones.length, capture });
}
