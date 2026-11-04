import assert from "node:assert/strict";
import { join } from "node:path";

export async function verifyPoseCardLibrary(page, output) {
    const settled = () => page.waitForFunction(() => {
        const w = poseNode.studioWidget;
        return !w._morphSolveInFlight && !w._pendingMorphSolve
            && w._managerPreviewRefreshGeneration === w._managerPreviewRefreshCompletedGeneration;
    });
    for (const [key, label] of [["age", "Age"], ["weight", "Weight"], ["muscle", "Muscle"], ["height", "Height"], ["head_size", "Head Size"]]) {
        const expected = await page.evaluate(key => {
            const w = poseNode.studioWidget, input = w.managerControls[key].input;
            input.value = key === "age" ? "60" : "0.8";
            input.dispatchEvent(new Event("input")); input.dispatchEvent(new Event("change"));
            return Number(w.sliders[key].slider.closest(".vnccs-ps-field").querySelector(".vnccs-ps-reset-btn").title.replace("Reset to ", ""));
        }, key);
        await settled();
        await page.getByRole("button", { name: `Reset ${label}`, exact: true }).click();
        await settled();
        assert.deepEqual(await page.evaluate(key => {
            const w = poseNode.studioWidget;
            return [w.meshParams[key], Number(w.managerControls[key].input.value), Number(w.sliders[key].slider.value)];
        }, key), [expected, expected, expected], `${label} uses Studio's reset default`);
    }
    await page.screenshot({ path: join(output, "manager-resets.png") });
    const poses = await page.evaluate(() => structuredClone(poseNode.studioWidget.poses));
    const chosen = { ...poses[5], prompt: "Selected card pose" };
    delete chosen.cameraParams;
    const spriteRepo = "MIUProject/VNCCS_PoseLibrary_Sprites";
    const assets = {
        "Single pose": { data: chosen, repository: "local_user_poses", category: "Local", size: [160, 384] },
        "Scene pose": { data: { type: "scene_pose", active_character_id: "selected", activeTab: 0,
            characters: [{ id: "selected", poses: [chosen] }] }, repository: "owner/studio-a", category: "Studio", size: [384, 160] },
        "Whole set": { data: { type: "pose_set", poses: [chosen, chosen] }, repository: "owner/studio-b", category: "Studio", size: [200, 200] },
        "Female sprite": { data: chosen, repository: spriteRepo, category: "Female", size: [160, 384] },
        "Male sprite": { data: chosen, repository: spriteRepo, category: "Male", size: [180, 384] },
        "Wide sprite": { data: chosen, repository: spriteRepo, category: "Male", size: [384, 160] },
    };
    const previews = await page.evaluate(assets => Object.fromEntries(Object.entries(assets).map(([name, { size }]) => {
        const canvas = document.createElement("canvas"), source = poseNode.studioWidget.managerGrid.querySelector("canvas");
        [canvas.width, canvas.height] = size;
        const scale = Math.min(canvas.width / source.width, canvas.height / source.height);
        canvas.getContext("2d").drawImage(source, (canvas.width - source.width * scale) / 2, 0, source.width * scale, source.height * scale);
        return [name, canvas.toDataURL().split(",")[1]];
    })), assets);
    const listRoute = "**/vnccs/pose_library/list?*", getRoute = "**/vnccs/pose_library/get/*";
    const reposRoute = "**/vnccs/pose_library/repositories", addRoute = "**/vnccs/pose_library/repositories/add";
    const previewRoute = "**/vnccs/pose_library/preview/*";
    let delay = false, release, requested, additions = 0, enabled = true, repoError = false;
    await page.route(reposRoute, route => {
        assert.equal(new URL(route.request().url()).pathname, "/vnccs/pose_library/repositories");
        return route.fulfill(repoError ? { status: 503, json: { error: "Repository service unavailable" } } :
        { json: { repositories: [{ repo_id: "owner/studio-a", enabled: true }, { repo_id: "owner/studio-b", enabled: true },
            ...(additions ? [{ repo_id: spriteRepo, enabled }] : [])] } });
    });
    await page.route(addRoute, route => {
        assert.equal(route.request().postDataJSON().repo_id, spriteRepo);
        additions++;
        return route.fulfill({ json: { refreshed: { repo_id: spriteRepo, enabled: true, status: "success" } } });
    });
    await page.route(listRoute, route => route.fulfill({ json: { poses: Object.entries(assets).map(([name, asset]) =>
        ({ name, ...asset, asset_type: "pose", has_preview: true })) } }));
    const assetName = route => decodeURIComponent(new URL(route.request().url()).pathname.split("/").at(-1));
    await page.route(previewRoute, route => previews[assetName(route)]
        ? route.fulfill({ contentType: "image/png", body: Buffer.from(previews[assetName(route)], "base64") }) : route.fallback());
    await page.route(getRoute, async route => {
        const name = assetName(route);
        if (delay) { requested?.(); await new Promise(resolve => { release = resolve; }); }
        await route.fulfill({ json: { pose: assets[name].data, asset_type: "pose" } });
    });
    const library = page.getByRole("dialog", { name: "Pose Library", exact: true });
    const names = () => library.locator(".vnccs-ps-library-item-name").allTextContents();
    const open = async (index, source = "Sprites") => {
        await page.getByRole("button", { name: `Library for Pose ${index + 1}`, exact: true }).click();
        await library.locator(".vnccs-ps-library-item").first().waitFor();
        if (source !== "Sprites") await library.getByRole("button", { name: source, exact: true }).click();
        assert.deepEqual(await page.evaluate(() => [poseNode.studioWidget.interfaceMode, poseNode.studioWidget.activeTab]), ["manager", index]);
    };
    const select = name => library.locator(".vnccs-ps-library-item").filter({ hasText: name }).click();
    const snapshot = () => page.evaluate(() => structuredClone(poseNode.studioWidget.poses));
    try {
        await page.evaluate(() => poseNode.studioWidget.refreshLibrary(false));
        for (const [width, height, zoom] of [[900, 800, .75], [1210, 940, .85], [1400, 1000, 1]]) {
            await page.evaluate(({ width, height, zoom }) => {
                poseNode.setSize([width, height]); app.canvas.ds.scale = zoom; app.canvas.centerOnNode(poseNode);
            }, { width, height, zoom });
            await open(5);
            assert.deepEqual(await names(), ["Female sprite", "Male sprite", "Wide sprite"]);
            await library.getByRole("button", { name: "Female", exact: true }).click();
            assert.deepEqual(await names(), ["Female sprite"]);
            await library.getByRole("button", { name: "Male", exact: true }).click();
            assert.deepEqual(await names(), ["Male sprite", "Wide sprite"]);
            await library.getByRole("button", { name: "All", exact: true }).click();
            await page.waitForFunction(() => [...document.querySelectorAll(".vnccs-pm-pose-picker img")].every(img => img.complete && img.naturalWidth));
            const geometry = await library.evaluate(overlay => {
                const a = overlay.getBoundingClientRect(), b = poseNode.studioWidget.container.getBoundingClientRect();
                const cards = [...overlay.querySelectorAll(".vnccs-ps-library-item")];
                const boxes = cards.map(card => card.getBoundingClientRect());
                return { local: overlay.parentElement === poseNode.studioWidget.container,
                    delta: ["x", "y", "width", "height"].map(key => Math.abs(a[key] - b[key])),
                    ratios: cards.map(card => {
                        const img = card.querySelector("img"), p = card.querySelector(".vnccs-ps-library-item-preview").getBoundingClientRect();
                        return Math.abs(p.width / p.height - img.naturalWidth / img.naturalHeight);
                    }),
                    gaps: boxes.slice(1).filter((box, i) => Math.abs(box.y - boxes[i].y) < 1).map((box, i) => box.x - boxes[i].right),
                    inside: boxes.every(box => box.x >= a.x && box.right <= a.right && box.y >= a.y),
                    noHorizontalScroll: poseNode.studioWidget.libraryGrid.scrollWidth <= poseNode.studioWidget.libraryGrid.clientWidth + 1,
                    selectorRight: overlay.querySelector(".vnccs-pm-pose-sources").getBoundingClientRect().right
                        >= overlay.querySelector(".vnccs-ps-library-categories").getBoundingClientRect().right - 40,
                    inspector: !!overlay.querySelector(".vnccs-ps-library-inspector.visible") };
            });
            assert.ok(geometry.local && geometry.delta.every(delta => delta <= 1));
            assert.ok(geometry.inside && geometry.noHorizontalScroll && geometry.selectorRight && !geometry.inspector, JSON.stringify(geometry));
            assert.ok(geometry.ratios.every(delta => delta < .01) && geometry.gaps.every(gap => gap > 0 && gap <= 8), JSON.stringify(geometry));
            await page.screenshot({ path: join(output, `pose-library-${width}.png`) });
            await library.locator(".vnccs-ps-library-item").last().scrollIntoViewIfNeeded();
            assert.ok(await library.locator(".vnccs-ps-library-item").last().evaluate(card =>
                card.getBoundingClientRect().bottom <= poseNode.studioWidget.libraryGrid.getBoundingClientRect().bottom + 1),
                "The last row is accessible in the scrolling gallery");
            await library.getByRole("button", { name: "PoseStudio", exact: true }).click();
            assert.deepEqual(await names(), ["Single pose", "Scene pose", "Whole set"]);
            await library.locator(".vnccs-ps-library-search").fill("Scene");
            await page.waitForFunction(() => document.querySelectorAll(".vnccs-pm-pose-picker .vnccs-ps-library-item").length === 1);
            assert.deepEqual(await names(), ["Scene pose"]);
            const before = await snapshot();
            await page.keyboard.press("Escape");
            await library.waitFor({ state: "detached" });
            assert.deepEqual(await snapshot(), before, "Cancel preserves every pose");
            await page.getByRole("button", { name: "Library for Pose 6", exact: true }).press("Enter");
            await library.waitFor();
            await library.locator(".vnccs-ps-modal-close").click();
        }
        assert.equal(additions, 1, "The default repository is only registered once");
        enabled = false;
        await open(5);
        await library.getByText("Sprite library is disabled. Enable it in repository settings.").waitFor();
        assert.equal(additions, 1, "Existing repository preferences are preserved");
        await page.keyboard.press("Escape"); enabled = true;
        repoError = true;
        await open(5);
        await library.getByText("Repository service unavailable Open repository settings to retry.").waitFor();
        await page.keyboard.press("Escape"); repoError = false;
        await page.evaluate(() => { poseNode.setSize([1210, 940]); app.canvas.ds.scale = .85; app.canvas.centerOnNode(poseNode); });
        for (const [index, name, source] of [[0, "Single pose", "PoseStudio"], [5, "Scene pose", "PoseStudio"], [11, "Female sprite", "Sprites"]]) {
            await open(index, source);
            const before = await snapshot();
            if (index === 11) await library.locator(".vnccs-ps-library-item").filter({ hasText: name }).press("Enter");
            else await select(name);
            await library.waitFor({ state: "detached" });
            await settled();
            const after = await snapshot();
            assert.equal(after.length, before.length);
            after.forEach((pose, i) => {
                if (i !== index) assert.deepEqual(pose, before[i], `Pose ${i + 1} must not change`);
            });
            assert.equal(after[index].prompt, chosen.prompt);
            for (const [bone, rotation] of Object.entries(chosen.bones)) rotation.forEach((value, axis) =>
                assert.ok(Math.abs(after[index].bones[bone][axis] - value) < 1e-5, `${name}: ${bone}:${axis}`));
            const serialized = await page.evaluate(() => JSON.parse(poseNode.serialize().widgets_values[0]));
            assert.equal(serialized.poses[index].prompt, chosen.prompt);
        }
        await open(2, "PoseStudio");
        const beforeSet = await snapshot();
        await select("Whole set");
        await library.waitFor({ state: "detached" });
        assert.deepEqual(await snapshot(), beforeSet, "A whole set cannot replace a single card");
        await page.locator(".vnccs-ps-modal").filter({ hasText: "Choose a single pose for this card." })
            .getByRole("button", { name: "OK", exact: true }).click();
        await open(2, "PoseStudio");
        const beforeLate = await snapshot();
        delay = true;
        const request = new Promise(resolve => { requested = resolve; });
        await select("Single pose");
        await request;
        await page.keyboard.press("Escape");
        assert.deepEqual(await snapshot(), beforeLate, "Closing the pending picker preserves every pose");
        await open(3);
        const beforeResponse = await snapshot();
        const response = page.waitForResponse(response => response.url().includes("/pose_library/get/"));
        release();
        await response;
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        assert.deepEqual(await snapshot(), beforeResponse, "A closed picker cannot apply a late response to another card");
        assert.ok(await library.isVisible(), "A late response cannot close the next picker");
        await page.keyboard.press("Escape");
        await page.screenshot({ path: join(output, "card-library-buttons.png") });
    } finally {
        release?.();
        for (const route of [listRoute, getRoute, reposRoute, addRoute, previewRoute]) await page.unroute(route);
    }
}
