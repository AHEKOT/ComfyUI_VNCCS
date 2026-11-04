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
    const mutationRoute = "**/vnccs/pose_library/repositories/**";
    const previewRoute = "**/vnccs/pose_library/preview/*", spriteRoute = "**/vnccs/pose-manager/sprites**";
    const imageImportRoute = "**/vnccs/sam3d/process_image_to_pose_json";
    let releaseImport;
    let delay = false, release, requested, initializations = 0, enabled = true, repoError = false, extraRepository = false;
    const studioMutations = [], managerMutations = [];
    await page.route(mutationRoute, route => {
        if (route.request().method() !== "GET") studioMutations.push(route.request().url());
        return route.fulfill({ status: 400, json: { error: "Studio settings must not be changed" } });
    });
    const records = Object.entries(assets).map(([name, asset], i) => ({ name, ...asset, asset_type: "pose", has_preview: true,
        ...(asset.repository === spriteRepo ? { id: String(i).padStart(32, "0"), manager_sprite_id: String(i).padStart(32, "0") } : {}) }));
    await page.route(listRoute, route => route.fulfill({ json: { poses: records.filter(record => !record.manager_sprite_id) } }));
    const assetName = route => decodeURIComponent(new URL(route.request().url()).pathname.split("/").at(-1));
    await page.route(previewRoute, route => previews[assetName(route)]
        ? route.fulfill({ contentType: "image/png", body: Buffer.from(previews[assetName(route)], "base64") }) : route.fallback());
    const waitForRelease = async () => {
        if (delay) { requested?.(); await new Promise(resolve => { release = resolve; }); }
    };
    await page.route(getRoute, async route => {
        await waitForRelease();
        await route.fulfill({ json: { pose: assets[assetName(route)].data, asset_type: "pose" } });
    });
    await page.route(spriteRoute, async route => {
        const parts = new URL(route.request().url()).pathname.split("/sprites")[1].split("/").filter(Boolean);
        if (parts.length) {
            const record = records.find(record => record.manager_sprite_id === parts[0]);
            if (parts[1] === "preview") return route.fulfill({ contentType: "image/png", body: Buffer.from(previews[record.name], "base64") });
            await waitForRelease();
            return route.fulfill({ json: { pose: record.data } });
        }
        if (repoError) return route.fulfill({ status: 503, json: { error: "Repository service unavailable" } });
        if (route.request().method() === "POST") {
            const data = route.request().postDataJSON();
            managerMutations.push(data);
            const { action, enabled: next } = data;
            if (action === "ensure") initializations++;
            if (action === "toggle") enabled = next;
            if (action === "add") extraRepository = true;
            if (action === "remove") extraRepository = false;
        }
        return route.fulfill({ json: { repositories: [{ repo_id: spriteRepo, title: "Sprite poses", builtin: false, enabled, pose_count: 3 },
            ...(extraRepository ? [{ repo_id: "owner/manager-extra", enabled: true, builtin: false, pose_count: 0 }] : [])],
            poses: enabled ? records.filter(record => record.manager_sprite_id) : [] } });
    });
    const library = page.getByRole("dialog", { name: "Pose Library", exact: true });
    const names = () => library.locator(".vnccs-ps-library-item-name").allTextContents();
    const checkPacking = async () => {
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const rows = await library.locator(".vnccs-ps-library-modal-grid").evaluate(grid => {
            const rect = grid.getBoundingClientRect(), style = getComputedStyle(grid), scale = rect.width / grid.offsetWidth;
            const width = (grid.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)) * scale;
            const left = rect.left + (grid.clientLeft + parseFloat(style.paddingLeft)) * scale;
            const grouped = new Map();
            for (const card of grid.querySelectorAll(".vnccs-ps-library-item")) {
                const box = card.getBoundingClientRect(), key = Math.round(box.top);
                if (!grouped.has(key)) grouped.set(key, []);
                grouped.get(key).push({ left: box.left, right: box.right, width: box.width,
                    name: card.querySelector(".vnccs-ps-library-item-name").textContent });
            }
            return { rows: [...grouped.values()], width, center: left + width / 2, gap: parseFloat(style.columnGap) * scale };
        });
        rows.rows.forEach((row, index) => {
            assert.ok(Math.abs((row[0].left + row.at(-1).right) / 2 - rows.center) <= 1, "Every row is centered");
            const used = row.at(-1).right - row[0].left;
            assert.ok(rows.rows.slice(index + 1).flat().every(card => used + rows.gap + card.width > rows.width - 1),
                "A later card must fill an earlier row whenever it fits");
        });
        return rows.rows.map(row => row.map(card => card.name));
    };
    const open = async (index, source = "Manager", empty = false) => {
        await page.getByRole("button", { name: `Library for Pose ${index + 1}`, exact: true }).click();
        await library.waitFor();
        if (!empty) await library.locator(".vnccs-ps-library-item").first().waitFor();
        if (source !== "Manager") await library.getByRole("button", { name: source, exact: true }).click();
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
            await checkPacking();
            const geometry = await library.evaluate(overlay => {
                const a = overlay.getBoundingClientRect(), b = poseNode.studioWidget.container.getBoundingClientRect();
                const cards = [...overlay.querySelectorAll(".vnccs-ps-library-item")];
                const boxes = cards.map(card => card.getBoundingClientRect());
                const header = overlay.querySelector(".vnccs-ps-library-modal-header").getBoundingClientRect();
                const capture = [...overlay.querySelectorAll("button")].find(button => button.textContent === "Capture pose from image");
                const captureBox = capture.getBoundingClientRect(), closeBox = overlay.querySelector(".vnccs-ps-modal-close").getBoundingClientRect();
                return { local: overlay.parentElement === poseNode.studioWidget.container,
                    captureFits: captureBox.left >= header.left && captureBox.right <= closeBox.left
                        && captureBox.top >= header.top && captureBox.bottom <= header.bottom
                        && capture.scrollWidth <= capture.clientWidth + 1 && capture.scrollHeight <= capture.clientHeight + 1,
                    delta: ["x", "y", "width", "height"].map(key => Math.abs(a[key] - b[key])),
                    ratios: cards.map(card => {
                        const img = card.querySelector("img"), p = card.querySelector(".vnccs-ps-library-item-preview").getBoundingClientRect();
                        return Math.abs(p.width / p.height - img.naturalWidth / img.naturalHeight);
                    }),
                    gaps: boxes.flatMap((box, i) => i && Math.abs(box.y - boxes[i - 1].y) < 1 ? [box.x - boxes[i - 1].right] : []),
                    inside: boxes.every(box => box.x >= a.x && box.right <= a.right && box.y >= a.y),
                    noHorizontalScroll: poseNode.studioWidget.libraryGrid.scrollWidth <= poseNode.studioWidget.libraryGrid.clientWidth + 1,
                    selectorRight: overlay.querySelector(".vnccs-pm-pose-sources").getBoundingClientRect().right
                        >= overlay.querySelector(".vnccs-ps-library-categories").getBoundingClientRect().right - 40,
                    inspector: !!overlay.querySelector(".vnccs-ps-library-inspector.visible") };
            });
            assert.ok(geometry.local && geometry.captureFits && geometry.delta.every(delta => delta <= 1));
            assert.ok(geometry.inside && geometry.noHorizontalScroll && geometry.selectorRight && !geometry.inspector, JSON.stringify(geometry));
            assert.ok(geometry.ratios.every(delta => delta < .01) && geometry.gaps.every(gap => gap > 0 && gap <= 8), JSON.stringify(geometry));
            await page.screenshot({ path: join(output, `pose-library-${width}.png`) });
            await library.locator(".vnccs-ps-library-item").last().scrollIntoViewIfNeeded();
            assert.ok(await library.locator(".vnccs-ps-library-item").last().evaluate(card =>
                card.getBoundingClientRect().bottom <= poseNode.studioWidget.libraryGrid.getBoundingClientRect().bottom + 1),
                "The last row is accessible in the scrolling gallery");
            await library.getByRole("button", { name: "Pose Studio", exact: true }).click();
            await page.waitForFunction(() => [...document.querySelectorAll(".vnccs-pm-pose-picker img")].every(img => img.complete && img.naturalWidth));
            assert.deepEqual((await names()).sort(), ["Scene pose", "Single pose", "Whole set"]);
            const packed = await checkPacking();
            if (width === 900) assert.deepEqual(packed[0], ["Single pose", "Whole set"], "A later narrow card fills the gap before a wide card");
            await page.screenshot({ path: join(output, `pose-library-packed-${width}.png`) });
            const scroll = await library.locator(".vnccs-ps-library-modal-grid").evaluate(grid => { grid.scrollTop = 60; return grid.scrollTop; });
            await page.evaluate(() => poseNode.studioWidget.renderLibrary());
            await checkPacking();
            assert.ok(Math.abs(await library.locator(".vnccs-ps-library-modal-grid").evaluate(grid => grid.scrollTop) - scroll) <= 1);
            await library.locator(".vnccs-ps-library-size-slider").evaluate(input => {
                input.value = "400"; input.dispatchEvent(new Event("input", { bubbles: true }));
            });
            await checkPacking();
            await library.locator(".vnccs-ps-library-size-slider").evaluate(input => {
                input.value = "320"; input.dispatchEvent(new Event("input", { bubbles: true }));
            });
            await checkPacking();
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
        assert.ok(initializations > 0);
        await open(5);
        await library.locator(".vnccs-ps-library-menu-btn").click();
        const settings = library.locator(".vnccs-ps-library-settings");
        await settings.getByRole("button", { name: "Disable", exact: true }).click();
        await settings.getByRole("button", { name: "Enable", exact: true }).waitFor();
        assert.equal(await settings.locator(".vnccs-ps-library-repo-card:visible").count(), 1);
        await settings.getByRole("button", { name: "Refresh", exact: true }).click();
        await settings.getByRole("button", { name: "Enable", exact: true }).click();
        await settings.getByRole("button", { name: "Disable", exact: true }).waitFor();
        await settings.locator(".vnccs-ps-library-repo-input").fill("owner/manager-extra");
        await settings.getByRole("button", { name: "Add Repository", exact: true }).click();
        const extra = settings.locator(".vnccs-ps-library-repo-card").filter({ hasText: "owner/manager-extra" });
        await extra.waitFor();
        await page.screenshot({ path: join(output, "sprite-settings-isolated.png") });
        await extra.getByRole("button", { name: "Remove", exact: true }).click();
        await extra.waitFor({ state: "detached" });
        await settings.getByRole("button", { name: "Back to library", exact: true }).click();
        await library.getByRole("button", { name: "Pose Studio", exact: true }).click();
        await library.locator(".vnccs-ps-library-menu-btn").click();
        await settings.getByText("Manager Repositories", { exact: true }).waitFor();
        assert.equal(await settings.locator(".vnccs-ps-library-repo-card:visible").count(), 1);
        await page.keyboard.press("Escape");
        enabled = false;
        await open(5, "Manager", true);
        await page.waitForFunction(() => document.querySelector(".vnccs-pm-pose-status")?.textContent === "");
        assert.deepEqual(await names(), []);
        await page.keyboard.press("Escape"); enabled = true;
        repoError = true;
        await open(5, "Manager", true);
        await library.getByText("Repository service unavailable Open repository settings to retry.").waitFor();
        await page.keyboard.press("Escape"); repoError = false;
        await page.evaluate(() => { poseNode.setSize([1210, 940]); app.canvas.ds.scale = .85; app.canvas.centerOnNode(poseNode); });
        for (const [index, name, source] of [[0, "Single pose", "Pose Studio"], [5, "Scene pose", "Pose Studio"], [11, "Female sprite", "Manager"]]) {
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
        await open(2, "Pose Studio");
        const beforeSet = await snapshot();
        await select("Whole set");
        await library.waitFor({ state: "detached" });
        assert.deepEqual(await snapshot(), beforeSet, "A whole set cannot replace a single card");
        await page.locator(".vnccs-ps-modal").filter({ hasText: "Choose a single pose for this card." })
            .getByRole("button", { name: "OK", exact: true }).click();
        await open(2);
        const beforeLate = await snapshot();
        delay = true;
        const request = new Promise(resolve => { requested = resolve; });
        await select("Female sprite");
        await request;
        await page.keyboard.press("Escape");
        assert.deepEqual(await snapshot(), beforeLate, "Closing the pending picker preserves every pose");
        await open(3);
        const beforeResponse = await snapshot();
        const response = page.waitForResponse(response => response.url().includes("/pose-manager/sprites/"));
        release();
        await response;
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        assert.deepEqual(await snapshot(), beforeResponse, "A closed picker cannot apply a late response to another card");
        assert.ok(await library.isVisible(), "A late response cannot close the next picker");
        await page.keyboard.press("Escape");
        await page.screenshot({ path: join(output, "card-library-buttons.png") });
        delay = false;
        await page.locator(".vnccs-ps-pose-preview").first().click();
        await page.evaluate(() => poseNode.studioWidget.showLibraryModal());
        await library.getByRole("button", { name: "Manager", exact: true }).waitFor();
        await library.locator(".vnccs-ps-library-menu-btn").click();
        await settings.getByText("Manager Repositories", { exact: true }).waitFor();
        await settings.getByRole("button", { name: "Disable", exact: true }).waitFor();
        await page.screenshot({ path: join(output, "editor-library-settings.png") });
        await page.keyboard.press("Escape");
        await page.evaluate(() => poseNode.studioWidget.setInterfaceMode("manager"));
        await open(4);
        const capture = library.getByRole("button", { name: "Capture pose from image", exact: true });
        let chooser = await Promise.all([page.waitForEvent("filechooser"), capture.click()]).then(([chooser]) => chooser);
        assert.equal(await chooser.element().getAttribute("accept"), "image/*");
        await chooser.setFiles([]);
        assert.ok(await library.isVisible(), "Cancelling image selection keeps the library open");
        const beforeImport = await snapshot();
        await page.evaluate(() => {
            const w = poseNode.studioWidget;
            window.nativeImageImporter = w.importSAM3DImageAsPose;
            window.importedImages = [];
            w.importSAM3DImageAsPose = async function (file) {
                importedImages.push({ name: file.name, type: file.type, tab: this.activeTab });
            };
        });
        const file = { name: "reference.png", mimeType: "image/png", buffer: Buffer.from(previews["Female sprite"], "base64") };
        chooser = await Promise.all([page.waitForEvent("filechooser"), capture.click()]).then(([chooser]) => chooser);
        await chooser.setFiles(file);
        await library.waitFor({ state: "detached" });
        assert.deepEqual(await page.evaluate(() => importedImages), [{ name: file.name, type: file.mimeType, tab: 4 }],
            "The native file handler delegates to Studio's image importer for the selected card");
        await page.evaluate(() => { poseNode.studioWidget.importSAM3DImageAsPose = nativeImageImporter; });
        const requestedImport = new Promise(resolve => {
            page.route(imageImportRoute, async route => {
                const wait = new Promise(release => { releaseImport = release; });
                resolve(route.request());
                await wait;
                await route.fulfill({ status: 503, json: { error: "Image analysis unavailable" } });
            });
        });
        await open(4);
        chooser = await Promise.all([page.waitForEvent("filechooser"), capture.click()]).then(([chooser]) => chooser);
        await chooser.setFiles(file);
        const importRequest = await requestedImport;
        assert.match(importRequest.headers()["content-type"], /multipart\/form-data/);
        const progress = page.locator(".vnccs-ps-modal-overlay").filter({ has: page.locator(".vnccs-ps-import-progress") });
        await progress.waitFor({ state: "visible" });
        assert.ok(await progress.evaluate(overlay => overlay.parentElement === poseNode.studioWidget.container));
        await page.screenshot({ path: join(output, "capture-pose-progress.png") });
        releaseImport();
        const error = page.locator(".vnccs-ps-modal").filter({ hasText: "Failed to import image with SAM 3D Body: Image analysis unavailable" });
        await error.waitFor({ state: "visible" });
        await progress.waitFor({ state: "detached" });
        assert.deepEqual(await snapshot(), beforeImport, "A failed import preserves all poses");
        await error.getByRole("button", { name: "OK", exact: true }).click();
        assert.deepEqual(studioMutations, [], "Manager must never register or change a Studio repository");
        for (const action of ["toggle", "refresh"]) assert.ok(managerMutations.some(data => data.action === action && data.repo_id === spriteRepo));
        for (const action of ["add", "remove"]) assert.ok(managerMutations.some(data => data.action === action && data.repo_id === "owner/manager-extra"));
        assert.ok(await page.evaluate(() => poseNode.studioWidget.libraryPoses.every(pose => !pose.manager_sprite_id)),
            "Manager sprites never leak into Studio's cached catalog");
    } finally {
        release?.();
        releaseImport?.();
        for (const route of [listRoute, getRoute, mutationRoute, previewRoute, spriteRoute, imageImportRoute]) await page.unroute(route);
    }
}
