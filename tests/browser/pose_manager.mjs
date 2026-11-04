// Run only against a disposable ComfyUI containing this checkout and VNCCS-Utils.
import assert from "node:assert/strict";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import { join } from "node:path";
import { verifyPoseSetSettings } from "./pose_set_settings.mjs";
import { verifyPoseCardLibrary } from "./pose_card_library.mjs";
import { verifyManagerLiveFraming } from "./pose_manager_live_framing.mjs";

const { chromium } = createRequire(import.meta.url)("playwright");
const [url, output, executablePath] = process.argv.slice(2);
assert.ok(["localhost", "127.0.0.1"].includes(new URL(url).hostname) && output);
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ executablePath, args: ["--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1650, height: 1170 } });
const errors = [], reports = [];
const moduleURLs = new Set();
page.on("request", request => {
    if (/\/(pose_manager_ui|pose_manager_previews|pose_manager_library|pose_set_library|preview_grid_layout)\.mjs/.test(request.url())) moduleURLs.add(request.url());
});
const setName = `Twelve poses ${Date.now()}`;
page.on("pageerror", error => errors.push(error.message));
const shot = name => page.screenshot({ path: join(output, `${name}.png`) });
async function ready(count) {
    await page.waitForFunction(count => {
        const w = window.poseNode?.studioWidget;
        return w && !w.container.inert && !w._poseDataRestoreError && w.interfaceMode === "manager"
            && w.managerGrid.children.length === count
            && w.managerGrid.querySelectorAll("canvas.vnccs-pm-live").length === count
            && w._managerPreviewRefreshGeneration === w._managerPreviewRefreshCompletedGeneration;
    }, count, { timeout: 150000 });
}
async function geometry() {
    return page.evaluate(() => {
        const w = poseNode.studioWidget, bounds = e => e.getBoundingClientRect();
        const root = bounds(w.container), sidebar = bounds(w.managerSidebar);
        const header = w.managerPanel.querySelector(".vnccs-ps-manager-header");
        const library = header.querySelector(".vnccs-pm-library");
        const button = bounds(library), section = bounds(w.managerSidebar.querySelector(".vnccs-ps-section"));
        const headerBounds = bounds(header), countBounds = bounds(header.querySelector(".vnccs-pm-count"));
        const inside = e => { const b = bounds(e); return b.width > 0 && b.height > 0
            && b.x >= root.x - 1 && b.y >= root.y - 1 && b.right <= root.right + 1 && b.bottom <= root.bottom + 1; };
        return { size: [...poseNode.size], count: w.poses.length, inside: [library, w.managerSidebar, w.managerGrid, ...w.managerGrid.children].every(inside),
            library: library.textContent, first: library === header.firstElementChild,
            above: button.bottom <= section.top, width: Math.abs(button.width - section.width),
            headerFits: button.top >= headerBounds.top && button.bottom <= headerBounds.bottom && button.right <= countBounds.left,
            buttonFits: library.scrollWidth <= library.clientWidth + 1 && library.scrollHeight <= library.clientHeight + 1,
            controlsFit: w.managerSidebar.scrollHeight <= w.managerSidebar.clientHeight + 1,
            sidebarWidth: sidebar.width, titleRemoved: !w.managerPanel.querySelector(".vnccs-ps-manager-title") };
    });
}
async function cardGeometry(card, count) {
    const result = await card.evaluate(card => {
        const preview = card.querySelector(".vnccs-set-preview"), rect = preview.getBoundingClientRect();
        const images = [...preview.querySelectorAll("img")];
        const bounds = images.map(image => image.getBoundingClientRect());
        const remove = card.querySelector(".vnccs-set-delete"), removeBox = remove?.getBoundingClientRect();
        const cardBox = card.getBoundingClientRect();
        return { count: images.length, width: card.offsetWidth,
            requestedWidth: parseFloat(getComputedStyle(card).getPropertyValue("--vnccs-ps-library-thumb-size")),
            rows: new Set(bounds.map(box => Math.round(box.y))).size,
            columns: new Set(bounds.map(box => Math.round(box.x))).size,
            onlyImages: [...preview.children].every(child => child.tagName === "IMG") && !preview.textContent.trim(),
            order: images.map(image => Number(new URL(image.src).pathname.split("/").at(-1))),
            inside: bounds.every(box => box.width > 0 && box.height > 0 && box.x >= rect.x - 1 && box.y >= rect.y - 1
                && box.right <= rect.right + 1 && box.bottom <= rect.bottom + 1),
            contained: images.every(image => getComputedStyle(image).objectFit === "contain"),
            deleteTopRight: remove && remove.parentElement === card && !remove.closest(".vnccs-set-load")
                && removeBox.x > cardBox.x + cardBox.width / 2 && removeBox.right <= cardBox.right
                && removeBox.y >= cardBox.y && removeBox.bottom < cardBox.y + cardBox.height / 2,
            noScroll: preview.scrollHeight <= preview.clientHeight + 1 && preview.scrollWidth <= preview.clientWidth + 1 };
    });
    assert.equal(result.count, count);
    assert.ok(result.deleteTopRight, "Delete is a separate button at the top right of the card");
    const [columns, rows] = { 1: [1, 1], 2: [2, 1], 5: [5, 1], 6: [6, 1], 7: [4, 2],
        8: [4, 2], 9: [5, 2], 10: [5, 2], 11: [6, 2], 12: [6, 2], 13: [5, 3] }[count];
    assert.equal(result.columns, columns);
    assert.equal(result.rows, rows);
    assert.ok(Math.abs(result.width - result.requestedWidth) <= 1, JSON.stringify(result));
    assert.ok(result.onlyImages && result.inside && result.contained && result.noScroll);
    assert.deepEqual(result.order, Array.from({ length: count }, (_, index) => index));
    return result;
}
async function compactPoseGrid() {
    const result = await page.evaluate(() => {
        const grid = poseNode.studioWidget.managerGrid, rows = new Map();
        const gridBox = grid.getBoundingClientRect();
        const cards = [...grid.children];
        const inside = (box, parent) => box.x >= parent.x - .5 && box.y >= parent.y - .5
            && box.right <= parent.right + .5 && box.bottom <= parent.bottom + .5;
        for (const card of grid.children) {
            const row = rows.get(card.offsetTop) || [];
            row.push({ x: card.offsetLeft, y: card.offsetTop, width: card.offsetWidth, height: card.offsetHeight });
            rows.set(card.offsetTop, row);
        }
        const lines = [...rows.values()];
        return { count: grid.children.length,
            renderedFits: cards.every(card => inside(card.getBoundingClientRect(), gridBox)),
            squareDelete: cards.every(card => {
                const remove = card.querySelector(".vnccs-ps-pose-card-delete").getBoundingClientRect();
                return Math.abs(remove.width - remove.height) <= .5
                    && inside(remove, card.querySelector(".vnccs-ps-pose-preview").getBoundingClientRect());
            }),
            equalAppearance: new Set(cards.map(card => {
                const style = getComputedStyle(card), preview = getComputedStyle(card.querySelector(".vnccs-ps-pose-preview"));
                return [style.boxShadow, style.transform, preview.borderColor, preview.backgroundColor].join("|");
            })).size === 1,
            buttonsFit: [...grid.children].every(card => {
                const button = card.querySelector(".vnccs-pm-pose-library"), preview = card.querySelector(".vnccs-ps-pose-preview");
                if (!button || button.textContent !== "Library") return false;
                const b = button.getBoundingClientRect(), p = preview.getBoundingClientRect(), c = card.getBoundingClientRect();
                return b.top >= p.bottom - 1 && b.bottom <= c.bottom + 1 && Math.abs(b.width - p.width) <= 1
                    && button.scrollWidth <= button.clientWidth + 1 && button.scrollHeight <= button.clientHeight + 1;
            }),
            horizontal: lines.flatMap(row => row.slice(1).map((card, i) => card.x - row[i].x - row[i].width)),
            vertical: lines.slice(1).map((row, i) => row[0].y - lines[i][0].y - lines[i][0].height),
            center: lines.map(row => Math.abs((row[0].x + row.at(-1).x + row.at(-1).width) / 2 - grid.clientWidth / 2)),
            fits: lines.flat().every(card => card.x >= -1 && card.y >= -1 && card.x + card.width <= grid.clientWidth + 1
                && card.y + card.height <= grid.clientHeight + 1) };
    });
    assert.ok(result.fits && result.renderedFits && result.squareDelete && result.equalAppearance
        && result.buttonsFit && result.center.every(delta => delta <= 1), JSON.stringify(result));
    assert.ok([...result.horizontal, ...result.vertical].every(gap => Math.abs(gap - 8) <= 1), JSON.stringify(result));
    return result;
}
try {
    await page.route("**/vnccs/pose_library/repositories/auto_refresh", route => route.fulfill({ json: { tasks: [] } }));
    await page.goto(url);
    await page.waitForFunction(() => window.LiteGraph?.registered_node_types?.VNCCS_PoseManager && window.app?.vueAppReady
        && (app.graph?._nodes?.length || document.body.innerText.includes("Node graph")), null, { timeout: 45000 });
    await page.keyboard.press("Escape");
    await page.evaluate(() => {
        window.mount = (saved, zoom = 1) => {
            app.graph.clear();
            const node = LiteGraph.createNode("VNCCS_PoseManager");
            app.graph.add(node);
            if (saved) node.configure({ ...saved, id: node.id, type: "VNCCS_PoseManager",
                inputs: saved.inputs.map(input => ({ ...input, link: null })),
                outputs: saved.outputs.map(output => ({ ...output, links: null })) });
            node.pos = [50, 80];
            app.canvas.ds.scale = zoom;
            app.canvas.centerOnNode(node);
            app.graph.setDirtyCanvas(true, true);
            window.poseNode = node;
        };
        mount();
    });
    await ready(1);
    const performance = await page.evaluate(async () => {
        const w = poseNode.studioWidget, viewer = w.viewer;
        const encode = HTMLCanvasElement.prototype.toDataURL, capture = viewer.capture;
        let encodes = 0;
        const values = [];
        HTMLCanvasElement.prototype.toDataURL = function () { encodes++; return encode.apply(this, arguments); };
        viewer.capture = function () { values.push(this.headScale); return capture.apply(this, arguments); };
        try {
            const input = w.managerControls.head_size.input;
            for (let i = 0; i < 12; i++) {
                input.value = String(1 + i * .02);
                input.dispatchEvent(new Event("input", { bubbles: true }));
                await new Promise(resolve => requestAnimationFrame(resolve));
            }
            input.dispatchEvent(new Event("change", { bubbles: true }));
            await w.awaitCurrentManagerPreviews();
            return { encodes, values, raw: w.poseCaptures,
                live: viewer.canvas.parentElement === w.managerGrid.querySelector(".vnccs-ps-pose-preview") };
        } finally {
            HTMLCanvasElement.prototype.toDataURL = encode;
            viewer.capture = capture;
        }
    });
    assert.equal(performance.encodes, 0);
    assert.equal(performance.live, true);
    assert.ok(performance.values.length >= 10);
    assert.equal(performance.values.at(-1), 1.22);
    assert.deepEqual(performance.raw, [null]);
    reports.push({ performance });
    for (const key of ["head_size", "weight", "muscle", "height", "age"]) {
    const slider = await page.evaluate(key => {
        const w = poseNode.studioWidget, input = w.managerControls[key].input;
        const rect = input.getBoundingClientRect();
        const original = w.viewer.capture, fit = w.computePoseManagerCaptureFraming;
        window.dragCaptures = [];
        window.fitTimes = [];
        window.readMannequinBounds = () => {
            const source = w.viewer.canvas, copy = document.createElement("canvas");
            copy.width = source.width; copy.height = source.height;
            const ctx = copy.getContext("2d"); ctx.drawImage(source, 0, 0);
            const pixels = ctx.getImageData(0, 0, copy.width, copy.height).data;
            let left = copy.width, top = copy.height, right = -1, bottom = -1;
            for (let y = 0; y < copy.height; y++) for (let x = 0; x < copy.width; x++) {
                const i = (y * copy.width + x) * 4;
                if (Math.max(...[0, 1, 2].map(c => Math.abs(pixels[i + c] - pixels[c]))) < 35) continue;
                left = Math.min(left, x); right = Math.max(right, x);
                top = Math.min(top, y); bottom = Math.max(bottom, y);
            }
            if (right < left) throw Error("The pointer test must measure a rendered mannequin");
            const rect = source.getBoundingClientRect();
            const scale = Math.min(rect.width / source.width, rect.height / source.height);
            return [left, top, right, bottom].map(value => value * scale);
        };
        w.computePoseManagerCaptureFraming = function (...args) {
            const start = performance.now();
            try { return fit.apply(this, args); }
            finally { fitTimes.push(performance.now() - start); }
        };
        w.viewer.capture = function (...args) {
            dragCaptures.push({ value: w.meshParams[key], framing: [args[2], args[4], args[5]], dragging: w._poseGestureActive });
            return original.apply(this, args);
        };
        window.restoreDragCapture = () => { w.viewer.capture = original; w.computePoseManagerCaptureFraming = fit; };
        return { x: rect.x, y: rect.y + rect.height / 2, width: rect.width };
    }, key);
    await page.mouse.move(slider.x + slider.width * .48, slider.y);
    await page.mouse.down();
    await page.mouse.move(slider.x + slider.width * .82, slider.y, { steps: 12 });
    await page.waitForFunction(() => !poseNode.studioWidget._morphSolveInFlight && !poseNode.studioWidget._pendingMorphSolve);
    await ready(1);
    const held = await page.evaluate(() => ({ ...dragCaptures.at(-1), bounds: readMannequinBounds() }));
    assert.ok(held.dragging, "The regression check must sample the still-held pointer");
    await page.mouse.up();
    await page.waitForFunction(() => !poseNode.studioWidget._morphSolveInFlight && !poseNode.studioWidget._pendingMorphSolve);
    await ready(1);
    const released = await page.evaluate(() => {
        restoreDragCapture();
        return { ...dragCaptures.at(-1), bounds: readMannequinBounds() };
    });
    assert.equal(held.value, released.value);
    assert.ok(held.bounds.every((value, index) => Math.abs(value - released.bounds[index]) <= 1),
        `${key}: releasing the pointer must not move the rendered mannequin by more than one pixel`);
    reports.push({ key, held, released, fitTimes: await page.evaluate(() => fitTimes) });
    }
    const pixels = await page.evaluate(async () => {
        const w = poseNode.studioWidget;
        const sample = source => {
            const ctx = document.createElement("canvas").getContext("2d");
            ctx.drawImage(source, 0, 0);
            return [...ctx.getImageData(0, 0, 1, 1).data].slice(0, 3);
        };
        const preview = sample(w.viewer.canvas);
        await w.awaitReadyForCompositeCapture();
        const image = new Image(); image.src = w.poseCaptures[0]; await image.decode();
        const original = w.poseCaptures[0];
        Object.getPrototypeOf(w).refreshAllManagerPreviews.call(w);
        const identical = w.poseCaptures[0] === original;
        w.scheduleAllManagerPreviewRefresh();
        await w.awaitCurrentManagerPreviews();
        return { preview, export: sample(image), identical };
    });
    assert.ok(pixels.preview.every(channel => channel < 64));
    assert.deepEqual(pixels.export, [255, 255, 255]);
    assert.ok(pixels.identical, "Full-size output must equal the native capture at the latest slider value");
    await page.evaluate(() => {
        const w = poseNode.studioWidget;
        w.managerGrid.children[0].click();
        if (w.interfaceMode !== "managerDetail" || w.viewer._captureBatch || w.viewer.canvas.closest(".vnccs-ps-pose-preview")) throw Error("Studio canvas was not released");
        w.managerBackBtn.click();
    });
    await ready(1);
    await shot("single");
    const workflows = ["Step1_CharacterCreator", "Step1_CharacterCloner", "Step2_CharacterClothes"];
    let saved;
    for (const [i, name] of workflows.entries()) {
        const graph = JSON.parse(await readFile(new URL(`../../workflows/VNCCS_3.2_${name}.json`, import.meta.url)));
        const node = graph.nodes.find(node => ["VNCCS_PoseStudio", "VNCCS_PoseManager"].includes(node.type));
        assert.ok(node, "The workflow fixture must contain its original pose node");
        await page.evaluate(({ node, zoom }) => mount(node, zoom), { node, zoom: [.85, .75, 1][i] });
        await ready(12);
        reports.push({ compactGrid: await compactPoseGrid() });
        const report = await geometry();
        assert.ok(report.inside && report.first && report.above && report.controlsFit);
        assert.equal(report.library, "Library");
        assert.ok(report.width <= 1 && report.headerFits && report.buttonFits, JSON.stringify(report));
        assert.ok(report.titleRemoved);
        const connections = graph.links.filter(link => link[1] === node.id);
        const target = graph.nodes.find(item => item.id === connections[0][3]);
        assert.ok(await page.evaluate(({ connections, type }) => {
            const target = LiteGraph.createNode(type); app.graph.add(target); target.pos = [4000, 80];
            return connections.every(link => { poseNode.connect(link[2], target, link[4]); return target.inputs[link[4]].link != null; });
        }, { connections, type: target.type }));
        reports.push(report);
        await shot(name);
        if (i === 0) saved = await page.evaluate(() => poseNode.serialize());
        const counts = i === 0 ? [1, 2, 3, 5, 7, 10, 11, 13, 16, 24] : [10, 13];
        for (const count of counts) {
            await page.evaluate(({ count, poses }) => {
                poseNode.studioWidget.loadPoseSetAsset({ type: "pose_set", poses: Array.from({ length: count }, (_, i) => poses[i % poses.length]) });
            }, { count, poses: JSON.parse(saved.widgets_values[0]).poses });
            await ready(count);
            reports.push({ size: node.size, compactGrid: await compactPoseGrid() });
            if (count === 10) {
                const first = page.locator(".vnccs-ps-manager-grid > .vnccs-ps-pose-card").first();
                const before = await first.boundingBox();
                await first.hover();
                reports.push({ hoveredGrid: await compactPoseGrid() });
                const after = await first.boundingBox();
                for (const key of ["x", "y", "width", "height"]) assert.ok(Math.abs(after[key] - before[key]) <= .5);
            }
            if ([10, 13].includes(count)) await shot(`grid-${i}-${count}`);
        }
    }
    await page.evaluate(saved => mount(saved, .85), saved);
    await ready(12);
    await verifyManagerLiveFraming(page, output);
    await verifyPoseSetSettings(page, output);
    await page.evaluate(saved => mount(saved, .85), saved);
    await ready(12);
    await verifyPoseCardLibrary(page, output);
    const selectedPoses = await page.evaluate(() => poseNode.serialize());
    await page.evaluate(saved => mount(saved, .85), selectedPoses);
    await ready(12);
    assert.deepEqual(await page.evaluate(() => [0, 5, 11].map(i => poseNode.studioWidget.poses[i].prompt)),
        Array(3).fill("Selected card pose"), "Library selections survive workflow restoration");
    await page.evaluate(saved => mount(saved, .85), saved);
    await ready(12);
    const opener = page.getByRole("button", { name: "Library", exact: true });
    await opener.click();
    const library = page.getByRole("dialog", { name: "Library", exact: true });
    const libraryBounds = await library.evaluate(overlay => {
        const node = window.poseNode.studioWidget.container;
        const a = overlay.getBoundingClientRect(), b = node.getBoundingClientRect();
        return { local: overlay.parentElement === node, delta: ["x", "y", "width", "height"].map(key => Math.abs(a[key] - b[key])),
            viewport: a.width < innerWidth && a.height < innerHeight };
    });
    assert.ok(libraryBounds.local && libraryBounds.viewport && libraryBounds.delta.every(value => value <= 1));
    await library.getByRole("button", { name: "Library settings", exact: true }).click();
    await library.getByRole("link", { name: "Export", exact: true }).waitFor();
    await library.getByRole("button", { name: "Back to library" }).click();
    await library.getByRole("button", { name: "Save current set" }).click();
    await library.getByLabel("Name", { exact: true }).fill(setName);
    await library.getByLabel("Description").fill("Saved pose order and character previews.");
    if (process.argv[5]) {
        await library.getByLabel("Preview character").selectOption({ label: process.argv[5] + " · Naked" });
    }
    assert.ok(await library.locator(".vnccs-set-form").evaluate(form => {
        const a = form.getBoundingClientRect(), b = form.parentElement.getBoundingClientRect();
        return a.x >= b.x && a.right <= b.right && a.y >= b.y && a.bottom <= b.bottom;
    }));
    await shot("save-form");
    await library.getByRole("button", { name: "Save set", exact: true }).click();
    const card = library.locator(".vnccs-set-card").filter({ hasText: setName });
    await card.waitFor();
    await card.scrollIntoViewIfNeeded();
    await page.waitForFunction(name => {
        const images = [...document.querySelectorAll(".vnccs-set-card")].find(card => card.textContent.includes(name))?.querySelectorAll("img");
        return images?.length === 12 && [...images].every(img => img.complete && img.naturalWidth);
    }, setName);
    const cardReports = [await cardGeometry(card, 12)];
    await shot("library");
    await library.locator(".vnccs-ps-library-search").fill(setName);
    await page.waitForFunction(() => document.querySelectorAll(".vnccs-set-card").length === 1);
    for (const [name, width, height, zoom] of [["narrow", 900, 800, .75], ["default", 1210, 940, .85], ["wide", 1400, 1000, 1]]) {
        await page.evaluate(({ width, height, zoom }) => {
            poseNode.setSize([width, height]); app.canvas.ds.scale = zoom; app.canvas.centerOnNode(poseNode);
        }, { width, height, zoom });
        cardReports.push(await cardGeometry(card, 12));
        await shot(`card-${name}`);
    }
    for (const size of [160, 520, 320]) {
        await library.locator(".vnccs-ps-library-size-slider").evaluate((input, size) => {
            input.value = size; input.dispatchEvent(new Event("input", { bubbles: true }));
        }, size);
        cardReports.push(await cardGeometry(card, 12));
    }
    await page.evaluate(() => {
        const w = poseNode.studioWidget;
        w.loadPoseSetAsset({ type: "pose_set", poses: w.poses.slice(0, 2) });
    });
    await ready(2);
    const beforeLoad = await page.evaluate(() => JSON.stringify(poseNode.studioWidget.poses));
    const detailRoute = /\/vnccs\/pose-sets\/local\/[^/]+$/;
    await page.route(detailRoute, route => route.fulfill({ status: 500, json: { error: "Set is unavailable." } }));
    await card.getByRole("button", { name: `Load set ${setName}`, exact: true }).click();
    await library.getByRole("status").filter({ hasText: "Set is unavailable." }).waitFor();
    assert.equal(await page.evaluate(() => JSON.stringify(poseNode.studioWidget.poses)), beforeLoad);
    await page.unroute(detailRoute);
    await card.getByRole("button", { name: `Load set ${setName}`, exact: true }).click();
    await library.waitFor({ state: "detached" });
    await ready(12);
    assert.deepEqual(await page.evaluate(() => poseNode.studioWidget.poses.map(pose => pose.bones)), JSON.parse(saved.widgets_values[0]).poses.map(pose => pose.bones));
    const catalog = await (await page.request.get(`${url}/vnccs/pose-sets`)).json();
    const item = catalog.sets.find(item => item.name === setName);
    for (const count of [1, 2, 5, 6, 7, 8, 9, 10, 11, 13]) {
        await page.route("**/vnccs/pose-sets", route => route.fulfill({ json: { sets: [{ ...item, count }], repositories: [] } }));
        const extraPreview = `**/vnccs/pose-sets/${item.source}/${item.id}/preview/12`;
        await page.route(extraPreview, async route => route.fulfill({ response: await route.fetch({
            url: `${url}/vnccs/pose-sets/${item.source}/${item.id}/preview/0` }) }));
        await opener.click();
        await card.waitFor();
        cardReports.push(await cardGeometry(card, count));
        if (count === 10) {
            await page.waitForFunction(() => [...document.querySelectorAll(".vnccs-set-preview img")]
                .every(image => image.complete && image.naturalWidth));
            await shot("library-ten-poses");
        }
        await page.keyboard.press("Escape");
        await library.waitFor({ state: "detached" });
        await page.unroute("**/vnccs/pose-sets");
        await page.unroute(extraPreview);
    }
    await opener.click();
    await library.locator(".vnccs-ps-library-search").fill(setName);
    await card.waitFor();
    const beforeDelete = await page.evaluate(() => JSON.stringify(poseNode.studioWidget.poses));
    const deletionRequests = [], deletionLoads = [];
    const recordDeletion = request => {
        if (request.url().endsWith("/vnccs/pose-sets/delete")) deletionRequests.push(request.postDataJSON());
        if (detailRoute.test(request.url())) deletionLoads.push(request.url());
    };
    page.on("request", recordDeletion);
    const remove = card.getByRole("button", { name: `Delete set ${setName}`, exact: true });
    const confirmation = library.getByRole("dialog", { name: "Delete set", exact: true });
    for (const [name, width, height, zoom] of [["narrow", 900, 800, .75], ["default", 1210, 940, .85], ["wide", 1400, 1000, 1]]) {
        await page.evaluate(({ width, height, zoom }) => {
            poseNode.setSize([width, height]); app.canvas.ds.scale = zoom; app.canvas.centerOnNode(poseNode);
        }, { width, height, zoom });
        await remove.click();
        await confirmation.waitFor();
        await page.waitForFunction(() => document.activeElement?.textContent === "Cancel");
        await confirmation.evaluate(dialog => Promise.all(dialog.parentElement.getAnimations({ subtree: true })
            .map(animation => animation.finished.catch(() => {}))));
        assert.equal(await confirmation.getByRole("button", { name: "Cancel", exact: true }).evaluate(el => el === document.activeElement), true);
        assert.ok(await confirmation.evaluate(dialog => {
            const outer = poseNode.studioWidget.container.getBoundingClientRect();
            return [dialog, ...dialog.querySelectorAll("button")].every(el => {
                const box = el.getBoundingClientRect();
                return box.x >= outer.x && box.y >= outer.y && box.right <= outer.right && box.bottom <= outer.bottom;
            }) && dialog.scrollWidth <= dialog.clientWidth + 1 && dialog.scrollHeight <= dialog.clientHeight + 1;
        }));
        await shot(`delete-confirmation-${name}`);
        if (name === "default") await page.keyboard.press("Escape");
        else await confirmation.getByRole("button", { name: "Cancel", exact: true }).click();
        await confirmation.waitFor({ state: "detached" });
        assert.ok(await remove.evaluate(el => el === document.activeElement));
        assert.equal(await page.evaluate(() => JSON.stringify(poseNode.studioWidget.poses)), beforeDelete);
    }
    assert.deepEqual(deletionRequests, [], "Opening or cancelling deletion sends no mutation");
    await page.route("**/vnccs/pose-sets/delete", route => route.fulfill({ status: 500, json: { error: "Delete denied." } }));
    await remove.press("Enter");
    await confirmation.getByRole("button", { name: "Delete", exact: true }).click();
    await library.getByRole("status").filter({ hasText: "Delete denied." }).waitFor();
    assert.ok(await card.isVisible(), "A failed delete preserves the card");
    await shot("delete-error");
    await page.unroute("**/vnccs/pose-sets/delete");
    await remove.click();
    await confirmation.waitFor();
    await page.evaluate(() => poseNode.studioWidget.closeSetLibrary());
    await library.waitFor({ state: "detached" });
    assert.equal(await page.locator(".vnccs-common-modal-overlay").count(), 0);
    assert.ok(await opener.evaluate(el => el === document.activeElement));
    await opener.click();
    await library.locator(".vnccs-ps-library-search").fill(setName);
    await remove.click();
    await confirmation.getByRole("button", { name: "Delete", exact: true }).click();
    await card.waitFor({ state: "detached" });
    assert.deepEqual(deletionRequests, [{ id: item.id }, { id: item.id }]);
    assert.deepEqual(deletionLoads, [], "Delete never triggers a set load");
    page.off("request", recordDeletion);
    assert.equal(await page.evaluate(() => JSON.stringify(poseNode.studioWidget.poses)), beforeDelete,
        "Deleting a saved set does not change the currently loaded poses");
    assert.ok(!(await (await page.request.get(`${url}/vnccs/pose-sets`)).json()).sets.some(set => set.id === item.id));
    await page.keyboard.press("Escape");
    await library.waitFor({ state: "detached" });
    assert.ok(await opener.evaluate(el => el === document.activeElement));
    const promptCheck = await page.evaluate(() => {
        const w = poseNode.studioWidget;
        const nativePrompt = Object.getPrototypeOf(w).generatePromptFromLights;
        w.exportParams.keepOriginalLighting = false;
        const lights = JSON.stringify(w.lightParams);
        const sceneLights = () => {
            const result = [];
            w.viewer.scene.traverse(object => {
                if (object.isLight) result.push([object.type, object.intensity, object.color.toArray(), object.position.toArray()]);
            });
            return result;
        };
        const before = sceneLights();
        const userPrompt = "Keep the authored lighting words and $& symbols.";
        w.setPosePrompt(w.activeTab, userPrompt);
        const cases = [];
        for (const template of ["", "Portrait\n<lighting>\n<user_prompt>\n<lighting>\nEnd", "<user_prompt>"]) {
            w.exportParams.prompt_template = template;
            const actual = w.generatePromptFromLights(w.lightParams);
            cases.push({ template, preservedTemplate: w.exportParams.prompt_template,
                native: nativePrompt.call(w, w.lightParams, userPrompt),
                actual,
                expected: nativePrompt.call(w, [], userPrompt) });
        }
        w.exportParams.prompt_template = "Portrait\n<lighting>\n<user_prompt>\nEnd";
        const expected = w.poses.map((_, index) => {
            const prompt = `Pose ${index + 1}: ${userPrompt}`;
            w.setPosePrompt(index, prompt);
            return `Portrait\n${prompt}\nEnd`;
        });
        w.syncToNode(false, { skipCaptureUpload: true });
        const saved = poseNode.serialize(), data = JSON.parse(saved.widgets_values[0]);
        data.lighting_prompts = expected.map(() => nativePrompt.call(w, w.lightParams, userPrompt));
        saved.widgets_values[0] = JSON.stringify(data);
        return { cases, expected, saved, lights, lightsAfter: JSON.stringify(w.lightParams), before, after: sceneLights() };
    });
    for (const item of promptCheck.cases) {
        assert.equal(item.actual, item.expected);
        assert.equal(item.preservedTemplate, item.template);
        if (item.template !== "<user_prompt>") assert.notEqual(item.actual, item.native, "Original Studio still includes lighting");
    }
    assert.ok(promptCheck.before.length > 0, "The scene contains rendered lights");
    assert.deepEqual(promptCheck.after, promptCheck.before);
    assert.equal(promptCheck.lightsAfter, promptCheck.lights);
    await page.evaluate(saved => mount(saved), promptCheck.saved);
    await ready(12);
    assert.deepEqual(await page.evaluate(() => poseNode.studioWidget.lightingPrompts), promptCheck.expected,
        "Restoring an older workflow replaces cached light descriptions");
    await shot("prompts-without-lighting");
    const ids = await page.evaluate(async () => {
        const images = LiteGraph.createNode("PreviewImage"); app.graph.add(images); images.pos = [4000, 80]; poseNode.connect(0, images, 0);
        const prompts = LiteGraph.createNode("PreviewAny"); app.graph.add(prompts); prompts.pos = [4000, 300]; poseNode.connect(1, prompts, 0);
        const { api } = await import("/scripts/api.js");
        return { images: images.id, prompts: prompts.id, prompt: (await api.queuePrompt(0, await app.graphToPrompt())).prompt_id };
    });
    let history;
    const deadline = Date.now() + 180000;
    while (!history && Date.now() < deadline) {
        history = (await (await page.request.get(`${url}/history/${ids.prompt}`)).json())[ids.prompt];
        if (!history) await new Promise(resolve => setTimeout(resolve, 500));
    }
    assert.equal(history?.status.status_str, "success", JSON.stringify(history?.status));
    assert.equal(history.outputs[ids.images].images.length, 12);
    assert.equal(history.outputs[ids.prompts].text.length, 12);
    assert.deepEqual(history.outputs[ids.prompts].text, promptCheck.expected);
    assert.equal(await page.evaluate(() => JSON.stringify(poseNode.studioWidget.lightParams)), promptCheck.lights);
    const damaged = structuredClone(saved); damaged.widgets_values[0] = '{"poses":';
    await page.evaluate(data => mount(data), damaged);
    await page.waitForFunction(() => Boolean(poseNode.studioWidget._poseDataRestoreError));
    assert.equal(await page.evaluate(() => poseNode.serialize().widgets_values[0]), damaged.widgets_values[0]);
    await page.evaluate(() => app.graph.clear());
    assert.deepEqual(errors, []);
    const version = (await readFile(new URL("../../web/vnccs_pose_manager.js", import.meta.url), "utf8"))
        .match(/pose_manager_ui\.mjs\?v=([^"\s]+)/)[1];
    assert.equal(moduleURLs.size, 5);
    assert.ok([...moduleURLs].every(url => new URL(url).searchParams.get("v") === version));
    await writeFile(join(output, "results.json"), JSON.stringify({ reports, libraryBounds, cardReports, pixels, executedImages: 12 }, null, 2));
    console.log("PASS: native single canvas, zero slider encodes, 12-pose workflows, node-local library, save/load, real execution, damaged state preservation.");
} catch (error) {
    console.error(await page.evaluate(() => ({ mode: window.poseNode?.studioWidget?.interfaceMode,
        error: window.poseNode?.studioWidget?._managerPreviewRefreshError,
        previews: window.poseNode?.studioWidget && {
            generation: poseNode.studioWidget._managerPreviewRefreshGeneration,
            completed: poseNode.studioWidget._managerPreviewRefreshCompletedGeneration,
            next: poseNode.studioWidget._managerPreviewRefreshNextIndex,
            canvases: poseNode.studioWidget.managerGrid.querySelectorAll("canvas.vnccs-pm-live").length,
        }, body: document.body.innerText.slice(-1800) })), errors);
    await shot("failure");
    throw error;
} finally { await browser.close(); }
