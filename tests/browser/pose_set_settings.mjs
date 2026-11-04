import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

export async function verifyPoseSetSettings(page, output) {
    const reports = [], requests = [], repositories = [];
    let failRefresh = false;
    const base = "**/vnccs/pose-sets";
    const mutations = "**/vnccs/pose-sets/repository";
    await page.route(base, async route => {
        const response = await route.fetch();
        await route.fulfill({ json: { ...await response.json(), repositories } });
    });
    await page.route(mutations, async route => {
        const data = route.request().postDataJSON();
        requests.push(data);
        if (data.action === "refresh" && failRefresh)
            return route.fulfill({ status: 400, json: { error: "Repository is unavailable; saved sets are preserved." } });
        if (data.action === "download") repositories.push({ id: "b".repeat(32), repo_id: data.repo_id, count: 3, enabled: true });
        if (data.action === "toggle") repositories[0].enabled = data.enabled;
        if (data.action === "remove") repositories.length = 0;
        const response = await page.request.get(new URL("/vnccs/pose-sets", page.url()).href);
        await route.fulfill({ json: { ...await response.json(), repositories } });
    });
    const library = page.getByRole("dialog", { name: "Library", exact: true });
    const inspect = () => page.evaluate(() => {
        const root = poseNode.studioWidget.container;
        const modal = root.querySelector(".vnccs-ps-library-modal");
        const panel = modal.querySelector(".vnccs-ps-library-settings");
        const box = panel.getBoundingClientRect(), node = root.getBoundingClientRect();
        const selectors = [".vnccs-ps-library-settings-head", ".vnccs-ps-library-settings-back",
            ".vnccs-ps-library-local-repo", ".vnccs-ps-library-repo-add", ".vnccs-ps-library-repo-input"];
        return {
            rows: selectors.map(selector => {
                const el = panel.querySelector(selector), r = el.getBoundingClientRect();
                return [r.x - box.x, r.y - box.y, r.width, r.height];
            }),
            controlsFit: panel.scrollHeight <= panel.clientHeight + 1,
            inside: [...modal.querySelectorAll("button,input,a")].filter(el => el.getBoundingClientRect().width).every(el => {
                const r = el.getBoundingClientRect();
                return r.x >= node.x - 1 && r.y >= node.y - 1 && r.right <= node.right + 1 && r.bottom <= node.bottom + 1;
            }),
            sizeInToolbar: !!modal.querySelector(".vnccs-ps-library-toolbar .vnccs-ps-library-size-slider"),
            customForm: !!panel.querySelector(".vnccs-set-form"),
        };
    });
    for (const [name, width, height, zoom] of [["narrow", 900, 800, .75], ["default", 1210, 940, .85], ["wide", 1400, 1000, 1]]) {
        await page.evaluate(({ width, height, zoom }) => {
            poseNode.setSize([width, height]); app.canvas.ds.scale = zoom; app.canvas.centerOnNode(poseNode);
        }, { width, height, zoom });
        await page.getByRole("button", { name: "Library", exact: true }).click();
        await library.getByRole("button", { name: "Library settings", exact: true }).click();
        await library.getByRole("link", { name: "Export", exact: true }).waitFor();
        const actual = await inspect();
        assert.ok(actual.inside && actual.controlsFit && actual.sizeInToolbar && !actual.customForm, JSON.stringify(actual));
        await page.screenshot({ path: join(output, `settings-${name}.png`) });
        await library.getByRole("button", { name: "Close library" }).click();
        // Render the dependency's unmodified settings in the same node for geometry parity.
        await page.evaluate(async () => {
            const w = poseNode.studioWidget;
            const host = window.settingsReference = Object.create(Object.getPrototypeOf(w));
            Object.assign(host, { container: w.container, repositoryProgressStates: {}, poseRepositories: [], localPoseRepository: {},
                libraryThumbSize: 320, isAnimationMode: () => false, refreshLibrary: () => {},
                refreshPoseRepositories() { this.renderPoseRepositorySettings(); } });
            host.showLibraryModal();
            host.libraryCategoriesEl.style.display = "none";
            await host.toggleLibrarySettings(true);
            await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        });
        const reference = await inspect();
        assert.ok(reference.inside && reference.controlsFit);
        for (let i = 0; i < actual.rows.length; i++) actual.rows[i].forEach((value, j) =>
            assert.ok(Math.abs(value - reference.rows[i][j]) <= 1, `${name} native row ${i}:${j}: ${value} / ${reference.rows[i][j]}`));
        await page.screenshot({ path: join(output, `settings-native-${name}.png`) });
        await page.evaluate(() => settingsReference._closeLibraryModal());
        reports.push({ name, actual, reference });
    }
    await page.getByRole("button", { name: "Library", exact: true }).click();
    await library.getByRole("button", { name: "Library settings", exact: true }).click();
    const input = library.locator(".vnccs-ps-library-repo-input");
    await input.fill("https://huggingface.co/owner/pose-sets/");
    await input.press("Enter");
    const remote = library.locator(".vnccs-ps-library-repo-list .vnccs-ps-library-repo-card");
    await remote.getByRole("button", { name: "Disable", exact: true }).waitFor();
    assert.equal(requests.at(-1).repo_id, "owner/pose-sets");
    assert.match(await remote.innerText(), /3 sets/);
    await remote.getByRole("button", { name: "Disable", exact: true }).click();
    await remote.getByRole("button", { name: "Enable", exact: true }).click();
    await remote.getByRole("button", { name: "Disable", exact: true }).waitFor();
    failRefresh = true;
    await remote.getByRole("button", { name: "Refresh", exact: true }).click();
    await library.locator(".vnccs-ps-library-repo-notice.error.visible").waitFor();
    assert.equal(await remote.count(), 1);
    assert.ok((await inspect()).controlsFit);
    await page.screenshot({ path: join(output, "settings-error.png") });
    failRefresh = false;
    await remote.getByRole("button", { name: "Refresh", exact: true }).click();
    await library.locator(".vnccs-ps-library-repo-notice.error.visible").waitFor({ state: "hidden" });
    await page.screenshot({ path: join(output, "settings-populated.png") });
    const downloadPromise = page.waitForEvent("download");
    await library.getByRole("link", { name: "Export", exact: true }).click();
    const download = await downloadPromise;
    assert.equal(download.suggestedFilename(), "pose_sets.json");
    assert.equal(JSON.parse(await readFile(await download.path(), "utf8")).version, 1);
    await remote.getByRole("button", { name: "Remove", exact: true }).click();
    await remote.waitFor({ state: "detached" });
    await library.getByRole("button", { name: "Back to library" }).click();
    assert.ok(await library.locator(".vnccs-ps-library-search").isEnabled());
    await library.getByRole("button", { name: "Library settings", exact: true }).click();
    await library.getByRole("link", { name: "Export", exact: true }).waitFor();
    await library.getByRole("button", { name: "Library settings", exact: true }).click();
    assert.ok(await library.locator(".vnccs-ps-library-search").isEnabled());
    await page.keyboard.press("Escape");
    await library.waitFor({ state: "detached" });
    await page.unroute(base);
    await page.unroute(mutations);
    await writeFile(join(output, "settings-results.json"), JSON.stringify({ reports, requests }, null, 2));
}
