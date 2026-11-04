import { checkedJSON, mediaURL } from "./vnccs_transport.js";
import { showModal } from "./vnccs_common.js";

const BASE = "/vnccs/pose-sets";
const post = (action, data) => checkedJSON(`${BASE}/${action}`, { method: "POST", body: JSON.stringify(data) });
function element(tag, className = "", text) {
    const node = document.createElement(tag);
    node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}
function button(text, action) {
    const node = element("button", "vnccs-ps-btn", text);
    node.type = "button";
    node.onclick = action;
    return node;
}
async function thumbnail(source) {
    const image = typeof source === "string" ? new Image() : source;
    if (!image) throw new Error("Wait for the pose previews to finish loading.");
    if (typeof source === "string") { image.src = source; await image.decode(); }
    const width = image.naturalWidth || image.width, height = image.naturalHeight || image.height;
    const scale = Math.min(1, 240 / width, 360 / height);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/webp", .86).split(",")[1];
}

export function openSetLibrary(w, opener, previewFor) {
    if (w.closeSetLibrary) return;
    // The native shell owns geometry and cleanup; only its content and data source differ.
    const host = Object.create(Object.getPrototypeOf(w));
    host.container = w.container;
    host.isAnimationMode = () => false;
    host.libraryThumbSizeStorageKey = "vnccsPoseSetPreviewSize";
    host.libraryThumbSize = host.loadLibraryThumbnailSize();
    host.repositoryProgressStates = {};
    let overlay, confirmation, screen = "list", busy = false, disposed = false;
    let catalog = { sets: [], repositories: [] }, listScroll = 0;
    const status = element("div", "vnccs-set-status");
    status.setAttribute("role", "status");
    const wasInert = w.managerPanel.inert;
    const active = () => !disposed && overlay?.isConnected;
    function close() {
        if (disposed) return;
        disposed = true;
        confirmation?.remove();
        host._closeLibraryModal?.();
        w.managerPanel.inert = wasInert;
        w.closeSetLibrary = null;
        if (opener.isConnected) opener.focus({ preventScroll: true });
    }
    async function run(action) {
        if (busy || !active()) return;
        busy = true;
        const focus = document.activeElement;
        host.libraryModal.inert = true;
        status.textContent = "Working…";
        try {
            await action();
            if (active()) status.textContent = "";
        } catch (error) {
            if (active()) {
                status.textContent = "";
                if (screen === "settings" && host.librarySettingsEl.querySelector(".vnccs-ps-library-repo-notice"))
                    host.showRepositoryNotice(error.message || String(error), true);
                else status.textContent = error.message || String(error);
            }
        } finally {
            busy = false;
            if (active()) {
                host.libraryModal.inert = false;
                (host.libraryModal.contains(focus) && !focus.disabled ? focus
                    : host.librarySettingsEl.querySelector("button") || host.librarySearchInput).focus({ preventScroll: true });
            }
        }
    }
    async function reload() {
        const next = await checkedJSON(BASE);
        if (!active()) return;
        catalog = next;
        render();
    }
    function render() {
        if (!active() || screen !== "list") return;
        const grid = host.libraryGrid;
        const scroll = grid.scrollTop;
        const search = host.librarySearchInput.value.trim().toLowerCase();
        grid.replaceChildren();
        for (const item of catalog.sets) {
            if (!`${item.name} ${item.description}`.toLowerCase().includes(search)) continue;
            const path = `${BASE}/${item.source}/${item.id}`;
            const card = element("div", "vnccs-set-card");
            const load = button("", () => run(async () => {
                const data = await checkedJSON(path);
                if (!active()) return;
                w.loadPoseSetAsset(data.asset);
                w.setInterfaceMode("manager");
                close();
            }));
            load.className = "vnccs-set-load";
            load.setAttribute("aria-label", `Load set ${item.name}`);
            const preview = element("div", "vnccs-set-preview");
            const rows = Math.ceil(item.count / 6);
            const columns = Math.ceil(item.count / rows);
            preview.style.setProperty("--set-columns", columns);
            preview.style.setProperty("--set-rows", rows);
            for (let i = 0; i < item.count; i++) preview.append(previewImage(path, i));
            const label = element("div", "vnccs-set-label");
            label.append(element("strong", "", item.name), element("span", "", `${item.count} ${item.count === 1 ? "pose" : "poses"}`));
            load.append(preview, label, element("p", "", item.description));
            card.append(load);
            if (item.source === "local") {
                const remove = button("×", () => confirmDelete(item));
                remove.className = "vnccs-set-delete";
                remove.title = "Delete set";
                remove.setAttribute("aria-label", `Delete set ${item.name}`);
                card.append(remove);
            }
            grid.append(card);
        }
        if (!grid.childElementCount) grid.append(element("p", "vnccs-set-empty", search ? "No matching sets." : "No saved sets yet."));
        grid.scrollTop = scroll;
    }
    function previewImage(path, index) {
        const image = element("img");
        image.alt = `Pose ${index + 1}`;
        image.loading = "lazy";
        image.src = mediaURL(`${path}/preview/${index}`);
        return image;
    }
    function back() {
        if (!active()) return;
        screen = "list";
        host.librarySettingsMode = false;
        host.librarySearchInput.disabled = false;
        host.librarySearchInput.placeholder = "Search sets";
        host.libraryWorkspace.classList.remove("settings-mode");
        host.librarySettingsEl.replaceChildren();
        status.textContent = "";
        render();
        host.libraryGrid.scrollTop = listScroll;
        host.librarySearchInput.focus({ preventScroll: true });
    }
    function panel(mode, title) {
        if (screen === "list") listScroll = host.libraryGrid.scrollTop;
        screen = mode;
        host.librarySettingsMode = false;
        host.librarySearchInput.disabled = false;
        status.textContent = "";
        host.libraryWorkspace.classList.add("settings-mode");
        const box = element("section", "vnccs-set-form");
        box.append(button("← Back", back), element("h2", "", title));
        host.librarySettingsEl.replaceChildren(box);
        box.querySelector("button").focus({ preventScroll: true });
        return box;
    }
    function confirmDelete(item) {
        if (busy || confirmation || !active()) return;
        host.libraryModal.inert = true;
        const dialog = showModal(overlay, "Delete set", () => element("div", "",
            `Delete "${item.name}"? This cannot be undone.`), [
            { text: "Cancel", autofocus: true },
            { text: "Delete", class: "danger", action: async () => {
                dialog.overlay.remove();
                await run(async () => {
                    catalog = await post("delete", { id: item.id });
                    render();
                });
            } },
        ]);
        confirmation = dialog.overlay;
        const remove = confirmation.remove;
        confirmation.remove = function () {
            if (confirmation === this) {
                confirmation = null;
                host.libraryModal.inert = false;
            }
            remove.call(this);
        };
    }
    function field(form, caption, input) {
        const label = element("label", "", caption);
        label.append(input);
        form.append(label);
        return input;
    }
    function save() {
        const box = panel("save", "Save current set");
        const form = element("form");
        const name = field(form, "Name", element("input"));
        name.required = true;
        name.maxLength = 160;
        const description = field(form, "Description", element("textarea"));
        description.rows = 2;
        description.maxLength = 4000;
        const source = field(form, "Preview character", element("select"));
        const fallback = element("option", "", "Current mannequins");
        fallback.value = "";
        source.append(fallback);
        form.append(element("small", "", "Character sprites are copied in pose order, one preview for each pose."));
        const submit = button("Save set");
        submit.type = "submit";
        form.append(submit);
        box.append(form);
        checkedJSON(`${BASE}/preview-sources`).then(items => {
            if (!active() || !source.isConnected) return;
            for (const item of items) {
                const option = element("option", "", `${item.character} · ${item.costume}`);
                option.value = JSON.stringify({ character: item.character, costume: item.costume });
                option.disabled = item.count < w.poses.length;
                source.append(option);
            }
        }).catch(error => { if (active() && source.isConnected) status.textContent = error.message; });
        form.onsubmit = event => {
            event.preventDefault();
            run(async () => {
                await w.awaitCurrentManagerPreviews();
                if (!active()) return;
                const poses = w.poses.map((pose, i) => ({ ...structuredClone(pose), prompt: w.getPosePrompt(i) }));
                const previews = source.value ? [] : await Promise.all(poses.map((_, i) => thumbnail(previewFor(i))));
                if (!active()) return;
                catalog = await post("save", { name: name.value, description: description.value,
                    asset: { type: "pose_set", poses }, previews,
                    preview_source: source.value ? JSON.parse(source.value) : null });
                back();
            });
        };
    }
    const renderSettings = host.renderPoseRepositorySettings;
    host.renderPoseRepositorySettings = () => {
        host.poseRepositories = catalog.repositories;
        host.localPoseRepository = {};
        renderSettings.call(host);
        const panel = host.librarySettingsEl;
        panel.querySelector(".vnccs-ps-library-settings-subtitle").textContent =
            "Pose set libraries on Hugging Face can be enabled, disabled, refreshed, or removed.";
        const local = panel.querySelector(".vnccs-ps-library-local-repo");
        local.querySelector(".vnccs-ps-library-repo-title").textContent = "Local Pose Set Library";
        local.querySelector(".vnccs-ps-library-repo-id").textContent = "PoseSetLibrary/local";
        local.querySelector(".vnccs-ps-library-repo-meta").textContent =
            `${catalog.sets.filter(item => item.source === "local").length} sets`;
        // Keep the native card; this library exports its manifest without remote publishing.
        const publish = local.querySelector(".publish");
        const download = element("a", publish.className, "Export");
        download.href = mediaURL(`${BASE}/export`);
        download.download = "pose_sets.json";
        download.title = "Export pose_sets.json for a public Hugging Face model repository";
        publish.replaceWith(download);
        for (const repo of catalog.repositories) {
            host.findRepositoryProgressCard(`repo:${repo.repo_id}`).querySelector(".vnccs-ps-library-repo-meta").textContent =
                `${repo.count} sets · ${repo.enabled ? "enabled" : "disabled"} · downloaded`;
        }
    };
    host.refreshPoseRepositories = () => run(async () => {
        await reload();
        if (active() && screen === "settings") host.renderPoseRepositorySettings();
    });
    function repositoryAction(action, repoId, enabled) {
        return run(async () => {
            const repo = catalog.repositories.find(item => item.repo_id === repoId);
            catalog = await post("repository", { action, repo_id: repoId, id: repo?.id, enabled });
            if (active() && screen === "settings") host.renderPoseRepositorySettings();
        });
    }
    host.addPoseRepository = () => repositoryAction("download", host.librarySettingsEl
        .querySelector(".vnccs-ps-library-repo-input").value.trim().replace(/^https?:\/\/huggingface\.co\//i, "").replace(/\/+$/, ""));
    host.togglePoseRepository = (repoId, enabled) => repositoryAction("toggle", repoId, enabled);
    host.refreshSinglePoseRepository = repoId => repositoryAction("refresh", repoId);
    host.removePoseRepository = repoId => repositoryAction("remove", repoId);
    const toggleSettings = host.toggleLibrarySettings;
    host.toggleLibrarySettings = async (force = null) => {
        if (!(force ?? screen !== "settings")) { back(); return; }
        if (screen === "list") listScroll = host.libraryGrid.scrollTop;
        screen = "settings";
        await toggleSettings.call(host, true);
        if (active() && screen === "settings") host.librarySettingsEl.querySelector("button")?.focus({ preventScroll: true });
    };
    host.refreshLibrary = () => run(reload);
    host.renderLibrary = render;
    host.showSaveToLibraryModal = save;
    host.showLibraryModal();
    overlay = host.libraryModal.parentElement;
    overlay.classList.add("vnccs-set-library");
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-label", "Library");
    const header = host.libraryModal.querySelector(".vnccs-ps-library-modal-header");
    header.querySelector(".vnccs-ps-library-modal-title").textContent = "Library";
    host.librarySearchInput.placeholder = "Search sets";
    header.querySelector(".vnccs-ps-library-save-current").textContent = "Save current set";
    const gear = host.libraryModal.querySelector(".vnccs-ps-library-menu-btn");
    gear.title = "Library settings";
    gear.setAttribute("aria-label", "Library settings");
    const closeButton = header.querySelector(".vnccs-ps-modal-close");
    closeButton.setAttribute("aria-label", "Close library");
    closeButton.onclick = close;
    overlay.onclick = event => { if (event.target === overlay) close(); };
    overlay.addEventListener("keydown", event => {
        if (event.key === "Escape") { event.stopPropagation(); close(); }
    });
    header.after(status);
    w.managerPanel.inert = true;
    w.closeSetLibrary = close;
    host.librarySearchInput.focus();
    run(reload);
}
