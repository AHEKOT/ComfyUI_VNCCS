import { api } from "../../scripts/api.js";
import { cacheIdentity, checkedJSON, mediaURL } from "./vnccs_transport.js";
import { createLoadingOverlay } from "./vnccs_common.js";
import { packPreviewRows } from "./preview_grid_layout.mjs?v=pose-manager-15";

const BASE = "/vnccs/pose-manager/sprites";
let helpersPromise;
async function studioHelpers() {
    if (!helpersPromise) helpersPromise = (async () => {
        const extensions = await api.getExtensions();
        const entry = extensions.find(path => path.split("?")[0].endsWith("/vnccs_pose_studio.js"));
        if (!entry) throw new Error("Pose Studio is unavailable.");
        return import(new URL("pose_studio/characters.mjs", new URL(entry, location.href)).href);
    })().catch(error => { helpersPromise = null; throw error; });
    return helpersPromise;
}

async function applyPose(widget, item, isCurrent) {
    const url = item.manager_sprite_id
        ? `${BASE}/${item.manager_sprite_id}?repository=${encodeURIComponent(item.repository)}`
        : `/vnccs/pose_library/get/${encodeURIComponent(widget.getLibraryPoseName(item))}${widget.getLibraryPoseQuery(item)}`;
    const [{ pose: asset, asset_type: type }, helpers] = await Promise.all([
        checkedJSON(url), studioHelpers(),
    ]);
    if (!isCurrent()) return;
    if (asset?.type === "pose_set" || asset?.animation || type === "animation") throw new Error("Choose a single pose for this card.");
    const source = helpers.extractActivePoseFromSceneAsset(asset) || asset;
    widget.validateImportedPose(source);
    // Import skeletal pose only; scene placement, camera, appearance and text stay local.
    const pose = structuredClone(Object.fromEntries([
        "bones", "bonePositions", "modelRotation", "ikEffectorPositions", "poleTargetPositions", "hipBonePosition",
    ].filter(key => source[key] !== undefined).map(key => [key, source[key]])));
    widget.viewer.recordState();
    widget.viewer.setPose(pose, true);
    widget.updateRotationSliders();
    widget.commitViewerPoseToCurrentEditor();
}

export const POSE_LIBRARY_CSS = `
.vnccs-pm-pose-picker .vnccs-ps-library-categories { flex-wrap:wrap; align-items:center; }
.vnccs-pm-pose-sources { display:flex; gap:6px; margin-left:auto; }
.vnccs-pm-pose-status { padding:0 18px 8px; font-size:12px; color:var(--text-secondary); }
.vnccs-pm-pose-picker .vnccs-ps-library-local-repo,.vnccs-pm-pose-picker .vnccs-ps-library-save-current { display:none; }
.vnccs-pm-pose-status:empty { display:none; }
.vnccs-pm-pose-picker .vnccs-ps-library-workspace { position:relative; }
.vnccs-pm-pose-picker .vnccs-pm-library-loading { box-sizing:border-box; padding:24px; text-align:center; gap:16px; background:var(--bg-primary,#0a0a0f); }
.vnccs-pm-library-loading[hidden],.vnccs-pm-pose-status[hidden] { display:none; }
.vnccs-pm-library-loading [hidden] { display:none !important; }
.vnccs-pm-library-loading .vnccs-common-spinner { border-top-color:var(--accent,#ff8fa3); }
.vnccs-pm-library-loading .vnccs-common-loading-text { color:var(--text-primary); font-size:20px; font-weight:600; }
.vnccs-pm-library-loading p { margin:0; max-width:440px; color:var(--text-secondary); font-size:14px; line-height:1.6; overflow-wrap:anywhere; white-space:pre-line; }
.vnccs-pm-library-loading progress { width:min(320px,85%); height:8px; accent-color:var(--accent,#ff8fa3); }
@media (prefers-reduced-motion:reduce) { .vnccs-pm-library-loading .vnccs-common-spinner { animation:none; } }
.vnccs-pm-pose-picker .vnccs-ps-library-workspace:not(.settings-mode) .vnccs-ps-library-modal-grid { display:flex; flex-wrap:wrap; justify-content:center; align-content:flex-start; align-items:flex-start; gap:8px; overflow-anchor:none; }
.vnccs-pm-pose-picker .vnccs-ps-library-item { box-sizing:border-box; flex:none; width:calc(var(--vnccs-ps-library-thumb-size,320px) * var(--pose-aspect,.416667)); max-width:100%; min-height:0; transition:border-color var(--ps-transition); }
.vnccs-pm-pose-picker .vnccs-ps-library-item:hover { transform:none; }
.vnccs-pm-pose-picker .vnccs-ps-library-item-preview { flex:none; aspect-ratio:var(--pose-aspect,.416667); border-radius:8px 8px 0 0; }
.vnccs-pm-pose-picker .vnccs-ps-library-item-preview img { width:100%; height:100%; }
.vnccs-pm-pose-picker .vnccs-ps-library-item-name { position:static; font-size:max(11px,calc(11px * var(--vnccs-ps-library-ui-scale))); }
`;

export function mountPoseLibrary(widget) {
    const show = widget.showLibraryModal;
    const selection = { source: "Manager", categories: { Manager: "All", "Pose Studio": "All" } };
    widget.showLibraryModal = function () { return openPoseLibrary(this, show, selection); };
    for (const name of ["showImportProgressModal", "showMessage"]) {
        const original = widget[name];
        widget[name] = function () {
            const previous = this.canvasContainer.lastElementChild;
            const result = original.apply(this, arguments);
            const modal = this.canvasContainer.lastElementChild;
            // Manager hides Studio's canvas panel, so its native import dialogs need the visible root.
            if (this.interfaceMode === "manager" && modal !== previous && modal?.matches(".vnccs-ps-modal-overlay"))
                this.container.append(modal);
            return result;
        };
    }
    // Studio is a read-only source; Manager must not start its repository refresh jobs.
    widget.autoRefreshEnabledPoseRepositories = () => {};
}

function openPoseLibrary(widget, show, selection) {
    show.call(widget);
    const index = widget.activeTab;
    const overlay = widget.libraryModal.parentElement;
    const nativeClose = widget._closeLibraryModal;
    const original = Object.fromEntries(["renderLibraryCategories", "renderLibrary", "selectLibraryPose",
        "getLibraryPreviewUrl", "refreshPoseRepositories", "togglePoseRepository", "refreshSinglePoseRepository",
        "addPoseRepository", "removePoseRepository", "renderPoseRepositorySettings", "renderLocalPoseRepositorySettings",
        "applyLibraryThumbnailSize", "updateLibraryLayoutScale"]
        .map(key => [key, widget[key]]));
    const wasInert = widget.managerPanel.inert, character = widget.activeCharacterId;
    widget.managerPanel.inert = true;
    widget._poseCardLibraryOpen = true;
    let closed = false, applying = false, busy = false, source = selection.source, catalog = null;
    let cards = [], layoutFrame = null;
    const schedulePacking = () => {
        if (closed || layoutFrame !== null) return;
        layoutFrame = requestAnimationFrame(() => {
            layoutFrame = null;
            const grid = widget.libraryGrid;
            if (closed || widget.librarySettingsMode || !grid?.clientWidth || !cards.length) return;
            const style = getComputedStyle(grid);
            const width = grid.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
            const widths = cards.map(card => parseFloat(getComputedStyle(card).width));
            const order = packPreviewRows(widths, width, parseFloat(style.columnGap)).flat();
            const focused = document.activeElement, top = grid.scrollTop, left = grid.scrollLeft;
            // Move the existing cards so keyboard navigation follows the packed rows too.
            order.forEach((index, position) => {
                if (grid.children[position] !== cards[index]) grid.insertBefore(cards[index], grid.children[position]);
            });
            if (grid.contains(focused) && document.activeElement !== focused) focused.focus({ preventScroll: true });
            grid.scrollTop = top;
            grid.scrollLeft = left;
        });
    };
    for (const name of ["applyLibraryThumbnailSize", "updateLibraryLayoutScale"]) {
        widget[name] = function () {
            const result = original[name].apply(this, arguments);
            schedulePacking();
            return result;
        };
    }
    const current = () => !closed && !widget._disposed && widget.activeCharacterId === character && widget.activeTab === index;
    const status = document.createElement("div");
    status.className = "vnccs-pm-pose-status";
    status.setAttribute("role", "status");
    widget.libraryCategoriesEl.after(status);
    const loading = createLoadingOverlay(widget.libraryWorkspace, "Loading pose library");
    loading.overlay.classList.add("vnccs-pm-library-loading");
    loading.overlay.setAttribute("role", "status");
    const loadingTitle = loading.overlay.querySelector(".vnccs-common-loading-text");
    const spinner = loading.overlay.querySelector(".vnccs-common-spinner");
    const detail = document.createElement("p"), progress = document.createElement("progress"), retry = document.createElement("button");
    progress.setAttribute("aria-label", "Poses ready");
    retry.type = "button";
    retry.className = "vnccs-ps-btn primary";
    retry.textContent = "Retry";
    retry.onclick = () => updateRepositories("ensure");
    loading.overlay.append(progress, detail, retry);
    let loadError = "", progressInfo = null, requestId = "";
    const renderLoading = () => {
        const initial = source === "Manager" && !catalog && !widget.librarySettingsMode;
        loading.overlay.hidden = !initial;
        status.hidden = initial;
        widget.libraryGrid.setAttribute("aria-busy", String(initial && !loadError));
        if (initial) { widget.libraryGrid.replaceChildren(); cards = []; }
        loadingTitle.textContent = loadError ? "Could not load poses" : progressInfo?.total ? "Preparing poses" : "Loading pose library";
        spinner.hidden = !!loadError;
        progress.hidden = !!loadError;
        retry.hidden = !loadError;
        detail.textContent = loadError || (progressInfo?.total
            ? `${progressInfo.completed} of ${progressInfo.total} poses ready\n${progressInfo.repository}`
            : "Preparing your pose library. Poses will appear here when ready.");
        if (progressInfo?.total) {
            progress.max = progressInfo.total;
            progress.value = progressInfo.completed;
        } else progress.removeAttribute("value");
    };
    const onProgress = event => {
        if (!current() || !busy || event.detail?.request_id !== requestId) return;
        progressInfo = event.detail;
        status.textContent = progressInfo.total
            ? `Preparing poses: ${progressInfo.completed} of ${progressInfo.total} ready · ${progressInfo.repository}`
            : `Loading repository: ${progressInfo.repository}`;
        renderLoading();
    };
    api.addEventListener("vnccs.pose_manager.library.progress", onProgress);
    widget.renderLocalPoseRepositorySettings = () => {};
    widget.renderPoseRepositorySettings = function () {
        const repositories = widget.poseRepositories;
        try {
            widget.poseRepositories = catalog?.repositories || [];
            original.renderPoseRepositorySettings.call(this);
            widget.librarySettingsEl.querySelector(".vnccs-ps-library-settings-title").textContent = "Manager Repositories";
            widget.librarySettingsEl.querySelector(".vnccs-ps-library-settings-subtitle").textContent = "Pose repositories used by Pose Manager. Pose Studio has its own repositories.";
            if (busy) widget.librarySettingsEl.querySelectorAll(".vnccs-ps-library-repo-action,.vnccs-ps-library-repo-add-btn").forEach(button => { button.disabled = true; });
            renderLoading();
        } finally { widget.poseRepositories = repositories; }
    };
    const updateRepositories = async (action, repo_id, enabled) => {
        if (busy || closed) return;
        const focused = widget.librarySettingsEl.contains(document.activeElement)
            ? (document.activeElement.classList.contains("refresh") ? ".refresh" : ".toggle") : null;
        busy = true;
        loadError = "";
        progressInfo = null;
        requestId = cacheIdentity();
        status.textContent = action ? "Updating Manager repositories…" : "Loading Manager repositories…";
        renderLoading();
        if (widget.librarySettingsMode) widget.renderPoseRepositorySettings();
        if (focused) overlay.focus({ preventScroll: true });
        try {
            const data = await checkedJSON(BASE, action ? { method: "POST", body: JSON.stringify({ action, repo_id, enabled, request_id: requestId }) } : {});
            if (!current()) return;
            catalog = data;
            status.textContent = "";
        } catch (error) {
            if (current()) {
                loadError = error.message;
                status.textContent = `${error.message} Open repository settings to retry.`;
            }
        } finally {
            busy = false;
            if (current()) widget.renderLibrary();
            if (current() && widget.librarySettingsMode) {
                widget.renderPoseRepositorySettings();
                if (focused) (widget.librarySettingsEl.querySelector(focused)
                    || widget.librarySettingsEl.querySelector(".vnccs-ps-library-settings-back"))?.focus({ preventScroll: true });
            }
        }
    };
    const close = () => {
        if (closed) return;
        closed = true;
        api.removeEventListener("vnccs.pose_manager.library.progress", onProgress);
        loading.remove();
        if (layoutFrame !== null) cancelAnimationFrame(layoutFrame);
        Object.assign(widget, original);
        nativeClose();
        widget._poseCardLibraryOpen = false;
        widget._libraryLoadToken = (widget._libraryLoadToken || 0) + 1;
        widget.managerPanel.inert = wasInert;
        widget.managerGrid.children[index]?.querySelector(".vnccs-pm-pose-library")?.focus({ preventScroll: true });
    };
    widget.getLibraryPreviewUrl = function (item) {
        return item.manager_sprite_id ? mediaURL(`${BASE}/${item.manager_sprite_id}/preview?repository=${encodeURIComponent(item.repository)}&v=${item.preview_mtime}`)
            : original.getLibraryPreviewUrl.apply(this, arguments);
    };
    widget.refreshPoseRepositories = function () {
        return updateRepositories();
    };
    widget.togglePoseRepository = function (id, enabled) {
        return updateRepositories("toggle", id, enabled);
    };
    widget.refreshSinglePoseRepository = function (id) {
        return updateRepositories("refresh", id);
    };
    widget.addPoseRepository = function () {
        return updateRepositories("add", this.librarySettingsEl.querySelector(".vnccs-ps-library-repo-input").value);
    };
    widget.removePoseRepository = function (id) {
        return updateRepositories("remove", id);
    };
    widget.renderLibraryCategories = function () {
        // Loading an empty catalog must not erase the user's category preference.
        this.libraryActiveCategory = selection.categories[source];
        original.renderLibraryCategories.apply(this, arguments);
        for (const button of this.libraryCategoriesEl.children) {
            const select = button.onclick;
            button.onclick = function (event) {
                selection.categories[source] = button.textContent;
                select.call(this, event);
                widget.libraryCategoriesEl.querySelector(":scope > .active")?.focus({ preventScroll: true });
            };
        }
        const selector = document.createElement("div");
        selector.className = "vnccs-pm-pose-sources";
        selector.setAttribute("role", "group");
        selector.setAttribute("aria-label", "Pose source");
        for (const label of ["Manager", "Pose Studio"]) {
            const button = document.createElement("button");
            button.type = "button";
            button.className = `vnccs-ps-library-category-chip${source === label ? " active" : ""}`;
            button.textContent = label;
            button.setAttribute("aria-pressed", String(source === label));
            button.onclick = () => {
                source = selection.source = label;
                this.libraryGrid.scrollTop = 0;
                this.renderLibrary();
                this.libraryCategoriesEl.querySelector(`[aria-pressed="true"]`)?.focus({ preventScroll: true });
            };
            selector.append(button);
        }
        this.libraryCategoriesEl.append(selector);
    };
    widget.renderLibrary = function () {
        const poses = this.libraryPoses;
        const top = this.libraryGrid.scrollTop, left = this.libraryGrid.scrollLeft;
        const aspects = new Map(cards.map(card => [card.querySelector("img")?.src, card.style.getPropertyValue("--pose-aspect")]));
        try {
            // Keep Manager assets out of Studio's cached catalog, including after closing the picker.
            this.libraryPoses = source === "Manager" ? (catalog?.poses || []) : (poses || [])
                .filter(pose => !this.isLibraryAnimation(pose));
            original.renderLibrary.apply(this, arguments);
        } finally { this.libraryPoses = poses; }
        cards = [...this.libraryGrid.querySelectorAll(".vnccs-ps-library-item")];
        cards.forEach(card => {
            card.removeAttribute("aria-pressed");
            const img = card.querySelector("img");
            if (!img) return;
            // Retain known dimensions so a rerender cannot collapse the scroll area while images reload.
            if (aspects.get(img.src)) card.style.setProperty("--pose-aspect", aspects.get(img.src));
            const resize = () => {
                if (!closed && img.naturalWidth && img.naturalHeight) {
                    card.style.setProperty("--pose-aspect", img.naturalWidth / img.naturalHeight);
                    schedulePacking();
                }
            };
            img.onload = resize;
            resize();
        });
        this.libraryGrid.scrollTop = top;
        this.libraryGrid.scrollLeft = left;
        renderLoading();
        if (source === "Manager" && catalog && !catalog.poses?.length && !busy) {
            const empty = this.libraryGrid.querySelector(".vnccs-ps-library-empty");
            if (empty) empty.textContent = "No poses available. Enable or add a repository in settings.";
        }
        schedulePacking();
    };
    widget.selectLibraryPose = async function (pose) {
        if (applying || !current()) return;
        applying = true;
        this.libraryGrid.setAttribute("aria-busy", "true");
        try {
            await applyPose(this, pose, current);
            if (current()) close();
        } catch (error) {
            if (current()) this.showMessage(`Failed to load pose: ${error.message}`, true);
        } finally {
            applying = false;
            if (current()) this.libraryGrid.removeAttribute("aria-busy");
        }
    };
    widget._closeLibraryModal = close;
    const capture = document.createElement("button"), imageInput = document.createElement("input");
    capture.type = "button";
    capture.className = "vnccs-ps-btn primary";
    capture.textContent = "Capture pose from image";
    imageInput.type = "file";
    imageInput.accept = "image/*";
    imageInput.hidden = true;
    capture.onclick = () => imageInput.click();
    imageInput.onchange = event => {
        const file = imageInput.files[0];
        if (!file || !current()) return;
        if (!file.type.startsWith("image/")) {
            widget.showMessage("Choose an image file.", true);
            imageInput.value = "";
            return;
        }
        close();
        widget.handleFileImport(event);
    };
    widget.libraryModal.querySelector(".vnccs-ps-library-header-actions").append(capture, imageInput);
    overlay.classList.add("vnccs-pm-pose-picker");
    overlay.tabIndex = -1;
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-label", "Pose Library");
    widget.libraryModal.querySelector(".vnccs-ps-modal-close").onclick = close;
    overlay.onclick = event => { if (event.target === overlay) close(); };
    overlay.onkeydown = event => {
        if (event.key === "Escape") { event.stopPropagation(); close(); }
    };
    widget.renderLibrary();
    widget.librarySearchInput.focus({ preventScroll: true });
    updateRepositories("ensure");
}
