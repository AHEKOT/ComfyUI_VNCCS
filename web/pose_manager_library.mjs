import { api } from "../../scripts/api.js";
import { checkedJSON, mediaURL } from "./vnccs_transport.js";
import { packPreviewRows } from "./preview_grid_layout.mjs?v=pose-manager-11";

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

async function applySprite(widget, item, isCurrent) {
    const [{ pose: asset }, helpers] = await Promise.all([
        checkedJSON(`${BASE}/${item.manager_sprite_id}?repository=${encodeURIComponent(item.repository)}`), studioHelpers(),
    ]);
    if (!isCurrent()) return;
    if (asset?.type === "pose_set" || asset?.animation) throw new Error("Choose a single pose for this card.");
    const pose = structuredClone(helpers.extractActivePoseFromSceneAsset(asset) || asset);
    widget.validateImportedPose(pose);
    const transform = helpers.extractActiveCharacterTransformFromSceneAsset(asset);
    const framing = pose.cameraParams;
    const angles = { yaw_deg: framing?.yaw_deg ?? asset.camera?.yaw_deg ?? 0,
        pitch_deg: framing?.pitch_deg ?? asset.camera?.pitch_deg ?? 0 };
    if (![angles.yaw_deg, angles.pitch_deg].every(value => Number.isFinite(Number(value))))
        throw new Error("Pose framing requires finite camera angles.");
    if (framing && !transform) helpers.cameraFramingToCharacterTransform(framing, widget.viewer.sceneCameraTarget);
    const projection = helpers.normalizeSAMProjectionFrame(pose.sam_projection ?? asset.sam_projection ?? asset.camera?.sam_projection);
    widget.viewer.recordState();
    widget.clearSAMCameraMode();
    widget.viewer.setPose(widget.stripSceneCameraFromPose(pose), true);
    if (transform) widget.applyLibraryPoseTransform(transform, angles);
    else if (framing) widget.applyLibraryPoseFraming(framing);
    if (projection) widget.applyLibrarySAMProjection(projection);
    widget.setPosePrompt(widget.activeTab, pose.prompt ?? "");
    widget.syncPromptFieldToActiveTab();
    widget.updateRotationSliders();
    widget.commitViewerPoseToCurrentEditor();
}

export const POSE_LIBRARY_CSS = `
.vnccs-pm-pose-picker .vnccs-ps-library-categories { flex-wrap:wrap; align-items:center; }
.vnccs-pm-pose-sources { display:flex; gap:6px; margin-left:auto; }
.vnccs-pm-pose-status { padding:0 18px 8px; font-size:12px; color:var(--text-secondary); }
.vnccs-pm-pose-picker .vnccs-ps-library-local-repo,.vnccs-pm-pose-picker .vnccs-ps-library-save-current { display:none; }
.vnccs-pm-pose-status:empty { display:none; }
.vnccs-pm-pose-picker .vnccs-ps-library-workspace:not(.settings-mode) .vnccs-ps-library-modal-grid { display:flex; flex-wrap:wrap; justify-content:center; align-content:flex-start; align-items:flex-start; gap:8px; overflow-anchor:none; }
.vnccs-pm-pose-picker .vnccs-ps-library-item { box-sizing:border-box; flex:none; width:calc(var(--vnccs-ps-library-thumb-size,320px) * var(--pose-aspect,.416667)); max-width:100%; min-height:0; transition:border-color var(--ps-transition); }
.vnccs-pm-pose-picker .vnccs-ps-library-item:hover { transform:none; }
.vnccs-pm-pose-picker .vnccs-ps-library-item-preview { flex:none; aspect-ratio:var(--pose-aspect,.416667); border-radius:8px 8px 0 0; }
.vnccs-pm-pose-picker .vnccs-ps-library-item-preview img { width:100%; height:100%; }
.vnccs-pm-pose-picker .vnccs-ps-library-item-name { position:static; font-size:max(11px,calc(11px * var(--vnccs-ps-library-ui-scale))); }
`;

export function mountPoseLibrary(widget) {
    const show = widget.showLibraryModal;
    widget.showLibraryModal = function () { return openPoseLibrary(this, show); };
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

function openPoseLibrary(widget, show) {
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
    let closed = false, applying = false, busy = false, source = "Manager", catalog = null;
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
    widget.renderLocalPoseRepositorySettings = () => {};
    widget.renderPoseRepositorySettings = function () {
        const repositories = widget.poseRepositories;
        try {
            widget.poseRepositories = catalog?.repositories || [];
            original.renderPoseRepositorySettings.call(this);
            widget.librarySettingsEl.querySelector(".vnccs-ps-library-settings-title").textContent = "Manager Repositories";
            widget.librarySettingsEl.querySelector(".vnccs-ps-library-settings-subtitle").textContent = "Pose repositories used by Pose Manager. Pose Studio has its own repositories.";
            if (busy) widget.librarySettingsEl.querySelectorAll(".vnccs-ps-library-repo-action,.vnccs-ps-library-repo-add-btn").forEach(button => { button.disabled = true; });
        } finally { widget.poseRepositories = repositories; }
    };
    const updateRepositories = async (action, repo_id, enabled) => {
        if (busy || closed) return;
        const focused = widget.librarySettingsEl.contains(document.activeElement)
            ? (document.activeElement.classList.contains("refresh") ? ".refresh" : ".toggle") : null;
        busy = true;
        status.textContent = action ? "Updating Manager repositories…" : "Loading Manager repositories…";
        if (widget.librarySettingsMode) widget.renderPoseRepositorySettings();
        if (focused) overlay.focus({ preventScroll: true });
        try {
            const data = await checkedJSON(BASE, action ? { method: "POST", body: JSON.stringify({ action, repo_id, enabled }) } : {});
            if (!current()) return;
            catalog = data;
            status.textContent = "";
            widget.renderLibrary();
        } catch (error) {
            if (current()) status.textContent = `${error.message} Open repository settings to retry.`;
        } finally {
            busy = false;
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
        original.renderLibraryCategories.apply(this, arguments);
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
                source = label;
                this.libraryActiveCategory = "All";
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
        schedulePacking();
    };
    widget.selectLibraryPose = async function (pose) {
        if (applying || !current()) return;
        applying = true;
        this.libraryGrid.setAttribute("aria-busy", "true");
        try {
            if (pose.manager_sprite_id) await applySprite(this, pose, current);
            else await this.loadFromLibrary(pose);
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
