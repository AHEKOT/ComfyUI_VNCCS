const SPRITE_REPOSITORY = "MIUProject/VNCCS_PoseLibrary_Sprites";
let repositoryRequest = null;

// Studio registers these routes directly, outside ComfyUI's /api transport.
async function studioJSON(route, options = {}) {
    const response = await fetch(`/vnccs/pose_library/${route}`, { cache: "no-store", ...options });
    const data = await response.json();
    if (!response.ok || data?.error) throw new Error(data?.error || `HTTP ${response.status}`);
    return data;
}

// Use Studio's repository cache, downloader and settings; only the picker belongs to Manager.
function ensureSpriteRepository(widget) {
    if (repositoryRequest) return repositoryRequest;
    const request = { taskId: "" };
    request.promise = (async () => {
        const find = data => data.repositories?.find(repo => repo.repo_id === SPRITE_REPOSITORY);
        let repo = find(await studioJSON("repositories"));
        if (!repo) {
            request.taskId = widget.createRepositoryTaskId("sprites");
            try {
                const data = await studioJSON("repositories/add", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ repo_id: SPRITE_REPOSITORY, task_id: request.taskId }),
                });
                repo = data.refreshed;
            } catch (error) {
                // Another node or browser may have registered the same repository first.
                repo = find(await studioJSON("repositories"));
                if (!repo || repo.status === "error") throw error;
            }
        }
        if (repo?.status === "error") throw new Error(repo.last_error || "Sprite repository download failed.");
        return repo;
    })().finally(() => { repositoryRequest = null; });
    repositoryRequest = request;
    return request;
}

export const POSE_LIBRARY_CSS = `
.vnccs-pm-pose-picker .vnccs-ps-library-categories { flex-wrap:wrap; align-items:center; }
.vnccs-pm-pose-sources { display:flex; gap:6px; margin-left:auto; }
.vnccs-pm-pose-status { padding:0 18px 8px; font-size:12px; color:var(--text-secondary); }
.vnccs-pm-pose-status:empty { display:none; }
.vnccs-pm-pose-picker .vnccs-ps-library-workspace:not(.settings-mode) .vnccs-ps-library-modal-grid { display:flex; flex-wrap:wrap; align-content:flex-start; align-items:flex-start; gap:8px; }
.vnccs-pm-pose-picker .vnccs-ps-library-item { box-sizing:border-box; flex:none; width:calc(var(--vnccs-ps-library-thumb-size,320px) * var(--pose-aspect,.416667)); max-width:100%; min-height:0; transition:border-color var(--ps-transition); }
.vnccs-pm-pose-picker .vnccs-ps-library-item:hover { transform:none; }
.vnccs-pm-pose-picker .vnccs-ps-library-item-preview { flex:none; aspect-ratio:var(--pose-aspect,.416667); border-radius:8px 8px 0 0; }
.vnccs-pm-pose-picker .vnccs-ps-library-item-preview img { width:100%; height:100%; }
.vnccs-pm-pose-picker .vnccs-ps-library-item-name { position:static; font-size:max(11px,calc(11px * var(--vnccs-ps-library-ui-scale))); }
`;

export function openPoseLibrary(widget, index) {
    widget.switchTab(index);
    widget.showLibraryModal();
    const overlay = widget.libraryModal.parentElement;
    const nativeClose = widget._closeLibraryModal;
    const original = Object.fromEntries(["getFilteredLibraryPoses", "renderLibraryCategories", "renderLibrary", "selectLibraryPose"]
        .map(key => [key, widget[key]]));
    const wasInert = widget.managerPanel.inert;
    widget.managerPanel.inert = true;
    widget._poseCardLibraryOpen = true;
    let closed = false, applying = false, polling = false, downloading = true;
    let source = "Sprites";
    const status = document.createElement("div");
    status.className = "vnccs-pm-pose-status";
    status.setAttribute("role", "status");
    widget.libraryCategoriesEl.after(status);
    const request = ensureSpriteRepository(widget);
    status.textContent = "Loading sprite library…";
    const progress = setInterval(async () => {
        if (polling || !request.taskId || closed || !downloading) return;
        polling = true;
        try {
            await widget.pollRepositoryProgress(request.taskId, { update: info => {
                if (!closed && downloading) status.textContent = `${info.message || "Downloading sprites…"} ${Math.round(info.progress || 0)}%`;
            } });
        } finally { polling = false; }
    }, 500);
    const close = () => {
        if (closed) return;
        closed = true;
        clearInterval(progress);
        Object.assign(widget, original);
        nativeClose();
        widget._poseCardLibraryOpen = false;
        widget._libraryLoadToken = (widget._libraryLoadToken || 0) + 1;
        widget.managerPanel.inert = wasInert;
        widget.managerGrid.children[index]?.querySelector(".vnccs-pm-pose-library")?.focus({ preventScroll: true });
    };
    const matchesSource = pose => !widget.isLibraryAnimation(pose)
        && (widget.getLibraryPoseMeta(pose).repository === SPRITE_REPOSITORY) === (source === "Sprites");
    widget.getFilteredLibraryPoses = function () {
        return original.getFilteredLibraryPoses.apply(this, arguments).filter(matchesSource);
    };
    widget.renderLibraryCategories = function () {
        const poses = this.libraryPoses;
        try {
            this.libraryPoses = (poses || []).filter(matchesSource);
            original.renderLibraryCategories.apply(this, arguments);
        } finally { this.libraryPoses = poses; }
        const selector = document.createElement("div");
        selector.className = "vnccs-pm-pose-sources";
        selector.setAttribute("role", "group");
        selector.setAttribute("aria-label", "Pose source");
        for (const label of ["Sprites", "PoseStudio"]) {
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
        original.renderLibrary.apply(this, arguments);
        this.libraryGrid?.querySelectorAll(".vnccs-ps-library-item").forEach(card => {
            card.removeAttribute("aria-pressed");
            const img = card.querySelector("img");
            if (!img) return;
            const resize = () => {
                if (!closed && img.naturalWidth && img.naturalHeight)
                    card.style.setProperty("--pose-aspect", img.naturalWidth / img.naturalHeight);
            };
            img.onload = resize;
            resize();
        });
    };
    widget.selectLibraryPose = async function (pose) {
        if (applying || closed) return;
        applying = true;
        this.libraryGrid.setAttribute("aria-busy", "true");
        try { await this.loadFromLibrary(pose); if (!closed) close(); }
        finally { applying = false; }
    };
    widget._closeLibraryModal = close;
    overlay.classList.add("vnccs-pm-pose-picker");
    overlay.setAttribute("role", "dialog");
    overlay.setAttribute("aria-label", "Pose Library");
    widget.libraryModal.querySelector(".vnccs-ps-modal-close").onclick = close;
    overlay.onclick = event => { if (event.target === overlay) close(); };
    overlay.onkeydown = event => {
        if (event.key === "Escape") { event.stopPropagation(); close(); }
    };
    widget.renderLibrary();
    widget.librarySearchInput.focus({ preventScroll: true });
    request.promise.then(async repo => {
        downloading = false;
        if (closed) return;
        status.textContent = repo?.enabled === false ? "Sprite library is disabled. Enable it in repository settings." : "";
        await widget.refreshLibrary(false);
    }).catch(error => {
        downloading = false;
        if (!closed) status.textContent = `${error.message} Open repository settings to retry.`;
    }).finally(() => clearInterval(progress));
}
