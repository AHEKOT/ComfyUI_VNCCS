import { injectStyles, SAKURA_THEME_CSS } from "./vnccs_common.js";
import { mountLivePreviews } from "./pose_manager_previews.mjs?v=pose-manager-10";
import { openSetLibrary } from "./pose_set_library.mjs?v=pose-manager-10";
import { fitPreviewGrid } from "./preview_grid_layout.mjs?v=pose-manager-10";
import { mountPoseLibrary, POSE_LIBRARY_CSS } from "./pose_manager_library.mjs?v=pose-manager-10";

const CSS = `
.vnccs-pm-shell,.vnccs-set-library { ${SAKURA_THEME_CSS} }
.vnccs-pm-shell .vnccs-ps-manager { background:var(--bg-primary); }
.vnccs-pm-shell .vnccs-ps-manager-header { background:var(--bg-secondary); gap:10px; }
.vnccs-pm-shell .vnccs-ps-manager-sidebar { zoom:1; box-sizing:border-box; width:21%; background:var(--bg-secondary); }
.vnccs-pm-shell .vnccs-pm-library { width:calc(21% - 12px); min-height:44px; flex-shrink:0; justify-content:center; gap:12px; padding:10px; border:1px solid var(--accent); border-radius:12px; background:var(--bg-elevated); color:var(--accent); font-size:14px; font-weight:700; letter-spacing:.8px; text-transform:uppercase; box-shadow:none; }
.vnccs-pm-shell .vnccs-pm-library:hover { background:var(--bg-hover); color:var(--accent-hover); border-color:var(--accent-hover); }
.vnccs-pm-shell .vnccs-pm-library svg { width:24px; height:24px; flex-shrink:0; }
.vnccs-pm-shell .vnccs-ps-manager-sidebar .vnccs-ps-section-title { font-size:11px; }
.vnccs-pm-shell .vnccs-ps-manager-sidebar .vnccs-ps-section-header { background:var(--bg-elevated); }
.vnccs-pm-reset-value { display:flex; align-items:center; gap:6px; }
.vnccs-pm-shell .vnccs-ps-manager-actions .vnccs-ps-btn { background:var(--bg-secondary); color:var(--text-primary); border:1px solid var(--accent-border); box-shadow:none; }
.vnccs-pm-shell .vnccs-ps-manager-auto-analysis { margin:6px; white-space:normal; }
.vnccs-pm-count { margin-right:auto; color:var(--text-secondary); font-size:12px; }
.vnccs-pm-shell .vnccs-ps-manager-grid { position:relative; gap:8px; }
.vnccs-pm-shell .vnccs-ps-pose-card { position:relative; display:grid; grid-template-rows:minmax(0,1fr) 28px; gap:4px; min-width:0; min-height:0; background:none; border:0; border-radius:8px; box-shadow:none; }
.vnccs-pm-shell .vnccs-ps-pose-card:is(:hover,.active) { border:0; box-shadow:none; transform:none; }
.vnccs-pm-shell .vnccs-ps-pose-preview { box-sizing:border-box; height:100%; background:var(--bg-elevated); border:1px solid #30303b; border-radius:8px; }
.vnccs-pm-pose-library { box-sizing:border-box; width:100%; min-width:0; height:28px; padding:0 4px; border:1px solid var(--accent-border); border-radius:6px; background:var(--bg-secondary); color:var(--accent); font:inherit; font-size:11px; cursor:pointer; }
.vnccs-pm-pose-library:hover { background:var(--bg-hover); color:var(--accent-hover); }
.vnccs-pm-shell .vnccs-pm-live-preview { position:relative; }
.vnccs-pm-shell .vnccs-pm-live-preview > :not(canvas.vnccs-pm-live) { display:none; }
.vnccs-pm-shell canvas.vnccs-pm-live { position:absolute; inset:0; width:100%; height:100%; object-fit:contain; pointer-events:none; }
.vnccs-pm-shell .vnccs-ps-pose-card-bottom { position:absolute; inset:5px 5px auto; display:flex; align-items:flex-start; justify-content:space-between; height:auto; padding:0; border:0; background:none; pointer-events:none; }
.vnccs-pm-shell .vnccs-ps-pose-card-name { padding:3px 6px; background:#0a0a0fdf; border-radius:5px; font-size:clamp(9px,calc(var(--pm-card-w)*.075),14px); }
.vnccs-pm-shell .vnccs-ps-pose-card-delete { box-sizing:border-box; width:24px; height:24px; min-width:0; min-height:0; padding:0; flex-shrink:0; display:grid; place-items:center; font-size:14px; line-height:1; pointer-events:auto; opacity:0; }
.vnccs-pm-shell .vnccs-ps-pose-card:is(:hover,:focus-within) .vnccs-ps-pose-card-delete { opacity:1; }
.vnccs-pm-shell button:focus-visible,.vnccs-set-library :focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.vnccs-set-library .vnccs-ps-library-modal { background:var(--bg-primary); }
.vnccs-set-library .vnccs-ps-library-categories { display:none; }
.vnccs-set-library .vnccs-ps-library-modal-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(0,min(100%,var(--vnccs-ps-library-thumb-size,320px)))); align-content:start; align-items:start; gap:14px; padding:18px; overflow:auto; }
.vnccs-set-card { position:relative; min-width:0; text-align:left; background:var(--bg-elevated); color:var(--text-primary); border:1px solid #30303b; border-radius:10px; padding:5px; font:inherit; }
.vnccs-set-card:hover { border-color:var(--accent); }
.vnccs-set-load { display:block; width:100%; padding:0; border:0; background:none; color:inherit; font:inherit; text-align:inherit; cursor:pointer; }
.vnccs-set-delete { position:absolute; top:9px; right:9px; width:28px; height:28px; padding:0; border:1px solid var(--error); border-radius:6px; background:var(--bg-secondary); color:var(--error); font:inherit; font-size:20px; cursor:pointer; }
.vnccs-set-delete:hover { background:var(--bg-hover); }
.vnccs-set-preview { display:grid; grid-template-columns:repeat(var(--set-columns),minmax(0,1fr)); grid-template-rows:repeat(var(--set-rows),minmax(0,1fr)); aspect-ratio:5/4; gap:3px; background:var(--bg-secondary); border-radius:6px; overflow:hidden; }
.vnccs-set-preview img { display:block; min-width:0; min-height:0; width:100%; height:100%; object-fit:contain; }
.vnccs-set-label { display:flex; gap:10px; justify-content:space-between; align-items:center; padding:10px 8px 0; }
.vnccs-set-label strong,.vnccs-set-card p { overflow-wrap:anywhere; }
.vnccs-set-label strong { font-size:16px; }
.vnccs-set-label span { color:var(--accent); font-size:12px; white-space:nowrap; }
.vnccs-set-card p,.vnccs-set-empty { font-size:14px; color:var(--text-secondary); margin:6px 8px 10px; }
.vnccs-set-library .vnccs-common-modal { box-sizing:border-box; max-width:calc(100% - 32px); background:var(--bg-secondary); border-color:var(--accent-border); border-radius:12px; color:var(--text-primary); font:inherit; font-size:14px; overflow-wrap:anywhere; }
.vnccs-set-library .vnccs-common-modal-title { color:var(--accent); border-color:var(--border); }
.vnccs-set-status:empty { display:none; }
.vnccs-set-status { padding:8px 18px; color:var(--accent-hover); }
.vnccs-set-library .vnccs-ps-library-settings:has(.vnccs-set-form) { zoom:1; padding:20px; }
.vnccs-set-library .vnccs-set-form .vnccs-ps-btn { min-height:36px; padding:8px 12px; font-size:14px; }
.vnccs-set-library a.vnccs-ps-library-repo-action { display:inline-flex; align-items:center; text-decoration:none; box-sizing:border-box; }
.vnccs-set-form { box-sizing:border-box; width:100%; max-width:580px; margin:auto; padding:16px; display:grid; gap:14px; font-size:14px; }
.vnccs-set-form form,.vnccs-set-form label { display:grid; gap:8px; }
.vnccs-set-form form { gap:14px; }
.vnccs-set-form input,.vnccs-set-form select,.vnccs-set-form textarea { box-sizing:border-box; width:100%; font:inherit; padding:8px; color:var(--text-primary); background:var(--bg-secondary); border:1px solid #383844; border-radius:6px; }
.vnccs-set-form small { color:var(--text-secondary); }
${POSE_LIBRARY_CSS}
`;

export function mountManager(widget) {
    mountPoseLibrary(widget);
    const generatePrompt = widget.generatePromptFromLights;
    // Omit light descriptions from Manager prompts without changing the rendered lights.
    widget.generatePromptFromLights = function (_lights, userPromptOverride) {
        return generatePrompt.call(this, [], userPromptOverride);
    };
    injectStyles(CSS, "vnccs-pose-manager-style");
    widget.container.classList.add("vnccs-pm-shell");
    widget.managerSidebar.querySelector(".vnccs-ps-section-content")
        .append(widget.managerAutoAnalyzeCheckbox.closest("label"));
    for (const [key, control] of Object.entries(widget.managerControls)) {
        const nativeReset = widget.sliders[key]?.slider.closest(".vnccs-ps-field")?.querySelector(".vnccs-ps-reset-btn");
        if (!nativeReset || !control.value) continue;
        const row = document.createElement("span"), reset = nativeReset.cloneNode(true);
        row.className = "vnccs-pm-reset-value";
        reset.type = "button";
        reset.setAttribute("aria-label", `Reset ${control.def.label}`);
        reset.onclick = event => {
            event.stopPropagation();
            nativeReset.click();
            widget.refreshPoseManagerControls();
            widget.scheduleAllManagerPreviewRefresh();
        };
        control.value.replaceWith(row);
        row.append(control.value, reset);
    }
    const count = document.createElement("span");
    count.className = "vnccs-pm-count";
    const library = document.createElement("button");
    library.type = "button";
    library.className = "vnccs-ps-btn vnccs-pm-library";
    library.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" aria-hidden="true"><path d="M3 5a2 2 0 0 1 2-2h5l3 3h6a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg><span>Library</span>';
    widget.managerPanel.querySelector(".vnccs-ps-manager-title").replaceWith(library, count);
    widget.layoutPoseManager = function () {
        const grid = this.managerGrid;
        if (!grid?.isConnected || !this.poses.length || !grid.children.length) return;
        const width = Math.max(1, grid.clientWidth), height = Math.max(1, grid.clientHeight);
        const aspect = Math.max(.05, Math.min(20, (Number(this.exportParams.view_width) || 1024) / (Number(this.exportParams.view_height) || 1024)));
        const key = `${grid.children.length}|${width}|${height}|${aspect}`;
        if (this._lastManagerLayoutKey === key) return;
        const layout = fitPreviewGrid(grid.children.length, width, height, aspect, 32);
        if (!layout) return;
        this._lastManagerLayoutKey = key;
        grid.style.gridTemplateColumns = `repeat(${layout.cols}, ${layout.trackW}px)`;
        grid.style.gridTemplateRows = `repeat(${layout.rows}, ${layout.trackH}px)`;
        [...grid.children].forEach((card, index) => {
            card.style.setProperty("--pm-card-w", `${layout.trackW}px`);
            card.style.setProperty("--pm-card-h", `${layout.trackH}px`);
            card.style.left = `${index >= grid.children.length - layout.lastRowCount ? layout.lastRowOffset : 0}px`;
        });
    };
    const previews = mountLivePreviews(widget);
    // A card picker must never replace the whole set or switch to animation mode.
    for (const name of ["loadPoseSetAsset", "loadAnimationLibraryAsset", "loadCharacterSceneLibraryAsset"]) {
        const original = widget[name];
        widget[name] = function () {
            if (this._poseCardLibraryOpen) throw new Error("Choose a single pose for this card.");
            return original.apply(this, arguments);
        };
    }
    library.onclick = () => openSetLibrary(widget, library, previews.get);
    const render = widget.renderPoseManager;
    widget.renderPoseManager = function () {
        previews.beforeRender();
        render.apply(this, arguments);
        count.textContent = `Poses · ${this.poses.length}`;
        this.poses.forEach((_, index) => {
            previews.decorate(index);
            const card = this.managerGrid.children[index];
            if (card.querySelector(".vnccs-pm-pose-library")) return;
            const button = document.createElement("button");
            button.type = "button";
            button.className = "vnccs-pm-pose-library";
            button.textContent = "Library";
            button.setAttribute("aria-label", `Library for Pose ${index + 1}`);
            button.onclick = event => { event.stopPropagation(); this.switchTab(index); this.showLibraryModal(); };
            button.onkeydown = event => event.stopPropagation();
            card.append(button);
        });
    };
    widget.renderPoseManager();
}
