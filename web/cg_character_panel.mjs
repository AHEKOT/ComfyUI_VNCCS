import { validateCharacters } from "./cg_characters.mjs";

export const CG_CHARACTER_PANEL_CSS = `
.vnccs-cg-creator .cg-character-panel { gap:12px; overflow:visible; padding:16px; }
.vnccs-cg-creator .cg-character-panel button { cursor:pointer; }
.vnccs-cg-creator .cg-character-panel .vnccs-common-creator-btn-secondary { background:var(--bg-surface); color:var(--text-primary); border:1px solid var(--accent-border); }
.vnccs-cg-creator .cg-character-panel>button { width:100%; min-height:48px; }
.vnccs-cg-creator .cg-character-tabs { display:flex; border-bottom:1px solid var(--accent-border); }
.vnccs-cg-creator .cg-character-tab { display:flex; min-width:200px; background:var(--bg-surface); border:1px solid var(--accent-border); border-bottom:3px solid var(--accent); border-radius:10px 10px 0 0; }
.vnccs-cg-creator .cg-character-tab button { padding:10px 12px; color:var(--text-primary); background:none; border:0; }
.vnccs-cg-creator .cg-character-tab [role=tab] { flex:1; }
.vnccs-cg-creator .cg-character-block { display:flex; flex-direction:column; gap:12px; padding:16px; border:1px solid var(--accent-border); border-radius:var(--radius-md); background:var(--bg-secondary); }
.vnccs-cg-creator .cg-character-block .vnccs-common-creator-section-title { margin:0; }
.vnccs-cg-creator .cg-character-field { display:grid; grid-template-columns:110px minmax(0,1fr); align-items:center; gap:12px; font-size:16px; }
.vnccs-cg-creator .cg-character-panel :is(input,select,button) { font-size:16px; }
.vnccs-cg-creator .cg-character-panel select { width:100%; min-height:42px; }
.vnccs-cg-creator .cg-character-slider { grid-template-columns:110px minmax(0,1fr) 84px; }
.vnccs-cg-creator .cg-character-slider input { min-width:0; width:100%; box-sizing:border-box; }
.vnccs-cg-creator .cg-sprite-grid { display:flex; flex-wrap:wrap; align-content:flex-start; gap:10px; max-height:214px; overflow:auto; padding:3px; overscroll-behavior:contain; }
.vnccs-cg-creator .cg-sprite { flex:0 0 auto; min-width:0; padding:0; height:200px; border:2px solid var(--border); border-radius:10px; background:var(--bg-primary); }
.vnccs-cg-creator .cg-sprite[aria-pressed=true] { border-color:var(--accent); }
.vnccs-cg-creator .cg-sprite img { width:auto; height:100%; display:block; }
.vnccs-cg-creator .cg-character-spatial { --cg-radar-size:260px; display:grid; grid-template-columns:390px minmax(0,1fr); gap:12px; }
.vnccs-cg-creator .cg-character-light { display:block; width:auto; height:auto; background:none; overflow:visible; pointer-events:auto; }
.vnccs-cg-creator .cg-character-light .vnccs-ps-light-body,.vnccs-cg-creator .cg-character-position-controls { display:grid; grid-template-rows:42px 28px 300px 24px; gap:10px; padding:0; }
.vnccs-cg-creator .cg-character-light .vnccs-ps-light-radar-wrap { grid-row:3; justify-content:center; padding:10px; }
.vnccs-cg-creator .cg-character-light .vnccs-ps-light-radar-canvas { width:var(--cg-radar-size); height:var(--cg-radar-size); box-sizing:content-box; }
.vnccs-cg-creator .cg-character-light .vnccs-ps-light-slider-vert-wrap { height:var(--cg-radar-size); width:44px; }
.vnccs-cg-creator .cg-character-light .vnccs-ps-light-h-label { white-space:nowrap; }
.vnccs-cg-creator .cg-character-light .vnccs-ps-light-slider-vert { height:calc(var(--cg-radar-size) - 52px); }
.vnccs-cg-creator .cg-character-light :is(.vnccs-ps-light-pos-label,.vnccs-ps-light-value,.vnccs-ps-light-h-val,.vnccs-ps-light-h-label) { font-size:14px; }
.vnccs-cg-creator .cg-character-light .vnccs-ps-light-slider::-webkit-slider-thumb { width:18px; height:18px; }
.vnccs-cg-creator .cg-character-light input[type=color] { width:100%; height:36px; }
.vnccs-cg-creator .cg-character-position-controls>.cg-character-slider { grid-template-columns:64px minmax(0,1fr) 64px; }
.vnccs-cg-creator .cg-character-radars { grid-row:3 / 5; display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:24px; }
.vnccs-cg-creator .cg-character-radars>div { display:grid; grid-template-rows:300px 24px; justify-items:center; align-items:center; gap:10px; }
.vnccs-cg-creator .cg-character-radars [data-character-radar-host] { min-width:0; }
.vnccs-cg-creator .cg-character-radars .vnccs-ps-radar-wrap { margin-top:0 !important; }
.vnccs-cg-creator .cg-character-radars canvas { width:var(--cg-radar-size) !important; height:var(--cg-radar-size) !important; }
.vnccs-cg-creator [data-character-radar] { touch-action:none; }
.vnccs-cg-creator [data-character-radar]:focus-visible { outline:2px solid var(--accent); }
.vnccs-cg-creator .cg-character-actions { display:flex; gap:12px; }
.vnccs-cg-creator .cg-character-actions>button { flex:1; min-height:46px; }
.vnccs-cg-creator .cg-character-actions .vnccs-common-creator-btn-secondary[aria-pressed=true] { background:var(--accent); color:var(--bg-primary); }
.vnccs-cg-creator .cg-character-panel button:disabled { cursor:default; opacity:.45; }
.vnccs-cg-creator .cg-character-error { margin:0; color:var(--error); overflow-wrap:anywhere; }
.cg-pose-library-host { position:absolute; inset:0; z-index:50; pointer-events:auto; }
.cg-pose-library-host .vnccs-ps-library-save-current,.cg-pose-library-host .vnccs-ps-library-menu-btn { display:none; }
`;
const buttonClass = "vnccs-common-creator-btn vnccs-common-creator-btn-secondary";
const primaryClass = "vnccs-common-creator-btn vnccs-common-creator-btn-primary";
const inputClass = "vnccs-common-creator-input";
const title = name => `<div class="vnccs-common-creator-section-title">${name}</div>`;
const slider = (label, key, min, max, step) => `<label class="cg-character-field cg-character-slider"><span>${label}</span><input class="vnccs-common-creator-slider" type="range" data-character-value="${key}" min="${min}" max="${max}" step="${step}" aria-label="${label}"><input class="${inputClass}" type="number" data-character-value="${key}" min="${min}" max="${max}" step="${step}" aria-label="Exact ${label.toLowerCase()}"></label>`;
const select = (label, key) => `<label class="cg-character-field"><span>${label}</span><select class="vnccs-common-creator-select" data-character-source="${key}" aria-label="${label}"></select></label>`;

export class CGCharacterPanel {
    constructor(widget) {
        this.widget = widget; this.events = new AbortController(); this.serial = 0; this.catalogKey = "";
        this.element = document.createElement("aside");
        this.element.className = "bg-panel cg-character-panel vnccs-common-creator-col"; this.element.hidden = true;
        this.element.innerHTML = `<button type="button" class="${primaryClass}" data-character-back>← Back to CG Creator</button>
            <div class="cg-character-tabs" role="tablist" aria-label="Characters"><div class="cg-character-tab"><button type="button" role="tab" aria-selected="true">Character 1</button><button type="button" data-character-delete aria-label="Delete character">×</button></div></div>
            <section class="cg-character-block">${title("Character")}
                <select class="vnccs-common-creator-select" data-character-source="character" aria-label="Character"></select>
                ${select("Costume", "costume")}
                <div class="cg-character-field"><span>Sprite</span><div class="cg-sprite-grid" data-character-sprites data-vnccs-own-navigation></div></div>
                ${select("Emotion", "emotion")}</section>
            <div class="cg-character-spatial">
                <section class="cg-character-block">${title("Lighting")}<div class="vnccs-pose-studio cg-character-light" data-character-radar-host="light"></div></section>
                <section class="cg-character-block">${title("Position")}<div class="cg-character-position-controls">${slider("Scale", "scale", .01, 5, .01)}
                    <div class="cg-character-radars">${["x", "y"].map(axis => `<div><div data-character-radar-host="${axis}"></div><span>${axis.toUpperCase()} POSITION</span></div>`).join("")}</div>
                </div></section>
            </div>
            <div class="cg-character-actions"><button type="button" class="${buttonClass}" data-character-mode="pose">Pose</button><button type="button" class="${buttonClass}" data-character-mode="move">Move</button><button type="button" class="${buttonClass}" data-character-mode="rotate">Rotate</button></div>
            <div class="cg-character-actions"><button type="button" class="${buttonClass}" data-character-reset>↺ Reset</button><button type="button" class="${buttonClass}" data-character-undo>↶ Undo</button><button type="button" class="${buttonClass}" data-character-redo>↷ Redo</button></div>
            <button type="button" class="${primaryClass}" data-character-library>Open Pose Library</button>
            <p class="cg-character-error" role="status" hidden></p>`;
        widget.container.querySelector(".bg-model-source").after(this.element);
        const on = (target, type, fn) => target.addEventListener(type, fn, { signal: this.events.signal });
        on(this.element, "click", event => { event.stopPropagation(); void this.click(event).catch(error => this.error(error)); });
        on(this.element, "input", event => { event.stopPropagation(); this.input(event); });
        on(this.element, "change", event => {
            event.stopPropagation();
            if (event.target.dataset.characterSource) void this.selectSource(event.target.dataset.characterSource, event.target.value).catch(error => this.error(error));
            else { widget.commitGesture(); this.refresh(); }
        });
        this.element.addEventListener("pointerdown", event => {
            if (!event.target.dataset.characterRadar) return;
            if (event.button !== 0 || !this.editable()) { event.preventDefault(); event.stopImmediatePropagation(); return; }
            this.radarDrag = event.target;
        }, { capture: true, signal: this.events.signal });
        for (const type of ["pointerdown", "pointermove", "pointerup", "pointercancel", "lostpointercapture"]) on(this.element, type, event => {
            if (event.target.dataset.characterRadar) event.stopPropagation();
        });
        on(this.element, "keydown", event => {
            const kind = event.target.dataset.characterRadar;
            if (!kind || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key) || !this.editable()) return;
            event.preventDefault(); event.stopPropagation();
            const [x, y] = this.radarValue(kind), delta = event.shiftKey ? .1 : .02;
            this.setRadar(kind, x + (event.key === "ArrowRight" ? delta : event.key === "ArrowLeft" ? -delta : 0),
                y + (event.key === "ArrowDown" ? delta : event.key === "ArrowUp" ? -delta : 0));
            widget.commitGesture();
        });
    }
    get core() { return this.widget.characters?.entries.values().next().value; }
    editable() { return this.widget.canEditCamera() && !this.widget.document?._recovery && !this.widget.document?.archived && !this.loading; }
    error(error) {
        if (this.disposed) return;
        const status = this.element.querySelector(".cg-character-error"); status.textContent = error?.message || String(error); status.hidden = false;
    }
    async show() {
        if (!this.core) return;
        this.element.hidden = false; this.widget.container.querySelector(".bg-model-source").hidden = true;
        this.widget.characters.select(this.widget.characters.entries.keys().next().value);
        this.refresh(); this.widget.resize();
        await Promise.all([this.mountRadars(), this.loadCatalog()]);
    }
    hide() {
        this.element.hidden = true; this.widget.container.querySelector(".bg-model-source").hidden = false;
        this.closeLibrary(); this.widget.resize();
    }
    options(key, options, value, empty) {
        const control = this.element.querySelector(`[data-character-source="${key}"]`);
        const values = [...options]; if (value && !values.includes(value)) values.unshift(value);
        control.replaceChildren(new Option(empty, ""), ...values.map(item => new Option(item, item)));
        control.value = value || "";
    }
    async loadCatalog() {
        const request = ++this.serial, core = this.core, source = { ...core?.source }, epoch = this.widget.epoch;
        if (!core) return;
        this.loading = true; this.refresh();
        try {
            const catalog = await this.widget.request("/characters");
            const data = source.character ? await this.widget.request(`/characters/${encodeURIComponent(source.character)}?${new URLSearchParams({ costume: source.costume || "" })}`) : null;
            if (this.disposed || request !== this.serial || epoch !== this.widget.epoch || this.core !== core) return;
            this.catalogKey = JSON.stringify([source.character, source.costume]); this.names = catalog.characters; this.catalog = data;
            this.element.querySelector(".cg-character-error").hidden = true;
        } catch (error) { if (request === this.serial) this.error(error); }
        finally { if (request === this.serial && !this.disposed) { this.loading = false; this.refresh(); } }
    }
    async selectSource(key, value) {
        if (!this.editable() || !this.core) return;
        const widget = this.widget, core = this.core, source = { ...core.source, [key]: value }, epoch = widget.epoch;
        if (key === "character") Object.assign(source, { costume: "", sprite: "", emotion: "" });
        if (key === "costume") Object.assign(source, { sprite: "", emotion: "" });
        if (key === "character" || key === "costume") {
            const request = ++this.serial; this.loading = true; this.refresh();
            try {
                const data = source.character ? await widget.request(`/characters/${encodeURIComponent(source.character)}?${new URLSearchParams({ costume: source.costume })}`) : null;
                if (this.disposed || request !== this.serial || epoch !== widget.epoch || core !== this.core) return;
                this.catalog = data; source.costume = data?.costume || "";
                source.sprite = data?.sprites[0]?.name || ""; source.emotion = source.sprite ? "Neutral" : "";
                this.loading = false;
                widget.characters.edit(rig => {
                    rig.source = source;
                    if (data) rig.setMesh({ ...rig.mesh, age: Math.max(1, Math.min(90, parseInt(data.age, 10) || 25)),
                        gender: String(data.gender).toLowerCase() === "male" ? 1 : String(data.gender).toLowerCase() === "female" ? 0 : .5 });
                });
                this.catalogKey = JSON.stringify([source.character, source.costume]);
                this.element.querySelector(".cg-character-error").hidden = true;
            } finally { if (request === this.serial) { this.loading = false; this.refresh(); } }
        } else {
            if (key === "sprite") {
                const emotions = this.catalog?.sprites.find(item => item.name === value)?.emotions || [];
                if (!emotions.includes(source.emotion)) source.emotion = emotions[0] || "";
            }
            widget.characters.edit(rig => { rig.source = source; });
        }
        widget.commitGesture(); this.refresh();
    }
    refresh() {
        if (this.disposed) return;
        const core = this.core;
        if (!core) { if (!this.element.hidden) this.hide(); return; }
        const source = core.source || {};
        if (!this.element.hidden && !this.loading && this.catalogKey && this.catalogKey !== JSON.stringify([source.character, source.costume])) void this.loadCatalog();
        const disabled = !this.editable();
        for (const control of this.element.querySelectorAll("button,input,select")) control.disabled = disabled;
        this.element.querySelector("[data-character-back]").disabled = false;
        this.element.querySelector("[data-character-undo]").disabled = disabled || !this.widget.history?.canUndo;
        this.element.querySelector("[data-character-redo]").disabled = disabled || !this.widget.history?.canRedo;
        for (const control of this.element.querySelectorAll("canvas")) control.setAttribute("aria-disabled", String(disabled));
        this.options("character", this.names || [], source.character, "Select character");
        this.options("costume", this.catalog?.costumes || [], source.costume, "Select costume");
        const emotions = this.catalog?.sprites.find(item => item.name === source.sprite)?.emotions || [];
        this.options("emotion", emotions, source.emotion, "Select emotion");
        const grid = this.element.querySelector("[data-character-sprites]");
        const gridKey = JSON.stringify([this.catalog?.character, this.catalog?.costume, this.catalog?.sprites]);
        if (gridKey !== this.gridKey) {
            this.gridKey = gridKey;
            grid.replaceChildren(...(this.catalog?.sprites || []).map(item => {
                const button = document.createElement("button"); button.type = "button"; button.className = "cg-sprite";
                button.dataset.characterSprite = item.name; button.setAttribute("aria-label", item.name);
                const image = document.createElement("img"); image.alt = item.name; image.loading = "lazy";
                image.src = this.widget.api.apiURL(`/vnccs/cg-creator/characters/${encodeURIComponent(this.catalog.character)}/sprite?${new URLSearchParams({ costume: this.catalog.costume, sprite: item.name })}`);
                button.append(image); return button;
            }));
        }
        for (const button of grid.children) {
            const selected = button.dataset.characterSprite === source.sprite;
            button.disabled = disabled; button.setAttribute("aria-pressed", String(selected));
            const image = button.querySelector("img"), emotion = selected && emotions.includes(source.emotion) ? source.emotion : "Neutral";
            if (image.dataset.emotion !== emotion) {
                image.dataset.emotion = emotion;
                image.src = this.widget.api.apiURL(`/vnccs/cg-creator/characters/${encodeURIComponent(this.catalog.character)}/sprite?${new URLSearchParams({ costume: this.catalog.costume, sprite: button.dataset.characterSprite, emotion })}`);
            }
        }
        for (const control of this.element.querySelectorAll("[data-character-value]")) {
            if (document.activeElement === control) continue;
            control.value = String(Math.round(core.skinnedMesh.scale.x / core.baseScale * 100) / 100);
        }
        for (const button of this.element.querySelectorAll("[data-character-mode]")) button.setAttribute("aria-pressed", String(button.dataset.characterMode === this.widget.characters.mode));
        this.refreshRadars();
        this.widget.customSelects.syncSelects();
    }
    input(event) {
        const key = event.target.dataset.characterValue;
        if (key !== "scale" || !this.editable() || !event.target.checkValidity()) return;
        const value = event.target.valueAsNumber;
        if (!Number.isFinite(value)) return;
        this.widget.characters.edit(core => core.skinnedMesh.scale.setScalar(Math.max(.001, Math.min(1000, core.baseScale * value))));
        this.refresh();
    }
    radarValue(kind) {
        const core = this.core;
        if (!core) return [0, 0];
        if (kind === "light") return [core.light.x / 100, core.light.z / 100];
        const span = core.baseScale * (this.widget.characters.runtime.maxY - this.widget.characters.runtime.minY) * 2;
        return kind === "x" ? [(core.skinnedMesh.position.x - core.origin[0]) / span, (core.skinnedMesh.position.z - core.origin[2]) / span]
            : [(core.skinnedMesh.position.x - core.origin[0]) / span, -(core.skinnedMesh.position.y - core.origin[1]) / span];
    }
    async mountRadars() {
        if (this.radars) return;
        const { PoseStudioWidget } = await import(new URL("../vnccs_pose_studio.js", this.widget.factoryRuntime.viewerURL).href);
        if (this.disposed || this.radars) return;
        const syncToNode = () => { this.radarDrag = null; this.widget.commitGesture(); this.refresh(); };
        this.lightRadar = Object.assign(Object.create(PoseStudioWidget.prototype), {
            exportParams: { keepOriginalLighting: false },
            lightListContainer: this.element.querySelector('[data-character-radar-host="light"]'),
            applyLighting: () => {
                if (!this.editable()) return;
                const { x, y, z, intensity, color } = this.radarLight;
                this.radarLightKey = JSON.stringify([x, y, z, intensity, color]);
                this.widget.characters.edit(core => core.setLight({ x, y, z, intensity, color }));
            }, syncToNode,
        });
        this.radars = ["x", "y"].map(kind => {
            const radar = Object.assign(Object.create(PoseStudioWidget.prototype), {
                exportParams: { view_width: 1024, view_height: 1024 },
                viewer: {
                    getCharacterFrameCenter: () => { const [x, y] = this.radarValue(kind); return { x, y: -y }; },
                    characterTranslationForFramePoint: (x, y) => ({ x, y, z: 0 }),
                },
                getActiveCharacter: () => ({ transform: {} }), persistActivePoseCameraParams() {}, syncToNode,
                applyCameraToViewer: () => this.setRadar(kind, radar.exportParams.cam_offset_x, -radar.exportParams.cam_offset_y),
            });
            const content = this.element.querySelector(`[data-character-radar-host="${kind}"]`);
            radar.createCameraRadar({ content });
            content.querySelectorAll("button").forEach(button => button.remove());
            content.querySelector("canvas").dataset.characterRadar = kind;
            return radar;
        });
        this.refreshRadars();
    }
    refreshRadars() {
        if (!this.radars || !this.core) return;
        const { x, y, z, intensity, color } = this.core.light, key = JSON.stringify([x, y, z, intensity, color]);
        if (key !== this.radarLightKey && this.radarDrag?.dataset.characterRadar !== "light") {
            this.radarLightKey = key; this.radarLight = { ...this.core.light, type: "directional" };
            const host = this.lightRadar.lightListContainer, focused = document.activeElement?.dataset.characterRadar === "light";
            this.lightRadar.lightParams = [this.radarLight];
            this.lightRadar.refreshLightUI();
            host.replaceChildren(host.querySelector(".vnccs-ps-light-body"));
            // The CG scene currently has one directional character light.
            host.querySelectorAll("option").forEach(option => { if (option.value !== "directional") option.remove(); });
            for (const [selector, label] of [["select", "Light type"], ["input[type=color]", "Light color"], [".vnccs-ps-light-slider", "Intensity"], [".vnccs-ps-light-slider-vert", "Light height"]]) host.querySelector(selector).setAttribute("aria-label", label);
            const canvas = host.querySelector("canvas"); canvas.dataset.characterRadar = "light"; canvas.tabIndex = 0;
            if (focused) canvas.focus();
        }
        for (const control of this.lightRadar.lightListContainer.querySelectorAll("input,select")) control.disabled = !this.editable();
        for (const radar of this.radars) radar.radarRedraw();
        for (const canvas of this.element.querySelectorAll("[data-character-radar]")) {
            const kind = canvas.dataset.characterRadar, [x, y] = this.radarValue(kind);
            canvas.tabIndex = 0; canvas.setAttribute("role", "slider");
            canvas.setAttribute("aria-label", kind === "light" ? "Light position" : `${kind.toUpperCase()} position`);
            canvas.setAttribute("aria-disabled", String(!this.editable()));
            canvas.setAttribute("aria-valuemin", kind === "light" ? "-1" : "-2.17");
            canvas.setAttribute("aria-valuemax", kind === "light" ? "1" : "2.17");
            canvas.setAttribute("aria-valuenow", String(x)); canvas.setAttribute("aria-valuetext", `${x.toFixed(2)}, ${y.toFixed(2)}`);
        }
    }
    setRadar(kind, x, y) {
        if (!this.editable()) return;
        if (kind === "light") { x = Math.max(-1, Math.min(1, x)); y = Math.max(-1, Math.min(1, y)); }
        this.widget.characters.edit(core => {
            if (kind === "light") core.setLight({ ...core.light, x: x * 100, z: y * 100 });
            else {
                const span = core.baseScale * (this.widget.characters.runtime.maxY - this.widget.characters.runtime.minY) * 2;
                core.skinnedMesh.position.x = core.origin[0] + x * span;
                if (kind === "x") core.skinnedMesh.position.z = core.origin[2] + y * span;
                else core.skinnedMesh.position.y = core.origin[1] - y * span;
            }
        });
        this.refresh();
    }
    async click(event) {
        const button = event.target.closest("button"); if (!button || button.disabled) return;
        if (button.hasAttribute("data-character-back")) { this.hide(); return; }
        if (!this.editable()) return;
        if (button.hasAttribute("data-character-delete")) {
            const widget = this.widget, core = this.core, epoch = widget.epoch;
            const id = widget.characters.entries.keys().next().value;
            const confirmed = await widget.dialog("Delete character", [], "Delete", { message: "Delete this character from the scene?", danger: true });
            if (!confirmed || this.disposed || epoch !== widget.epoch || core !== this.core || !this.editable()) return;
            widget.beginGesture();
            widget.characters.restore(widget.characters.snapshot().filter(character => character.id !== id));
            widget.commitGesture(); widget.updateControls();
        }
        else if (button.dataset.characterSprite) await this.selectSource("sprite", button.dataset.characterSprite);
        else if (button.dataset.characterMode) this.widget.characters.setMode(button.dataset.characterMode);
        else if (button.hasAttribute("data-character-reset")) this.widget.characters.reset();
        else if (button.hasAttribute("data-character-undo")) this.widget.history.undo();
        else if (button.hasAttribute("data-character-redo")) this.widget.history.redo();
        else if (button.hasAttribute("data-character-library")) await this.openLibrary();
        this.refresh();
    }
    async openLibrary() {
        const epoch = this.widget.epoch, core = this.core;
        const [{ PoseStudioWidget }, { extractActivePoseFromSceneAsset }] = await Promise.all([
            import(new URL("../vnccs_pose_studio.js", this.widget.factoryRuntime.viewerURL).href),
            import(new URL("../pose_studio/characters.mjs", this.widget.factoryRuntime.viewerURL).href),
        ]);
        if (this.disposed || epoch !== this.widget.epoch || core !== this.core || this.element.hidden) return;
        this.closeLibrary();
        const container = document.createElement("div"); container.className = "vnccs-pose-studio cg-pose-library-host";
        this.widget.container.append(container);
        const library = Object.assign(Object.create(PoseStudioWidget.prototype), { container, libraryPoses: [], libraryThumbSize: 240,
            libraryThumbSizeStorageKey: "vnccsPoseLibraryPreviewSize", repositoryProgressStates: {}, isAnimationMode: () => false,
            showMessage: message => this.error(message), renderLibraryInspector() {} });
        const filtered = library.getFilteredLibraryPoses;
        library.getFilteredLibraryPoses = () => filtered.call(library).filter(item => !library.isLibraryAnimation(item));
        library.selectLibraryPose = async item => {
            if (library.applying) return;
            library.applying = true;
            try {
                const response = await fetch(`/vnccs/pose_library/get/${encodeURIComponent(library.getLibraryPoseName(item))}${library.getLibraryPoseQuery(item)}`, { signal: this.events.signal });
                const result = await response.json(); if (!response.ok) throw new Error(result.error || "Could not load pose.");
                if (library._disposed || this.disposed || epoch !== this.widget.epoch || core !== this.core) return;
                const pose = extractActivePoseFromSceneAsset(result.pose) || result.pose;
                if (!pose || pose.animation || pose.type === "pose_set") throw new Error("Choose a single pose.");
                const candidate = { bones: pose.bones || {}, bonePositions: pose.bonePositions || {}, modelRotation: pose.modelRotation || [0, 0, 0] };
                const snapshot = this.widget.characters.snapshot()[0];
                validateCharacters([{ ...snapshot, pose: candidate }]);
                if (library._disposed || this.disposed || epoch !== this.widget.epoch || core !== this.core) return;
                this.widget.characters.edit(rig => rig.setPose(candidate, true)); this.widget.commitGesture(); this.closeLibrary(); this.refresh();
            } catch (error) { if (!library._disposed) this.error(error); }
            finally { library.applying = false; }
        };
        this.library = library; library.showLibraryModal();
        const close = library._closeLibraryModal;
        library._closeLibraryModal = () => { library._disposed = true; close(); container.remove(); if (this.library === library) this.library = null; };
        library.libraryModal.querySelector(".vnccs-ps-modal-close").onclick = () => library._closeLibraryModal();
        library.libraryModal.parentElement.onclick = event => { if (event.target === library.libraryModal?.parentElement) library._closeLibraryModal(); };
        container.addEventListener("keydown", event => { if (event.key === "Escape") { event.stopPropagation(); library._closeLibraryModal(); } });
        library.librarySearchInput.focus();
    }
    closeLibrary() { this.library?._closeLibraryModal?.(); }
    dispose() { this.disposed = true; ++this.serial; this.events.abort(); this.closeLibrary(); this.element.remove(); }
}
