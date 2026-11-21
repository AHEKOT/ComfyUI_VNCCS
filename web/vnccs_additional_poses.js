import { app } from "../../scripts/app.js";
import { createSakuraInteractions, createSakuraSelect } from "./sakura_interactions.mjs";
import { checkedJSON, watchConnection } from "./vnccs_transport.js";
import {
    SAKURA_THEME_CSS, SAKURA_INTERACTION_CSS, SAKURA_CREATOR_CONTROL_CSS,
    createRequestGuard, createSpritePreviewNavigator, enableMiddleMouseCanvasPan, injectStyles,
    registerCleanup, syncDOMWidgetWidthSoon,
} from "./vnccs_common.js";

const NODE = "VNCCS_AdditionalPoses";
const WIDGET = "additional_poses_ui";
const CATALOG = "/vnccs/additional_poses/catalog";
const FIELDS = ["character", "costume", "emotion"];
const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
${SAKURA_INTERACTION_CSS}
${SAKURA_CREATOR_CONTROL_CSS}
.vnccs-additional-poses {
    ${SAKURA_THEME_CSS}
    width: 100%;
    overflow: hidden;
    color: var(--text-primary);
    font: 12px var(--font);
}
.vnccs-additional-poses-fields {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    flex-shrink: 0;
}
.vnccs-additional-poses .vnccs-common-creator-select:disabled { opacity: .55; cursor: default; }
.vnccs-additional-poses :is(.vnccs-common-creator-select, .vnccs-common-creator-sprite-nav-btn):focus-visible {
    outline: 1px solid var(--accent);
    outline-offset: 2px;
}
.vnccs-additional-poses-preview img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
.vnccs-additional-poses-status {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 12px;
    text-align: center;
    overflow-wrap: anywhere;
    color: var(--text-secondary);
}
.vnccs-additional-poses .vnccs-common-creator-sprite-nav { display: flex; flex-shrink: 0; visibility: hidden; }
.vnccs-additional-poses .vnccs-common-creator-sprite-nav.is-visible { visibility: visible; }
`;

function registerPoseStudioSource(sync) {
    if (sync._vnccsAdditionalPosesPatched) return;
    const findSourceNode = sync.findSourceNode;
    const registerStudio = sync.registerStudio;
    const findAdditionalPoses = () => app.graph?._nodes?.find(node =>
        node.type === NODE && node._vnccsGetPoseStudioValues?.());
    sync.findSourceNode = function () {
        return findSourceNode?.apply(this, arguments) || findAdditionalPoses() || null;
    };
    sync.registerStudio = function (studio) {
        const result = registerStudio?.apply(this, arguments);
        const values = findAdditionalPoses()?._vnccsGetPoseStudioValues?.();
        if (values) this.applyToStudio?.(studio, values, { initial: true });
        return result;
    };
    sync._vnccsAdditionalPosesPatched = true;
}

class AdditionalPosesWidget {
    constructor(node) {
        this.node = node;
        this.dataWidget = node.widgets.find(item => item.name === "widget_data");
        this.dataWidget.hidden = true;
        this.dataWidget.computeSize = () => [0, -4];
        this.characters = [];
        this.costumes = [];
        this.emotions = [];
        this.loadGuard = createRequestGuard(node);
        node._vnccsGetPoseStudioValues = () => this.poseStudioValues;
        registerCleanup(node, () => {
            clearTimeout(this.poseSyncTimer);
            delete node._vnccsGetPoseStudioValues;
        });

        injectStyles(CSS, "vnccs-additional-poses");
        const root = document.createElement("div");
        root.className = "vnccs-additional-poses vnccs-common-creator-col vnccs-common-ui-host";
        this.ui = createSakuraInteractions(root, node);
        registerCleanup(node, () => this.ui.dispose());
        enableMiddleMouseCanvasPan(root, node);
        const fields = document.createElement("div");
        fields.className = "vnccs-additional-poses-fields";
        this.selects = Object.fromEntries(FIELDS.map(field => {
            const label = document.createElement("label");
            label.className = "vnccs-common-creator-field";
            const caption = document.createElement("span");
            caption.className = "vnccs-common-creator-label";
            caption.textContent = field[0].toUpperCase() + field.slice(1);
            const select = createSakuraSelect(this.ui, `vnccs-additional-${node.id}-${field}`, [], "",
                value => this.choose(field, value), "vnccs-common-creator-select");
            select.ariaLabel = caption.textContent;
            label.append(caption, select);
            fields.append(label);
            return [field, select];
        }));

        const preview = document.createElement("div");
        preview.className = "vnccs-common-creator-preview-container vnccs-common-ui-preview-host vnccs-additional-poses-preview";
        const image = document.createElement("img");
        image.style.display = "none";
        image.alt = "";
        const aura = document.createElement("div");
        aura.className = "vnccs-common-ui-preview-aura";
        const placeholder = document.createElement("div");
        placeholder.className = "vnccs-additional-poses-status";
        placeholder.setAttribute("role", "status");
        const hint = document.createElement("span");
        hint.className = "vnccs-common-ui-placeholder-hint vnccs-common-creator-placeholder-hint";
        placeholder.append(hint);
        preview.append(aura, image, placeholder);

        const nav = document.createElement("div");
        nav.className = "vnccs-common-creator-sprite-nav";
        const prevButton = this.arrow("Previous sprite", "M15 6l-6 6 6 6");
        const nextButton = this.arrow("Next sprite", "M9 6l6 6-6 6");
        const countLabel = document.createElement("span");
        countLabel.className = "vnccs-common-creator-sprite-nav-count";
        nav.append(prevButton, countLabel, nextButton);
        this.navigator = createSpritePreviewNavigator({
            node, image, placeholder, nav, prevButton, nextButton, countLabel, interactions: this.ui,
            onSelect: index => {
                const name = this.sprites()[index].name;
                if (name === this.state.sprite) return;
                this.state.sprite = name;
                this.persist();
            },
        });
        root.append(fields, preview, nav);
        node.addDOMWidget(WIDGET, "ui", root, { serialize: false, hideOnZoom: false, getMinHeight: () => 380 });
        watchConnection(node, () => this.state && this.load(true), registerCleanup);
        const initialize = setTimeout(() => { if (!this.state) this.restore(); }, 0);
        registerCleanup(node, () => clearTimeout(initialize));
    }

    arrow(label, path) {
        const button = document.createElement("button");
        button.type = "button";
        button.ariaLabel = button.title = label;
        button.className = "vnccs-common-creator-sprite-nav-btn";
        button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="${path}"/></svg>`;
        return button;
    }

    restore() {
        this.poseStudioValues = null;
        this.costumes = [];
        this.emotions = [];
        try {
            const saved = JSON.parse(this.dataWidget.value || "{}");
            if (!saved || typeof saved !== "object" || Array.isArray(saved)) throw new Error();
            this.state = Object.fromEntries([...FIELDS, "sprite"].map(field => [field, typeof saved[field] === "string" ? saved[field] : ""]));
            this.restoreError = false;
        } catch {
            this.state = {};
            this.restoreError = true;
        }
        this.load();
    }

    persist() {
        const value = JSON.stringify(this.state);
        if (this.dataWidget.value === value) return;
        this.dataWidget.value = value;
        this.node.graph?.change?.();
    }

    choose(field, value) {
        const index = FIELDS.indexOf(field);
        this.restoreError = false;
        this.state[field] = value;
        for (const dependent of FIELDS.slice(index + 1)) this.state[dependent] = "";
        this.state.sprite = "";
        if (field === "character") {
            this.poseStudioValues = null;
            this.costumes = [];
        }
        if (field !== "emotion") this.emotions = [];
        if (field === "emotion") this.selectFirstSprite();
        this.persist();
        if (field === "emotion") this.showPreview();
        else this.load();
    }

    renderChoices() {
        const available = [this.characters, this.costumes, this.emotions.map(emotion => emotion.name)];
        FIELDS.forEach((field, index) => {
            const value = this.state[field] || "";
            const choices = available[index].map(name => ({ value: name, label: name }));
            if (value && !available[index].includes(value)) choices.unshift({ value, label: value });
            const select = this.selects[field];
            select.setChoices(choices, value);
            select.disabled = !available[index].length;
            if (this.restoreError && choices.length) {
                select.value = "";
                select.firstElementChild.textContent = "—";
            }
        });
    }

    sprites() {
        return this.emotions.find(emotion => emotion.name === this.state.emotion)?.sprites || [];
    }

    selectFirstSprite() {
        this.state.sprite ||= this.sprites()[0]?.name || "";
    }

    syncPoseStudioValues(attempt = 0) {
        clearTimeout(this.poseSyncTimer);
        const values = this.node._vnccsGetPoseStudioValues?.();
        if (!values) return;
        const sync = window.__vnccsPoseStudioCharacterCreatorSync;
        if (!sync) {
            if (attempt < 40) this.poseSyncTimer = setTimeout(() => this.syncPoseStudioValues(attempt + 1), 250);
            return;
        }
        registerPoseStudioSource(sync);
        sync.sourceSignatures?.set(this.node, values.signature);
        for (const studio of sync.studios || []) sync.applyToStudio?.(studio, values, { force: true });
    }

    async load(quiet = false) {
        const current = this.loadGuard();
        if (!quiet) {
            this.renderChoices();
            this.navigator.loadList("", [], -1, "Loading sprites…");
        }
        try {
            const { characters } = await checkedJSON(CATALOG);
            if (!current()) return;
            this.characters = characters;
            const previous = { ...this.state };
            const character = this.state.character ||= this.restoreError ? "" : characters[0] || "";
            this.renderChoices();
            if (character) {
                const query = new URLSearchParams({ character });
                const [{ costumes }, profile] = await Promise.all([
                    checkedJSON(`${CATALOG}?${query}`),
                    checkedJSON(`/vnccs/character_info?${query}`).catch(() => null),
                ]);
                if (!current()) return;
                if (profile) {
                    const { age } = profile;
                    const savedAge = typeof age === "number" || typeof age === "string" ? Number(age) : NaN;
                    const poseAge = Number.isFinite(savedAge) && savedAge > 0
                        ? Math.max(1, Math.min(90, Math.round(savedAge))) : NaN;
                    const sex = String(profile.sex || profile.gender || "").trim().toLowerCase();
                    const gender = sex === "male" ? 1 : sex === "female" ? 0 : NaN;
                    const values = Number.isFinite(poseAge) || Number.isFinite(gender) ? {
                        age: poseAge,
                        gender,
                        signature: `${Number.isFinite(poseAge) ? poseAge : "?"}|${Number.isFinite(gender) ? gender : "?"}`,
                    } : null;
                    if (values?.signature !== this.poseStudioValues?.signature) {
                        this.poseStudioValues = values;
                        this.syncPoseStudioValues();
                    }
                }
                this.costumes = costumes;
                const costume = this.state.costume ||= costumes[0] || "";
                this.renderChoices();
                if (costume) {
                    query.set("costume", costume);
                    const { emotions } = await checkedJSON(`${CATALOG}?${query}`);
                    if (!current() || this.state.character !== character || this.state.costume !== costume) return;
                    this.emotions = emotions;
                    if (!this.state.emotion && emotions.length) this.state.emotion = emotions[0].name;
                    this.selectFirstSprite();
                    this.renderChoices();
                }
            }
            if (!this.restoreError && [...FIELDS, "sprite"].some(field => previous[field] !== this.state[field])) this.persist();
            this.showPreview();
        } catch (error) {
            if (!current()) return;
            this.navigator.loadList("", [], -1, error.message || "Unable to load sprites.");
        }
    }

    showPreview() {
        const sprites = this.sprites();
        const index = sprites.findIndex(sprite => sprite.name === this.state.sprite);
        let message;
        if (this.restoreError) message = "Could not restore the saved selection. Choose a character to continue.";
        else if (!this.state.character) message = "No saved characters.";
        else if (!this.state.costume) message = "No saved costumes.";
        else if (this.state.sprite) message = `Sprite '${this.state.sprite}' is missing.`;
        else message = "No sprites in this costume.";
        this.navigator.loadList(this.state.character || "", sprites.map(sprite => sprite.url), index, message);
    }
}

app.registerExtension({
    name: "VNCCS.AdditionalPoses",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const { onNodeCreated, onConfigure, onResize, onSerialize } = nodeType.prototype;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            this._vnccsAdditionalPoses = new AdditionalPosesWidget(this);
            this.setSize([360, Math.max(this.size?.[1] || 0, 620)]);
            syncDOMWidgetWidthSoon(this, WIDGET);
        };
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            this._vnccsAdditionalPoses?.restore();
            syncDOMWidgetWidthSoon(this, WIDGET);
        };
        nodeType.prototype.onResize = function () {
            onResize?.apply(this, arguments);
            this._vnccsAdditionalPoses?.ui._closeDropdown(false);
            syncDOMWidgetWidthSoon(this, WIDGET);
        };
        nodeType.prototype.onSerialize = function (serialized) {
            onSerialize?.apply(this, arguments);
            const index = this.widgets.findIndex(widget => widget.name === "widget_data");
            if (index >= 0 && Array.isArray(serialized?.widgets_values)) serialized.widgets_values[index] = this.widgets[index].value;
        };
    },
});
