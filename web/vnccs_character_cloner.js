import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { showModal as showCommonModal, createLoadingOverlay, injectStyles } from "./vnccs_common.js";

// --- STYLES: Sakura Archive Design System ---
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Variables ── */
.vnccs-cloner-container {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-elevated: #1a1a26;
    --bg-surface: #22222e;
    --bg-hover: #2a2a38;
    --text-primary: #e8e8f0;
    --text-secondary: #9898a8;
    --text-muted: #5e5e70;
    --accent: #ff8fa3;
    --accent-hover: #ffb6c8;
    --accent-glow: rgba(255, 143, 163, 0.3);
    --accent-subtle: rgba(255, 143, 163, 0.1);
    --accent-border: rgba(255, 143, 163, 0.22);
    --accent-lavender: #b8a9e8;
    --success: #00d68f;
    --warning: #ffaa00;
    --error: #ff4757;
    --border: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(255, 255, 255, 0.12);
    --font: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --shadow-subtle: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-elevated: 0 8px 32px rgba(0, 0, 0, 0.5);
    --transition: 0.2s ease;
}

/* ── Container ── */
.vnccs-cloner-container {
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 13px;
    width: 100%;
    height: 100%;
    overflow: hidden;
    box-sizing: border-box;
    padding: 12px;
    gap: 10px;
    zoom: 0.67;
    background-image: radial-gradient(ellipse at 20% 0%, rgba(255, 143, 163, 0.04) 0%, transparent 60%),
                      radial-gradient(ellipse at 80% 100%, rgba(184, 169, 232, 0.03) 0%, transparent 60%);
}

/* ── Rows ── */
.vnccs-cloner-top-row {
    display: grid;
    grid-template-columns: 40% 60%;
    gap: 10px;
    flex: 1;
    min-height: 0;
    width: 100%;
}

.vnccs-cloner-bottom-row {
    display: grid;
    grid-template-columns: 40% 60%;
    gap: 10px;
    height: 80px;
    min-height: 80px;
    width: 100%;
    flex-shrink: 0;
}

/* ── Columns ── */
.vnccs-cloner-col {
    display: flex;
    flex-direction: column;
    background: rgba(20, 16, 30, 0.88);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-lg);
    padding: 14px 16px;
    gap: 8px;
    overflow-y: auto;
    height: 100%;
    box-sizing: border-box;
    position: relative;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
.vnccs-cloner-col::before {
    content: '';
    position: absolute;
    top: 0; left: 18%; right: 18%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 143, 163, 0.6), transparent);
    border-radius: 1px;
}
.vnccs-cloner-col::-webkit-scrollbar { width: 4px; }
.vnccs-cloner-col::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

/* ── Section Titles ── */
.vnccs-cloner-section-title {
    font-size: 10px;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 2px;
}
.vnccs-cloner-section-title::before {
    content: '';
    width: 3px;
    height: 12px;
    background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
    border-radius: 2px;
    box-shadow: 0 0 8px var(--accent-glow);
    flex-shrink: 0;
}

/* ── Fields ── */
.vnccs-cloner-field {
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin-bottom: 3px;
    flex-shrink: 0;
}
.vnccs-cloner-label {
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Inputs ── */
.vnccs-cloner-input, .vnccs-cloner-textarea {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: var(--text-primary);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    font-family: var(--font);
    font-size: 12px;
    width: 100%;
    box-sizing: border-box;
    transition: all var(--transition);
}
.vnccs-cloner-select {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: var(--text-primary);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    font-family: var(--font);
    font-size: 12px;
    width: 100%;
    box-sizing: border-box;
    transition: all var(--transition);
}
.vnccs-cloner-input:focus, .vnccs-cloner-select:focus, .vnccs-cloner-textarea:focus {
    outline: none;
    border-color: var(--accent-border);
    background: rgba(255, 143, 163, 0.03);
    box-shadow: 0 0 0 3px rgba(255, 143, 163, 0.05);
}

/* ── Buttons ── */
.vnccs-cloner-btn-row {
    display: flex;
    gap: 6px;
    margin-top: auto;
    flex-shrink: 0;
}
.vnccs-cloner-btn {
    flex: 1;
    padding: 8px 12px;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    font-family: var(--font);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 10px;
    color: white;
    text-align: center;
    transition: all var(--transition);
}
.vnccs-cloner-btn-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
    color: #1a1525;
    box-shadow: 0 4px 20px rgba(255, 143, 163, 0.25);
    position: relative;
    overflow: hidden;
}
.vnccs-cloner-btn-primary::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.22) 45%, rgba(255,255,255,0.32) 50%, rgba(255,255,255,0.22) 55%, transparent 100%);
    transform: translateX(-120%) skewX(-15deg);
    animation: clonerBtnShimmer 3.5s ease-in-out infinite;
    pointer-events: none;
}
@keyframes clonerBtnShimmer {
    0%   { transform: translateX(-120%) skewX(-15deg); opacity: 1; }
    35%  { transform: translateX(120%)  skewX(-15deg); opacity: 1; }
    100% { transform: translateX(120%)  skewX(-15deg); opacity: 0; }
}
.vnccs-cloner-btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(255, 143, 163, 0.45), 0 0 0 1px var(--accent-glow);
}
.vnccs-cloner-btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

.vnccs-cloner-btn-success {
    background: rgba(0, 214, 143, 0.15);
    border: 1px solid rgba(0, 214, 143, 0.3);
    color: var(--success);
}
.vnccs-cloner-btn-success:hover { background: rgba(0, 214, 143, 0.25); border-color: var(--success); }

.vnccs-cloner-btn-danger {
    background: rgba(255, 71, 87, 0.15);
    border: 1px solid rgba(255, 71, 87, 0.3);
    color: var(--error);
}
.vnccs-cloner-btn-danger:hover { background: rgba(255, 71, 87, 0.25); border-color: var(--error); }

.vnccs-cloner-btn-upload {
    background: rgba(255, 143, 163, 0.08);
    border: 1px dashed var(--accent-border);
    color: var(--accent);
    border-radius: var(--radius-md);
    padding: 12px 24px;
    cursor: pointer;
    font-family: var(--font);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    transition: all var(--transition);
}
.vnccs-cloner-btn-upload:hover {
    background: rgba(255, 143, 163, 0.15);
    border-color: var(--accent);
    box-shadow: 0 0 16px var(--accent-subtle);
}

/* ── Image Grid ── */
.vnccs-cloner-img-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.vnccs-cloner-thumb {
    width: 60px;
    height: 60px;
    object-fit: cover;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all var(--transition);
}
.vnccs-cloner-thumb:hover { border-color: var(--accent); box-shadow: 0 0 8px var(--accent-subtle); }
.vnccs-cloner-thumb.generating {
    border: 2px solid var(--accent);
    animation: clonerPulse 1s infinite alternate;
    box-shadow: 0 0 12px var(--accent-glow);
}
@keyframes clonerPulse { from { opacity: 0.6; } to { opacity: 1; } }

/* ── Toggle Switch ── */
.vnccs-cloner-toggle-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    user-select: none;
}
.vnccs-cloner-toggle-wrap span {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
}
.vnccs-cloner-toggle {
    position: relative;
    width: 36px;
    height: 20px;
    flex-shrink: 0;
}
.vnccs-cloner-toggle input { opacity: 0; width: 0; height: 0; position: absolute; }
.vnccs-cloner-toggle-track {
    position: absolute;
    inset: 0;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    transition: all var(--transition);
}
.vnccs-cloner-toggle input:checked + .vnccs-cloner-toggle-track {
    background: rgba(255, 143, 163, 0.25);
    border-color: var(--accent-border);
    box-shadow: 0 0 8px var(--accent-subtle);
}
.vnccs-cloner-toggle-thumb {
    position: absolute;
    top: 3px;
    left: 3px;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all var(--transition);
    box-shadow: 0 1px 4px rgba(0,0,0,0.4);
}
.vnccs-cloner-toggle input:checked ~ .vnccs-cloner-toggle-thumb {
    transform: translateX(16px);
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent-glow);
}

/* ── Preview Container ── */
.vnccs-cloner-preview-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    color: var(--text-muted);
    pointer-events: none;
}
.vnccs-cloner-preview-placeholder svg { opacity: 0.35; }
.vnccs-cloner-preview-placeholder-text {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Loading Overlay ── */
.vnccs-cloner-loading-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(10, 10, 15, 0.92);
    backdrop-filter: blur(10px);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 1000; pointer-events: auto; gap: 16px;
    border-radius: var(--radius-lg);
}
.vnccs-cloner-spinner {
    width: 44px;
    height: 44px;
    position: relative;
}
.vnccs-cloner-spinner::before, .vnccs-cloner-spinner::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
    border: 3px solid transparent;
}
.vnccs-cloner-spinner::before {
    border-top-color: var(--accent);
    border-right-color: rgba(255, 143, 163, 0.3);
    animation: clonerSpin 1s linear infinite;
    box-shadow: 0 0 18px rgba(255, 143, 163, 0.2);
}
.vnccs-cloner-spinner::after {
    inset: 7px;
    border-bottom-color: rgba(184, 169, 232, 0.6);
    border-left-color: rgba(184, 169, 232, 0.2);
    animation: clonerSpin 1.4s linear infinite reverse;
}
@keyframes clonerSpin { to { transform: rotate(360deg); } }
.vnccs-cloner-loading-text {
    color: var(--text-secondary);
    font-family: var(--font);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.vnccs-cloner-loading-dots::after {
    content: '';
    animation: clonerDots 1.5s steps(4, end) infinite;
}
@keyframes clonerDots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
}
`;

app.registerExtension({
    name: "VNCCS.CharacterCloner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "CharacterCloner") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const node = this;
                node.setSize([900, 700]); // 2-Column fitting

                // 1. Setup CSS
                injectStyles(STYLE, "vnccs-character-cloner");

                // 2. Cleanup Widgets
                const cleanup = () => {
                    if (!node.widgets) return;
                    for (const w of node.widgets) {
                        if (w.name !== "ui" && w.name !== "widget_data") {
                            w.hidden = true;
                        }
                    }
                };
                cleanup();
                const origDraw = node.onDrawBackground;
                node.onDrawBackground = function (ctx) {
                    cleanup();
                    if (origDraw) origDraw.apply(this, arguments);
                };

                // 3. State
                let dataWidget = node.widgets ? node.widgets.find(w => w.name === "widget_data") : null;
                if (!dataWidget) {
                    dataWidget = node.addWidget("text", "widget_data", "{}", (v) => { }, { serialize: true });
                }
                dataWidget.hidden = true;

                const state = {
                    character: "",
                    source_images: [], // List of filenames
                    character_info: {
                        sex: "female", age: 18, race: "human", skin_color: "",
                        hair: "", eyes: "", face: "", body: "", additional_details: "",
                        nsfw: false, aesthetics: "masterpiece, best quality",
                        negative_prompt: "bad quality, worst quality",
                        lora_prompt: "", background_color: "Green"
                    }
                };

                const els = {};
                const saveState = () => {
                    if (dataWidget) dataWidget.value = JSON.stringify(state);
                };

                const loadState = () => {
                    if (dataWidget && dataWidget.value && dataWidget.value !== "{}") {
                        try {
                            console.log("[VNCCS] Loading State from widget_data:", dataWidget.value);
                            const parsed = JSON.parse(dataWidget.value);
                            Object.assign(state, parsed);
                            // Update UI
                            updateUIFromState();
                            console.log("[VNCCS] State loaded successfully.");
                        } catch (e) {
                            console.error("[VNCCS] Failed to load state:", e);
                        }
                    }
                };

                // React to external changes (e.g. reload)
                if (dataWidget) dataWidget.callback = loadState;

                // Explicitly handle graph configuration (Restoration on Reload)
                const origConfigure = node.onConfigure;
                node.onConfigure = function () {
                    if (origConfigure) origConfigure.apply(this, arguments);
                    console.log("[VNCCS] onConfigure triggered.");
                    loadState();
                };

                // UI Builders
                const createField = (lbl, key, type = "text", opts = [], targetObj = state.character_info) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-cloner-field";

                    if (type === "checkbox") {
                        const toggleWrap = document.createElement("label");
                        toggleWrap.className = "vnccs-cloner-toggle-wrap";

                        const toggle = document.createElement("div");
                        toggle.className = "vnccs-cloner-toggle";

                        const inp = document.createElement("input");
                        inp.type = "checkbox";
                        inp.checked = !!targetObj[key];
                        inp.onchange = (e) => { targetObj[key] = e.target.checked; saveState(); };

                        const track = document.createElement("div");
                        track.className = "vnccs-cloner-toggle-track";
                        const thumb = document.createElement("div");
                        thumb.className = "vnccs-cloner-toggle-thumb";

                        toggle.appendChild(inp);
                        toggle.appendChild(track);
                        toggle.appendChild(thumb);

                        const labelSpan = document.createElement("span");
                        labelSpan.textContent = lbl;

                        toggleWrap.appendChild(toggle);
                        toggleWrap.appendChild(labelSpan);
                        wrap.appendChild(toggleWrap);
                        els[key] = inp;
                        return wrap;
                    }

                    const header = document.createElement("div");
                    header.style.display = "flex"; header.style.justifyContent = "space-between";
                    header.innerHTML = `<div class="vnccs-cloner-label">${lbl}</div>`;
                    wrap.appendChild(header);

                    let inp;
                    if (type === "select") {
                        inp = document.createElement("select"); inp.className = "vnccs-cloner-select";
                        opts.forEach(v => inp.add(new Option(v, v)));
                        inp.value = targetObj[key] || opts[0];
                        inp.onchange = (e) => { targetObj[key] = e.target.value; saveState(); };
                    } else if (type === "number") {
                        inp = document.createElement("input"); inp.className = "vnccs-cloner-input";
                        inp.type = "number";
                        inp.value = targetObj[key];
                        inp.onchange = (e) => { targetObj[key] = parseFloat(e.target.value); saveState(); };
                    } else {
                        inp = document.createElement("input"); inp.className = "vnccs-cloner-input";
                        inp.value = targetObj[key] || "";
                        inp.onchange = (e) => { targetObj[key] = e.target.value; saveState(); };
                    }
                    els[key] = inp;
                    wrap.appendChild(inp);
                    return wrap;
                };

                // Define renderThumbs placeholder (populated later)
                let renderThumbs = () => { };

                const updateUIFromState = () => {
                    // Fields
                    for (let k in state.character_info) {
                        if (els[k]) {
                            let val = state.character_info[k];
                            // Normalize background_color case to match dropdown options
                            if (k === "background_color" && typeof val === "string" && val) {
                                val = val.charAt(0).toUpperCase() + val.slice(1).toLowerCase();
                                state.character_info[k] = val;
                            }
                            if (els[k].type === "checkbox") els[k].checked = val;
                            else els[k].value = val;
                        }
                    }
                    // Images
                    if (renderThumbs) renderThumbs();
                };

                // --- Helpers (Hoisted) ---
                // Modal Helper — delegates to vnccs_common showModal
                const showModal = (title, contentFunc, buttons) => {
                    const mappedButtons = buttons.map(b => ({
                        ...b,
                        class: b.class?.includes("danger") ? "danger" : b.class?.includes("primary") ? "primary" : undefined
                    }));
                    return showCommonModal(container, title, contentFunc, mappedButtons);
                };

                const loadChar = async (name, skipInfoLoad = false) => {
                    if (!name || name === "None") {
                        state.char_preview_url = null;
                        updateUIFromState();
                        return;
                    }
                    try {
                        // Skip loading character_info if restoring from widget_data
                        if (!skipInfoLoad) {
                            const r = await api.fetchApi(`/vnccs/config?name=${encodeURIComponent(name)}`);
                            if (r.ok) {
                                const d = await r.json();
                                if (d.character_info) {
                                    Object.assign(state.character_info, d.character_info);
                                    updateUIFromState();
                                }
                            }
                        } else {
                            // Still update UI from current state
                            updateUIFromState();
                        }

                        // Load Preview Image URL Logic
                        const ts = Date.now();
                        const sheetUrl = `/vnccs/get_character_sheet_preview?character=${encodeURIComponent(name)}&t=${ts}`;
                        const cacheUrl = `/vnccs/get_cached_preview?character=${encodeURIComponent(name)}&t=${ts}`;

                        // Helper to find valid one
                        const checkImage = (url) => {
                            return new Promise((resolve) => {
                                const img = new Image();
                                img.onload = () => resolve(url);
                                img.onerror = () => resolve(null);
                                img.src = url;
                            });
                        };

                        let finalUrl = await checkImage(sheetUrl);
                        if (!finalUrl) finalUrl = await checkImage(cacheUrl);

                        state.char_preview_url = finalUrl;
                        updateUIFromState(); // Trigger renderThumbs

                    } catch (e) {
                        console.error(e);
                        state.char_preview_url = null;
                        updateUIFromState();
                    }
                };

                const loadCharList = async () => {
                    try {
                        const r = await api.fetchApi("/vnccs/context_lists");
                        if (!r.ok) return;
                        const d = await r.json();

                        els.charSelect.innerHTML = "";
                        if (!d.characters || !d.characters.length) els.charSelect.add(new Option("None", ""));
                        else d.characters.forEach(c => els.charSelect.add(new Option(c, c)));

                        // Set Value
                        if (state.character && Array.from(els.charSelect.options).some(o => o.value === state.character)) {
                            els.charSelect.value = state.character;
                        } else if (els.charSelect.options.length > 0) {
                            state.character = els.charSelect.options[0].value;
                            els.charSelect.value = state.character;
                            saveState();
                        }

                        // Wait for loadChar to finish so preview updates
                        // Skip loading info if we already have widget_data (restoring session)
                        const hasWidgetData = state.character_info && state.character_info.hair !== undefined;
                        if (state.character) await loadChar(state.character, hasWidgetData);

                    } catch (e) { console.error(e); }
                };

                const doCreate = () => {
                    let inpRef;
                    showModal("New Character", () => {
                        const inp = document.createElement("input");
                        inp.className = "vnccs-cloner-input";
                        inp.placeholder = "Name...";
                        inpRef = inp;
                        return inp;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "Create", class: "vnccs-btn-primary",
                            action: async (ol, btn) => {
                                const n = inpRef.value.trim();
                                if (!n) return true;
                                try {
                                    await api.fetchApi(`/vnccs/create?name=${encodeURIComponent(n)}`);
                                    const exists = Array.from(els.charSelect.options).some(o => o.value === n);
                                    if (!exists) els.charSelect.add(new Option(n, n));
                                    state.character = n; els.charSelect.value = n;
                                    await loadChar(n);
                                    saveState();
                                    return false;
                                } catch (e) {
                                    showModal("Error", () => { const d = document.createElement("div"); d.innerText = "Create Failed: " + e; return d; }, [{ text: "Close" }]);
                                    return true;
                                }
                            }
                        }
                    ]);
                    inpRef.focus();
                };

                const doDelete = () => {
                    const charName = state.character;
                    if (!charName || charName === "None") return;
                    showModal("Delete Character", () => {
                        const d = document.createElement("div");
                        d.innerHTML = `Are you sure you want to delete <b>${charName}</b>?`;
                        return d;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "DELETE", class: "vnccs-btn-danger",
                            action: async (ol, btn) => {
                                try {
                                    const r = await api.fetchApi(`/vnccs/delete?name=${encodeURIComponent(charName)}`);
                                    if (r.ok) {
                                        const idx = Array.from(els.charSelect.options).findIndex(o => o.value === charName);
                                        if (idx > -1) els.charSelect.remove(idx);
                                        if (els.charSelect.options.length > 0) state.character = els.charSelect.options[0].value;
                                        else state.character = "";
                                        els.charSelect.value = state.character;
                                        await loadChar(state.character);
                                        saveState();
                                        return false;
                                    }
                                } catch (e) {
                                    showModal("Error", () => { const d = document.createElement("div"); d.innerText = "" + e; return d; }, [{ text: "Close" }]);
                                    return false;
                                }
                            }
                        }
                    ]);
                };

                // --- LAYOUT ---
                const container = document.createElement("div");
                container.className = "vnccs-cloner-container";

                // Top Row
                const topRow = document.createElement("div");
                topRow.className = "vnccs-cloner-top-row";

                // COL 1: SOURCE IMAGES
                // COL 1: SOURCE IMAGES (Left Column)
                const colSrc = document.createElement("div");
                colSrc.className = "vnccs-cloner-col";

                // --- CHARACTER SELECTOR (Moved to Left Top) ---
                colSrc.innerHTML = '<div class="vnccs-cloner-section-title">Character Select</div>';

                const charRow = document.createElement("div");
                charRow.className = "vnccs-cloner-field";

                const charSel = document.createElement("select");
                charSel.className = "vnccs-cloner-select";
                charSel.onchange = async (e) => {
                    state.character = e.target.value;
                    await loadChar(state.character);
                    saveState();
                };
                els.charSelect = charSel;
                charRow.appendChild(charSel);
                colSrc.appendChild(charRow);

                // (Preview removed: Using Native Preview Window below)

                const btnRow = document.createElement("div");
                btnRow.className = "vnccs-cloner-btn-row";

                const btnNew = document.createElement("button");
                btnNew.className = "vnccs-cloner-btn vnccs-cloner-btn-success";
                btnNew.innerText = "NEW";
                btnNew.onclick = doCreate;

                const btnDel = document.createElement("button");
                btnDel.className = "vnccs-cloner-btn vnccs-cloner-btn-danger";
                btnDel.innerText = "DEL";
                btnDel.onclick = doDelete;

                btnRow.appendChild(btnNew);
                btnRow.appendChild(btnDel);
                colSrc.appendChild(btnRow);

                // --- SOURCE IMAGES SECTION ---
                const srcHeader = document.createElement("div");
                srcHeader.className = "vnccs-cloner-section-title";
                srcHeader.innerText = "Source Images";
                srcHeader.style.marginTop = "15px";
                colSrc.appendChild(srcHeader);

                const imgList = document.createElement("div");
                imgList.className = "vnccs-cloner-img-list";
                colSrc.appendChild(imgList);

                // --- IMAGE PREVIEW/UPLOAD AREA ---
                // Container for the Large Preview
                const previewContainer = document.createElement("div");
                previewContainer.className = "vnccs-cloner-preview-container";
                previewContainer.style.flex = "1";
                previewContainer.style.position = "relative";
                previewContainer.style.border = "1px solid rgba(255, 143, 163, 0.15)";
                previewContainer.style.borderRadius = "12px";
                previewContainer.style.background = "radial-gradient(circle, rgba(255, 143, 163, 0.04) 1px, transparent 1px), rgba(15, 12, 22, 0.6)";
                previewContainer.style.backgroundSize = "20px 20px, 100% 100%";
                previewContainer.style.display = "flex";
                previewContainer.style.flexDirection = "column";
                previewContainer.style.alignItems = "center";
                previewContainer.style.justifyContent = "center";
                previewContainer.style.overflow = "hidden";
                previewContainer.style.minHeight = "300px";

                // The Large Image Element
                const previewImg = document.createElement("img");
                previewImg.style.maxWidth = "100%";
                previewImg.style.maxHeight = "100%";
                previewImg.style.objectFit = "contain";
                previewImg.style.display = "none";
                previewContainer.appendChild(previewImg);

                // Placeholder (shown when no image)
                const previewPlaceholder = document.createElement("div");
                previewPlaceholder.className = "vnccs-cloner-preview-placeholder";
                previewPlaceholder.innerHTML = `
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="7" r="4" stroke="#ff8fa3" stroke-width="1.5"/>
                        <path d="M4 20c0-4 3.582-7 8-7s8 3 8 7" stroke="#ff8fa3" stroke-width="1.5" stroke-linecap="round"/>
                        <rect x="1" y="1" width="22" height="22" rx="4" stroke="#ff8fa3" stroke-width="1" stroke-dasharray="3 3" opacity="0.4"/>
                    </svg>
                    <div class="vnccs-cloner-preview-placeholder-text">Source Preview</div>
                `;
                previewContainer.appendChild(previewPlaceholder);

                // Overlay Controls (Upload Button) - Always visible or overlay?
                // User wants standard behaviour: Empty -> Upload Button. Filled -> Image + Mini Overlay?
                // Let's make the click on image trigger upload if empty?
                // Or keep the upload button below?
                // User said "Huge square with +upload images IS the place for preview"

                const uploadOverlay = document.createElement("div");
                uploadOverlay.className = "vnccs-cloner-upload-overlay";
                uploadOverlay.style.position = "absolute";
                uploadOverlay.style.inset = "0";
                uploadOverlay.style.display = "flex";
                uploadOverlay.style.alignItems = "center";
                uploadOverlay.style.justifyContent = "center";
                uploadOverlay.style.cursor = "pointer";
                uploadOverlay.style.zIndex = "10";

                // If image exists, make overlay transparent or subtle?
                // Let's replicate the Button look but centered
                const uploadBtn = document.createElement("button");
                uploadBtn.className = "vnccs-cloner-btn vnccs-cloner-btn-upload";
                uploadBtn.innerText = "+ UPLOAD IMAGES";
                uploadBtn.style.padding = "10px 20px";
                uploadOverlay.appendChild(uploadBtn);

                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.multiple = true;
                fileInput.accept = "image/*";
                fileInput.style.display = "none";
                fileInput.onchange = async (e) => {
                    if (e.target.files.length) {
                        uploadBtn.innerText = "UPLOADING...";
                        for (const file of e.target.files) {
                            try {
                                const body = new FormData();
                                body.append("image", file);
                                const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                                const json = await resp.json();
                                if (json.name) {
                                    state.source_images.push({
                                        name: json.name,
                                        type: json.type || "input",
                                        subfolder: json.subfolder || ""
                                    });
                                }
                            } catch (err) {
                                showModal("Upload Error", () => { const d = document.createElement("div"); d.innerText = "Upload Failed: " + err; return d; }, [{ text: "Close" }]);
                            }
                        }
                        uploadBtn.innerText = "+ UPLOAD IMAGES";
                        saveState();
                        renderThumbs();
                    }
                };

                // Click anywhere on overlay triggers upload
                uploadOverlay.onclick = (e) => {
                    // If clicking button, it propagates. 
                    // But if we want to support clicking image to swap?
                    // For now, let's keep it simple: Button triggers input.
                    if (!previewImg.src || previewImg.style.display === "none") {
                        fileInput.click();
                    } else {
                        // If image is showing, maybe we don't want giant click area?
                        // User can still drag/drop (not implemented yet) or use button.
                        // Let's stick to button click.
                        if (e.target === uploadBtn) fileInput.click();
                        // If clicking background (not button), do nothing?
                    }
                };

                // Allow clicking button specifically even if overlay is transparent
                uploadBtn.onclick = (e) => { e.stopPropagation(); fileInput.click(); };

                previewContainer.appendChild(uploadOverlay);
                colSrc.appendChild(previewContainer);
                colSrc.appendChild(fileInput);


                // COL 2: ATTRIBUTES (Auto Gen)
                const colAttr = document.createElement("div");
                colAttr.className = "vnccs-cloner-col";

                const attrHeader = document.createElement("div");
                attrHeader.className = "vnccs-cloner-section-title";
                attrHeader.innerText = "Attributes";

                const autoGenBtn = document.createElement("button");
                autoGenBtn.className = "vnccs-cloner-btn vnccs-cloner-btn-primary";
                autoGenBtn.style.padding = "10px 12px";
                autoGenBtn.style.fontSize = "12px";
                autoGenBtn.style.flex = "0 0 auto"; // Prevent stretching
                autoGenBtn.innerText = "Analyze Captions (Qwen2-VL-7B)";
                autoGenBtn.onclick = async () => {
                    if (autoGenBtn.disabled) return;

                    if (!state.source_images.length) {
                        showModal("No Images", (m) => {
                            const d = document.createElement("div");
                            d.innerText = "Please upload at least one source image to analyze.";
                            return d;
                        }, [{ text: "OK", class: "vnccs-btn-primary" }]);
                        return;
                    }

                    // USE SELECTED IMAGE
                    const selIdx = (typeof state.selected_idx === 'number') ? state.selected_idx : 0;
                    const imgName = state.source_images[selIdx];

                    // Show loading overlay
                    const loading = createLoadingOverlay(container, "Analyzing image");

                    autoGenBtn.innerText = "ANALYZING...";
                    autoGenBtn.disabled = true;

                    // Visual feedback highlighting selected thumb
                    const thumbs = imgList.querySelectorAll("img");
                    let thumb = null;
                    if (thumbs[selIdx]) {
                        thumb = thumbs[selIdx];
                        thumb.classList.add("generating");
                    }

                    try {
                        const r = await api.fetchApi("/vnccs/cloner_auto_generate", {
                            method: "POST",
                            body: JSON.stringify({ image_name: imgName })
                        });

                        if (r.ok) {
                            const data = await r.json();
                            console.log("[VNCCS] Auto-Gen Success. Data:", data);

                            // Merge into state
                            Object.assign(state.character_info, data);
                            console.log("[VNCCS] State Updated:", state.character_info);

                            updateUIFromState();
                            console.log("[VNCCS] UI Updated.");
                            saveState();
                        } else {
                            console.log("[VNCCS] Auto-Gen Failed. Status:", r.status);
                            // Check for structured errors (404 for model, 500 for mmproj/other)
                            let err = null;
                            try { err = await r.json(); } catch (e) { }

                            if (err && (err.error === "MODEL_MISSING" || err.error === "MMPROJ_MISSING" || err.error === "DEPENDENCY_MISSING")) {

                                // CASE A: Dependency Error (llama-cpp-python)
                                if (err.error === "DEPENDENCY_MISSING") {
                                    showModal("Dependency Missing", (m) => {
                                        const d = document.createElement("div");
                                        d.innerHTML = `
                                            <div style="font-size:13px; margin-bottom:10px;">
                                                <b>Missing AI Library</b><br/><br/>
                                                ${err.message}<br/>
                                                <b>Model:</b> ${err.model_name}<br/><br/>
                                                <span style="color:#f88">Correct 'llama-cpp-python' version required.</span><br/>
                                                Please install the JamePeng fork or a compatible version manually.
                                            </div>
                                        `;
                                        return d;
                                    }, [
                                        { text: "OK", class: "vnccs-btn-danger" }
                                    ]);
                                    return;
                                }

                                // CASE B: Regular Missing Model Files
                                showModal("Model Missing", (m) => {
                                    const d = document.createElement("div");
                                    const missingMsg = err.error === "MMPROJ_MISSING" ? "The Vision Projector (mmproj) is missing." : `Required Model: ${err.model_name || 'QwenVL'}`;
                                    d.innerHTML = `
                                        <div style="font-size:13px; margin-bottom:10px;">
                                            <b>${missingMsg}</b><br/><br/>
                                            This component is required for character analysis. Would you like to download it now?<br/>
                                            <span style="color:#aaa; font-size:11px;">Verification & Download will start automatically.</span>
                                        </div>
                                    `;
                                    return d;
                                }, [
                                    { text: "Cancel" },
                                    {
                                        text: "DOWNLOAD & INSTALL", class: "vnccs-btn-success",
                                        action: async (ol, btn) => {
                                            // Trigger Download
                                            try {
                                                const dl = await api.fetchApi("/vnccs/cloner_download_model", { method: "POST" });
                                                if (dl.status === 409) {
                                                    // already downloading
                                                    startProgressPolling();
                                                    return false;
                                                }
                                                if (dl.ok) {
                                                    ol.remove(); // Close prompt
                                                    startProgressPolling(); // View Progress
                                                    return false;
                                                }
                                            } catch (e) {
                                                showModal("Error", () => { const d = document.createElement("div"); d.innerText = "Download trigger failed: " + e; return d; }, [{ text: "Close" }]);
                                            }
                                            return true;
                                        }
                                    }
                                ]);
                                return; // Exit normally
                            }

                            // Fallback for real errors
                            const txt = err && err.message ? err.message : (await r.text());
                            showModal("Error", (m) => {
                                const d = document.createElement("div");
                                d.innerText = "Auto-Gen Failed: " + txt;
                                d.style.color = "red";
                                return d;
                            }, [{ text: "Close" }]);
                        }
                    } catch (e) {
                        showModal("Error", (m) => {
                            const d = document.createElement("div"); d.innerText = "Script Error: " + e; return d;
                        }, [{ text: "Close" }]);
                    } finally {
                        loading.remove();
                        autoGenBtn.innerText = "Analyze Captions (Qwen2-VL-7B)";
                        autoGenBtn.disabled = false;
                        if (thumb) thumb.classList.remove("generating");
                    }
                };

                // Helper: Progress Polling
                const startProgressPolling = () => {
                    const { overlay, modal } = showModal("Downloading Model...", (m) => {
                        const d = document.createElement("div");
                        d.style.width = "100%";
                        d.innerHTML = `
                             <div style="margin-bottom:5px; font-size:12px;" id="vnccs-dl-status">Starting...</div>
                             <div style="width:100%; height:20px; background:#444; border-radius:4px; overflow:hidden;">
                                <div id="vnccs-dl-bar" style="width:0%; height:100%; background:#2e7d32; transition: width 0.5s;"></div>
                             </div>
                             <div style="text-align:right; font-size:11px; color:#aaa; margin-top:5px;" id="vnccs-dl-pct">0%</div>
                        `;
                        return d;
                    }, []); // No buttons, auto-close

                    const statusEl = modal.querySelector("#vnccs-dl-status");
                    const barEl = modal.querySelector("#vnccs-dl-bar");
                    const pctEl = modal.querySelector("#vnccs-dl-pct");

                    const interval = setInterval(async () => {
                        try {
                            const r = await api.fetchApi("/vnccs/cloner_download_status");
                            if (r.ok) {
                                const d = await r.json();
                                if (d.status === "completed") {
                                    clearInterval(interval);
                                    statusEl.innerText = "Download Complete!";
                                    barEl.style.width = "100%";
                                    pctEl.innerText = "100%";
                                    setTimeout(() => overlay.remove(), 1000);
                                    // Maybe retry generation automatically? No, user can click.
                                } else if (d.status === "error") {
                                    clearInterval(interval);
                                    statusEl.innerText = "Error: " + d.error;
                                    statusEl.style.color = "red";
                                } else {
                                    statusEl.innerText = `Downloading ${d.current_file}...`;
                                    barEl.style.width = d.progress + "%";
                                    pctEl.innerText = d.progress + "%";
                                }
                            }
                        } catch (e) {
                            clearInterval(interval);
                            overlay.remove();
                        }
                    }, 1000);
                };

                // Add Header
                colAttr.appendChild(attrHeader);

                // Add Button (Below header, above fields)
                autoGenBtn.style.width = "100%";
                autoGenBtn.style.marginBottom = "10px";
                autoGenBtn.style.textAlign = "center";
                colAttr.appendChild(autoGenBtn);



                // Other fields
                // colAttr.appendChild(createField("Name", "character", "text", [], state)); // REMOVE THIS

                colAttr.appendChild(createField("Background", "background_color", "select", ["Green", "Blue"]));
                colAttr.appendChild(createField("Sex", "sex", "select", ["female", "male"]));
                colAttr.appendChild(createField("Age", "age", "number"));
                colAttr.appendChild(createField("Race", "race", "text"));
                colAttr.appendChild(createField("Skin Color", "skin_color", "text"));
                colAttr.appendChild(createField("Hair", "hair", "text"));
                colAttr.appendChild(createField("Eyes", "eyes", "text"));
                colAttr.appendChild(createField("Face", "face", "text"));
                colAttr.appendChild(createField("Body", "body", "text"));
                colAttr.appendChild(createField("Details", "additional_details", "text"));
                colAttr.appendChild(createField("Aesthetics", "aesthetics", "text"));
                colAttr.appendChild(createField("NSFW", "nsfw", "checkbox"));

                // Assemble Top Row
                topRow.appendChild(colSrc);
                topRow.appendChild(colAttr);
                container.appendChild(topRow);

                // --- BOTTOM ROW (Prompts) --
                const botRow = document.createElement("div");
                botRow.className = "vnccs-cloner-bottom-row";

                // Read-only Prompts? Or generated?
                // Standard textareas
                const createTA = (lbl, key) => {
                    const w = document.createElement("div");
                    w.className = "vnccs-cloner-col";
                    w.innerHTML = `<div class="vnccs-cloner-label">${lbl}</div>`;
                    const t = document.createElement("textarea");
                    t.className = "vnccs-cloner-textarea";
                    t.style.flex = "1";
                    t.style.resize = "none";
                    // We don't bind this to state input, but output?
                    // Actually, output prompts are generated from attributes. 
                    // Users might want to ADD to it. 
                    // For now, let's bind to lora_prompt or neg_prompt
                    if (key === "lora_prompt") {
                        t.value = state.character_info.lora_prompt;
                        t.onchange = (e) => { state.character_info.lora_prompt = e.target.value; saveState(); }
                    } else if (key === "negative_prompt") {
                        t.value = state.character_info.negative_prompt;
                        t.onchange = (e) => { state.character_info.negative_prompt = e.target.value; saveState(); }
                    }
                    // If positive, it's auto-generated... maybe just show extra prompt field?
                    // Let's stick to "LoRA Trigger / Manual Prompt" and "Negative"
                    w.appendChild(t);
                    return w;
                };

                botRow.appendChild(createTA("Extra / LoRA Prompt", "lora_prompt"));
                botRow.appendChild(createTA("Negative Prompt", "negative_prompt"));

                container.appendChild(botRow);

                // ADD WIDGET
                node.addDOMWidget("ui", "ui", container, { serialize: false, hideOnZoom: false });

                // Define Render Thumbs Logic
                renderThumbs = () => {
                    imgList.innerHTML = "";
                    const images = state.source_images;

                    // Default Selection
                    if (typeof state.selected_idx !== 'number' || state.selected_idx < 0 || state.selected_idx >= images.length) {
                        state.selected_idx = 0;
                    }

                    // Preview Logic
                    if (images.length === 0) {
                        if (state.char_preview_url) {
                            previewImg.src = state.char_preview_url;
                            previewImg.style.display = "block";
                            previewPlaceholder.style.display = "none";
                            uploadOverlay.style.opacity = "0";
                            previewContainer.onmouseenter = () => uploadOverlay.style.opacity = "1";
                            previewContainer.onmouseleave = () => uploadOverlay.style.opacity = "0";
                        } else {
                            previewImg.style.display = "none";
                            previewImg.src = "";
                            previewPlaceholder.style.display = "flex";
                            uploadOverlay.style.opacity = "1";
                            uploadOverlay.style.background = "transparent";
                            uploadBtn.style.display = "block";
                            previewContainer.onmouseenter = null;
                            previewContainer.onmouseleave = null;
                        }
                    } else {
                        // Show Selected Image
                        const imgObj = images[state.selected_idx];
                        let name = "", type = "input", sub = "";
                        if (typeof imgObj === 'string') { name = imgObj; }
                        else if (imgObj && imgObj.name) { name = imgObj.name; type = imgObj.type || "input"; sub = imgObj.subfolder || ""; }

                        if (name) {
                            const params = new URLSearchParams();
                            params.append("filename", name);
                            params.append("type", type);
                            if (sub) params.append("subfolder", sub);
                            previewImg.src = api.apiURL("/view?" + params.toString());
                            previewImg.style.display = "block";
                            previewPlaceholder.style.display = "none";
                            uploadOverlay.style.opacity = "0";
                            previewContainer.onmouseenter = () => uploadOverlay.style.opacity = "1";
                            previewContainer.onmouseleave = () => uploadOverlay.style.opacity = "0";
                        }
                    }

                    // Thumbnails Loop
                    images.forEach((imgObj, idx) => {
                        let name = "", type = "input", sub = "";
                        if (typeof imgObj === 'string') { name = imgObj; }
                        else if (imgObj && imgObj.name) { name = imgObj.name; type = imgObj.type || "input"; sub = imgObj.subfolder || ""; }
                        if (!name) return;

                        const wrap = document.createElement("div");
                        wrap.style.position = "relative";
                        wrap.style.display = "inline-block";
                        wrap.style.margin = "2px";
                        wrap.style.border = "2px solid transparent";
                        wrap.style.borderRadius = "8px";
                        if (idx === state.selected_idx) {
                            wrap.style.borderColor = "#ff8fa3";
                            wrap.style.boxShadow = "0 0 10px rgba(255, 143, 163, 0.3)";
                        }

                        const img = document.createElement("img");
                        const params = new URLSearchParams();
                        params.append("filename", name);
                        params.append("type", type);
                        if (sub) params.append("subfolder", sub);
                        img.src = api.apiURL("/view?" + params.toString());
                        img.className = "vnccs-cloner-thumb";
                        img.style.width = "60px"; // Ensure styling
                        img.style.height = "60px";
                        img.style.objectFit = "cover";
                        img.style.cursor = "pointer";

                        img.onclick = () => {
                            state.selected_idx = idx;
                            saveState();
                            renderThumbs();
                        };

                        // Delete X
                        const delBtn = document.createElement("div");
                        delBtn.innerText = "×";
                        delBtn.style.position = "absolute";
                        delBtn.style.top = "0";
                        delBtn.style.right = "0";
                        delBtn.style.background = "rgba(0,0,0,0.7)";
                        delBtn.style.color = "white";
                        delBtn.style.cursor = "pointer";
                        delBtn.style.padding = "0 5px";
                        delBtn.style.fontSize = "14px";
                        delBtn.style.borderRadius = "0 0 0 4px";
                        delBtn.onclick = (e) => {
                            e.stopPropagation();
                            showModal("Remove Image", () => {
                                const d = document.createElement("div");
                                d.innerText = "Are you sure you want to remove this image?";
                                return d;
                            }, [
                                { text: "Cancel" },
                                {
                                    text: "Remove",
                                    class: "vnccs-btn-danger",
                                    action: () => {
                                        state.source_images.splice(idx, 1);
                                        if (state.selected_idx >= state.source_images.length) state.selected_idx = Math.max(0, state.source_images.length - 1);
                                        saveState();
                                        renderThumbs();
                                        return false;
                                    }
                                }
                            ]);
                        };

                        wrap.appendChild(img);
                        wrap.appendChild(delBtn);
                        imgList.appendChild(wrap);
                    });
                };

                // Initialize
                loadState();
                loadCharList();
            };
        }
    }
});
