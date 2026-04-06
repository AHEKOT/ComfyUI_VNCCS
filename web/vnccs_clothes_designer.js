import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { registerCleanup, showModal as showCommonModal, showMessage } from "./vnccs_common.js";

const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
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
    --accent-glow: rgba(255,143,163,0.3);
    --accent-subtle: rgba(255,143,163,0.1);
    --accent-border: rgba(255,143,163,0.22);
    --accent-lavender: #b8a9e8;
    --success: #00d68f;
    --warning: #ffaa00;
    --error: #ff4757;
    --border: rgba(255,255,255,0.06);
    --border-hover: rgba(255,255,255,0.12);
    --font: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --transition: 0.2s ease;
}

.vnccs-container {
    display: flex; flex-direction: column;
    background: var(--bg-primary); color: var(--text-primary);
    font-family: var(--font); font-size: 13px;
    width: 100%; height: 100%; overflow: hidden; box-sizing: border-box;
    padding: 12px; gap: 12px; pointer-events: none; zoom: 0.67;
}
.vnccs-top-row {
    display: grid; grid-template-columns: 30% 35% 35%; gap: 12px;
    flex: 1; min-height: 0; width: 100%;
}
.vnccs-col {
    display: flex; flex-direction: column;
    background: rgba(20,16,30,0.88);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-lg);
    padding: 16px; gap: 10px;
    overflow-y: auto; height: 100%; box-sizing: border-box; pointer-events: auto;
    position: relative; box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.vnccs-col::before {
    content: '';
    position: absolute; top: 0; left: 18%; right: 18%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,143,163,0.5), transparent);
    border-radius: 1px;
}
.vnccs-col::-webkit-scrollbar { width: 4px; }
.vnccs-col::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

.vnccs-section-title {
    font-size: 10px; font-weight: 700; color: var(--accent);
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-bottom: 6px; flex-shrink: 0;
    display: flex; align-items: center; gap: 8px;
}
.vnccs-section-title::before {
    content: ''; width: 3px; height: 12px; flex-shrink: 0;
    background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
    border-radius: 2px; box-shadow: 0 0 8px var(--accent-glow);
}

.vnccs-field { display: flex; flex-direction: column; gap: 5px; margin-bottom: 6px; flex-shrink: 0; }
.vnccs-label {
    color: var(--text-secondary); font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
}

.vnccs-input, .vnccs-textarea {
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    color: var(--text-primary); border-radius: var(--radius-md);
    padding: 8px 12px; font-family: var(--font); font-size: 12px;
    width: 100%; box-sizing: border-box; transition: all var(--transition);
}
.vnccs-textarea { resize: none; min-height: 40px; }
.vnccs-select {
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    color: var(--text-primary); border-radius: var(--radius-md);
    padding: 8px 12px; font-family: var(--font); font-size: 12px;
    width: 100%; box-sizing: border-box; zoom: 1.5; transition: all var(--transition);
}
.vnccs-input:focus, .vnccs-select:focus, .vnccs-textarea:focus {
    outline: none; border-color: var(--accent-border);
    background: rgba(255,143,163,0.04);
    box-shadow: 0 0 0 3px rgba(255,143,163,0.06);
}

.vnccs-btn {
    padding: 10px; border: none; border-radius: var(--radius-md); cursor: pointer;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
    font-size: 11px; font-family: var(--font); color: white;
    text-align: center; flex: 1; transition: all var(--transition);
    position: relative; overflow: hidden;
}
.vnccs-btn-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
    color: #1a1525; box-shadow: 0 4px 16px rgba(255,143,163,0.25);
}
.vnccs-btn-primary::after {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
    transform: translateX(-120%) skewX(-15deg);
    animation: cd-shimmer 3.5s ease-in-out infinite; pointer-events: none;
}
@keyframes cd-shimmer {
    0% { transform: translateX(-120%) skewX(-15deg); opacity: 1; }
    35% { transform: translateX(120%) skewX(-15deg); opacity: 1; }
    100% { transform: translateX(120%) skewX(-15deg); opacity: 0; }
}
.vnccs-btn-primary:hover:not(:disabled) { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(255,143,163,0.4); }
.vnccs-btn-success {
    background: rgba(0,214,143,0.15); color: var(--success); border: 1px solid rgba(0,214,143,0.3);
}
.vnccs-btn-success:hover:not(:disabled) { background: rgba(0,214,143,0.25); transform: translateY(-1px); }
.vnccs-btn-danger {
    background: rgba(255,71,87,0.15); color: var(--error); border: 1px solid rgba(255,71,87,0.3);
}
.vnccs-btn-danger:hover:not(:disabled) { background: rgba(255,71,87,0.25); transform: translateY(-1px); }
.vnccs-btn:disabled {
    background: rgba(255,255,255,0.04) !important; color: var(--text-muted) !important;
    cursor: not-allowed; box-shadow: none !important; transform: none !important;
}

.vnccs-btn-row { display: flex; gap: 8px; margin-top: auto; flex-shrink: 0; flex-wrap: wrap; }
.vnccs-row { display: flex; gap: 8px; align-items: center; }

/* Preview */
.vnccs-preview-container {
    flex: 1;
    background: radial-gradient(circle, rgba(255,143,163,0.04) 1px, transparent 1px), rgba(10,10,15,0.7);
    background-size: 20px 20px, 100% 100%;
    border: 1px solid var(--border); border-radius: var(--radius-md);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden; position: relative; min-height: 0;
}
.vnccs-preview-img {
    width: 100%; height: 100%; object-fit: contain;
    animation: cd-fadein 0.4s ease;
}
@keyframes cd-fadein { from { opacity: 0; } to { opacity: 1; } }
.vnccs-placeholder {
    display: flex; flex-direction: column; align-items: center; gap: 10px;
    color: var(--text-muted); font-size: 11px; letter-spacing: 0.05em;
}
.vnccs-placeholder-icon { width: 48px; height: 48px; opacity: 0.25; }

/* Tab bar */
.cd-tab-bar {
    display: flex; border-bottom: 1px solid var(--border);
    margin: 0 -16px 12px; background: transparent;
}
.cd-tab {
    flex: 1; text-align: center; padding: 10px 8px; cursor: pointer;
    font-size: 10px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;
    color: var(--text-muted); border-bottom: 2px solid transparent;
    transition: all var(--transition);
}
.cd-tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.cd-tab:hover:not(.active) { color: var(--text-secondary); }

/* Clone upload area */
.cd-upload-area {
    border: 1px dashed rgba(255,143,163,0.3);
    background: rgba(255,143,163,0.04);
    border-radius: var(--radius-md);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: all var(--transition); position: relative;
    min-height: 200px; overflow: hidden;
}
.cd-upload-area:hover { border-color: var(--accent); background: rgba(255,143,163,0.08); }
.cd-upload-hint { color: var(--text-muted); font-size: 11px; text-align: center; }

/* Loading overlay */
.vnccs-loading-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(10,10,15,0.92); backdrop-filter: blur(8px);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 1000; pointer-events: auto; gap: 16px; border-radius: var(--radius-lg);
}
.vnccs-spinner {
    width: 44px; height: 44px; position: relative;
}
.vnccs-spinner::before, .vnccs-spinner::after {
    content: ''; position: absolute; inset: 0; border-radius: 50%; border: 3px solid transparent;
}
.vnccs-spinner::before {
    border-top-color: var(--accent); border-right-color: rgba(255,143,163,0.3);
    animation: cd-spin 1s linear infinite;
    box-shadow: 0 0 18px rgba(255,143,163,0.2);
}
.vnccs-spinner::after {
    inset: 7px;
    border-bottom-color: rgba(184,169,232,0.6); border-left-color: rgba(184,169,232,0.2);
    animation: cd-spin 1.4s linear infinite reverse;
}
@keyframes cd-spin { to { transform: rotate(360deg); } }
.vnccs-loading-text { color: var(--text-primary); font-size: 13px; font-weight: 600; }
.vnccs-loading-dots::after {
    content: ''; animation: cd-dots 1.5s steps(4,end) infinite;
}
@keyframes cd-dots {
    0%,20% { content: ''; } 40% { content: '.'; } 60% { content: '..'; } 80%,100% { content: '...'; }
}
`;

app.registerExtension({
    name: "VNCCS.ClothesDesigner",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "ClothesDesigner") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const node = this;
                node.setSize([1280, 800]);

                // CSS Injections
                const style = document.createElement("style");
                style.innerHTML = STYLE;
                document.head.appendChild(style);

                const cleanup = () => {
                    if (!node.widgets) return;
                    for (const w of node.widgets) {
                        if (w.name !== "widget_data") w.hidden = true;
                    }
                };
                cleanup();

                // Widget Data Sync
                let dataWidget = node.widgets ? node.widgets.find(w => w.name === "widget_data") : null;
                if (!dataWidget) {
                    dataWidget = node.addWidget("text", "widget_data", "{}", (v) => { }, { serialize: true });
                }
                dataWidget.hidden = true;

                // State
                const defaultState = {
                    character: "",
                    costume: "Naked",
                    activeTab: "generate", // 'generate' or 'clone'
                    clone_image: null,     // { name, type, subfolder }
                    costume_info: {
                        top: "", bottom: "", head: "", shoes: "", face: ""
                    },
                    gen_settings: {
                        unet_name: "", vae_name: "", clip_name: "",
                        lightning_lora: "None", lightning_lora_strength: 1.0,
                        service_lora: "None", service_lora_strength: 1.0,
                        background_color: "Green",
                        seed: 0, seed_mode: "fixed", steps: 4, cfg: 1.0, sampler: "euler", scheduler: "simple",
                        lora_stack: [{ name: "", strength: 1.0 }, { name: "", strength: 1.0 }, { name: "", strength: 1.0 }]
                    }
                };

                // Restore State Logic
                let saved = {};
                try {
                    // Priority 1: Widget Data (Graph)
                    if (dataWidget && dataWidget.value && dataWidget.value !== "{}") {
                        saved = JSON.parse(dataWidget.value);
                    } else {
                        // Priority 2: LocalStorage (Session)
                        const ls = localStorage.getItem("VNCCS_ClothesDesigner_State");
                        if (ls) saved = JSON.parse(ls);
                    }
                } catch (e) { console.warn("[VNCCS] ClothesDesigner: Error loading state", e); }

                const state = {
                    ...defaultState,
                    ...saved,
                    activeTab: saved.activeTab || "generate", // Ensure valid tab
                    costume_info: { ...defaultState.costume_info, ...(saved.costume_info || {}) },
                    gen_settings: { ...defaultState.gen_settings, ...(saved.gen_settings || {}) }
                };

                const els = {};

                const saveState = () => {
                    if (dataWidget) dataWidget.value = JSON.stringify(state);
                    try {
                        localStorage.setItem("VNCCS_ClothesDesigner_State", JSON.stringify(state));
                    } catch (e) { console.warn("[VNCCS] ClothesDesigner: Error saving to localStorage", e); }
                };

                // Randomize seed on each workflow run if mode is "randomize"
                node.onSerialize = function (o) {
                    if (state.gen_settings.seed_mode === "randomize") {
                        state.gen_settings.seed = Math.floor(Math.random() * 10000000000000);
                        if (els.seed) els.seed.value = state.gen_settings.seed;
                    }
                    if (dataWidget) dataWidget.value = JSON.stringify(state);
                };

                const saveCostumeToBackend = async () => {
                    if (!state.character || !state.costume) return;
                    try {
                        await api.fetchApi("/vnccs/save_costume", {
                            method: "POST",
                            body: JSON.stringify({
                                character: state.character,
                                costume: state.costume,
                                info: state.costume_info
                            })
                        });
                    } catch (e) { console.error("Save failed", e); }
                };

                // Modal Helper — delegates to vnccs_common showModal
                const showModal = (title, contentFunc, buttons) => {
                    const mappedButtons = buttons.map(b => ({
                        ...b,
                        class: b.class?.includes("danger") ? "danger" : b.class?.includes("primary") ? "primary" : undefined
                    }));
                    return showCommonModal(container, title, contentFunc, mappedButtons);
                };

                const showInfo = (title, msg) => {
                    showModal(title, () => {
                        const d = document.createElement("div");
                        d.innerText = msg;
                        d.style.padding = "10px 0";
                        return d;
                    }, [{ text: "OK", class: "vnccs-btn-primary" }]);
                };

                // UI Builders
                const createField = (key, placeholder, multiline = true) => {
                    const wrap = document.createElement("div"); wrap.className = "vnccs-field";
                    const l = document.createElement("div"); l.className = "vnccs-label";
                    l.innerText = key.toUpperCase();
                    wrap.appendChild(l);

                    const inp = document.createElement(multiline ? "textarea" : "input");
                    inp.className = multiline ? "vnccs-textarea" : "vnccs-input";
                    if (placeholder) inp.placeholder = placeholder;

                    inp.value = state.costume_info[key] || "";

                    if (multiline) {
                        inp.style.resize = "none";
                        inp.style.overflow = "hidden";
                        const autoResize = () => {
                            inp.style.height = "auto";
                            inp.style.height = inp.scrollHeight + "px";
                        };
                        inp.addEventListener("input", autoResize);
                        setTimeout(autoResize, 10);
                        inp.autoResize = autoResize;
                    }

                    inp.onchange = (e) => {
                        state.costume_info[key] = e.target.value;
                        saveState();
                        saveCostumeToBackend();
                    };
                    els[key] = inp;
                    wrap.appendChild(inp);
                    return wrap;
                };

                // Helper to render Attributes Panel (Original Tab)
                const renderAttributes = (container) => {
                    container.innerHTML = '';
                    container.appendChild(createField("top", "e.g. White t-shirt"));
                    container.appendChild(createField("bottom", "e.g. Blue jeans"));
                    container.appendChild(createField("shoes", "e.g. Sneakers"));
                    container.appendChild(createField("head", "e.g. Hat"));
                    container.appendChild(createField("face", "Face features (e.g. glasses)"));
                };

                // Helper to render Clone Panel (New Tab)
                const renderClonePanel = (container) => {
                    container.innerHTML = '';

                    const desc = document.createElement("div");
                    desc.style.fontSize = "12px"; desc.style.color = "#aaa"; desc.style.marginBottom = "10px";
                    desc.innerText = "Upload an image to clone clothes from. The AI will attempt to transfer the outfit to your character.";
                    container.appendChild(desc);

                    // Preview / Upload Area
                    const pContainer = document.createElement("div");
                    pContainer.className = "vnccs-preview-container";
                    pContainer.style.height = "250px";
                    pContainer.style.background = "#151515";
                    pContainer.style.position = "relative";
                    pContainer.style.display = "flex";
                    pContainer.style.alignItems = "center";
                    pContainer.style.justifyContent = "center";
                    pContainer.style.border = "1px solid #444";

                    const img = document.createElement("img");
                    img.style.maxWidth = "100%"; img.style.maxHeight = "100%"; img.style.objectFit = "contain";
                    img.style.display = "none";
                    if (state.clone_image) {
                        img.src = api.apiURL(`/view?filename=${encodeURIComponent(state.clone_image.name)}&type=${state.clone_image.type}&subfolder=${state.clone_image.subfolder}`);
                        img.style.display = "block";
                    }
                    pContainer.appendChild(img);

                    // Upload Button Overlay
                    const overlay = document.createElement("div");
                    overlay.style.position = "absolute"; overlay.style.inset = "0";
                    overlay.style.display = "flex"; overlay.style.alignItems = "center"; overlay.style.justifyContent = "center";
                    overlay.style.cursor = "pointer";

                    const btn = document.createElement("button");
                    btn.className = "vnccs-btn";
                    btn.style.background = "#444"; btn.style.border = "1px dashed #666";
                    btn.innerText = state.clone_image ? "REPLACE IMAGE" : "+ UPLOAD IMAGE";
                    if (state.clone_image) { btn.style.opacity = "0.8"; btn.style.fontSize = "10px"; btn.style.padding = "4px 8px"; btn.style.position = "absolute"; btn.style.bottom = "10px"; }

                    overlay.appendChild(btn);

                    const fileInp = document.createElement("input");
                    fileInp.type = "file"; fileInp.accept = "image/*"; fileInp.style.display = "none";
                    fileInp.onchange = async (e) => {
                        if (e.target.files.length) {
                            const file = e.target.files[0];
                            try {
                                btn.innerText = "UPLOADING...";
                                const body = new FormData();
                                body.append("image", file);
                                const resp = await api.fetchApi("/upload/image", { method: "POST", body });
                                const json = await resp.json();
                                state.clone_image = {
                                    name: json.name, type: json.type || "input", subfolder: json.subfolder || ""
                                };
                                saveState();
                                // Update View
                                renderClonePanel(container);
                            } catch (err) {
                                showInfo("Upload Failed", err.toString());
                            }
                        }
                    };

                    overlay.onclick = (e) => {
                        e.stopPropagation();
                        fileInp.click();
                    };

                    pContainer.appendChild(overlay);
                    container.appendChild(pContainer);
                };


                // --- MAIN LAYOUT ---
                const container = document.createElement("div"); container.className = "vnccs-container";
                const topRow = document.createElement("div"); topRow.className = "vnccs-top-row";

                // --- COL 1: DESIGN STUDIO ---
                const colLeft = document.createElement("div"); colLeft.className = "vnccs-col";
                colLeft.innerHTML = '<div class="vnccs-section-title">Design Studio</div>';

                // Character Select
                const charRow = document.createElement("div"); charRow.className = "vnccs-field";
                charRow.innerHTML = '<div class="vnccs-label">CHARACTER</div>';
                const charSel = document.createElement("select"); charSel.className = "vnccs-select";
                charSel.onchange = async (e) => {
                    state.character = e.target.value;
                    await loadCostumes();
                    updatePreviewImage();
                    saveState();
                };
                charRow.appendChild(charSel);
                els.charSelect = charSel;
                colLeft.appendChild(charRow);

                // Costume Select
                const costRow = document.createElement("div"); costRow.className = "vnccs-field";
                costRow.innerHTML = '<div class="vnccs-label">COSTUME (Select to Edit)</div>';
                const costSel = document.createElement("select"); costSel.className = "vnccs-select";
                costSel.onchange = async (e) => {
                    state.costume = e.target.value;
                    await loadCostumeInfo();
                    updatePreviewImage();
                    saveState();
                };
                els.costSel = costSel;
                costRow.appendChild(costSel);
                colLeft.appendChild(costRow);

                // Action Buttons
                const actionRow = document.createElement("div"); actionRow.className = "vnccs-btn-row";
                actionRow.style.marginBottom = "10px";

                const btnNewCostume = document.createElement("button");
                btnNewCostume.className = "vnccs-btn vnccs-btn-success";
                btnNewCostume.innerText = "NEW";
                btnNewCostume.style.fontSize = "10px";
                btnNewCostume.onclick = () => {
                    showModal("New Costume Name", () => {
                        const inp = document.createElement("input"); inp.className = "vnccs-input";
                        return inp;
                    }, [{ text: "Cancel" }, {
                        text: "CREATE", class: "vnccs-btn-primary", action: async (ol, btn) => {
                            const n = ol.querySelector("input").value.trim();
                            if (n) {
                                await api.fetchApi("/vnccs/save_costume", {
                                    method: "POST", body: JSON.stringify({ character: state.character, costume: n, info: {} })
                                });
                                await loadCostumes();
                                state.costume = n;
                                costSel.value = n;
                                await loadCostumeInfo();
                                updatePreviewImage();
                                saveState();
                                return false;
                            }
                            return true;
                        }
                    }]);
                };
                actionRow.appendChild(btnNewCostume);

                const btnDelCostume = document.createElement("button");
                btnDelCostume.className = "vnccs-btn vnccs-btn-danger";
                btnDelCostume.innerText = "DELETE";
                btnDelCostume.style.fontSize = "10px";
                btnDelCostume.onclick = () => {
                    if (state.costume === "Naked") { showInfo("Warning", "Cannot delete Naked costume."); return; }
                    showModal("Delete", () => {
                        const d = document.createElement("div");
                        d.innerText = "Delete " + state.costume + "?";
                        return d;
                    },
                        [{ text: "Cancel" }, { text: "DELETE", class: "vnccs-btn-danger", action: async () => { showInfo("Not Implemented", "Manual fix required."); return false; } }]);
                };
                actionRow.appendChild(btnDelCostume);
                els.btnDel = btnDelCostume;
                colLeft.appendChild(actionRow);

                // Generate Button
                const btnGen = document.createElement("button");
                btnGen.className = "vnccs-btn vnccs-btn-primary";
                btnGen.innerText = "GENERATE PREVIEW";
                btnGen.style.width = "100%"; btnGen.style.marginBottom = "5px";
                btnGen.style.flex = "0 0 auto"; // Prevent vertical stretching
                btnGen.onclick = async () => {
                    if (!state.character) { showInfo("Error", "Select Character"); return; }
                    if (btnGen.disabled) return;

                    if (state.gen_settings.seed_mode === "randomize") {
                        state.gen_settings.seed = Math.floor(Math.random() * 10000000000000);
                        if (els.seed) els.seed.value = state.gen_settings.seed;
                    }
                    if (!state.gen_settings.unet_name) { showInfo("Error", "No UNET Selected"); return; }

                    // Show loading overlay
                    const loadingOverlay = document.createElement('div');
                    loadingOverlay.className = 'vnccs-loading-overlay';
                    loadingOverlay.innerHTML = `
                        <div class="vnccs-spinner"></div>
                        <div class="vnccs-loading-text">Generating preview<span class="vnccs-loading-dots"></span></div>
                    `;
                    container.appendChild(loadingOverlay);

                    saveCostumeToBackend();
                    saveState();
                    btnGen.innerText = "GENERATING..."; btnGen.disabled = true;
                    try {
                        const r = await api.fetchApi("/vnccs/clothes_preview", { method: "POST", body: JSON.stringify(state) });
                        if (r.ok) {
                            const d = await r.json();
                            if (d.image) {
                                els.previewImg.src = "data:image/png;base64," + d.image;
                                els.previewImg.style.display = "block";
                                els.placeholder.style.display = "none";
                            }
                        } else showInfo("Error", "Failed");
                    } catch (e) { showInfo("Error", e.toString()); }
                    finally {
                        loadingOverlay.remove();
                        btnGen.innerText = "GENERATE PREVIEW / SAVE"; btnGen.disabled = false;
                    }
                };
                els.btnGen = btnGen;
                colLeft.appendChild(btnGen);

                // Preview
                const frame = document.createElement("div"); frame.className = "vnccs-preview-container";
                frame.style.marginTop = "5px";
                frame.innerHTML = `<div class="vnccs-placeholder">
                    <svg class="vnccs-placeholder-icon" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M16 12h16M12 20l4-8h16l4 8v20H12V20z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                        <path d="M20 40V28h8v12" stroke="currentColor" stroke-width="2"/>
                    </svg>
                    No Preview
                </div>`;
                const pImg = document.createElement("img"); pImg.className = "vnccs-preview-img"; pImg.style.display = "none";
                pImg.onclick = () => window.open(pImg.src, "_blank");
                frame.appendChild(pImg);
                els.previewImg = pImg; els.placeholder = frame.querySelector(".vnccs-placeholder");
                colLeft.appendChild(frame);

                topRow.appendChild(colLeft);

                // --- COL 2: MIDDLE PANEL (Tabs) ---
                const colMid = document.createElement("div"); colMid.className = "vnccs-col";
                colMid.style.paddingTop = "0"; // Reset padding for tabs

                // Tab Header
                const tabBar = document.createElement("div");
                tabBar.className = "cd-tab-bar";

                const createTab = (id, label) => {
                    const t = document.createElement("div");
                    t.className = "cd-tab" + (state.activeTab === id ? " active" : "");
                    t.innerText = label;
                    t.onclick = () => {
                        state.activeTab = id;
                        saveState();
                        refreshTabs();
                    };
                    return t;
                };

                const contentArea = document.createElement("div");
                contentArea.style.flex = "1"; contentArea.style.display = "flex"; contentArea.style.flexDirection = "column";

                const refreshTabs = () => {
                    tabBar.innerHTML = '';
                    tabBar.appendChild(createTab("generate", "GENERATE CLOTHES"));
                    tabBar.appendChild(createTab("clone", "CLONE CLOTHES"));

                    if (state.activeTab === "generate") renderAttributes(contentArea);
                    else renderClonePanel(contentArea);
                };

                refreshTabs(); // Initial Render

                colMid.appendChild(tabBar);
                colMid.appendChild(contentArea);
                topRow.appendChild(colMid);

                // --- COL 3: SETTINGS ---
                const colRight = document.createElement("div"); colRight.className = "vnccs-col";
                colRight.innerHTML = '<div class="vnccs-section-title">Settings (Qwen)</div>';

                const createSetting = (lbl, key, type = "text", list = []) => {
                    const w = document.createElement("div"); w.className = "vnccs-field";
                    w.innerHTML = `<div class="vnccs-label">${lbl}</div>`;
                    let inp;
                    if (type === "select") {
                        inp = document.createElement("select"); inp.className = "vnccs-select";
                        list.forEach(i => inp.add(new Option(i, i)));

                        // Critical: Sync state with visual default
                        // If state is empty, use first item in list
                        let val = state.gen_settings[key] || list[0];

                        // Normalize background_color case to match dropdown options
                        if (key === "background_color" && typeof val === "string" && val) {
                            val = val.charAt(0).toUpperCase() + val.slice(1).toLowerCase();
                            state.gen_settings[key] = val;
                        }

                        inp.value = val;

                        // Force update state if it was empty/undefined
                        if (!state.gen_settings[key] && val) {
                            state.gen_settings[key] = val;
                        }

                        inp.onchange = (e) => {
                            state.gen_settings[key] = e.target.value;
                            saveState();
                        };
                    } else if (type === "number") {
                        inp = document.createElement("input"); inp.className = "vnccs-input"; inp.type = "number";
                        inp.value = state.gen_settings[key];
                        inp.onchange = (e) => { state.gen_settings[key] = Number(e.target.value); saveState(); };
                    }
                    els[key] = inp;
                    w.appendChild(inp);
                    colRight.appendChild(w);
                };

                // Helper for LoRA with strength
                const createLoraField = (lbl, keyName, keyStr, loras) => {
                    const w = document.createElement("div"); w.className = "vnccs-field";
                    w.innerHTML = `<div class="vnccs-label">${lbl}</div>`;

                    const row = document.createElement("div"); row.className = "vnccs-row";

                    const sel = document.createElement("select"); sel.className = "vnccs-select"; sel.style.flex = "1";
                    loras.forEach(l => sel.add(new Option(l, l)));
                    sel.value = state.gen_settings[keyName] || "None";
                    sel.onchange = (e) => { state.gen_settings[keyName] = e.target.value; saveState(); };

                    const num = document.createElement("input"); num.className = "vnccs-input";
                    num.type = "number"; num.step = "0.01"; num.style.width = "60px";
                    num.value = state.gen_settings[keyStr] !== undefined ? state.gen_settings[keyStr] : 1.0;
                    num.onchange = (e) => { state.gen_settings[keyStr] = Number(e.target.value); saveState(); };

                    row.appendChild(sel);
                    row.appendChild(num);
                    w.appendChild(row);
                    colRight.appendChild(w);

                    els[keyName] = sel; els[keyStr] = num;
                };

                // Initial Load
                (async () => {
                    const r = await api.fetchApi("/vnccs/context_lists");
                    const d = await r.json();

                    // 1. Try UnetLoaderGGUF (Custom Node) - Primary for GGUF
                    let unets = [];
                    try {
                        const rGGUF = await api.fetchApi("/object_info/UnetLoaderGGUF");
                        const dGGUF = await rGGUF.json();
                        unets = dGGUF.UnetLoaderGGUF.input.required.unet_name[0];
                        console.log("VNCCS: Loaded GGUF models from UnetLoaderGGUF");
                    } catch (e) {
                        console.log("VNCCS: UnetLoaderGGUF not found, trying UNETLoader.");
                        // 2. Try Standard UNETLoader
                        try {
                            const rStd = await api.fetchApi("/object_info/UNETLoader");
                            const dStd = await rStd.json();
                            unets = dStd.UNETLoader.input.required.unet_name[0];
                        } catch (e2) {
                            console.warn("VNCCS: No UNET loader found, list empty.");
                            unets = []; // Do NOT use checkpoints
                        }
                    }

                    colRight.innerHTML = '<div class="vnccs-section-title">Settings (Qwen)</div>';

                    createSetting("UNET (GGUF)", "unet_name", "select", unets.length ? unets : ["No Models Found"]);
                    createSetting("CLIP", "clip_name", "select", d.checkpoints);
                    try {
                        const r3 = await api.fetchApi("/object_info/CLIPLoader");
                        const d3 = await r3.json();
                        els.clip_name.innerHTML = "";
                        d3.CLIPLoader.input.required.clip_name[0].forEach(o => els.clip_name.add(new Option(o, o)));
                    } catch (e) { console.warn("[VNCCS] ClothesDesigner: Error loading CLIP models", e); }

                    createSetting("VAE", "vae_name", "select", []);
                    try {
                        const r4 = await api.fetchApi("/object_info/VAELoader");
                        const d4 = await r4.json();
                        els.vae_name.innerHTML = "";
                        d4.VAELoader.input.required.vae_name[0].forEach(o => els.vae_name.add(new Option(o, o)));
                    } catch (e) { console.warn("[VNCCS] ClothesDesigner: Error loading VAE models", e); }

                    // LoRAs
                    const loraList = ["None", ...d.loras];
                    createLoraField("Lightning LoRA", "lightning_lora", "lightning_lora_strength", loraList);
                    createLoraField("Service LoRA", "service_lora", "service_lora_strength", loraList);

                    createSetting("Background", "background_color", "select", ["Green", "Blue"]);
                    // createSetting("Seed", "seed", "number");

                    // --- Custom Seed Control ---
                    const seedWrap = document.createElement("div"); seedWrap.className = "vnccs-field";
                    seedWrap.innerHTML = '<div class="vnccs-label">Seed</div>';

                    const seedRow = document.createElement("div"); seedRow.className = "vnccs-row";
                    seedRow.style.display = "flex"; seedRow.style.gap = "5px";

                    const seedInp = document.createElement("input"); seedInp.className = "vnccs-input";
                    seedInp.type = "number";
                    seedInp.value = state.gen_settings.seed || 0;
                    seedInp.style.flex = "2";
                    seedInp.onchange = (e) => {
                        state.gen_settings.seed = Number(e.target.value); // Use Number() or parseInt()
                        saveState();
                    };
                    els.seed = seedInp;

                    const seedMode = document.createElement("select"); seedMode.className = "vnccs-select";
                    seedMode.style.flex = "1.5";
                    ["fixed", "randomize"].forEach(x => seedMode.add(new Option(x, x)));
                    seedMode.value = state.gen_settings.seed_mode || "fixed";
                    seedMode.onchange = (e) => {
                        state.gen_settings.seed_mode = e.target.value;
                        saveState();
                    };
                    els.seed_mode = seedMode;

                    seedRow.appendChild(seedInp);
                    seedRow.appendChild(seedMode);
                    seedWrap.appendChild(seedRow);
                    colRight.appendChild(seedWrap);
                    createSetting("Steps", "steps", "number");
                    createSetting("CFG", "cfg", "number");

                    // Restore values
                    if (state.gen_settings.unet_name) els.unet_name.value = state.gen_settings.unet_name;
                    if (state.gen_settings.clip_name) els.clip_name.value = state.gen_settings.clip_name;
                    if (state.gen_settings.vae_name) els.vae_name.value = state.gen_settings.vae_name;

                    // Restore LoRAs
                    if (els.lightning_lora) els.lightning_lora.value = state.gen_settings.lightning_lora || "None";
                    if (els.lightning_lora_strength) els.lightning_lora_strength.value = state.gen_settings.lightning_lora_strength || 1.0;
                    if (els.service_lora) els.service_lora.value = state.gen_settings.service_lora || "None";
                    if (els.service_lora_strength) els.service_lora_strength.value = state.gen_settings.service_lora_strength || 1.0;

                    if (state.gen_settings.steps) els.steps.value = state.gen_settings.steps;
                    if (state.gen_settings.cfg) els.cfg.value = state.gen_settings.cfg;
                    if (state.gen_settings.seed_mode && els.seed_mode) els.seed_mode.value = state.gen_settings.seed_mode;

                    // SYNC Defaults for Models (Critical Fix for Empty String error)
                    if (!state.gen_settings.unet_name && els.unet_name.options.length) {
                        state.gen_settings.unet_name = els.unet_name.options[0].value;
                        els.unet_name.value = state.gen_settings.unet_name;
                    }
                    if (!state.gen_settings.clip_name && els.clip_name.options.length) {
                        state.gen_settings.clip_name = els.clip_name.options[0].value;
                        els.clip_name.value = state.gen_settings.clip_name;
                    }
                    if (!state.gen_settings.vae_name && els.vae_name.options.length) {
                        state.gen_settings.vae_name = els.vae_name.options[0].value;
                        els.vae_name.value = state.gen_settings.vae_name;
                    }
                    saveState(); // Ensure defaults are persisted immediately

                    // Char List
                    els.charSelect.innerHTML = "";
                    d.characters.forEach(c => els.charSelect.add(new Option(c, c)));
                    if (state.character) els.charSelect.value = state.character;
                    else if (d.characters.length) { state.character = d.characters[0]; els.charSelect.value = state.character; }

                    await loadCostumes();
                    updatePreviewImage();
                })();

                topRow.appendChild(colRight);
                container.appendChild(topRow);

                // Functions
                const loadCostumes = async () => {
                    const c = state.character;
                    if (!c) return;
                    const r = await api.fetchApi(`/vnccs/list_costumes?character=${encodeURIComponent(c)}`);
                    let list = await r.json();

                    // Filter 'Naked' from display list
                    const displayList = list.filter(i => i !== "Naked");

                    els.costSel.innerHTML = "";
                    displayList.forEach(i => els.costSel.add(new Option(i, i)));

                    // Logic: If only Naked exists (displayList empty), prevent generation/deletion
                    if (displayList.length === 0) {
                        state.costume = "Naked";
                        if (els.btnGen) els.btnGen.disabled = true;
                        if (els.btnDel) els.btnDel.disabled = true;
                        if (els.costSel) els.costSel.disabled = true;
                        // Disable Inputs
                        ["top", "bottom", "head", "face", "shoes"].forEach(k => { if (els[k]) els[k].disabled = true; });
                    } else {
                        if (els.btnGen) els.btnGen.disabled = false;
                        if (els.btnDel) els.btnDel.disabled = false;
                        if (els.costSel) els.costSel.disabled = false;
                        // Enable Inputs
                        ["top", "bottom", "head", "face", "shoes"].forEach(k => { if (els[k]) els[k].disabled = false; });

                        // Select default if current is Naked or invalid
                        if (state.costume === "Naked" || !displayList.includes(state.costume)) {
                            state.costume = displayList[0];
                        }
                    }

                    if (els.costSel.options.length > 0) {
                        els.costSel.value = state.costume;
                    }

                    await loadCostumeInfo();
                };

                const loadCostumeInfo = async () => {
                    const c = state.character;
                    const cos = state.costume;
                    const r = await api.fetchApi(`/vnccs/get_costume?character=${encodeURIComponent(c)}&costume=${encodeURIComponent(cos)}`);
                    const info = await r.json();

                    state.costume_info = {
                        top: info.top || "",
                        bottom: info.bottom || "",
                        head: info.head || "",
                        face: info.face || "",
                        shoes: info.shoes || ""
                    };

                    for (const k in state.costume_info) {
                        if (els[k]) {
                            els[k].value = state.costume_info[k];
                            if (els[k].autoResize) els[k].autoResize();
                        }
                    }
                };

                const updatePreviewImage = async (forceCache = false) => {
                    if (!state.character) return;
                    const ts = Date.now();
                    let url = `/vnccs/get_preview?character=${encodeURIComponent(state.character)}&costume=${encodeURIComponent(state.costume)}&ts=${ts}`;
                    if (forceCache) url += "&force_cache=true";

                    // Check validity first to show message
                    try {
                        const r = await fetch(url);
                        if (!r.ok) {
                            if (r.status === 400) {
                                const txt = await r.text();
                                showModal("Character don't have body yet. Complete Step1 workflow first.", () => {
                                    const d = document.createElement("div"); d.innerText = txt; return d;
                                }, [
                                    {
                                        text: "OK", class: "vnccs-btn-primary", action: async (overlay) => {
                                            // Auto-switch to first valid character
                                            overlay.remove(); // Close modal immediately

                                            // Helper to check validity
                                            const checkChar = async (c) => {
                                                try {
                                                    const testUrl = `/vnccs/get_preview?character=${encodeURIComponent(c)}&costume=Naked&ts=${Date.now()}`;
                                                    const testR = await fetch(testUrl);
                                                    return testR.ok;
                                                } catch (e) { return false; }
                                            };

                                            const opts = els.charSelect.options;
                                            for (let i = 0; i < opts.length; i++) {
                                                const candidate = opts[i].value;
                                                // Check validity (even if it's the current one, though we know current failed)
                                                // Actually, try next ones mainly, but user said "reset to first valid", implies order 0..N
                                                if (await checkChar(candidate)) {
                                                    console.log("Found valid character:", candidate);
                                                    state.character = candidate;
                                                    els.charSelect.value = candidate;
                                                    saveState();
                                                    await loadCostumes();
                                                    updatePreviewImage();
                                                    return false; // Done
                                                }
                                            }
                                            showInfo("Error", "No valid characters found in the list.");
                                            return false;
                                        }
                                    }
                                ]);
                            }
                            els.previewImg.style.display = "none";
                            els.placeholder.style.display = "block";
                            return;
                        }
                    } catch (e) { console.warn("[VNCCS] ClothesDesigner: Error in preview update", e); }

                    els.previewImg.src = url;
                    els.previewImg.style.display = "block";
                    els.placeholder.style.display = "none";
                    els.previewImg.onerror = () => {
                        els.previewImg.style.display = "none";
                        els.placeholder.style.display = "block";
                    };
                };

                const onPreviewUpdated = (event) => {
                    const targetId = String(event.detail?.node_id);
                    const myId = String(node.id);
                    if (targetId === myId) {
                        console.log("VNCCS Preview Update Received for Node:", myId);
                        updatePreviewImage(true);
                    }
                };
                api.addEventListener("vnccs.preview.updated", onPreviewUpdated);
                registerCleanup(node, () => api.removeEventListener("vnccs.preview.updated", onPreviewUpdated));

                node.addDOMWidget("clothes_designer_ui", "ui", container, {
                    serialize: false,
                    hideOnZoom: false
                });

            };
        }
    }
});
