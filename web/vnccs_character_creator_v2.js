import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// --- STYLES: 3-Column Grid Layout ---
const STYLE = `
/* Main Host */
.vnccs-container {
    display: flex;
    flex-direction: column;
    background: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 16px;
    width: 100%;
    height: 100%;
    overflow: hidden; /* Main container shouldn't scroll, inner parts will */
    box-sizing: border-box;
    padding: 10px;
    gap: 10px;
    pointer-events: none; /* Allow canvas zoom/pan in gaps */
    zoom: 0.67; /* Scale down 1.5x as requested */
}

/* TOP ROW: 3 Columns */
.vnccs-top-row {
    display: grid;
    grid-template-columns: 30% 35% 35%; /* Adjust ratios as needed */
    gap: 10px;
    flex: 1; /* Takes remaining height */
    min-height: 0; /* Important for scroll */
    width: 100%;
}

/* BOTTOM ROW: Prompts - aligned with Top Row Columns */
.vnccs-bottom-row {
    display: grid;
    grid-template-columns: 30% 35% 35%; /* Matching top row */
    gap: 10px;
    height: 75px; 
    min-height: 75px;
    width: 100%;
    flex-shrink: 0;
    pointer-events: auto; 
}

/* Common Section/Column Styles */
.vnccs-col {
    display: flex;
    flex-direction: column;
    background: #252525;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 10px;
    gap: 10px;
    overflow-y: auto; /* Columns scroll independently */
    height: 100%;
    box-sizing: border-box;
    pointer-events: auto; /* Allow events to fall through gaps */
}
.vnccs-section-title {
    font-size: 14px;
    font-weight: bold;
    color: #fff;
    border-bottom: 2px solid #444;
    padding-bottom: 5px;
    margin-bottom: 5px;
    text-transform: uppercase;
    flex-shrink: 0;
    pointer-events: auto; /* Allow interaction */
}

/* Interactive Elements - Re-enable events */
.vnccs-field, 
.vnccs-btn-row > *, 
.vnccs-preview-container, 
.vnccs-lora-item,
.vnccs-textarea-wrapper,
.vnccs-slider-container,
.vnccs-input,
.vnccs-select,
.vnccs-textarea {
    pointer-events: auto;
}

/* Scrollbar */
.vnccs-col::-webkit-scrollbar { width: 6px; }
.vnccs-col::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }

/* Fields */
.vnccs-field { display: flex; flex-direction: column; gap: 4px; margin-bottom: 5px; flex-shrink: 0; }
.vnccs-label { color: #aaa; font-size: 11px; font-weight: 600; }
.vnccs-input, .vnccs-textarea {
    background: #151515; border: 1px solid #444; color: #fff;
    border-radius: 4px; padding: 6px; font-family: inherit; font-size: 12px;
    width: 100%; box-sizing: border-box;
}
.vnccs-select {
    background: #151515; border: 1px solid #444; color: #fff;
    border-radius: 4px; padding: 6px; font-family: inherit; 
    font-size: 12px;
    width: 100%; box-sizing: border-box;
    zoom: 1.5; /* Counteract container zoom to fix dropdown menu size */
    padding: 4px;
}
.vnccs-input:focus, .vnccs-select:focus, .vnccs-textarea:focus { border-color: #5b96f5; outline: none; }

/* Slider */
.vnccs-slider-container {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #151515;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 4px 8px;
}
.vnccs-slider {
    flex: 1;
    accent-color: #3558c7;
    cursor: pointer;
    height: 4px;
}
.vnccs-slider-val {
    width: 48px;
    text-align: right;
    font-size: 12px;
    color: #fff;
    font-family: inherit;
    background: transparent;
    border: none;
}
.vnccs-slider-val:focus { outline: none; border-bottom: 1px solid #5b96f5; }

/* Specialized Components */

/* PREVIEW COLUMN (Left) */
.vnccs-preview-container {
    flex: 1;
    background: #000;
    border: 1px solid #333;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
    min-height: 0; 
}
.vnccs-preview-img {
    width: 100%;
    height: 100%;
    object-fit: contain; /* Scales with container */
}
.vnccs-placeholder { color: #555; text-align: center; }

/* ATTRIBUTES (Center) */
/* Just uses standard fields */

/* GENERATION (Right) */
.vnccs-lora-stack {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-top: 10px;
    border-top: 1px solid #444;
    padding-top: 10px;
}
.vnccs-lora-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
    background: #1a1a1a;
    padding: 5px;
    border-radius: 4px;
    border: 1px solid #333;
}
.vnccs-lora-row {
    display: flex;
    gap: 5px;
    align-items: center;
}
.vnccs-slider-val { width: 35px; text-align: right; font-size: 10px; color: #aaa; }

/* Buttons */
.vnccs-btn-row { display: flex; gap: 10px; margin-top: auto; flex-shrink: 0; }
.vnccs-btn {
    flex: 1; padding: 10px; border: none; border-radius: 4px;
    cursor: pointer; font-weight: bold; text-transform: uppercase;
    font-size: 12px; color: white;
    text-align: center;
}
.vnccs-btn-primary { background: #3558c7; } .vnccs-btn-primary:hover { background: #4264d9; }
.vnccs-btn-success { background: #2e7d32; } .vnccs-btn-success:hover { background: #388e3c; }
.vnccs-btn-danger { background: #d32f2f; } .vnccs-btn-danger:hover { background: #b71c1c; }
.vnccs-btn-disabled { background: #333; color: #666; cursor: default; }

/* Bottom Textareas */
.vnccs-textarea-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    background: #252525;
    padding: 5px;
    border-radius: 6px;
    border: 1px solid #333;
}
.vnccs-textarea-wrapper textarea {
    flex: 1;
    resize: none;
    border: none;
    background: transparent;
    padding: 5px;
}
.vnccs-textarea-label {
    font-size: 10px; color: #888; text-transform: uppercase; font-weight: bold; padding: 0 5px;
}

/* Modal */
.vnccs-modal-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center;
    z-index: 1000;
    pointer-events: auto;
}
.vnccs-modal {
    background: #252525; border: 1px solid #444; padding: 20px; border-radius: 8px;
    width: 300px; display: flex; flex-direction: column; gap: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    max-height: 80vh; /* Safety for tall content */
}

/* Loading Overlay */
.vnccs-loading-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.9);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 1000; pointer-events: auto; gap: 20px;
}
.vnccs-spinner {
    width: 50px; height: 50px;
    border: 4px solid #333; border-top-color: #5b96f5;
    border-radius: 50%;
    animation: vnccs-spin 1s linear infinite;
}
@keyframes vnccs-spin { to { transform: rotate(360deg); } }
.vnccs-loading-text {
    color: #fff; font-size: 16px; font-weight: bold;
}
.vnccs-loading-dots::after {
    content: '';
    animation: vnccs-dots 1.5s steps(4, end) infinite;
}
@keyframes vnccs-dots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
}

/* Tag Constructor Styles */
.vnccs-tag-btn {
    width: 20px; height: 20px;
    background: #333; color: #fff; border: 1px solid #555;
    border-radius: 4px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; margin-left: auto; /* Push to right */
    flex-shrink: 0;
}
.vnccs-tag-btn:hover { background: #444; color: #5b96f5; border-color: #5b96f5; }

.vnccs-tag-grid {
    display: flex; flex-wrap: wrap; gap: 5px;
    max-height: 300px; overflow-y: auto;
    padding: 5px; background: #1a1a1a; border-radius: 4px;
}
.vnccs-tag-chip {
    padding: 4px 8px; background: #333; border: 1px solid #444; border-radius: 12px;
    font-size: 11px; color: #ccc; cursor: pointer; select-user: none;
}
.vnccs-tag-chip:hover { background: #444; border-color: #666; }
.vnccs-tag-chip.selected { background: #3558c7; color: white; border-color: #5b96f5; }

.vnccs-tag-category { font-size: 11px; color: #888; margin-top: 5px; text-transform: uppercase; font-weight: bold; }
`;

app.registerExtension({
    name: "VNCCS.CharacterCreatorV2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "CharacterCreatorV2") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const node = this;
                node.setSize([1280, 800]); // Default wide 3-column

                // 1. Setup CSS
                const style = document.createElement("style");
                style.innerHTML = STYLE;
                document.head.appendChild(style);

                // 2. Strict Widget Cleanup
                const cleanup = () => {
                    if (!node.widgets) return;
                    for (const w of node.widgets) {
                        if (w.name !== "ui" && w.name !== "widget_data") {
                            w.hidden = true;
                            //w.computeSize = () => [0, -4]; // Cause issues sometimes?
                        }
                    }
                };
                cleanup();

                // Keep keeping them hidden
                const origDraw = node.onDrawBackground;
                node.onDrawBackground = function (ctx) {
                    cleanup();
                    if (origDraw) origDraw.apply(this, arguments);
                };

                // Override onSerialize to guarantee state is written to widget before execution
                const origSerialize = node.onSerialize;
                node.onSerialize = function (o) {
                    if (origSerialize) origSerialize.apply(this, arguments);

                    // Critical Sync: Ensure widget_data receives latest state
                    const w = node.widgets ? node.widgets.find(w => w.name === "widget_data") : null;
                    if (w) {
                        w.value = JSON.stringify(state);
                    } else {
                        // Should have been created, but safety net
                        console.warn("[VNCCS] widget_data missing on serialize, creating...");
                        node.addWidget("text", "widget_data", JSON.stringify(state), (v) => { }, { serialize: true });
                    }
                };

                // 3. State & Widget Setup
                // Ensure 'widget_data' widget exists (ComfyUI backend hidden inputs don't always create widgets automatically)
                let dataWidget = node.widgets ? node.widgets.find(w => w.name === "widget_data") : null;
                if (!dataWidget) {
                    // Create it manually if missing. 
                    // Type "text" is safe, we'll hide it.
                    // serialize: true is default for widgets added this way? We check opts.
                    dataWidget = node.addWidget("text", "widget_data", "{}", (v) => { }, { serialize: true });
                }
                // Ensure it's hidden (cleanup hides everything else, but let's be explicit)
                if (dataWidget) dataWidget.hidden = true;

                const state = {
                    preview_valid: false, // Smart Cache Flag
                    preview_source: "gen", // "gen" or "sheet" - tracks what user sees
                    character: "",
                    character_info: {
                        sex: "female", age: 18, race: "human", skin_color: "",
                        hair: "", eyes: "", face: "", body: "", additional_details: "",
                        nsfw: false, aesthetics: "masterpiece, best quality",
                        negative_prompt: "bad quality, worst quality",
                        lora_prompt: "", background_color: "Green"
                    },
                    gen_settings: {
                        ckpt_name: "", sampler: "euler", scheduler: "normal",
                        steps: 20, cfg: 8.0, seed: 0, seed_mode: "fixed",
                        dmd_lora_name: "", dmd_lora_strength: 1.0,
                        age_lora_name: "",
                        lora_stack: [
                            { name: "", strength: 1.0 },
                            { name: "", strength: 1.0 },
                            { name: "", strength: 1.0 },
                            { name: "", strength: 1.0 },
                            { name: "", strength: 1.0 }
                        ]
                    }
                };

                let TAG_DATA = null;

                const els = {};
                const saveState = (isValid = false) => {
                    const persistData = {
                        gen_settings: state.gen_settings,
                        character: state.character
                    };
                    localStorage.setItem("VNCCS_V2_Settings", JSON.stringify(persistData));

                    // Mark cache validity
                    state.preview_valid = isValid;

                    const w = node.widgets.find(x => x.name === "widget_data");
                    if (w) w.value = JSON.stringify(state);
                };
                const loadState = () => {
                    // 1. Try Widget Data (Graph Persistence)
                    const w = node.widgets.find(x => x.name === "widget_data");
                    if (w && w.value && w.value !== "{}") {
                        try {
                            const parsed = JSON.parse(w.value);
                            // Merge from Graph Data
                            if (parsed.character) state.character = parsed.character;
                            if (parsed.character_info) Object.assign(state.character_info, parsed.character_info);
                            if (parsed.gen_settings) {
                                Object.assign(state.gen_settings, parsed.gen_settings);
                                // Ensure stack length just in case
                                while (state.gen_settings.lora_stack.length < 5) {
                                    state.gen_settings.lora_stack.push({ name: "", strength: 1.0 });
                                }
                            }
                            if (parsed.preview_valid !== undefined) state.preview_valid = parsed.preview_valid;

                            console.log("[VNCCS V2] Loaded state from graph widget. Character:", state.character);
                            return; // Found graph data, stop here (don't overwrite with global storage defaults)
                        } catch (e) { console.error("Error loading widget data", e); }
                    }

                    // 2. Fallback to LocalStorage (Global Preferences for new nodes)
                    try {
                        const s = localStorage.getItem("VNCCS_V2_Settings");
                        if (s) {
                            const parsed = JSON.parse(s);
                            // Check if new structure (has 'gen_settings' key) or legacy (flat gen_settings)
                            if (parsed.gen_settings) {
                                // New structure
                                Object.assign(state.gen_settings, parsed.gen_settings);
                                if (parsed.character) state.character = parsed.character;
                            } else {
                                // Legacy assumption
                                Object.assign(state.gen_settings, parsed);
                            }

                            while (state.gen_settings.lora_stack.length < 5) {
                                state.gen_settings.lora_stack.push({ name: "", strength: 1.0 });
                            }
                        }
                    } catch (e) { }
                };

                // 4. UI Builders
                const createField = (lbl, key, type = "text", opts = [], targetObj = state.character_info) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-field";

                    if (type === "checkbox") {
                        // Row layout for checkbox
                        wrap.style.flexDirection = "row";
                        wrap.style.alignItems = "center";
                        wrap.style.gap = "8px";
                        const inp = document.createElement("input");
                        inp.type = "checkbox";
                        inp.checked = !!targetObj[key];
                        inp.onchange = (e) => {
                            targetObj[key] = e.target.checked;
                            saveState();
                        };
                        const l = document.createElement("label");
                        l.className = "vnccs-label";
                        l.innerText = lbl;
                        wrap.appendChild(inp);
                        wrap.appendChild(l);
                        els[key] = inp;
                        return wrap;
                    }

                    // Header Row: Label + Optional Tag Button
                    const header = document.createElement("div");
                    header.style.display = "flex";
                    header.style.alignItems = "center";
                    header.style.justifyContent = "space-between";
                    header.innerHTML = `<div class="vnccs-label">${lbl}</div>`;

                    // Fields that support tag constructor
                    // hair, eyes, race, body, skin_color(maybe), face, details
                    const tagSupported = ["hair", "eyes", "race", "body", "face", "additional_details"].includes(key);
                    if (tagSupported && type === "text") {
                        const btn = document.createElement("div");
                        btn.className = "vnccs-tag-btn";
                        btn.innerHTML = "✎"; // Pencil or List icon
                        btn.title = "Open Tag Constructor";
                        btn.onclick = () => openTagConstructor(key, inp);
                        header.appendChild(btn);
                    }

                    wrap.appendChild(header);

                    let inp;
                    if (type === "select") {
                        inp = document.createElement("select"); inp.className = "vnccs-select";
                        opts.forEach(v => inp.add(new Option(v, v)));
                        inp.value = targetObj[key] || opts[0];
                        inp.onchange = (e) => { targetObj[key] = e.target.value; saveState(); };
                    } else if (type === "number") {
                        inp = document.createElement("input"); inp.className = "vnccs-input";
                        inp.type = "number";
                        if (opts.step) inp.step = opts.step;
                        inp.value = targetObj[key];
                        inp.onchange = (e) => { targetObj[key] = parseFloat(e.target.value); saveState(); };
                    } else {
                        inp = document.createElement("input"); inp.className = "vnccs-input";
                        inp.value = targetObj[key] || "";
                        inp.onchange = (e) => { targetObj[key] = e.target.value; saveState(); };
                    }
                    els[key] = inp; // Register for updates
                    wrap.appendChild(inp);
                    return wrap;
                };

                const createSlider = (lbl, key, min, max, step, targetObj = state.gen_settings) => {
                    const wrap = document.createElement("div");
                    wrap.className = "vnccs-field";
                    wrap.innerHTML = `<div class="vnccs-label">${lbl}</div>`;

                    const container = document.createElement("div");
                    container.className = "vnccs-slider-container";

                    const range = document.createElement("input");
                    range.type = "range"; range.className = "vnccs-slider";
                    range.min = min; range.max = max; range.step = step;
                    range.value = targetObj[key];

                    const num = document.createElement("input");
                    num.type = "number"; num.className = "vnccs-slider-val";
                    num.step = step;
                    num.value = targetObj[key];

                    // Sync
                    range.oninput = (e) => {
                        num.value = e.target.value;
                        targetObj[key] = parseFloat(e.target.value);
                        saveState();
                    };
                    num.onchange = (e) => {
                        let v = parseFloat(e.target.value);
                        if (v < min) v = min; if (v > max) v = max;
                        num.value = v; range.value = v;
                        targetObj[key] = v;
                        saveState();
                    };

                    container.appendChild(range);
                    container.appendChild(num);
                    wrap.appendChild(container);

                    els[key] = { range, num }; // composite ref
                    return wrap;
                };

                // 5. Build Layout
                const container = document.createElement("div");
                container.className = "vnccs-container";

                // HACK: Forward MMB (Middle Click) to Canvas for Panning
                container.addEventListener("mousedown", (e) => {
                    if (e.button === 1) { // Middle Click
                        e.preventDefault();
                        e.stopPropagation();
                        const canvasEl = app.canvas.canvas; // HTMLCanvasElement
                        if (canvasEl) {
                            // Clone event to dispatch to canvas
                            const evt = new MouseEvent("mousedown", {
                                bubbles: true,
                                cancelable: true,
                                view: window,
                                detail: e.detail,
                                screenX: e.screenX,
                                screenY: e.screenY,
                                clientX: e.clientX,
                                clientY: e.clientY,
                                ctrlKey: e.ctrlKey,
                                altKey: e.altKey,
                                shiftKey: e.shiftKey,
                                metaKey: e.metaKey,
                                button: 1,
                                buttons: 4
                            });
                            canvasEl.dispatchEvent(evt);
                        }
                    }
                });

                // --- TOP ROW ---
                const topRow = document.createElement("div");
                topRow.className = "vnccs-top-row";

                // COL 1: LEFT (Preview)
                const colLeft = document.createElement("div");
                colLeft.className = "vnccs-col";
                colLeft.innerHTML = '<div class="vnccs-section-title">Character Select</div>';

                const charSel = document.createElement("select"); charSel.className = "vnccs-select";
                charSel.onchange = async (e) => {
                    state.character = e.target.value;
                    await loadChar(state.character);
                    saveState(true);
                };
                els.charSelect = charSel;
                colLeft.appendChild(charSel);

                const btnRow = document.createElement("div");
                btnRow.className = "vnccs-btn-row";
                const btnGen = document.createElement("button");
                btnGen.className = "vnccs-btn vnccs-btn-primary";
                btnGen.innerText = "GENERATE PREVIEW";
                btnGen.onclick = () => doGenerate();
                els.btnGen = btnGen;

                const btnNew = document.createElement("button");
                btnNew.className = "vnccs-btn vnccs-btn-success";
                btnNew.innerText = "CREATE NEW";
                btnNew.onclick = () => doCreate();

                const btnDel = document.createElement("button");
                btnDel.className = "vnccs-btn vnccs-btn-danger";
                btnDel.innerText = "DELETE";
                // Shared Modal Helper
                const showModal = (title, contentFunc, buttons) => {
                    const overlay = document.createElement("div"); overlay.className = "vnccs-modal-overlay";
                    const m = document.createElement("div"); m.className = "vnccs-modal";
                    m.innerHTML = `<div class="vnccs-section-title">${title}</div>`;

                    // Content
                    const content = contentFunc(m);
                    if (content) m.appendChild(content);

                    // Buttons
                    const row = document.createElement("div"); row.className = "vnccs-btn-row";
                    buttons.forEach(b => {
                        const btn = document.createElement("button");
                        btn.className = `vnccs-btn ${b.class || ""}`;
                        btn.innerText = b.text;
                        btn.onclick = async () => {
                            if (b.action) {
                                const keepOpen = await b.action(overlay, btn);
                                if (!keepOpen) overlay.remove();
                            } else {
                                overlay.remove();
                            }
                        }
                        row.appendChild(btn);
                    });
                    m.appendChild(row);

                    overlay.appendChild(m);
                    container.appendChild(overlay);
                    return { overlay, modal: m, content };
                };

                const doCreate = () => {
                    let inpRef;
                    const { content } = showModal("New Character", () => {
                        const inp = document.createElement("input");
                        inp.className = "vnccs-input";
                        inp.placeholder = "Name...";
                        inpRef = inp;
                        return inp;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "Create",
                            class: "vnccs-btn-primary",
                            action: async (ol, btn) => {
                                const n = inpRef.value.trim();
                                if (!n) return true; // Keep open
                                try {
                                    await api.fetchApi(`/vnccs/create?name=${encodeURIComponent(n)}`);
                                    const exists = Array.from(els.charSelect.options).some(o => o.value === n);
                                    if (!exists) els.charSelect.add(new Option(n, n));

                                    state.character = n; // Updates internal state immediately
                                    els.charSelect.value = n; // Update UI
                                    await loadChar(n);
                                    saveState();
                                    return false; // Close
                                } catch (e) {
                                    alert("Create Failed: " + e);
                                    return true;
                                }
                            }
                        }
                    ]);
                    inpRef.focus();
                };

                const doDelete = () => {
                    const charName = state.character;
                    if (!charName || charName === "None" || charName === "Unknown") {
                        alert("Please select a character to delete.");
                        return;
                    }

                    showModal("Delete Character", () => {
                        const div = document.createElement("div");
                        div.innerHTML = `
                                <div style="font-size:14px; text-align:center;">
                                    Are you sure you want to <b>PERMANENTLY DELETE</b><br/>
                                    <span style="color:#fff; font-weight:bold;">'${charName}'</span>?<br/><br/>
                                    <span style="font-size:12px; color:#aaa;">This action cannot be undone.</span>
                                </div>
                            `;
                        return div;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "CONFIRM DELETE",
                            class: "vnccs-btn-danger",
                            action: async (ol, btn) => {
                                try {
                                    btn.innerText = "DELETING...";
                                    btn.disabled = true;
                                    const r = await api.fetchApi(`/vnccs/delete?name=${encodeURIComponent(charName)}`);
                                    if (r.ok) {
                                        const opts = Array.from(els.charSelect.options);
                                        const idx = opts.findIndex(o => o.value === charName);
                                        if (idx > -1) els.charSelect.remove(idx);

                                        if (els.charSelect.options.length > 0) state.character = els.charSelect.options[0].value;
                                        else state.character = "";

                                        els.charSelect.value = state.character;
                                        await loadChar(state.character);
                                        saveState();
                                        return false;
                                    } else {
                                        const err = await r.json();
                                        alert("Delete Failed: " + (err.error || "Unknown Error"));
                                        btn.innerText = "CONFIRM DELETE";
                                        btn.disabled = false;
                                        return true;
                                    }
                                } catch (e) {
                                    alert("Delete Failed: " + e);
                                    return false; // Close on crash to avoid Stuck UI
                                }
                            }
                        }
                    ]);
                };

                btnDel.onclick = () => doDelete();

                const openTagConstructor = async (fieldKey, inputEl) => {
                    // 1. Ensure Data
                    if (!TAG_DATA) {
                        try {
                            const r = await api.fetchApi("/vnccs/get_tags");
                            if (r.ok) TAG_DATA = await r.json();
                            else throw new Error("Failed to load tags");
                        } catch (e) {
                            alert("Error loading tag database: " + e);
                            return;
                        }
                    }

                    // 2. Determine Categories
                    // Mapping: widget_key -> [json_keys...]
                    const map = {
                        "hair": ["hair_color", "hairstyles"],
                        "eyes": ["eyes"], // special handling for nested
                        "face": ["eyes"], // special handling (nested under eyes in user update)
                        "race": ["races"],
                        "body": ["breast_size"],
                        "additional_details": ["details"]
                    };

                    const categories = map[fieldKey] || [];
                    if (!categories.length) return;

                    // 3. Collect Tags
                    const allTags = [];
                    categories.forEach(cat => {
                        let data = TAG_DATA.tags[cat];
                        if (cat === "eyes") {
                            if (fieldKey === "eyes") {
                                // Flatten eyes
                                if (TAG_DATA.tags.eyes.colors) allTags.push({ header: "Eye Colors", items: TAG_DATA.tags.eyes.colors });
                                if (TAG_DATA.tags.eyes.features) allTags.push({ header: "Eye Features", items: TAG_DATA.tags.eyes.features });
                            } else if (fieldKey === "face") {
                                // Face Characteristics
                                if (TAG_DATA.tags.eyes.face_characteristics) allTags.push({ header: "Face Characteristics", items: TAG_DATA.tags.eyes.face_characteristics });
                            }
                        } else if (data) {
                            // Arrays
                            // format header nicel
                            const h = cat.replace("_", " ").toUpperCase();
                            allTags.push({ header: h, items: data });
                        }
                    });

                    if (!allTags.length) {
                        alert("No tags found for this category.");
                        return;
                    }

                    // 4. Modal
                    // Parse current values to pre-select
                    const currentVals = inputEl.value.split(",").map(s => s.trim().toLowerCase()).filter(Boolean);
                    const selected = new Set(currentVals);

                    showModal(`Tag Constructor: ${fieldKey}`, (modal) => {
                        const container = document.createElement("div");
                        container.className = "vnccs-tag-grid";

                        // Fix width for tag modal
                        modal.style.width = "500px";

                        allTags.forEach(group => {
                            if (group.header) {
                                const h = document.createElement("div");
                                h.className = "vnccs-tag-category";
                                h.style.width = "100%";
                                h.innerText = group.header;
                                container.appendChild(h);
                            }

                            group.items.forEach(item => {
                                const t = item.tag;
                                const chip = document.createElement("div");
                                chip.className = "vnccs-tag-chip";
                                chip.innerText = item.label || t;
                                if (selected.has(t.replace(/_/g, " "))) chip.classList.add("selected");
                                // Logic: check 't' (underscore) against currentVals (spaces usually)
                                // Prompt usually uses spaces? "long hair". tag is "long_hair".
                                // Let's normalize.
                                const normTag = t.replace(/_/g, " ");
                                if (currentVals.includes(normTag) || currentVals.includes(t)) {
                                    chip.classList.add("selected");
                                    selected.add(normTag);
                                }

                                chip.onclick = () => {
                                    const useTag = t.replace(/_/g, " "); // Use spaces for prompt readability? Or Keep underscores?
                                    // Danbooru acts weird. Usually spaces are better for readability in UI prompts unless model specific.
                                    // Comfy/SD usually handles spaces. detailed prompts usually "long hair".

                                    if (selected.has(useTag)) {
                                        selected.delete(useTag);
                                        chip.classList.remove("selected");
                                    } else {
                                        selected.add(useTag);
                                        chip.classList.add("selected");
                                    }
                                };
                                container.appendChild(chip);
                            });
                        });

                        return container;
                    }, [
                        { text: "Cancel" },
                        {
                            text: "APPLY",
                            class: "vnccs-btn-primary",
                            action: () => {
                                const final = Array.from(selected).join(", ");
                                inputEl.value = final;
                                // Trigger change
                                inputEl.dispatchEvent(new Event('change'));
                                return false; // Close
                            }
                        }
                    ]);
                };

                btnDel.onclick = () => doDelete();

                btnRow.appendChild(btnGen);
                btnRow.appendChild(btnNew);
                btnRow.appendChild(btnDel);
                colLeft.appendChild(btnRow);

                const frame = document.createElement("div");
                frame.className = "vnccs-preview-container";
                frame.innerHTML = '<div class="vnccs-placeholder">No Preview</div>';
                const img = document.createElement("img");
                img.className = "vnccs-preview-img"; img.style.display = "none";
                frame.appendChild(img);
                els.previewImg = img; els.placeholder = frame.querySelector(".vnccs-placeholder");
                colLeft.appendChild(frame);

                topRow.appendChild(colLeft);

                // COL 2: CENTER (Attributes)
                const colCenter = document.createElement("div");
                colCenter.className = "vnccs-col";
                colCenter.innerHTML = '<div class="vnccs-section-title">Attributes</div>';

                colCenter.appendChild(createField("Background", "background_color", "select", ["Green", "Blue"]));
                colCenter.appendChild(createField("Gender", "sex", "select", ["female", "male"]));
                colCenter.appendChild(createSlider("Age", "age", 1, 100, 1, state.character_info));
                colCenter.appendChild(createField("Race", "race"));
                colCenter.appendChild(createField("Skin Color", "skin_color"));
                colCenter.appendChild(createField("Body Type", "body"));
                colCenter.appendChild(createField("Face Features", "face"));
                colCenter.appendChild(createField("Hair Style", "hair"));
                colCenter.appendChild(createField("Eye Color", "eyes"));
                colCenter.appendChild(createField("Details", "additional_details"));
                colCenter.appendChild(createField("NSFW Mode", "nsfw", "checkbox"));

                topRow.appendChild(colCenter);

                // COL 3: RIGHT (Generation)
                const colRight = document.createElement("div");
                colRight.className = "vnccs-col";
                colRight.innerHTML = '<div class="vnccs-section-title">Generation</div>';

                // Checkpoint
                const wrapCkpt = document.createElement("div"); wrapCkpt.className = "vnccs-field";
                wrapCkpt.innerHTML = '<div class="vnccs-label">Checkpoint (SDXL)</div>';
                const sCkpt = document.createElement("select"); sCkpt.className = "vnccs-select";
                sCkpt.onchange = (e) => { state.gen_settings.ckpt_name = e.target.value; saveState(); };
                els.ckptSelect = sCkpt; wrapCkpt.appendChild(sCkpt); colRight.appendChild(wrapCkpt);

                colRight.appendChild(createField("Sampler", "sampler", "select", [], state.gen_settings));
                els.sampler = colRight.lastChild.querySelector("select");

                colRight.appendChild(createField("Scheduler", "scheduler", "select", [], state.gen_settings));
                els.scheduler = colRight.lastChild.querySelector("select");

                // Sliders for Steps & CFG
                colRight.appendChild(createSlider("Steps", "steps", 1, 100, 1));
                colRight.appendChild(createSlider("CFG", "cfg", 1, 20, 0.1));

                // SEED Section (Rebalanced)
                const seedWrap = document.createElement("div"); seedWrap.className = "vnccs-field";
                seedWrap.innerHTML = '<div class="vnccs-label">Seed</div>';

                const seedRow = document.createElement("div");
                seedRow.style.display = "flex";
                seedRow.style.gap = "5px";
                seedRow.style.width = "100%"; // Ensure fill

                const seedInp = document.createElement("input"); seedInp.className = "vnccs-input";
                seedInp.type = "number"; seedInp.value = state.gen_settings.seed || 0;
                seedInp.style.flex = "1.5"; // Give seed more space, but not overwhelming
                seedInp.onchange = (e) => {
                    state.gen_settings.seed = parseInt(e.target.value);
                    saveState();
                };
                els.seed = seedInp;

                const seedMode = document.createElement("select"); seedMode.className = "vnccs-select";
                seedMode.style.flex = "1"; // Less space for mode
                // Since .vnccs-select has zoom:1.5, we might need to be careful with flex calc?
                // Actually zoom affects computed width. If we set width 100% in CSS, flex controls container width.
                // The zoom might make it overflow its flex item container.
                seedMode.style.width = "100%"; // Inner width

                ["fixed", "randomize"].forEach(x => seedMode.add(new Option(x, x)));
                seedMode.value = state.gen_settings.seed_mode || "fixed";
                seedMode.onchange = (e) => {
                    state.gen_settings.seed_mode = e.target.value;
                    saveState();
                };
                els.seed_mode = seedMode;

                seedRow.appendChild(seedInp);
                // Wrap seedMode in a div to contain the zoom overflow issues comfortably?
                // Let's just put it directly first. 
                const smWrap = document.createElement("div"); smWrap.style.flex = "1"; smWrap.style.minWidth = "0";
                smWrap.appendChild(seedMode);
                seedRow.appendChild(smWrap);

                seedWrap.appendChild(seedRow);
                colRight.appendChild(seedWrap);

                // --- LoRA Section ---
                const loraHeader = document.createElement("div");
                loraHeader.className = "vnccs-section-title";
                loraHeader.style.marginTop = "10px";
                loraHeader.innerText = "LoRa Stack";
                colRight.appendChild(loraHeader);

                // DMD2 LoRA
                const dmdWrap = document.createElement("div"); dmdWrap.className = "vnccs-lora-item";
                dmdWrap.innerHTML = '<div class="vnccs-label">DMD2 LoRA Model</div>';
                const dmdRow = document.createElement("div"); dmdRow.className = "vnccs-lora-row";
                const dmdSel = document.createElement("select"); dmdSel.className = "vnccs-select";
                dmdSel.style.flex = "2";
                dmdSel.onchange = (e) => { state.gen_settings.dmd_lora_name = e.target.value; saveState(); };
                els.dmdSelect = dmdSel;

                const dmdStr = document.createElement("input");
                dmdStr.type = "checkbox";
                dmdStr.checked = (state.gen_settings.dmd_lora_strength || 0) > 0;
                dmdStr.style.flex = "0 0 auto"; // Prevent stretching
                dmdStr.style.margin = "0 10px";
                dmdStr.onchange = (e) => {
                    state.gen_settings.dmd_lora_strength = e.target.checked ? 1.0 : 0.0;
                    saveState();
                };

                dmdRow.appendChild(dmdSel); dmdRow.appendChild(dmdStr);
                dmdWrap.appendChild(dmdRow);
                colRight.appendChild(dmdWrap);
                els.dmdSlider = dmdStr; // Renamed ref for logic compat, though it's an input now

                // Age LoRA
                const ageWrap = document.createElement("div"); ageWrap.className = "vnccs-lora-item";
                ageWrap.innerHTML = '<div class="vnccs-label">Age LoRA (Auto Strength)</div>';
                const ageSel = document.createElement("select"); ageSel.className = "vnccs-select";
                ageSel.onchange = (e) => { state.gen_settings.age_lora_name = e.target.value; saveState(); };
                els.ageSelect = ageSel;
                ageWrap.appendChild(ageSel);
                colRight.appendChild(ageWrap);

                // Stack (5 Slots)
                const stackContainer = document.createElement("div");
                stackContainer.className = "vnccs-lora-stack";
                els.loraStackSelects = [];

                for (let i = 0; i < 5; i++) {
                    const item = document.createElement("div"); item.className = "vnccs-lora-item";
                    const row = document.createElement("div"); row.className = "vnccs-lora-row";

                    const sel = document.createElement("select"); sel.className = "vnccs-select";
                    sel.style.flex = "2";
                    sel.onchange = (e) => { state.gen_settings.lora_stack[i].name = e.target.value; saveState(); };

                    const rng = document.createElement("input"); rng.className = "vnccs-input";
                    rng.type = "number"; rng.step = "0.05"; rng.style.flex = "1";

                    rng.onchange = (e) => {
                        state.gen_settings.lora_stack[i].strength = parseFloat(e.target.value);
                        saveState();
                    };

                    row.appendChild(sel);
                    row.appendChild(rng);
                    item.appendChild(row);
                    stackContainer.appendChild(item);

                    // Ref for population
                    els.loraStackSelects.push({ sel, rng, idx: i });
                }
                colRight.appendChild(stackContainer);

                topRow.appendChild(colRight);
                container.appendChild(topRow);

                // --- BOTTOM ROW (Prompts) ---
                const bottomRow = document.createElement("div");
                bottomRow.className = "vnccs-bottom-row";

                const createText = (lbl, key) => {
                    const w = document.createElement("div"); w.className = "vnccs-textarea-wrapper";
                    w.innerHTML = `<div class="vnccs-textarea-label">${lbl}</div>`;
                    const t = document.createElement("textarea");
                    t.value = state.character_info[key] || "";
                    t.oninput = (e) => { state.character_info[key] = e.target.value; saveState(); };
                    w.appendChild(t);
                    els[key] = t;
                    return w;
                }
                bottomRow.appendChild(createText("Aesthetics", "aesthetics"));
                bottomRow.appendChild(createText("Negative Prompt", "negative_prompt"));
                bottomRow.appendChild(createText("LoRA Trigger", "lora_prompt"));

                container.appendChild(bottomRow);

                // Inject UI
                this.addDOMWidget("ui", "ui", container, { serialize: false });

                // 6. Logic

                // EVENT LISTENER for Backend Updates
                api.addEventListener("vnccs.preview.updated", (e) => {
                    // Check if this event is for ME
                    // Note: node.id is string or int depending on context, usually string in api inputs.
                    if (e.detail.node_id == node.id) {
                        const charName = e.detail.character;
                        console.log(`[VNCCS] Preview Update Event received for '${charName}' (Node ${node.id})`);
                        // Optional: verify character matches current state
                        if (charName === state.character) {
                            console.log("[VNCCS] Character matches. Refreshing local preview...");
                            // Force reload from cache endpoint
                            els.previewImg.src = `/vnccs/get_cached_preview?character=${encodeURIComponent(charName)}&t=${Date.now()}`;
                            els.previewImg.style.display = "block";
                            els.placeholder.style.display = "none";

                            state.preview_valid = true;
                            // state.preview_source = "gen"; // Logic suggests we are now up to date
                            saveState(true);
                        } else {
                            console.warn(`[VNCCS] Ignoring update: Current character '${state.character}' != Update character '${charName}'`);
                        }
                    }
                });

                const init = async () => {
                    loadState();
                    try {
                        const r = await api.fetchApi("/vnccs/context_lists");
                        const d = await r.json();

                        const pop = (el, items, none = false) => {
                            if (!el) return; el.innerHTML = "";
                            if (none) el.add(new Option("None", ""));
                            (items || []).forEach(x => el.add(new Option(x, x)));
                        };

                        pop(els.ckptSelect, d.checkpoints);
                        pop(els.sampler, d.samplers);
                        pop(els.scheduler, d.schedulers);

                        // Populate LoRA selectors
                        const loras = d.loras || [];
                        pop(els.dmdSelect, loras, true);
                        pop(els.ageSelect, loras, true);
                        els.loraStackSelects.forEach(o => pop(o.sel, loras, true));

                        els.charSelect.innerHTML = "";
                        if (!d.characters || !d.characters.length) els.charSelect.add(new Option("None", ""));
                        else d.characters.forEach(c => els.charSelect.add(new Option(c, c)));

                        // Restore values or Default
                        const g = state.gen_settings;

                        if (g.ckpt_name) {
                            els.ckptSelect.value = g.ckpt_name;
                        } else if (els.ckptSelect.options.length > 0) {
                            // Default to first option if not set
                            g.ckpt_name = els.ckptSelect.options[0].value;
                            els.ckptSelect.value = g.ckpt_name;
                            saveState(); // Ensure this default is saved immediately
                        }
                        if (g.sampler) els.sampler.value = g.sampler;
                        if (g.scheduler) els.scheduler.value = g.scheduler;

                        if (g.steps && els.steps) { els.steps.range.value = g.steps; els.steps.num.value = g.steps; }
                        if (g.cfg && els.cfg) { els.cfg.range.value = g.cfg; els.cfg.num.value = g.cfg; }

                        if (g.seed !== undefined && els.seed) els.seed.value = g.seed;
                        if (g.seed_mode && els.seed_mode) els.seed_mode.value = g.seed_mode;

                        // Restore LoRAs
                        if (g.dmd_lora_name) els.dmdSelect.value = g.dmd_lora_name;
                        if (g.dmd_lora_strength !== undefined) {
                            els.dmdSlider.checked = (g.dmd_lora_strength > 0);
                        } else {
                            els.dmdSlider.checked = true;
                        }

                        if (g.age_lora_name) els.ageSelect.value = g.age_lora_name;
                        // Restore Stack
                        g.lora_stack.forEach((item, i) => {
                            if (i < els.loraStackSelects.length) {
                                const ref = els.loraStackSelects[i];
                                if (item.name) ref.sel.value = item.name;
                                ref.rng.value = item.strength;
                            }
                        });

                        if (state.character) {
                            // Ensure it's in the list (handle potential race/latency)
                            const exists = Array.from(els.charSelect.options).some(o => o.value === state.character);
                            if (!exists) {
                                els.charSelect.add(new Option(state.character, state.character));
                            }
                            els.charSelect.value = state.character;
                        }
                        else if (d.characters && d.characters.length) {
                            state.character = d.characters[0];
                            els.charSelect.value = state.character;
                        }

                        // If we have valid widget_data for this character, skip loading info from backend
                        // to preserve user's unsaved edits from previous session
                        const hasWidgetData = state.character_info && state.character_info.hair !== undefined;
                        if (state.character) await loadChar(state.character, hasWidgetData);

                        // CRITICAL: Ensure the widget is synced with whatever state we just loaded/defaulted
                        // If we loaded a character, we assume the preview is valid (or at least we want to try to use cache)
                        // AND we explicitly set source to 'sheet' because loadChar does that.
                        saveState(!!state.character);

                    } catch (e) { console.error(e); }
                };

                const loadChar = async (n, skipInfoLoad = false) => {
                    if (!n) return;
                    try {
                        // 1. Fetch Info (skip if restoring from widget_data)
                        if (!skipInfoLoad) {
                            const r = await api.fetchApi(`/vnccs/character_info?character=${encodeURIComponent(n)}`);
                            const i = await r.json();

                            const defaultInfo = {
                                sex: "female", age: 18, race: "human", skin_color: "",
                                hair: "", eyes: "", face: "", body: "", additional_details: "",
                                nsfw: false, aesthetics: "masterpiece, best quality",
                                negative_prompt: "bad quality, worst quality",
                                lora_prompt: "", background_color: "Green"
                            };

                            // Reset & Assign
                            Object.assign(state.character_info, defaultInfo);
                            Object.assign(state.character_info, i);
                        }

                        // Update Fields from current state
                        Object.keys(state.character_info).forEach(k => {
                            if (els[k]) {
                                let val = state.character_info[k];
                                // Normalize background_color case to match dropdown options
                                if (k === "background_color" && typeof val === "string" && val) {
                                    val = val.charAt(0).toUpperCase() + val.slice(1).toLowerCase();
                                    state.character_info[k] = val; // Update state too
                                }
                                if (els[k].range && els[k].num) {
                                    // Slider Composite
                                    els[k].range.value = val;
                                    els[k].num.value = val;
                                }
                                else if (els[k].type === "checkbox") els[k].checked = !!val;
                                else els[k].value = val;
                            }
                        });

                        // 2. Fetch Preview Image
                        // Strategy: Try Sheet -> Fail -> Try Cache -> Fail -> Placeholder

                        // Function to try loading cache
                        const tryCache = () => {
                            console.log("[VNCCS] Trying to load cached preview...");
                            const cacheUrl = `/vnccs/get_cached_preview?character=${encodeURIComponent(n)}&t=${Date.now()}`;

                            // Set up one-time error handler for cache failure
                            els.previewImg.onerror = () => {
                                console.warn("[VNCCS] Both sheet and cache preview failed.");
                                els.previewImg.style.display = "none";
                                els.placeholder.innerText = "No Preview Image";
                                els.placeholder.style.display = "block";
                                // Clear onerror to prevent potential loop
                                els.previewImg.onerror = null;
                            };

                            els.previewImg.src = cacheUrl;
                            // Update state to reflect we are trying/using gen(cache) source
                            state.preview_source = "gen";
                        };

                        // First try Sheet
                        els.previewImg.onerror = () => {
                            console.warn("[VNCCS] Sheet preview load failed. Fallback to cache.");
                            tryCache();
                        };

                        els.previewImg.src = `/vnccs/get_character_sheet_preview?character=${encodeURIComponent(n)}&t=${Date.now()}`;
                        els.previewImg.style.display = "block";
                        els.placeholder.style.display = "none";
                        state.preview_source = "sheet";

                    } catch (e) { console.error(e); }
                };

                const doGenerate = async () => {
                    if (!state.gen_settings.ckpt_name) { alert("Select Checkpoint"); return; }
                    if (els.btnGen.disabled) return;

                    if (state.gen_settings.seed_mode === "randomize") {
                        state.gen_settings.seed = Math.floor(Math.random() * 10000000000000);
                        if (els.seed) els.seed.value = state.gen_settings.seed;
                    }

                    // Show loading overlay
                    const loadingOverlay = document.createElement('div');
                    loadingOverlay.className = 'vnccs-loading-overlay';
                    loadingOverlay.innerHTML = `
                        <div class="vnccs-spinner"></div>
                        <div class="vnccs-loading-text">Generating preview<span class="vnccs-loading-dots"></span></div>
                    `;
                    container.appendChild(loadingOverlay);

                    els.btnGen.innerText = "GENERATING...";
                    els.btnGen.disabled = true;
                    saveState();

                    try {
                        const payload = {
                            character: state.character,
                            character_info: state.character_info,
                            gen_settings: { ...state.gen_settings }
                        };
                        // Clean stack in the copy
                        payload.gen_settings.lora_stack = payload.gen_settings.lora_stack.filter(x => x.name && x.name !== "None");

                        const r = await api.fetchApi("/vnccs/preview_generate", { method: "POST", body: JSON.stringify(payload) });

                        if (!r.ok) {
                            const errText = await r.text();
                            throw new Error(`Server Error (${r.status}): ${errText}`);
                        }

                        const d = await r.json();
                        if (d.image) {
                            els.previewImg.src = "data:image/png;base64," + d.image;
                            els.previewImg.style.display = "block";
                            els.placeholder.style.display = "none";
                            // Successful generation -> Cache is Valid AND source is 'gen'
                            state.preview_source = "gen";
                            saveState(true);
                        }
                    } catch (e) { alert("Error: " + e); }
                    finally {
                        loadingOverlay.remove();
                        els.btnGen.innerText = "GENERATE PREVIEW";
                        els.btnGen.disabled = false;
                    }
                };



                // 7. Graph Restore Hook / Main Entry Point
                let initialized = false;
                node.onConfigure = function () {
                    if (initialized) return;
                    // Prevent double init if called multiple times
                    // But we might need to re-load state if configure happens again? 
                    // Usually configure happens once on load.

                    initialized = true;
                    // Check if widget_data has value NOW
                    const w = node.widgets.find(x => x.name === "widget_data");
                    if (w && w.value) {
                        // We have data, init will use it via loadState
                    }

                    init();
                };

                // Fallback for new nodes (onConfigure runs on add too, but just in case)
                setTimeout(() => {
                    if (!initialized) {
                        initialized = true;
                        init();
                    }
                }, 100);
            };
        }
    }
});
