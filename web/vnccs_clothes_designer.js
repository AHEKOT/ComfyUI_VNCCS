import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE = `
/* Reusing core consistency styles from V2 */
.vnccs-container {
    display: flex; flex-direction: column; background: #1e1e1e; color: #e0e0e0;
    font-family: 'Consolas', 'Monaco', monospace; font-size: 16px;
    width: 100%; height: 100%; overflow: hidden; box-sizing: border-box;
    padding: 10px; gap: 10px; pointer-events: none; zoom: 0.67;
}
.vnccs-top-row {
    display: grid; grid-template-columns: 30% 35% 35%; gap: 10px;
    flex: 1; min-height: 0; width: 100%;
}
.vnccs-col {
    display: flex; flex-direction: column; background: #252525;
    border: 1px solid #333; border-radius: 6px; padding: 10px; gap: 10px;
    overflow-y: auto; height: 100%; box-sizing: border-box; pointer-events: auto;
}
.vnccs-section-title {
    font-size: 14px; font-weight: bold; color: #fff; border-bottom: 2px solid #444;
    padding-bottom: 5px; margin-bottom: 5px; text-transform: uppercase; flex-shrink: 0;
}
.vnccs-field { display: flex; flex-direction: column; gap: 4px; margin-bottom: 5px; flex-shrink: 0; }
.vnccs-label { color: #aaa; font-size: 11px; font-weight: 600; }
.vnccs-input, .vnccs-textarea, .vnccs-select {
    background: #151515; border: 1px solid #444; color: #fff;
    border-radius: 4px; padding: 6px; font-family: inherit; font-size: 12px;
    width: 100%; box-sizing: border-box;
}
.vnccs-textarea { resize: vertical; min-height: 40px; }
.vnccs-select { zoom: 1.5; padding: 4px; } 

.vnccs-btn {
    padding: 8px; border: none; border-radius: 4px; cursor: pointer;
    font-weight: bold; text-transform: uppercase; font-size: 12px; color: white;
    text-align: center; flex: 1; /* Match V2 flex buttons */
}
.vnccs-btn-primary { background: #3558c7; } .vnccs-btn-primary:hover { background: #4264d9; }
.vnccs-btn-success { background: #2e7d32; } .vnccs-btn-success:hover { background: #388e3c; }
.vnccs-btn-danger { background: #d32f2f; } .vnccs-btn-danger:hover { background: #b71c1c; }

.vnccs-btn-row { display: flex; gap: 10px; margin-top: auto; flex-shrink: 0; flex-wrap: wrap; }

.vnccs-row { display: flex; gap: 5px; align-items: center; }

/* Preview */
.vnccs-preview-container {
    flex: 1; background: #000; border: 1px solid #333; border-radius: 4px;
    display: flex; align-items: center; justify-content: center; overflow: hidden;
    position: relative; min-height: 0;
}
.vnccs-preview-img { width: 100%; height: 100%; object-fit: contain; }
.vnccs-placeholder { color: #555; text-align: center; }

/* Modal */
.vnccs-modal-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center;
    z-index: 1000; pointer-events: auto;
}
.vnccs-modal {
    background: #252525; border: 1px solid #444; padding: 20px; border-radius: 8px;
    min-width: 300px; max-width: 90%; display: flex; flex-direction: column; gap: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
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
                } catch (e) { }

                const state = {
                    ...defaultState,
                    ...saved,
                    costume_info: { ...defaultState.costume_info, ...(saved.costume_info || {}) },
                    gen_settings: { ...defaultState.gen_settings, ...(saved.gen_settings || {}) }
                };

                const els = {};

                const saveState = () => {
                    if (dataWidget) dataWidget.value = JSON.stringify(state);
                    // Persist to LocalStorage for session restoration
                    try {
                        localStorage.setItem("VNCCS_ClothesDesigner_State", JSON.stringify(state));
                    } catch (e) { }
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

                // Shared Modal Helper
                const showModal = (title, contentFunc, buttons) => {
                    const overlay = document.createElement("div"); overlay.className = "vnccs-modal-overlay";
                    const m = document.createElement("div"); m.className = "vnccs-modal";
                    m.innerHTML = `<div class="vnccs-section-title">${title}</div>`;

                    const content = contentFunc(m);
                    if (content) m.appendChild(content);

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
                        // Initial resize after append (needs small delay or observer, or just call it)
                        // calling immediately might not work if not in DOM, but we can try or use requestAnimationFrame
                        setTimeout(autoResize, 10);

                        // Also hook into value setting if needed, but onchange covers user input.
                        // We need to trigger this when loading data too.
                        // We'll expose it or just trigger input event? 
                        // Let's add it to the element so we can call it later.
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

                // Container
                const container = document.createElement("div"); container.className = "vnccs-container";

                // --- TOP ROW ---
                const topRow = document.createElement("div"); topRow.className = "vnccs-top-row";

                // --- COL 1: PREVIEW & SELECTION ---
                const colLeft = document.createElement("div"); colLeft.className = "vnccs-col";
                colLeft.innerHTML = '<div class="vnccs-section-title">Design Studio</div>';

                // 1. Character Dropdown
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

                // 2. Costume Dropdown (Moved to Left)
                const costRow = document.createElement("div"); costRow.className = "vnccs-field";
                costRow.innerHTML = '<div class="vnccs-label">COSTUME (Select to Edit)</div>';
                const costSel = document.createElement("select"); costSel.className = "vnccs-select";
                costSel.onchange = async (e) => {
                    state.costume = e.target.value;
                    await loadCostumeInfo();
                    updatePreviewImage(); // Also update preview when switching costume
                    saveState();
                };
                els.costSel = costSel;
                costRow.appendChild(costSel);
                colLeft.appendChild(costRow);

                // 3. Action Buttons (Create/Delete Costume - NOT Character)
                const actionRow = document.createElement("div"); actionRow.className = "vnccs-btn-row";
                actionRow.style.marginBottom = "10px";

                const btnNewCostume = document.createElement("button");
                btnNewCostume.className = "vnccs-btn vnccs-btn-success";
                btnNewCostume.innerText = "NEW COSTUME";
                btnNewCostume.onclick = () => {
                    showModal("New Costume Name", () => {
                        const inp = document.createElement("input"); inp.className = "vnccs-input";
                        return inp;
                    }, [{ text: "Cancel" }, {
                        text: "CREATE", class: "vnccs-btn-primary", action: async (ol, btn) => {
                            const n = ol.querySelector("input").value.trim();
                            if (n) {
                                await api.fetchApi("/vnccs/save_costume", {
                                    method: "POST", body: JSON.stringify({
                                        character: state.character, costume: n, info: {}
                                    })
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

                const btnDelCostume = document.createElement("button");
                btnDelCostume.className = "vnccs-btn vnccs-btn-danger";
                btnDelCostume.innerText = "DELETE COSTUME";
                btnDelCostume.onclick = () => {
                    if (state.costume === "Naked") { showInfo("Warning", "Cannot delete Naked costume."); return; }
                    showModal("Delete Costume", () => {
                        return document.createElement("div").innerHTML = `Delete <b>${state.costume}</b>?`;
                    }, [{ text: "Cancel" }, {
                        text: "DELETE", class: "vnccs-btn-danger", action: async () => {
                            // Assuming we have a delete endpoint? Or just save empty? 
                            // Implementation plan didn't specify delete endpoint.
                            // But we can just "forget" it. Actually we need to remove folder.
                            // For now, let's just alert "Not fully implemented" or try to remove from list?
                            // VNCCS lists come from folders.
                            // Let's rely on manual deletion or implement if critical.
                            // User asked for "Delete" button.
                            // alert("Deletion of costume folders not yet implemented in backend API. Please delete folder manually.");
                            // console.warn("Deletion of costume folders not yet implemented in backend API. Please delete folder manually.");
                            showInfo("Not Implemented", "Deletion of costume folders not yet implemented in backend API. Please delete folder manually.");
                            return false;
                        }
                    }]);
                };

                actionRow.appendChild(btnNewCostume);
                actionRow.appendChild(btnDelCostume);
                colLeft.appendChild(actionRow);

                // 4. Generate Button
                const btnGen = document.createElement("button");
                btnGen.className = "vnccs-btn vnccs-btn-primary";
                btnGen.innerText = "GENERATE PREVIEW";
                btnGen.style.width = "100%";
                btnGen.style.marginBottom = "5px";
                btnGen.style.flex = "0 0 auto"; // Prevent vertical stretching
                btnGen.onclick = async () => {
                    if (!state.character) { showInfo("Error", "Select Character"); return; }

                    if (state.gen_settings.seed_mode === "randomize") {
                        state.gen_settings.seed = Math.floor(Math.random() * 10000000000000);
                        if (els.seed) els.seed.value = state.gen_settings.seed;
                    }

                    // Failsafe: Ensure UNET is set
                    if (!state.gen_settings.unet_name && els.unet_name && els.unet_name.value) {
                        state.gen_settings.unet_name = els.unet_name.value;
                        saveState();
                    }
                    if (!state.gen_settings.unet_name) {
                        showInfo("Error", "No UNET Model Selected. Please select a model.");
                        return;
                    }

                    saveCostumeToBackend();
                    saveState();
                    btnGen.innerText = "GENERATING...";
                    btnGen.disabled = true;
                    try {
                        const r = await api.fetchApi("/vnccs/clothes_preview", {
                            method: "POST", body: JSON.stringify(state)
                        });
                        if (r.ok) {
                            const d = await r.json();
                            if (d.image) {
                                els.previewImg.src = "data:image/png;base64," + d.image;
                                els.previewImg.style.display = "block";
                                els.placeholder.style.display = "none";
                            }
                        } else {
                            showInfo("Error", "Generation Failed");
                        }
                    } catch (e) { showInfo("Error", e.toString()); }
                    btnGen.innerText = "GENERATE PREVIEW / SAVE";
                    btnGen.disabled = false;
                };
                els.btnGen = btnGen;
                colLeft.appendChild(btnGen);

                // Preview Frame
                const frame = document.createElement("div");
                frame.className = "vnccs-preview-container";
                frame.style.marginTop = "5px";
                frame.innerHTML = '<div class="vnccs-placeholder">No Preview</div>';

                const pImg = document.createElement("img");
                pImg.className = "vnccs-preview-img"; pImg.style.display = "none";
                pImg.onclick = () => { window.open(pImg.src, "_blank"); };
                frame.appendChild(pImg);

                els.previewImg = pImg;
                els.placeholder = frame.querySelector(".vnccs-placeholder");

                colLeft.appendChild(frame);
                topRow.appendChild(colLeft);

                // --- COL 2: ATTRIBUTES (Costume Data) ---
                const colMid = document.createElement("div"); colMid.className = "vnccs-col";
                colMid.innerHTML = '<div class="vnccs-section-title">Costume Attributes</div>';

                // Attributes only
                colMid.appendChild(createField("top", "e.g. White t-shirt"));
                colMid.appendChild(createField("bottom", "e.g. Blue jeans"));
                colMid.appendChild(createField("shoes", "e.g. Sneakers"));
                colMid.appendChild(createField("head", "e.g. Hat"));
                colMid.appendChild(createField("face", "Face features (e.g. glasses)"));

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
                        const val = state.gen_settings[key] || list[0];
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
                    } catch (e) { }

                    createSetting("VAE", "vae_name", "select", []);
                    try {
                        const r4 = await api.fetchApi("/object_info/VAELoader");
                        const d4 = await r4.json();
                        els.vae_name.innerHTML = "";
                        d4.VAELoader.input.required.vae_name[0].forEach(o => els.vae_name.add(new Option(o, o)));
                    } catch (e) { }

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

                    els.costSel.innerHTML = "";
                    list.forEach(i => els.costSel.add(new Option(i, i)));

                    // Always default to Naked to ensure fresh view when switching characters
                    if (list.includes("Naked")) {
                        state.costume = "Naked";
                    } else if (list.length) {
                        state.costume = list[0];
                    }
                    els.costSel.value = state.costume;

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

                const updatePreviewImage = async () => {
                    if (!state.character) return;
                    const ts = Date.now();
                    const url = `/vnccs/get_preview?character=${encodeURIComponent(state.character)}&costume=${encodeURIComponent(state.costume)}&ts=${ts}`;

                    // Check validity first to show message
                    try {
                        const r = await fetch(url);
                        if (!r.ok) {
                            if (r.status === 400) {
                                const txt = await r.text();
                                showInfo("Character Incomplete", txt);
                            }
                            els.previewImg.style.display = "none";
                            els.placeholder.style.display = "block";
                            return;
                        }
                    } catch (e) { }

                    els.previewImg.src = url;
                    els.previewImg.style.display = "block";
                    els.placeholder.style.display = "none";
                    els.previewImg.onerror = () => {
                        els.previewImg.style.display = "none";
                        els.placeholder.style.display = "block";
                    };
                };

                node.addDOMWidget("clothes_designer_ui", "ui", container, {
                    serialize: false,
                    hideOnZoom: false
                });

            };
        }
    }
});
