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
    font-size: 13px;
    width: 100%;
    height: 100%;
    overflow: hidden; /* Main container shouldn't scroll, inner parts will */
    box-sizing: border-box;
    padding: 10px;
    gap: 10px;
    pointer-events: none; /* Allow canvas zoom/pan in gaps */
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

/* BOTTOM ROW: Prompts */
.vnccs-bottom-row {
    display: flex;
    flex-direction: row;
    gap: 10px;
    height: 75px; /* Reduced height per user request */
    min-height: 75px;
    width: 100%;
    flex-shrink: 0;
    pointer-events: auto; /* Re-enable interaction */
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
    pointer-events: auto; /* Re-enable interaction */
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
}

/* Scrollbar */
.vnccs-col::-webkit-scrollbar { width: 6px; }
.vnccs-col::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }

/* Fields */
.vnccs-field { display: flex; flex-direction: column; gap: 4px; margin-bottom: 5px; flex-shrink: 0; }
.vnccs-label { color: #aaa; font-size: 11px; font-weight: 600; }
.vnccs-input, .vnccs-select, .vnccs-textarea {
    background: #151515; border: 1px solid #444; color: #fff;
    border-radius: 4px; padding: 6px; font-family: inherit; font-size: 12px;
    width: 100%; box-sizing: border-box;
}
.vnccs-input:focus { border-color: #5b96f5; outline: none; }

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
}
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

                // 3. State
                const state = {
                    character: "",
                    character_info: {
                        sex: "female", age: 18, race: "human", skin_color: "",
                        hair: "", eyes: "", face: "", body: "", additional_details: "",
                        nsfw: false, aesthetics: "masterpiece, best quality",
                        negative_prompt: "bad quality, worst quality",
                        lora_prompt: "", background_color: ""
                    },
                    gen_settings: {
                        ckpt_name: "", sampler: "euler", scheduler: "normal",
                        steps: 20, cfg: 8.0, seed: 0,
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

                const els = {};
                const saveState = () => {
                    localStorage.setItem("VNCCS_V2_Settings", JSON.stringify(state.gen_settings));
                    const w = node.widgets.find(x => x.name === "widget_data");
                    if (w) w.value = JSON.stringify(state);
                };
                const loadState = () => {
                    try {
                        const s = localStorage.getItem("VNCCS_V2_Settings");
                        if (s) {
                            const parsed = JSON.parse(s);
                            // Merge carefully to preserve structure
                            Object.assign(state.gen_settings, parsed);
                            // Ensure stack length
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

                    wrap.innerHTML = `<div class="vnccs-label">${lbl}</div>`;
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

                // 5. Build Layout
                const container = document.createElement("div");
                container.className = "vnccs-container";

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
                    saveState();
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

                btnRow.appendChild(btnGen);
                btnRow.appendChild(btnNew);
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

                colCenter.appendChild(createField("Gender", "sex", "select", ["female", "male"]));
                colCenter.appendChild(createField("Age", "age", "number"));
                colCenter.appendChild(createField("Race", "race"));
                colCenter.appendChild(createField("Skin Mode", "skin_color")); // Label changed to "Color"?
                colCenter.appendChild(createField("Body Type", "body"));
                colCenter.appendChild(createField("Face Features", "face"));
                colCenter.appendChild(createField("Hair Style", "hair"));
                colCenter.appendChild(createField("Eye Color", "eyes"));
                colCenter.appendChild(createField("Outfit / Details", "additional_details")); // Simplified to input for compactness, or keep input
                colCenter.appendChild(createField("NSFW Mode", "nsfw", "checkbox"));
                colCenter.appendChild(createField("Background Color", "background_color"));

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

                const rowGen = document.createElement("div"); rowGen.className = "vnccs-field";
                rowGen.style.flexDirection = "row"; rowGen.style.gap = "10px";
                rowGen.appendChild(createField("Steps", "steps", "number", {}, state.gen_settings));
                rowGen.appendChild(createField("CFG", "cfg", "number", { step: 0.1 }, state.gen_settings));
                colRight.appendChild(rowGen);

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

                const dmdStr = document.createElement("input"); dmdStr.className = "vnccs-input";
                dmdStr.type = "number"; dmdStr.step = "0.05"; dmdStr.style.flex = "1";
                dmdStr.onchange = (e) => {
                    state.gen_settings.dmd_lora_strength = parseFloat(e.target.value);
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

                        // Restore values
                        const g = state.gen_settings;
                        if (g.ckpt_name) els.ckptSelect.value = g.ckpt_name;
                        if (g.sampler) els.sampler.value = g.sampler;
                        if (g.scheduler) els.scheduler.value = g.scheduler;

                        if (g.steps && els.steps) els.steps.value = g.steps;
                        if (g.cfg && els.cfg) els.cfg.value = g.cfg;

                        // Restore LoRAs
                        if (g.dmd_lora_name) els.dmdSelect.value = g.dmd_lora_name;
                        if (g.dmd_lora_strength !== undefined) {
                            els.dmdSlider.value = g.dmd_lora_strength;
                        } else {
                            els.dmdSlider.value = 1.0;
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

                        if (state.character) els.charSelect.value = state.character;
                        else if (d.characters.length) { state.character = d.characters[0]; els.charSelect.value = state.character; }

                        if (state.character) await loadChar(state.character);

                    } catch (e) { console.error(e); }
                };

                const loadChar = async (n) => {
                    if (!n) return;
                    try {
                        // 1. Fetch Info
                        const r = await api.fetchApi(`/vnccs/character_info?character=${encodeURIComponent(n)}`);
                        const i = await r.json();
                        Object.assign(state.character_info, i);

                        // Update Fields
                        Object.keys(state.character_info).forEach(k => {
                            if (els[k]) {
                                if (els[k].type === "checkbox") els[k].checked = !!state.character_info[k];
                                else els[k].value = state.character_info[k];
                            }
                        });

                        // 2. Fetch Preview Image (Existing endpoint logic or direct file?)
                        // Currently backend returns image in JSON for 'preview_generate'?
                        // No, for loading existing character, user might want to see the stored sheet or a generated preview.
                        // Emotion Studio has `/vnccs/get_character_sheet_preview`.
                        // Let's use that if available.
                        els.previewImg.src = `/vnccs/get_character_sheet_preview?character=${encodeURIComponent(n)}&t=${Date.now()}`;
                        els.previewImg.style.display = "block";
                        els.placeholder.style.display = "none";
                        els.previewImg.onerror = () => {
                            els.previewImg.style.display = "none";
                            els.placeholder.innerText = "No Preview Image";
                            els.placeholder.style.display = "block";
                        }

                    } catch (e) { console.error(e); }
                };

                const doGenerate = async () => {
                    if (!state.gen_settings.ckpt_name) { alert("Select Checkpoint"); return; }
                    els.btnGen.innerText = "BUSY...";
                    els.btnGen.disabled = true;
                    saveState();

                    try {
                        const payload = {
                            ...state.gen_settings,
                            character_info: state.character_info
                        };
                        // Clean stack
                        payload.lora_stack = payload.lora_stack.filter(x => x.name && x.name !== "None");

                        const r = await fetch("/vnccs/preview_generate", { method: "POST", body: JSON.stringify(payload) });
                        const d = await r.json();
                        if (d.image) {
                            els.previewImg.src = "data:image/png;base64," + d.image;
                            els.previewImg.style.display = "block";
                            els.placeholder.style.display = "none";
                        }
                    } catch (e) { alert("Error: " + e); }
                    finally {
                        els.btnGen.innerText = "GENERATE PREVIEW";
                        els.btnGen.disabled = false;
                    }
                };

                const doCreate = () => {
                    const overlay = document.createElement("div"); overlay.className = "vnccs-modal-overlay";
                    const m = document.createElement("div"); m.className = "vnccs-modal";
                    m.innerHTML = `<div class="vnccs-section-title">New Character</div>`;
                    const inp = document.createElement("input"); inp.className = "vnccs-input"; inp.placeholder = "Name...";

                    const row = document.createElement("div"); row.className = "vnccs-btn-row";
                    const bC = document.createElement("button"); bC.className = "vnccs-btn"; bC.innerText = "Cancel";
                    bC.onclick = () => overlay.remove();
                    const bO = document.createElement("button"); bO.className = "vnccs-btn vnccs-btn-primary"; bO.innerText = "Create";
                    bO.onclick = async () => {
                        const n = inp.value.trim();
                        if (!n) return;
                        await fetch(`/vnccs/create?name=${encodeURIComponent(n)}`);
                        overlay.remove();
                        await init();
                    };
                    row.appendChild(bC); row.appendChild(bO);
                    m.appendChild(inp); m.appendChild(row);
                    overlay.appendChild(m);
                    container.appendChild(overlay);
                    inp.focus();
                };

                setTimeout(init, 50);
            };
        }
    }
});
