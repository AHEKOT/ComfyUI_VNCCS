import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// --- STYLES: Monolithic Centered Layout ---
const STYLE = `
/* Main Host: Centers the Monolithic UI */
.vnccs-container {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 13px;
    width: 100%;
    height: 100%;
    overflow: auto; /* Scroll if node is too small */
    padding: 0;
}

/* The Grid Wrapper: Keeps columns glued together and holds bottom row */
.vnccs-layout-grid {
    display: flex;
    flex-direction: column; /* Vertical stack */
    gap: 20px;
    height: fit-content;
    max-height: 100%;
    padding: 20px;
    box-sizing: border-box;
    width: fit-content; 
    margin: auto;
}

/* TOP ROW */
.vnccs-main-row {
    display: flex;
    flex-direction: row;
    gap: 20px;
    height: fit-content;
}

/* BOTTOM ROW */
.vnccs-bottom-row {
    display: flex;
    flex-direction: row;
    gap: 20px;
    width: 100%;
}
.vnccs-bottom-row .vnccs-section {
    flex: 1; /* Expand textareas equally */
    min-width: 0;
}

/* Side Columns */
.vnccs-col {
    display: flex;
    flex-direction: column;
    gap: 15px;
    /* overflow-y: auto; -- removed to prevent double scrollbars in monolithic layout */
    height: fit-content;
    width: 320px; 
    flex-shrink: 0;
}

/* Scrollbar */
.vnccs-col::-webkit-scrollbar { width: 6px; }
.vnccs-col::-webkit-scrollbar-thumb { background: #444; border-radius: 3px; }

/* Center Column */
.vnccs-center {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start; /* Start from top in the stack */
    gap: 15px;
    height: fit-content;
    width: 330px; 
    flex-shrink: 0;
}

/* Preview Frame: FIXED SIZE */
.vnccs-preview-frame {
    width: 320px;
    height: 768px;
    background: #000;
    border: 2px solid #333;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    position: relative;
    flex-shrink: 0;
}

.vnccs-preview-img { 
    width: 100%; 
    height: 100%; 
    object-fit: cover; 
    border-radius: 4px; 
}
.vnccs-placeholder { color: #555; text-align: center; }

/* Fields */
.vnccs-section {
    background: #252525;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.vnccs-header {
    font-size: 11px; font-weight: bold; color: #fff;
    border-bottom: 2px solid #333; padding-bottom: 5px; margin-bottom: 5px;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.vnccs-field { display: flex; flex-direction: column; gap: 4px; }
.vnccs-label { color: #aaa; font-size: 11px; font-weight: 600; }
.vnccs-input, .vnccs-select, .vnccs-textarea {
    background: #151515; border: 1px solid #444; color: #fff;
    border-radius: 4px; padding: 6px; font-family: inherit; font-size: 12px;
}
.vnccs-input:focus { border-color: #5b96f5; outline: none; }
.vnccs-textarea { min-height: 80px; resize: vertical; width: 100%; box-sizing: border-box; }

/* Buttons */
.vnccs-footer { display: flex; gap: 10px; width: 100%; max-width: 320px; }
.vnccs-btn {
    flex: 1; padding: 12px; border: none; border-radius: 4px;
    cursor: pointer; font-weight: bold; text-transform: uppercase;
    font-size: 12px; color: white;
}
.vnccs-btn-primary { background: #3558c7; } .vnccs-btn-primary:hover { background: #4264d9; }
.vnccs-btn-success { background: #2e7d32; } .vnccs-btn-success:hover { background: #388e3c; }
.vnccs-btn-disabled { background: #333; color: #666; cursor: default; }

.vnccs-checkbox-row { flex-direction: row; align-items: center; gap: 8px; }
.vnccs-row { display: flex; gap: 8px; }

/* MODAL */
.vnccs-modal-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center;
    z-index: 1000; border-radius: 6px;
}
.vnccs-modal {
    background: #252525; border: 1px solid #444; padding: 20px; border-radius: 8px;
    width: 300px; display: flex; flex-direction: column; gap: 15px; box-shadow: 0 10px 25px rgba(0,0,0,0.5);
}
.vnccs-modal-title { font-weight: bold; color: #fff; text-transform: uppercase; font-size: 12px; }
.vnccs-modal-footer { display: flex; gap: 10px; justify-content: flex-end; }
.vnccs-btn-sm { padding: 8px 12px; font-size: 11px; }
`;

app.registerExtension({
    name: "VNCCS.CharacterCreatorV2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "CharacterCreatorV2") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const node = this;
                node.setSize([1200, 1100]); // Increased to fit the bottom row

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
                            w.computeSize = () => [0, -4];
                        }
                    }
                };
                cleanup();
                const origDraw = node.onDrawBackground;
                node.onDrawBackground = function (ctx) {
                    cleanup();
                    if (origDraw) origDraw.apply(this, arguments);
                };

                // Add onResize to help the DOM widget follow node height
                this.onResize = function (size) {
                    if (this.widgets && this.widgets[0]) {
                        this.widgets[0].computedHeight = size[1];
                    }
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
                        lora_name: "", lora_strength: 1.0
                    }
                };

                // UI Referencing
                const els = {};
                const saveState = () => {
                    localStorage.setItem("VNCCS_V2_Settings", JSON.stringify(state.gen_settings));
                    const w = node.widgets.find(x => x.name === "widget_data");
                    if (w) w.value = JSON.stringify(state); // Sync everything
                };
                const loadState = () => {
                    try {
                        const s = localStorage.getItem("VNCCS_V2_Settings");
                        if (s) state.gen_settings = { ...state.gen_settings, ...JSON.parse(s) };
                    } catch (e) { }
                };

                // 4. UI Builders
                const createField = (lbl, key, type = "text", opts = [], target = "character_info") => {
                    const wrap = document.createElement("div");
                    // FIXED: Access state[target][key] dynamically.

                    if (type === "checkbox") {
                        wrap.className = "vnccs-field vnccs-checkbox-row";
                        const inp = document.createElement("input");
                        inp.type = "checkbox";
                        inp.checked = !!state[target][key];
                        inp.onchange = (e) => {
                            state[target][key] = e.target.checked;
                            saveState();
                        };
                        const l = document.createElement("label"); l.className = "vnccs-label"; l.innerText = lbl;
                        wrap.appendChild(inp); wrap.appendChild(l);
                        els[key] = inp;
                        return wrap;
                    }

                    wrap.className = "vnccs-field";
                    wrap.innerHTML = `<div class="vnccs-label">${lbl}</div>`;

                    let inp;
                    if (type === "select") {
                        inp = document.createElement("select"); inp.className = "vnccs-select";
                        opts.forEach(v => {
                            const o = document.createElement("option"); o.value = v; o.innerText = v; inp.appendChild(o);
                        });
                        inp.value = state[target][key] || opts[0];
                        inp.onchange = (e) => {
                            state[target][key] = e.target.value;
                            saveState();
                        };
                    } else if (type === "number") {
                        inp = document.createElement("input"); inp.className = "vnccs-input";
                        inp.type = "number"; if (opts.step) inp.step = opts.step;
                        inp.value = state[target][key];
                        inp.onchange = (e) => {
                            state[target][key] = parseFloat(e.target.value);
                            saveState();
                        };
                    } else if (type === "textarea") {
                        inp = document.createElement("textarea"); inp.className = "vnccs-textarea";
                        inp.value = state[target][key] || "";
                        inp.oninput = (e) => {
                            state[target][key] = e.target.value;
                            saveState();
                        };
                    } else {
                        inp = document.createElement("input"); inp.className = "vnccs-input";
                        inp.value = state[target][key] || "";
                        inp.oninput = (e) => {
                            state[target][key] = e.target.value;
                            saveState();
                        };
                    }
                    wrap.appendChild(inp);
                    els[key] = inp;
                    return wrap;
                };

                // 5. Layout
                const container = document.createElement("div");
                container.className = "vnccs-container";

                // MAIN GRID (TOP + BOTTOM)
                const layoutGrid = document.createElement("div");
                layoutGrid.className = "vnccs-layout-grid";

                // TOP BLOCK (3 Columns)
                const mainRow = document.createElement("div");
                mainRow.className = "vnccs-main-row";

                // LEFT (Params)
                const left = document.createElement("div"); left.className = "vnccs-col";
                const sChar = document.createElement("div"); sChar.className = "vnccs-section";
                sChar.innerHTML = '<div class="vnccs-header">Character</div>';
                const charSel = document.createElement("select"); charSel.className = "vnccs-select";
                charSel.onchange = async (e) => {
                    const v = e.target.value;
                    state.character = v;
                    await loadChar(v);
                    saveState();
                };
                els.charSelect = charSel;
                sChar.appendChild(charSel);
                left.appendChild(sChar);

                const sAttr = document.createElement("div"); sAttr.className = "vnccs-section";
                sAttr.innerHTML = '<div class="vnccs-header">Attributes</div>';
                sAttr.appendChild(createField("Gender", "sex", "select", ["female", "male"]));
                sAttr.appendChild(createField("Age", "age", "number"));
                sAttr.appendChild(createField("Race", "race"));
                sAttr.appendChild(createField("Skin Color", "skin_color"));
                sAttr.appendChild(createField("Body Type", "body"));
                sAttr.appendChild(createField("Face Features", "face"));
                sAttr.appendChild(createField("Hair Style", "hair"));
                sAttr.appendChild(createField("Eye Color", "eyes"));
                sAttr.appendChild(createField("Outfit / Details", "additional_details", "textarea"));
                sAttr.appendChild(createField("NSFW Mode", "nsfw", "checkbox"));
                sAttr.appendChild(createField("Background Color", "background_color"));
                left.appendChild(sAttr);

                mainRow.appendChild(left);

                // CENTER (Preview)
                const center = document.createElement("div"); center.className = "vnccs-center";

                const frame = document.createElement("div"); frame.className = "vnccs-preview-frame";
                frame.innerHTML = '<div class="vnccs-placeholder">No Preview<br>(320x768)</div>';
                const img = document.createElement("img");
                img.className = "vnccs-preview-img"; img.style.display = "none";
                frame.appendChild(img);
                els.previewImg = img; els.placeholder = frame.querySelector(".vnccs-placeholder");

                const footer = document.createElement("div"); footer.className = "vnccs-footer";
                const btnGen = document.createElement("button");
                btnGen.className = "vnccs-btn vnccs-btn-primary";
                btnGen.innerText = "GENERATE PREVIEW";
                btnGen.onclick = () => doGenerate();
                els.btnGen = btnGen;

                const btnNew = document.createElement("button");
                btnNew.className = "vnccs-btn vnccs-btn-success";
                btnNew.innerText = "CREATE NEW";
                btnNew.onclick = () => doCreate();

                footer.appendChild(btnGen); footer.appendChild(btnNew);

                center.appendChild(frame);
                center.appendChild(footer);
                mainRow.appendChild(center);

                // RIGHT (Settings)
                const right = document.createElement("div"); right.className = "vnccs-col";
                const sGen = document.createElement("div"); sGen.className = "vnccs-section";
                sGen.innerHTML = '<div class="vnccs-header">Generation</div>';

                const dC = document.createElement("div"); dC.className = "vnccs-field";
                dC.innerHTML = '<div class="vnccs-label">Checkpoint (SDXL)</div>';
                const sC = document.createElement("select"); sC.className = "vnccs-select";
                sC.onchange = (e) => { state.gen_settings.ckpt_name = e.target.value; saveState(); };
                els.ckptSelect = sC; dC.appendChild(sC); sGen.appendChild(dC);

                sGen.appendChild(createField("Sampler", "sampler", "select", [], "gen_settings"));
                els.sampler = sGen.lastChild.querySelector("select");

                sGen.appendChild(createField("Scheduler", "scheduler", "select", [], "gen_settings"));
                els.scheduler = sGen.lastChild.querySelector("select");

                const row = document.createElement("div"); row.className = "vnccs-row";
                row.appendChild(createField("Steps", "steps", "number", {}, "gen_settings"));
                row.appendChild(createField("CFG", "cfg", "number", { step: 0.1 }, "gen_settings"));
                // Fix layout of row
                row.querySelector(".vnccs-field:first-child").style.flex = "1";
                row.querySelector(".vnccs-field:last-child").style.flex = "1";
                sGen.appendChild(row);

                // DMD2 LoRA moved here (used to be in sAdv)
                const dL = document.createElement("div"); dL.className = "vnccs-field";
                dL.innerHTML = '<div class="vnccs-label">DMD2 LoRA Model</div>';
                const rL = document.createElement("div"); rL.className = "vnccs-row";
                const sL = document.createElement("select"); sL.className = "vnccs-select"; sL.style.flex = "2";
                sL.onchange = (e) => { state.gen_settings.lora_name = e.target.value; saveState(); };
                els.loraSelect = sL;
                const iL = document.createElement("input"); iL.className = "vnccs-input"; iL.type = "number"; iL.step = "0.1"; iL.style.flex = "1";
                iL.value = 1.0; iL.onchange = (e) => { state.gen_settings.lora_strength = parseFloat(e.target.value); saveState(); };
                els.loraStrength = iL;
                rL.appendChild(sL); rL.appendChild(iL); dL.appendChild(rL); sGen.appendChild(dL);

                right.appendChild(sGen);
                mainRow.appendChild(right);

                layoutGrid.appendChild(mainRow);

                // BOTTOM BLOCK (Prompt Areas)
                const bottomRow = document.createElement("div");
                bottomRow.className = "vnccs-bottom-row";

                const sAes = document.createElement("div"); sAes.className = "vnccs-section";
                sAes.appendChild(createField("Aesthetics", "aesthetics", "textarea"));
                bottomRow.appendChild(sAes);

                const sNeg = document.createElement("div"); sNeg.className = "vnccs-section";
                sNeg.appendChild(createField("Negative Prompt", "negative_prompt", "textarea"));
                bottomRow.appendChild(sNeg);

                const sLora = document.createElement("div"); sLora.className = "vnccs-section";
                sLora.appendChild(createField("LoRA Trigger (Prompt)", "lora_prompt", "textarea"));
                bottomRow.appendChild(sLora);

                layoutGrid.appendChild(bottomRow);

                container.appendChild(layoutGrid);
                this.addDOMWidget("ui", "ui", container, { serialize: false });

                // 6. Logic
                const init = async () => {
                    loadState();
                    try {
                        const r = await api.fetchApi("/vnccs/context_lists");
                        const d = await r.json();

                        const pop = (el, i, none) => {
                            if (!el) return; el.innerHTML = "";
                            if (none) el.appendChild(new Option("None", ""));
                            (i || []).forEach(x => el.appendChild(new Option(x, x)));
                        };

                        pop(els.ckptSelect, d.checkpoints);
                        pop(els.sampler, d.samplers);
                        pop(els.scheduler, d.schedulers);
                        pop(els.loraSelect, d.loras, true);

                        els.charSelect.innerHTML = "";
                        if (!d.characters || !d.characters.length) els.charSelect.appendChild(new Option("None", ""));
                        else d.characters.forEach(c => els.charSelect.appendChild(new Option(c, c)));

                        // Restore Gen
                        const g = state.gen_settings;
                        if (g.ckpt_name) els.ckptSelect.value = g.ckpt_name;
                        if (g.sampler) els.sampler.value = g.sampler;
                        if (g.scheduler) els.scheduler.value = g.scheduler;
                        if (g.lora_name) els.loraSelect.value = g.lora_name;
                        if (g.lora_strength) els.loraStrength.value = g.lora_strength;
                        if (els.steps) els.steps.value = g.steps;
                        if (els.cfg) els.cfg.value = g.cfg;

                        if (state.character) els.charSelect.value = state.character;
                        else if (d.characters.length) { state.character = d.characters[0]; els.charSelect.value = state.character; }

                        if (state.character) await loadChar(state.character);

                    } catch (e) { console.error(e); }
                };

                const loadChar = async (n) => {
                    if (!n) return;
                    try {
                        const r = await api.fetchApi(`/vnccs/character_info?character=${encodeURIComponent(n)}`);
                        const i = await r.json();
                        state.character_info = { ...state.character_info, ...i };
                        // Update Fields
                        Object.keys(i).forEach(k => {
                            if (els[k]) {
                                if (els[k].type === "checkbox") els[k].checked = !!i[k];
                                else els[k].value = i[k];
                            }
                        });
                        // Explicit Extra
                        if (els.lora_prompt) els.lora_prompt.value = i.lora_prompt || "";

                    } catch (e) { }
                };

                const doCreate = () => {
                    const overlay = document.createElement("div");
                    overlay.className = "vnccs-modal-overlay";
                    const m = document.createElement("div");
                    m.className = "vnccs-modal";
                    m.innerHTML = `<div class="vnccs-modal-title">New Character Name:</div>`;
                    const inp = document.createElement("input");
                    inp.className = "vnccs-input";
                    inp.placeholder = "Enter name...";
                    m.appendChild(inp);

                    const foot = document.createElement("div");
                    foot.className = "vnccs-modal-footer";

                    const btnCan = document.createElement("button");
                    btnCan.className = "vnccs-btn vnccs-btn-sm";
                    btnCan.style.background = "#444";
                    btnCan.innerText = "CANCEL";
                    btnCan.onclick = () => overlay.remove();

                    const btnOk = document.createElement("button");
                    btnOk.className = "vnccs-btn vnccs-btn-primary vnccs-btn-sm";
                    btnOk.innerText = "CREATE";
                    btnOk.onclick = async () => {
                        const n = inp.value.trim();
                        if (!n) return;
                        await fetch("/vnccs/create", { method: "POST", body: JSON.stringify({ name: n }) });
                        overlay.remove();
                        await init();
                    };

                    foot.appendChild(btnCan);
                    foot.appendChild(btnOk);
                    m.appendChild(foot);
                    overlay.appendChild(m);
                    container.appendChild(overlay);
                    inp.focus();
                };

                const doGenerate = async () => {
                    if (!state.gen_settings.ckpt_name) { alert("Select Checkpoint"); return; }
                    els.btnGen.innerText = "GENERATING...";
                    els.btnGen.disabled = true;
                    saveState();

                    try {
                        // Send RAW INFO, Backend does the prompt logic now!
                        const payload = {
                            ...state.gen_settings,
                            character_info: state.character_info
                        };
                        console.log("[VNCCS V2] Sending Payload:", payload); // DEBUG
                        const r = await fetch("/vnccs/preview_generate", { method: "POST", body: JSON.stringify(payload) });
                        const d = await r.json();
                        if (d.image) {
                            els.previewImg.src = "data:image/png;base64," + d.image;
                            els.previewImg.style.display = "block";
                            els.placeholder.style.display = "none";
                        }
                    } catch (e) { alert(e); }
                    finally {
                        els.btnGen.innerText = "GENERATE PREVIEW";
                        els.btnGen.disabled = false;
                    }
                };

                setTimeout(init, 50);
            };
        }
    }
});
