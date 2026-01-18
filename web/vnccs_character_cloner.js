import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// --- STYLES: 2-Column Grid Layout (Source / Attributes) ---
const STYLE = `
/* Main Host */
.vnccs-cloner-container {
    display: flex;
    flex-direction: column;
    background: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Consolas', 'Monaco', monospace;
    font-size: 16px;
    width: 100%;
    height: 100%;
    overflow: hidden;
    box-sizing: border-box;
    padding: 10px;
    gap: 10px;
    zoom: 0.67; 
}

/* TOP ROW: 2 Columns now */
.vnccs-top-row {
    display: grid;
    grid-template-columns: 40% 60%; 
    gap: 10px;
    flex: 1;
    min-height: 0;
    width: 100%;
}

/* BOTTOM ROW: Prompts - aligned with Top Row */
.vnccs-bottom-row {
    display: grid;
    grid-template-columns: 40% 60%;
    gap: 10px;
    height: 75px; 
    min-height: 75px;
    width: 100%;
    flex-shrink: 0;
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
    overflow-y: auto;
    height: 100%;
    box-sizing: border-box;
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
    display: flex; 
    justify-content: space-between;
    align-items: center;
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
    zoom: 1.5;
    padding: 4px;
}
.vnccs-input:focus, .vnccs-select:focus, .vnccs-textarea:focus { border-color: #5b96f5; outline: none; }

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
.vnccs-btn-upload { background: #555; border: 1px dashed #777; } .vnccs-btn-upload:hover { background: #666; border-color: #999; }

/* Image List */
.vnccs-img-list {
    display: flex; flex-wrap: wrap; gap: 5px;
}
.vnccs-thumb {
    width: 80px; height: 80px; object-fit: cover;
    border: 1px solid #444; border-radius: 4px;
    cursor: pointer;
}
.vnccs-thumb:hover { border-color: #fff; }
.vnccs-thumb.generating {
    border: 2px solid #3558c7;
    animation: pulse 1s infinite alternate;
}

@keyframes pulse { from { opacity: 0.6; } to { opacity: 1; } }

/* Tag Styles */
.vnccs-tag-btn {
    width: 20px; height: 20px;
    background: #333; color: #fff; border: 1px solid #555;
    border-radius: 4px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; margin-left: auto;
    flex-shrink: 0;
}

/* Modal */
.vnccs-modal-overlay {
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.7); z-index: 100;
    display: flex; align-items: center; justify-content: center;
}
.vnccs-modal {
    background: #252525; border: 1px solid #555; border-radius: 6px;
    padding: 15px; width: 300px; display: flex; flex-direction: column; gap: 10px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
}

.vnccs-btn-row {
    display: flex;
    gap: 5px;
    margin-top: 5px;
    width: 100%;
}
.vnccs-btn {
    flex: 1;
    padding: 6px;
    background: #444; color: white;
    border: 1px solid #666; border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
}
.vnccs-btn:hover { background: #555; }
.vnccs-btn-primary { background: #007bff; border-color: #0056b3; }
.vnccs-btn-primary:hover { background: #0056b3; }
.vnccs-btn-success { background: #28a745; border-color: #1e7e34; }
.vnccs-btn-success:hover { background: #218838; }
.vnccs-btn-danger { background: #dc3545; border-color: #bd2130; }
.vnccs-btn-danger:hover { background: #c82333; }

.vnccs-thumb {
    width: 80px; height: 80px; object-fit: cover;
    border: 1px solid #555; border-radius: 4px;
    cursor: pointer;
}
.vnccs-thumb:hover { opacity: 0.8; border-color: #f00; }
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
                const style = document.createElement("style");
                style.innerHTML = STYLE;
                document.head.appendChild(style);

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
                        lora_prompt: ""
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
                    wrap.className = "vnccs-field";

                    if (type === "checkbox") {
                        wrap.style.flexDirection = "row";
                        wrap.style.alignItems = "center";
                        wrap.style.gap = "8px";
                        const inp = document.createElement("input");
                        inp.type = "checkbox";
                        inp.checked = !!targetObj[key];
                        inp.onchange = (e) => { targetObj[key] = e.target.checked; saveState(); };
                        wrap.appendChild(inp);
                        wrap.appendChild(document.createTextNode(lbl));
                        els[key] = inp;
                        return wrap;
                    }

                    const header = document.createElement("div");
                    header.style.display = "flex"; header.style.justifyContent = "space-between";
                    header.innerHTML = `<div class="vnccs-label">${lbl}</div>`;
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
                        inp.value = targetObj[key];
                        inp.onchange = (e) => { targetObj[key] = parseFloat(e.target.value); saveState(); };
                    } else {
                        inp = document.createElement("input"); inp.className = "vnccs-input";
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
                            if (els[k].type === "checkbox") els[k].checked = state.character_info[k];
                            else els[k].value = state.character_info[k];
                        }
                    }
                    // Images
                    if (renderThumbs) renderThumbs();
                };

                // --- Helpers (Hoisted) ---
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

                const loadChar = async (name) => {
                    if (!name || name === "None") {
                        state.char_preview_url = null;
                        updateUIFromState();
                        return;
                    }
                    try {
                        const r = await api.fetchApi(`/vnccs/config?name=${encodeURIComponent(name)}`);
                        if (r.ok) {
                            const d = await r.json();
                            if (d.character_info) {
                                Object.assign(state.character_info, d.character_info);
                                updateUIFromState();
                            }
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
                        if (state.character) await loadChar(state.character);

                    } catch (e) { console.error(e); }
                };

                const doCreate = () => {
                    let inpRef;
                    showModal("New Character", () => {
                        const inp = document.createElement("input");
                        inp.className = "vnccs-input";
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
                topRow.className = "vnccs-top-row";

                // COL 1: SOURCE IMAGES
                // COL 1: SOURCE IMAGES (Left Column)
                const colSrc = document.createElement("div");
                colSrc.className = "vnccs-col";

                // --- CHARACTER SELECTOR (Moved to Left Top) ---
                colSrc.innerHTML = '<div class="vnccs-section-title">Character Select</div>';

                const charRow = document.createElement("div");
                charRow.className = "vnccs-field";

                const charSel = document.createElement("select");
                charSel.className = "vnccs-select";
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
                btnRow.className = "vnccs-btn-row";

                const btnNew = document.createElement("button");
                btnNew.className = "vnccs-btn vnccs-btn-success";
                btnNew.innerText = "NEW";
                btnNew.onclick = doCreate;

                const btnDel = document.createElement("button");
                btnDel.className = "vnccs-btn vnccs-btn-danger";
                btnDel.innerText = "DEL";
                btnDel.onclick = doDelete;

                btnRow.appendChild(btnNew);
                btnRow.appendChild(btnDel);
                colSrc.appendChild(btnRow);

                // --- SOURCE IMAGES SECTION ---
                const srcHeader = document.createElement("div");
                srcHeader.className = "vnccs-section-title";
                srcHeader.innerText = "Source Images";
                srcHeader.style.marginTop = "15px";
                colSrc.appendChild(srcHeader);

                const imgList = document.createElement("div");
                imgList.className = "vnccs-img-list";
                colSrc.appendChild(imgList);

                // --- IMAGE PREVIEW/UPLOAD AREA ---
                // Container for the Large Preview
                const previewContainer = document.createElement("div");
                previewContainer.className = "vnccs-preview-container";
                previewContainer.style.flex = "1";
                previewContainer.style.position = "relative";
                previewContainer.style.border = "1px solid #444";
                previewContainer.style.borderRadius = "4px";
                previewContainer.style.background = "#151515";
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

                // Overlay Controls (Upload Button) - Always visible or overlay?
                // User wants standard behaviour: Empty -> Upload Button. Filled -> Image + Mini Overlay?
                // Let's make the click on image trigger upload if empty?
                // Or keep the upload button below?
                // User said "Huge square with +upload images IS the place for preview"

                const uploadOverlay = document.createElement("div");
                uploadOverlay.className = "vnccs-upload-overlay";
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
                uploadBtn.className = "vnccs-btn vnccs-btn-upload";
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
                colAttr.className = "vnccs-col";

                const attrHeader = document.createElement("div");
                attrHeader.className = "vnccs-section-title";
                attrHeader.innerText = "Attributes";

                const autoGenBtn = document.createElement("button");
                autoGenBtn.className = "vnccs-btn vnccs-btn-primary";
                autoGenBtn.style.padding = "10px 12px";
                autoGenBtn.style.fontSize = "12px";
                autoGenBtn.style.flex = "0 0 auto"; // Prevent stretching
                autoGenBtn.innerText = "AUTO GENERATE (Qwen2-VL-7B)";
                autoGenBtn.onclick = async () => {
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

                    autoGenBtn.innerText = "ANALYZING...";
                    autoGenBtn.disabled = true;

                    // Visual feedback highlighting selected thumb
                    const thumbs = imgList.querySelectorAll("img");
                    if (thumbs[selIdx]) thumbs[selIdx].classList.add("generating");

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
                        autoGenBtn.innerText = "AUTO GENERATE (QWEN)";
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
                botRow.className = "vnccs-bottom-row";

                // Read-only Prompts? Or generated?
                // Standard textareas
                const createTA = (lbl, key) => {
                    const w = document.createElement("div");
                    w.className = "vnccs-col";
                    w.innerHTML = `<div class="vnccs-label">${lbl}</div>`;
                    const t = document.createElement("textarea");
                    t.className = "vnccs-textarea";
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
                            uploadOverlay.style.opacity = "0";
                            previewContainer.onmouseenter = () => uploadOverlay.style.opacity = "1";
                            previewContainer.onmouseleave = () => uploadOverlay.style.opacity = "0";
                        } else {
                            previewImg.style.display = "none";
                            previewImg.src = "";
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
                        if (idx === state.selected_idx) {
                            wrap.style.borderColor = "#3558c7";
                            wrap.style.borderRadius = "4px";
                        }

                        const img = document.createElement("img");
                        const params = new URLSearchParams();
                        params.append("filename", name);
                        params.append("type", type);
                        if (sub) params.append("subfolder", sub);
                        img.src = api.apiURL("/view?" + params.toString());
                        img.className = "vnccs-thumb";
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
