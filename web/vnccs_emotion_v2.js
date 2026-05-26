import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { syncDOMWidgetWidth, syncDOMWidgetWidthSoon } from "./vnccs_common.js";

// --- CSS STYLES: Sakura Archive Design System ---
const STYLE = `
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Variables ── */
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
    --accent-glow: rgba(255, 143, 163, 0.3);
    --accent-subtle: rgba(255, 143, 163, 0.1);
    --accent-border: rgba(255, 143, 163, 0.22);
    --accent-lavender: #b8a9e8;
    --success: #00d68f;
    --error: #ff4757;
    --border: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(255, 255, 255, 0.12);
    --font: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --transition: 0.2s ease;
}

/* ── Container ── */
.ems-container {
    display: flex;
    flex-direction: row;
    gap: 10px;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 24px;
    padding: 10px;
    border-radius: 8px;
    width: 100%;
    height: 100%;
    max-height: 100%;
    box-sizing: border-box;
    overflow: hidden;
    background-image: radial-gradient(ellipse at 15% 0%, rgba(255, 143, 163, 0.05) 0%, transparent 55%),
                      radial-gradient(ellipse at 85% 100%, rgba(184, 169, 232, 0.04) 0%, transparent 55%);
}

/* ── Columns ── */
.ems-left-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 200px;
    overflow: hidden;
}

.ems-right-col {
    flex: 3;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 300px;
    overflow: hidden;
}

/* ── Sections ── */
.ems-section {
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-lg);
    background: rgba(20, 16, 30, 0.88);
    padding: 8px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
}
.ems-section::before {
    content: '';
    position: absolute;
    top: 0; left: 18%; right: 18%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 143, 163, 0.6), transparent);
    border-radius: 1px;
    pointer-events: none;
}

/* ── Character Header ── */
.ems-char-header {
    color: var(--accent);
    padding: 4px 8px 8px;
    font-weight: 700;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.ems-char-preview-container {
    flex: 1;
    background: radial-gradient(circle, rgba(255, 143, 163, 0.04) 1px, transparent 1px), rgba(10, 8, 16, 0.6);
    background-size: 18px 18px, 100% 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-md);
    border: 1px solid var(--border);
    overflow: hidden;
    position: relative;
}

.ems-char-preview {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* ── Costumes ── */
.ems-costumes-header {
    font-weight: 700;
    color: var(--accent);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    padding-left: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ems-costumes-header::before {
    content: '';
    width: 3px;
    height: 10px;
    background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
    border-radius: 2px;
    box-shadow: 0 0 6px var(--accent-glow);
    flex-shrink: 0;
}
.ems-costumes-list {
    background: rgba(10, 8, 16, 0.5);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    color: var(--text-primary);
    max-height: 120px;
    overflow-y: auto;
}
.ems-costumes-list::-webkit-scrollbar { height: 3px; width: 3px; }
.ems-costumes-list::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

/* ── Costume Toggle Items ── */
.ems-checkbox-item {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    user-select: none;
}
.ems-checkbox-item span {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.em-toggle {
    position: relative;
    width: 34px;
    height: 18px;
    flex-shrink: 0;
}
.em-toggle input { opacity: 0; width: 0; height: 0; position: absolute; }
.em-toggle-track {
    position: absolute;
    inset: 0;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all var(--transition);
}
.em-toggle input:checked + .em-toggle-track {
    background: rgba(255, 143, 163, 0.22);
    border-color: var(--accent-border);
    box-shadow: 0 0 8px var(--accent-subtle);
}
.em-toggle-thumb {
    position: absolute;
    top: 3px; left: 3px;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all var(--transition);
}
.em-toggle input:checked ~ .em-toggle-thumb {
    transform: translateX(16px);
    background: var(--accent);
    box-shadow: 0 0 6px var(--accent-glow);
}

/* ── Emotions Grid ── */
.ems-emotions-container {
    flex: 1;
    background: rgba(8, 6, 14, 0.5);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 10px;
    overflow-y: auto;
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    min-height: 200px;
    scrollbar-width: thin;
    scrollbar-color: var(--accent-border) transparent;
}
.ems-emotions-container::-webkit-scrollbar { width: 4px; }
.ems-emotions-container::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

.ems-emotion-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: pointer;
    transition: all 0.15s ease;
    width: 100%;
    padding: 6px;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    box-sizing: border-box;
    background: transparent;
}
.ems-emotion-item:hover {
    background: rgba(255, 143, 163, 0.06);
    border-color: var(--accent-border);
}
.ems-emotion-item.selected {
    background: rgba(255, 143, 163, 0.15);
    border-color: var(--accent);
    box-shadow: 0 0 12px var(--accent-subtle), inset 0 0 8px rgba(255, 143, 163, 0.05);
}
.ems-emotion-item.selected .ems-emotion-label {
    color: var(--accent-hover);
    font-weight: 600;
}

.ems-emotion-img {
    width: 100%;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    transition: border-color var(--transition);
}
.ems-emotion-item.selected .ems-emotion-img {
    border-color: var(--accent-border);
}
.ems-emotion-label {
    font-size: 20px;
    color: var(--text-secondary);
    text-align: center;
    margin-top: 4px;
    font-family: var(--font);
    font-weight: 500;
    word-break: break-word;
    width: 100%;
    line-height: 1.2;
}

/* ── Footer / Button ── */
.ems-footer {
    margin-top: 6px;
    display: flex;
    justify-content: center;
}
.ems-btn {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
    color: #1a1525;
    border: none;
    border-radius: var(--radius-md);
    padding: 10px 32px;
    font-family: var(--font);
    font-weight: 700;
    font-size: 22px;
    letter-spacing: 0.5px;
    cursor: pointer;
    box-shadow: 0 4px 20px rgba(255, 143, 163, 0.25);
    position: relative;
    overflow: hidden;
    transition: all var(--transition);
    text-transform: uppercase;
}
.ems-btn::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 45%, rgba(255,255,255,0.28) 50%, rgba(255,255,255,0.2) 55%, transparent 100%);
    transform: translateX(-120%) skewX(-15deg);
    animation: emBtnShimmer 3.5s ease-in-out infinite;
    pointer-events: none;
}
@keyframes emBtnShimmer {
    0%   { transform: translateX(-120%) skewX(-15deg); opacity: 1; }
    35%  { transform: translateX(120%)  skewX(-15deg); opacity: 1; }
    100% { transform: translateX(120%)  skewX(-15deg); opacity: 0; }
}
.ems-btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(255, 143, 163, 0.45);
}
.ems-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
.ems-btn.cancel {
    background: rgba(255, 71, 87, 0.15);
    color: var(--error);
    border: 1px solid rgba(255, 71, 87, 0.3);
    box-shadow: none;
}
.ems-btn.cancel::after { display: none; }
.ems-btn.cancel:hover:not(:disabled) {
    background: rgba(255, 71, 87, 0.28);
    box-shadow: 0 4px 16px rgba(255, 71, 87, 0.2);
}

/* ── Search Input ── */
.ems-search-input {
    width: 100%;
    padding: 8px 14px;
    margin-top: 6px;
    margin-bottom: 6px;
    background: rgba(255, 255, 255, 0.04);
    color: var(--text-primary);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: var(--radius-md);
    font-size: 14px;
    font-family: var(--font);
    box-sizing: border-box;
    transition: all var(--transition);
}
.ems-search-input:focus {
    outline: none;
    border-color: var(--accent-border);
    background: rgba(255, 143, 163, 0.03);
    box-shadow: 0 0 0 3px rgba(255, 143, 163, 0.05);
}
.ems-search-input::placeholder { color: var(--text-muted); }

/* ── Custom Select Style ── */
.em-select {
    width: 100%;
    padding: 6px 10px;
    background: rgba(255, 255, 255, 0.05);
    color: var(--text-primary);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-md);
    font-family: var(--font);
    font-size: 13px;
    font-weight: 500;
    transition: all var(--transition);
    cursor: pointer;
}
.em-select:focus {
    outline: none;
    border-color: var(--accent-border);
    box-shadow: 0 0 0 2px rgba(255, 143, 163, 0.08);
}

/* ── Confirm Modal ── */
.ems-modal-backdrop {
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    border-radius: 8px;
}
.ems-modal {
    background: rgba(22, 18, 34, 0.98);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-lg);
    padding: 24px 28px 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    min-width: 280px;
    max-width: 360px;
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 143, 163, 0.08);
    position: relative;
}
.ems-modal::before {
    content: '';
    position: absolute;
    top: 0; left: 18%; right: 18%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255, 143, 163, 0.6), transparent);
    border-radius: 1px;
}
.ems-modal-text {
    font-family: var(--font);
    font-size: 13px;
    color: var(--text-primary);
    line-height: 1.6;
}
.ems-modal-text strong {
    color: var(--accent);
    font-weight: 700;
}
.ems-modal-actions {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
}
.ems-modal-btn {
    padding: 8px 20px;
    border-radius: var(--radius-md);
    font-family: var(--font);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    cursor: pointer;
    border: none;
    transition: all var(--transition);
}
.ems-modal-btn--cancel {
    background: rgba(255, 255, 255, 0.06);
    color: var(--text-secondary);
    border: 1px solid var(--border);
}
.ems-modal-btn--cancel:hover {
    background: rgba(255, 255, 255, 0.1);
    color: var(--text-primary);
}
.ems-modal-btn--confirm {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%);
    color: #1a1525;
    box-shadow: 0 4px 16px rgba(255, 143, 163, 0.25);
}
.ems-modal-btn--confirm:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(255, 143, 163, 0.4);
}
`;

// Inject Styles
const styleEl = document.createElement("style");
styleEl.textContent = STYLE;
document.head.appendChild(styleEl);

// --- MAIN EXTENSION ---

app.registerExtension({
    name: "VNCCS.EmotionGeneratorV2",

    async setup() {
        const origQueuePrompt = app.queuePrompt.bind(app);
        app.queuePrompt = async function(...args) {
            const nodes = app.graph?._nodes?.filter(n => n.type === "EmotionGeneratorV2") || [];
            for (const node of nodes) {
                if (node._validateBeforeQueue && !node._validateBeforeQueue()) {
                    return; // block queue, modal already shown inside _validateBeforeQueue
                }
            }
            return origQueuePrompt(...args);
        };
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EmotionGeneratorV2") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const node = this;

                // Set default size
                node.setSize([920, 650]);
                syncDOMWidgetWidthSoon(node, "emotion_ui_v2");

                // Get Widgets
                const charWidget = node.widgets.find(w => w.name === "character");
                const costumesDataWidget = node.widgets.find(w => w.name === "costumes_data");
                const emotionsDataWidget = node.widgets.find(w => w.name === "emotions_data");

                // Hide data widgets
                if (costumesDataWidget) costumesDataWidget.hidden = true;
                if (emotionsDataWidget) emotionsDataWidget.hidden = true;

                // State
                let state = {
                    character: charWidget ? charWidget.value : "",
                    costumes: [],
                    selectedCostumes: new Set(),
                    emotions: [],
                    selectedEmotions: new Set(),
                    searchTerm: ""
                };

                // Create UI Container
                const container = document.createElement("div");
                container.className = "ems-container";

                // --- LEFT COL ---
                const leftCol = document.createElement("div");
                leftCol.className = "ems-left-col";

                // Character Header
                const charSection = document.createElement("div");
                charSection.className = "ems-section";
                charSection.style.flex = "1";

                const charHeader = document.createElement("div");
                charHeader.className = "ems-char-header";
                charHeader.innerText = "Character select";

                const previewContainer = document.createElement("div");
                previewContainer.className = "ems-char-preview-container";

                const charImg = document.createElement("img");
                charImg.className = "ems-char-preview";
                charImg.style.display = "none";
                previewContainer.appendChild(charImg);

                const charPlaceholder = document.createElement("div");
                charPlaceholder.style.cssText = "display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;color:rgba(255,143,163,0.35);pointer-events:none;";
                charPlaceholder.innerHTML = `
                    <svg width="52" height="52" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="7" r="4" stroke="#ff8fa3" stroke-width="1.5"/>
                        <path d="M4 20c0-4 3.582-7 8-7s8 3 8 7" stroke="#ff8fa3" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                    <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:1px;font-family:'Sora',sans-serif;">Character</div>
                `;
                previewContainer.appendChild(charPlaceholder);

                // Show/hide placeholder when image loads
                charImg.onload = () => { charImg.style.display = "block"; charPlaceholder.style.display = "none"; };
                charImg.onerror = () => { charImg.style.display = "none"; charPlaceholder.style.display = "flex"; };

                charSection.appendChild(charHeader);
                charSection.appendChild(previewContainer);
                leftCol.appendChild(charSection);

                container.appendChild(leftCol);

                // --- RIGHT COL ---
                const rightCol = document.createElement("div");
                rightCol.className = "ems-right-col";

                // Costumes
                const costumesSection = document.createElement("div");
                costumesSection.className = "ems-section";

                const costumesHeader = document.createElement("div");
                costumesHeader.className = "ems-costumes-header";
                costumesHeader.innerText = "Selected costumes";
                costumesSection.appendChild(costumesHeader);

                const costumesList = document.createElement("div");
                costumesList.className = "ems-costumes-list";
                costumesSection.appendChild(costumesList);

                rightCol.appendChild(costumesSection);

                // Emotions
                const emotionsSection = document.createElement("div");
                emotionsSection.className = "ems-section";
                emotionsSection.style.flex = "1";

                const emotionsGrid = document.createElement("div");
                emotionsGrid.className = "ems-emotions-container";
                emotionsSection.appendChild(emotionsGrid);

                // Search Input
                const searchInput = document.createElement("input");
                searchInput.className = "ems-search-input";
                searchInput.placeholder = "Search emotions (name or description)...";
                searchInput.oninput = (e) => {
                    state.searchTerm = e.target.value;
                    renderEmotions();
                    updateButtonState();
                };
                emotionsSection.appendChild(searchInput);

                // Footer (Select All)
                const footer = document.createElement("div");
                footer.className = "ems-footer";
                const btnAll = document.createElement("button");
                btnAll.className = "ems-btn";
                btnAll.innerText = "Select ALL";
                footer.appendChild(btnAll);
                emotionsSection.appendChild(footer);

                rightCol.appendChild(emotionsSection);
                container.appendChild(rightCol);

                // Custom Select in Header
                const charSelect = document.createElement("select");
                charSelect.className = "em-select";

                charHeader.appendChild(charSelect);

                if (charWidget) {
                    if (charWidget.options.values) {
                        charWidget.options.values.forEach(v => {
                            const opt = document.createElement("option");
                            opt.value = v;
                            opt.innerText = v;
                            if (v === charWidget.value) opt.selected = true;
                            charSelect.appendChild(opt);
                        });
                    }
                    charSelect.onchange = () => {
                        charWidget.value = charSelect.value;
                        state.character = charSelect.value;
                        fetchCharacterData(charSelect.value);
                        if (charWidget.callback) charWidget.callback(charSelect.value);
                    };
                    charWidget.hidden = true;
                }

                // Add Prompt Style Select (Top)
                const styleWidget = node.widgets.find(w => w.name === "prompt_style");

                const styleContainer = document.createElement("div");
                styleContainer.className = "ems-section";
                styleContainer.style.marginBottom = "10px";
                styleContainer.style.padding = "5px 10px";

                const styleSelect = document.createElement("select");
                styleSelect.className = "em-select";

                if (styleWidget && styleWidget.options.values) {
                    styleWidget.options.values.forEach(v => {
                        const opt = document.createElement("option");
                        opt.value = v;
                        opt.innerText = v;
                        if (v === styleWidget.value) opt.selected = true;
                        styleSelect.appendChild(opt);
                    });
                    // Sync
                    styleSelect.onchange = () => {
                        styleWidget.value = styleSelect.value;
                        if (styleWidget.callback) styleWidget.callback(styleSelect.value);
                    };
                    styleWidget.hidden = true;
                }
                styleContainer.appendChild(styleSelect);

                // Insert Style Container at the TOP of Left Col
                // Currently Left Col has charSection. We can prepend.
                leftCol.prepend(styleContainer);


                const widget = node.addDOMWidget("emotion_ui_v2", "ui", container, {
                    serialize: true,
                    hideOnZoom: false,
                    getValue() { return undefined; },
                    setValue(v) { }
                });
                syncDOMWidgetWidthSoon(node, "emotion_ui_v2");

                // Fix layout after tab switch: ComfyUI detaches/reattaches DOM widgets
                // without triggering onResize. Use ResizeObserver on the node's canvas
                // element to detect when the widget becomes visible again and reapply sizes.
                function applySize() {
                    const [w, h] = node.size;
                    container.style.width = (w - 20) + "px";
                    container.style.height = (h - 60) + "px";
                }

                const origDraw = node.onDrawBackground;
                node.onDrawBackground = function (ctx) {
                    applySize();
                    if (origDraw) origDraw.apply(this, arguments);
                };

                function showAlert(html) {
                    const backdrop = document.createElement("div");
                    backdrop.className = "ems-modal-backdrop";

                    const modal = document.createElement("div");
                    modal.className = "ems-modal";

                    const text = document.createElement("div");
                    text.className = "ems-modal-text";
                    text.innerHTML = html;

                    const actions = document.createElement("div");
                    actions.className = "ems-modal-actions";

                    const btnOk = document.createElement("button");
                    btnOk.className = "ems-modal-btn ems-modal-btn--confirm";
                    btnOk.innerText = "OK";
                    btnOk.onclick = () => backdrop.remove();

                    actions.appendChild(btnOk);
                    modal.appendChild(text);
                    modal.appendChild(actions);
                    backdrop.appendChild(modal);
                    container.appendChild(backdrop);
                    btnOk.focus();
                }

                node._validateBeforeQueue = function() {
                    if (state.selectedCostumes.size === 0 && state.selectedEmotions.size === 0) {
                        showAlert(`<strong>Nothing selected</strong><br>Please select at least one costume and one emotion before running.`);
                        return false;
                    }
                    if (state.selectedCostumes.size === 0) {
                        showAlert(`<strong>No costumes selected</strong><br>Please enable at least one costume.`);
                        return false;
                    }
                    if (state.selectedEmotions.size === 0) {
                        showAlert(`<strong>No emotions selected</strong><br>Please select at least one emotion.`);
                        return false;
                    }
                    return true;
                };

                function showConfirm(html, onConfirm) {
                    const backdrop = document.createElement("div");
                    backdrop.className = "ems-modal-backdrop";

                    const modal = document.createElement("div");
                    modal.className = "ems-modal";

                    const text = document.createElement("div");
                    text.className = "ems-modal-text";
                    text.innerHTML = html;

                    const actions = document.createElement("div");
                    actions.className = "ems-modal-actions";

                    const btnCancel = document.createElement("button");
                    btnCancel.className = "ems-modal-btn ems-modal-btn--cancel";
                    btnCancel.innerText = "Cancel";
                    btnCancel.onclick = () => backdrop.remove();

                    const btnOk = document.createElement("button");
                    btnOk.className = "ems-modal-btn ems-modal-btn--confirm";
                    btnOk.innerText = "Proceed";
                    btnOk.onclick = () => { backdrop.remove(); onConfirm(); };

                    actions.appendChild(btnCancel);
                    actions.appendChild(btnOk);
                    modal.appendChild(text);
                    modal.appendChild(actions);
                    backdrop.appendChild(modal);
                    container.appendChild(backdrop);
                }

                function restoreStateFromWidgets() {
                    // 1. Character
                    if (charWidget && charWidget.value) {
                        state.character = charWidget.value;
                        charSelect.value = charWidget.value;
                        fetchCharacterData(state.character);
                    }

                    // 2. Style
                    if (styleWidget && styleWidget.value) {
                        styleSelect.value = styleWidget.value;
                    }

                    // 3. Costumes & Emotions (from hidden text strings)
                    if (costumesDataWidget && costumesDataWidget.value) {
                        try {
                            const savedCostumes = JSON.parse(costumesDataWidget.value);
                            state.selectedCostumes = new Set(savedCostumes);
                            renderCostumes();
                        } catch (e) { }
                    }
                    if (emotionsDataWidget && emotionsDataWidget.value) {
                        try {
                            const savedEmotions = JSON.parse(emotionsDataWidget.value);
                            state.selectedEmotions = new Set(savedEmotions);
                            // renderEmotions is called after fetch("/vnccs/get_emotions"), need to wait?
                            // No, renderEmotions() just needs state.emotions to be populated.
                            // The fetch happens async.
                        } catch (e) { }
                    }
                }

                // Apply size on explicit resize too
                node.onResize = function (size) {
                    syncDOMWidgetWidth(node, "emotion_ui_v2");
                    requestAnimationFrame(() => syncDOMWidgetWidth(node, "emotion_ui_v2"));
                    applySize();
                }

                // Helper
                function getFilteredEmotions() {
                    if (!state.searchTerm) return state.emotions;
                    const term = state.searchTerm.toLowerCase();
                    return state.emotions.filter(e => {
                        const nameMatch = e.safe_name.toLowerCase().includes(term);
                        const descMatch = (e.description || "").toLowerCase().includes(term);
                        return nameMatch || descMatch;
                    });
                }

                // Button Logic & Text Update
                function updateButtonState() {
                    const filtered = getFilteredEmotions();
                    if (filtered.length === 0) {
                        btnAll.innerText = "No Emotions Found";
                        btnAll.disabled = true;
                        btnAll.classList.remove("cancel");
                        return;
                    }
                    btnAll.disabled = false;

                    const allFilteredSelected = filtered.every(e => state.selectedEmotions.has(e.safe_name));

                    if (allFilteredSelected) {
                        btnAll.innerText = "Cancel Selection";
                        btnAll.classList.add("cancel");
                    } else {
                        btnAll.innerText = "Select ALL";
                        btnAll.classList.remove("cancel");
                    }
                }

                btnAll.onclick = () => {
                    const filtered = getFilteredEmotions();
                    if (filtered.length === 0) return;

                    const allFilteredSelected = filtered.every(e => state.selectedEmotions.has(e.safe_name));

                    if (allFilteredSelected) {
                        // Deselect visible
                        filtered.forEach(e => state.selectedEmotions.delete(e.safe_name));
                        renderEmotions();
                        updateEmotionsData();
                    } else {
                        // Select All Visible
                        const numEmotions = filtered.length;
                        const numCostumes = state.selectedCostumes.size;
                        const total = numEmotions * numCostumes;

                        const html = `Select <strong>${numEmotions}</strong> visible emotion(s) for <strong>${numCostumes}</strong> costume(s)?<br>Total: <strong>${total}</strong> images.`;
                        showConfirm(html, () => {
                            filtered.forEach(e => state.selectedEmotions.add(e.safe_name));
                            renderEmotions();
                            updateEmotionsData();
                        });
                    }
                };

                // Functions
                function updateCostumesData() {
                    const list = Array.from(state.selectedCostumes);
                    if (costumesDataWidget) costumesDataWidget.value = JSON.stringify(list);
                }

                function updateEmotionsData() {
                    const list = Array.from(state.selectedEmotions);
                    if (emotionsDataWidget) emotionsDataWidget.value = JSON.stringify(list);
                    updateButtonState();
                }

                function renderCostumes() {
                    costumesList.innerHTML = "";
                    state.costumes.forEach(c => {
                        const lbl = document.createElement("label");
                        lbl.className = "ems-checkbox-item";

                        const toggle = document.createElement("div");
                        toggle.className = "em-toggle";

                        const chk = document.createElement("input");
                        chk.type = "checkbox";
                        chk.checked = state.selectedCostumes.has(c);
                        chk.onchange = () => {
                            if (chk.checked) state.selectedCostumes.add(c);
                            else state.selectedCostumes.delete(c);
                            updateCostumesData();
                        };

                        const track = document.createElement("div");
                        track.className = "em-toggle-track";
                        const thumb = document.createElement("div");
                        thumb.className = "em-toggle-thumb";

                        toggle.appendChild(chk);
                        toggle.appendChild(track);
                        toggle.appendChild(thumb);

                        const span = document.createElement("span");
                        span.innerText = c;

                        lbl.appendChild(toggle);
                        lbl.appendChild(span);
                        costumesList.appendChild(lbl);
                    });
                }

                function renderEmotions() {
                    emotionsGrid.innerHTML = "";
                    const filtered = getFilteredEmotions();
                    filtered.forEach(e => {
                        const div = document.createElement("div");
                        const selected = state.selectedEmotions.has(e.safe_name);
                        div.className = "ems-emotion-item" + (selected ? " selected" : "");
                        div.title = e.description || "";

                        const img = document.createElement("img");
                        img.className = "ems-emotion-img";
                        img.src = `/vnccs/get_emotion_image?name=${encodeURIComponent(e.safe_name)}`;
                        img.onerror = () => { img.style.display = 'none'; };

                        const lbl = document.createElement("div");
                        lbl.className = "ems-emotion-label";
                        lbl.innerText = e.safe_name;

                        div.appendChild(img);
                        div.appendChild(lbl);

                        div.onclick = () => {
                            if (state.selectedEmotions.has(e.safe_name)) {
                                state.selectedEmotions.delete(e.safe_name);
                            } else {
                                state.selectedEmotions.add(e.safe_name);
                            }
                            renderEmotions();
                            updateEmotionsData();
                        };

                        emotionsGrid.appendChild(div);
                    });
                    updateButtonState();
                }

                async function fetchCharacterData(charName) {
                    if (!charName || charName === "Character Name") return;

                    // Preview (randomize to force reload from disk)
                    charImg.src = `/vnccs/get_character_pose_preview?character=${encodeURIComponent(charName)}&t=${Date.now()}`;

                    // Costumes
                    try {
                        const res = await fetch(`/vnccs/get_character_costumes?character=${encodeURIComponent(charName)}`);
                        const validCostumes = await res.json();
                        state.costumes = validCostumes || [];

                        // FIX: Only reset to "all" if no saved selection exists
                        // Otherwise, filter saved selection to only include valid costumes
                        if (costumesDataWidget && costumesDataWidget.value) {
                            try {
                                const saved = JSON.parse(costumesDataWidget.value);
                                state.selectedCostumes = new Set(saved.filter(c => state.costumes.includes(c)));
                            } catch (e) {
                                state.selectedCostumes = new Set(state.costumes);
                            }
                        } else {
                            state.selectedCostumes = new Set(state.costumes);
                        }

                        renderCostumes();
                        updateCostumesData();
                    } catch (e) {
                        console.error("Error fetching costumes", e);
                    }
                }

                // Initial Load
                fetch("/vnccs/get_emotions").then(async (res) => {
                    if (res.ok) {
                        const data = await res.json();
                        let flat = [];
                        for (let cat in data) {
                            data[cat].forEach(e => flat.push({ ...e, category: cat }));
                        }
                        state.emotions = flat;
                        renderEmotions();

                        // NOW restore selection state (after list loaded)
                        restoreStateFromWidgets();
                        // Re-render to show restored selections
                        renderEmotions();
                        renderCostumes();
                        updateButtonState();
                    }
                });

                if (state.character) {
                    fetchCharacterData(state.character);
                }

                // Hook callback
                if (charWidget) {
                    const originalCb = charWidget.callback;
                    charWidget.callback = function (v) {
                        state.character = v;
                        if (charSelect.value !== v) charSelect.value = v;
                        fetchCharacterData(v);
                        if (originalCb) originalCb(v);
                    };
                }
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                onConfigure?.apply(this, arguments);
                syncDOMWidgetWidth(this, "emotion_ui_v2");
                setTimeout(() => syncDOMWidgetWidth(this, "emotion_ui_v2"), 100);
            };
        }
    }
});
