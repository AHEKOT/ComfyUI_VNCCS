import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// --- CSS STYLES ---
const STYLE = `
.vnccs-container {
    display: flex;
    flex-direction: row;
    gap: 10px;
    background: #1e1e1e;
    color: white;
    font-family: monospace;
    font-size: 24px; /* Base size doubled (was implicit ~12px) */
    padding: 10px;
    border-radius: 8px;
    width: 100%;
    height: 100%;
    max-height: 100%;
    box-sizing: border-box;
    overflow: hidden; 
}

/* Left Column */
.vnccs-left-col {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 200px;
    overflow: hidden;
}

/* Right Column */
.vnccs-right-col {
    flex: 3;
    display: flex;
    flex-direction: column;
    gap: 10px;
    min-width: 300px;
    overflow: hidden;
}

/* Sections - Gray Theme */
.vnccs-section {
    border: 2px solid #555; 
    border-radius: 12px;
    background: #333;
    padding: 5px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Character Header inside Left Col */
.vnccs-char-header {
    background: #444;
    color: white;
    padding: 5px 10px;
    font-weight: bold;
    font-size: 16px; /* Reduced to match user request */
    border-bottom: 2px solid #555;
    margin-bottom: 5px;
    border-radius: 8px 8px 0 0;
}

.vnccs-char-preview-container {
    flex: 1;
    background: #222; /* Dark bg for image */
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 0 0 8px 8px;
    padding: 10px;
    overflow: hidden; 
}

.vnccs-char-preview {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

/* Costumes (Top Right) */
.vnccs-costumes-header {
    font-weight: bold;
    color: #ddd;
    margin-bottom: 5px;
    padding-left: 5px;
    /* Inherits base font-size 24px */
}
.vnccs-costumes-list {
    background: #444;
    border-radius: 8px;
    padding: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    color: white;
    max-height: 150px;
    overflow-y: auto;
}
.vnccs-checkbox-item {
    display: flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
    font-weight: bold;
    user-select: none;
    margin-right: 15px; /* Increase spacing for larger text */
}

/* Emotions (Bottom Right) */
.vnccs-emotions-container {
    flex: 1;
    background: #2a2a2a; /* Darker gray for grid bg */
    border-radius: 8px;
    padding: 10px;
    overflow-y: auto;
    display: grid;
    /* 4 Columns for larger preview */
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    min-height: 200px;
}

.vnccs-emotion-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: pointer;
    transition: all 0.1s ease;
    width: 100%; 
    padding: 5px;
    border-radius: 8px;
    border: 2px solid transparent;
    box-sizing: border-box;
}
.vnccs-emotion-item:hover {
    background-color: #444; /* Subtle highlight on hover */
}
.vnccs-emotion-item.selected {
    background-color: #007bff; /* Bright Blue Background */
    border-color: #0056b3;
    transform: scale(0.98);
}

.vnccs-emotion-item.selected .vnccs-emotion-img {
    border: none; /* Remove image border */
    box-shadow: none;
}

.vnccs-emotion-item.selected .vnccs-emotion-label {
    color: white; /* White text on blue bg */
    font-weight: bold;
}

.vnccs-emotion-img {
    width: 100%; 
    aspect-ratio: 1 / 1; 
    object-fit: cover;
    border-radius: 4px; 
}
.vnccs-emotion-label {
    font-size: 22px; /* Doubled from 11px */
    color: #ccc;
    text-align: center;
    margin-top: 5px;
    font-family: monospace;
    word-break: break-word;
    width: 100%;
    line-height: 1.1; /* Tighter line height for large text */
}

/* Footer (Select All) */
.vnccs-footer {
    margin-top: 5px;
    display: flex;
    justify-content: center;
}
.vnccs-btn {
    background: #555;
    color: white;
    border: 2px solid #333;
    border-radius: 5px;
    padding: 10px 30px; /* Larger padding */
    font-weight: bold;
    cursor: pointer;
    font-family: monospace;
    font-size: 24px; /* Doubled */
}
.vnccs-btn:hover {
    background: #777;
}

/* Search Input */
.vnccs-search-input {
    width: 100%;
    padding: 8px;
    margin-top: 5px;
    margin-bottom: 5px;
    background: #444;
    color: white;
    border: 1px solid #555;
    border-radius: 4px;
    font-size: 16px;
    font-family: monospace;
    box-sizing: border-box;
}
.vnccs-search-input:focus {
    outline: none;
    border-color: #007bff;
}
`;

// Inject Styles
const styleEl = document.createElement("style");
styleEl.textContent = STYLE;
document.head.appendChild(styleEl);

// --- MAIN EXTENSION ---

app.registerExtension({
    name: "VNCCS.EmotionGeneratorV2",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EmotionGeneratorV2") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const node = this;

                // Set default size
                node.setSize([920, 650]);

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
                container.className = "vnccs-container";

                // --- LEFT COL ---
                const leftCol = document.createElement("div");
                leftCol.className = "vnccs-left-col";

                // Character Header
                const charSection = document.createElement("div");
                charSection.className = "vnccs-section";
                charSection.style.flex = "1";

                const charHeader = document.createElement("div");
                charHeader.className = "vnccs-char-header";
                charHeader.innerText = "Character select";

                const previewContainer = document.createElement("div");
                previewContainer.className = "vnccs-char-preview-container";

                const charImg = document.createElement("img");
                charImg.className = "vnccs-char-preview";
                previewContainer.appendChild(charImg);

                charSection.appendChild(charHeader);
                charSection.appendChild(previewContainer);
                leftCol.appendChild(charSection);

                container.appendChild(leftCol);

                // --- RIGHT COL ---
                const rightCol = document.createElement("div");
                rightCol.className = "vnccs-right-col";

                // Costumes
                const costumesSection = document.createElement("div");
                costumesSection.className = "vnccs-section";

                const costumesHeader = document.createElement("div");
                costumesHeader.className = "vnccs-costumes-header";
                costumesHeader.innerText = "Selected costumes";
                costumesSection.appendChild(costumesHeader);

                const costumesList = document.createElement("div");
                costumesList.className = "vnccs-costumes-list";
                costumesSection.appendChild(costumesList);

                rightCol.appendChild(costumesSection);

                // Emotions
                const emotionsSection = document.createElement("div");
                emotionsSection.className = "vnccs-section";
                emotionsSection.style.flex = "1";

                const emotionsGrid = document.createElement("div");
                emotionsGrid.className = "vnccs-emotions-container";
                emotionsSection.appendChild(emotionsGrid);

                // Search Input
                const searchInput = document.createElement("input");
                searchInput.className = "vnccs-search-input";
                searchInput.placeholder = "Search emotions (name or description)...";
                searchInput.oninput = (e) => {
                    state.searchTerm = e.target.value;
                    renderEmotions();
                    updateButtonState();
                };
                emotionsSection.appendChild(searchInput);

                // Footer (Select All)
                const footer = document.createElement("div");
                footer.className = "vnccs-footer";
                const btnAll = document.createElement("button");
                btnAll.className = "vnccs-btn";
                btnAll.innerText = "Select ALL";
                footer.appendChild(btnAll);
                emotionsSection.appendChild(footer);

                rightCol.appendChild(emotionsSection);
                container.appendChild(rightCol);

                // Custom Select in Header
                const charSelect = document.createElement("select");
                charSelect.style.width = "100%";
                charSelect.style.marginTop = "5px";
                charSelect.style.padding = "5px";
                charSelect.style.fontWeight = "bold";
                // Style select for dark theme
                charSelect.style.background = "#555";
                charSelect.style.color = "white";
                charSelect.style.border = "1px solid #333";

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
                styleContainer.className = "vnccs-section";
                styleContainer.style.marginBottom = "10px";
                styleContainer.style.padding = "5px 10px";
                
                const styleSelect = document.createElement("select");
                styleSelect.style.width = "100%";
                styleSelect.style.padding = "5px";
                styleSelect.style.fontWeight = "bold";
                styleSelect.style.background = "#555";
                styleSelect.style.color = "white";
                styleSelect.style.border = "1px solid #333";
                
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


                function restoreStateFromWidgets() {
                     // 1. Character
                     if(charWidget && charWidget.value) {
                         state.character = charWidget.value;
                         charSelect.value = charWidget.value;
                         fetchCharacterData(state.character);
                     }
                     
                     // 2. Style
                     if(styleWidget && styleWidget.value) {
                         styleSelect.value = styleWidget.value;
                     }

                     // 3. Costumes & Emotions (from hidden text strings)
                    if(costumesDataWidget && costumesDataWidget.value) {
                        try {
                            const savedCostumes = JSON.parse(costumesDataWidget.value);
                            state.selectedCostumes = new Set(savedCostumes);
                            renderCostumes();
                        } catch(e) {}
                    }
                    if(emotionsDataWidget && emotionsDataWidget.value) {
                        try {
                            const savedEmotions = JSON.parse(emotionsDataWidget.value);
                            state.selectedEmotions = new Set(savedEmotions);
                            // renderEmotions is called after fetch("/vnccs/get_emotions"), need to wait?
                            // No, renderEmotions() just needs state.emotions to be populated.
                            // The fetch happens async.
                        } catch(e) {}
                    }
                }

                // Explicitly set dimensions on resize
                node.onResize = function (size) {
                    const [w, h] = size;
                    container.style.width = (w - 20) + "px";
                    container.style.height = (h - 60) + "px";
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
                        btnAll.style.background = "#333";
                        return;
                    }
                    btnAll.disabled = false;
                    
                    const allFilteredSelected = filtered.every(e => state.selectedEmotions.has(e.safe_name));
                    
                    if (allFilteredSelected) {
                        btnAll.innerText = "Cancel Selection";
                        btnAll.style.background = "#d32f2f"; // Red for cancel
                    } else {
                        btnAll.innerText = "Select ALL";
                        btnAll.style.background = "#555"; // Default gray
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

                        const msg = `Select all ${numEmotions} visible emotions for ${numCostumes} costume(s)?\nTotal: ${total} images.\n\nProceed?`;
                        if (confirm(msg)) {
                            filtered.forEach(e => state.selectedEmotions.add(e.safe_name));
                            renderEmotions();
                            updateEmotionsData();
                        }
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
                        lbl.className = "vnccs-checkbox-item";
                        const chk = document.createElement("input");
                        chk.type = "checkbox";
                        chk.checked = state.selectedCostumes.has(c);
                        chk.onchange = () => {
                            if (chk.checked) state.selectedCostumes.add(c);
                            else state.selectedCostumes.delete(c);
                            updateCostumesData();
                        };
                        const span = document.createElement("span");
                        span.innerText = c;
                        lbl.appendChild(chk);
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
                        div.className = "vnccs-emotion-item" + (selected ? " selected" : "");
                        div.title = e.description || "";

                        const img = document.createElement("img");
                        img.className = "vnccs-emotion-img";
                        img.src = `/vnccs/get_emotion_image?name=${encodeURIComponent(e.safe_name)}`;
                        img.onerror = () => { img.style.display = 'none'; };

                        const lbl = document.createElement("div");
                        lbl.className = "vnccs-emotion-label";
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
                    charImg.src = `/vnccs/get_character_sheet_preview?character=${encodeURIComponent(charName)}&t=${Date.now()}`;

                    // Costumes
                    try {
                        const res = await fetch(`/vnccs/get_character_costumes?character=${encodeURIComponent(charName)}`);
                        const validCostumes = await res.json();
                        state.costumes = validCostumes || [];
                        state.selectedCostumes = new Set(state.costumes);
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
        }
    }
});
