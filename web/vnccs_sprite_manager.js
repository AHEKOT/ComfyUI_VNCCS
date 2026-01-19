import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE = `
/* VNCCS Sprite Manager Styles */
.vnccs-sm-container {
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
    pointer-events: none;
    zoom: 0.67;
}

/* Header Row */
.vnccs-sm-header {
    display: flex;
    align-items: center;
    gap: 20px;
    flex-shrink: 0;
    background: #252525;
    padding: 10px 15px;
    border-radius: 6px;
    border: 1px solid #333;
    pointer-events: auto;
}

.vnccs-sm-header-group {
    display: flex;
    align-items: center;
    gap: 8px;
}

.vnccs-sm-label {
    color: #888;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
}

.vnccs-sm-select {
    background: #151515;
    border: 1px solid #444;
    color: #fff;
    border-radius: 4px;
    padding: 6px 10px;
    font-family: inherit;
    font-size: 14px;
    min-width: 150px;
    zoom: 1.5;
}

.vnccs-sm-emotion-group {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-left: auto;
}

/* Cleanup Button */
.vnccs-sm-cleanup-group {
    display: flex;
    align-items: center;
    gap: 5px;
}

.vnccs-sm-btn-cleanup {
    background: #5c3d99;
    border: none;
    border-radius: 4px;
    color: white;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: bold;
    cursor: pointer;
    white-space: nowrap;
}

.vnccs-sm-btn-cleanup:hover {
    background: #7352b8;
}

.vnccs-sm-help-icon {
    width: 18px;
    height: 18px;
    background: #444;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: bold;
    color: #aaa;
    cursor: help;
    position: relative;
}

.vnccs-sm-help-icon:hover {
    background: #555;
    color: #fff;
}

.vnccs-sm-tooltip {
    display: none;
    position: absolute;
    top: 25px;
    left: 50%;
    transform: translateX(-50%);
    background: #333;
    color: #eee;
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: normal;
    white-space: nowrap;
    z-index: 100;
    border: 1px solid #555;
}

.vnccs-sm-help-icon:hover .vnccs-sm-tooltip {
    display: block;
}

/* Modal */
.vnccs-sm-modal-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    pointer-events: auto;
}

.vnccs-sm-modal {
    background: #252525;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
    min-width: 400px;
    max-width: 600px;
    max-height: 80%;
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.vnccs-sm-modal-title {
    font-size: 16px;
    font-weight: bold;
    color: #fff;
    border-bottom: 1px solid #444;
    padding-bottom: 10px;
}

.vnccs-sm-modal-content {
    flex: 1;
    overflow-y: auto;
    max-height: 300px;
}

.vnccs-sm-modal-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.vnccs-sm-modal-list li {
    padding: 6px 10px;
    margin: 3px 0;
    background: #1a1a1a;
    border-radius: 4px;
    font-size: 12px;
    display: flex;
    justify-content: space-between;
}

.vnccs-sm-modal-list li .reason {
    color: #888;
    font-size: 10px;
}

.vnccs-sm-modal-buttons {
    display: flex;
    gap: 10px;
    justify-content: flex-end;
}

.vnccs-sm-modal-btn {
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    font-weight: bold;
    cursor: pointer;
}

.vnccs-sm-modal-btn-cancel {
    background: #444;
    color: #fff;
}

.vnccs-sm-modal-btn-confirm {
    background: #d32f2f;
    color: #fff;
}

.vnccs-sm-modal-btn-confirm:hover {
    background: #b71c1c;
}

/* Loading State */
.vnccs-sm-btn-create.loading {
    background: #666;
    cursor: wait;
    pointer-events: none;
}

.vnccs-sm-loading-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.9);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    pointer-events: auto;
    gap: 20px;
}

.vnccs-sm-spinner {
    width: 50px;
    height: 50px;
    border: 4px solid #333;
    border-top-color: #5b96f5;
    border-radius: 50%;
    animation: vnccs-sm-spin 1s linear infinite;
}

@keyframes vnccs-sm-spin {
    to { transform: rotate(360deg); }
}

.vnccs-sm-loading-text {
    color: #fff;
    font-size: 16px;
    font-weight: bold;
}

.vnccs-sm-loading-dots::after {
    content: '';
    animation: vnccs-sm-dots 1.5s steps(4, end) infinite;
}

@keyframes vnccs-sm-dots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
}

/* Main Content - Costumes Area */
.vnccs-sm-costumes-area {
    flex: 1;
    display: flex;
    align-items: center;
    background: #151515;
    border: 1px solid #333;
    border-radius: 6px;
    overflow: hidden;
    min-height: 0;
    pointer-events: auto;
}

.vnccs-sm-scroll-btn {
    flex-shrink: 0;
    width: 40px;
    height: 100%;
    background: #252525;
    border: none;
    color: #888;
    font-size: 24px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background 0.2s, color 0.2s;
}

.vnccs-sm-scroll-btn:hover {
    background: #333;
    color: #fff;
}

.vnccs-sm-scroll-btn:disabled {
    color: #333;
    cursor: default;
    background: #1a1a1a;
}

.vnccs-sm-costumes-scroll {
    flex: 1;
    display: flex;
    flex-direction: row;
    gap: 15px;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 15px;
    height: 100%;
    box-sizing: border-box;
    scroll-behavior: smooth;
}

/* Hide scrollbar but keep functionality */
.vnccs-sm-costumes-scroll::-webkit-scrollbar {
    height: 8px;
}

.vnccs-sm-costumes-scroll::-webkit-scrollbar-track {
    background: #1a1a1a;
}

.vnccs-sm-costumes-scroll::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 4px;
}

.vnccs-sm-costume-card {
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    background: #1e1e1e;
    border: 2px solid #333;
    border-radius: 8px;
    overflow: hidden;
    cursor: pointer;
    transition: border-color 0.2s, transform 0.1s;
    height: 100%;
    box-sizing: border-box;
}

.vnccs-sm-costume-card:hover {
    border-color: #5b96f5;
}

.vnccs-sm-costume-card.selected {
    border-color: #5b96f5;
    background: #1a2a4a;
}

.vnccs-sm-costume-img-container {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 5px;
    min-height: 0;
    width: 100%;
}

.vnccs-sm-costume-img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
}

.vnccs-sm-costume-label {
    flex-shrink: 0;
    font-size: 11px;
    padding: 6px 10px;
    text-align: center;
    width: 100%;
    background: #252525;
    border-top: 1px solid #333;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    box-sizing: border-box;
}

/* Footer Row */
.vnccs-sm-footer {
    flex-shrink: 0;
    display: flex;
    pointer-events: auto;
}

.vnccs-sm-btn-create {
    flex: 1;
    padding: 12px 20px;
    background: #2e7d32;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-weight: bold;
    font-size: 14px;
    color: white;
    text-transform: uppercase;
    transition: background 0.2s;
}

.vnccs-sm-btn-create:hover {
    background: #388e3c;
}

/* Empty State */
.vnccs-sm-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    height: 100%;
    color: #555;
    font-size: 14px;
}

.vnccs-sm-no-cleanup {
    color: #4caf50;
    font-size: 12px;
    padding: 20px;
    text-align: center;
}
`;

// Inject styles once
if (!document.getElementById("vnccs-sm-styles")) {
    const styleEl = document.createElement("style");
    styleEl.id = "vnccs-sm-styles";
    styleEl.textContent = STYLE;
    document.head.appendChild(styleEl);
}

app.registerExtension({
    name: "VNCCS.SpriteManager",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SpriteManager") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const node = this;
            node.setSize([1000, 450]);

            // Hide all widgets except widget_data
            const cleanup = () => {
                if (!node.widgets) return;
                for (const w of node.widgets) {
                    if (w.name !== "widget_data") w.hidden = true;
                }
            };
            cleanup();

            // Widget Data Sync
            let dataWidget = node.widgets?.find(w => w.name === "widget_data");
            if (!dataWidget) {
                dataWidget = node.addWidget("text", "widget_data", "{}", () => { }, { serialize: true });
            }
            dataWidget.hidden = true;

            // State
            const state = {
                character: "",
                costumes: [],
                emotions: ["neutral"],
                currentEmotionIndex: 0
            };

            // === BUILD UI ===
            const container = document.createElement("div");
            container.className = "vnccs-sm-container";

            // === HEADER ROW ===
            const header = document.createElement("div");
            header.className = "vnccs-sm-header";

            // Character Group
            const charGroup = document.createElement("div");
            charGroup.className = "vnccs-sm-header-group";

            const charLabel = document.createElement("span");
            charLabel.className = "vnccs-sm-label";
            charLabel.innerText = "Character:";

            const charSelect = document.createElement("select");
            charSelect.className = "vnccs-sm-select";

            charGroup.appendChild(charLabel);
            charGroup.appendChild(charSelect);
            header.appendChild(charGroup);

            // Emotion Group
            const emotionGroup = document.createElement("div");
            emotionGroup.className = "vnccs-sm-emotion-group";

            const emotionLabel = document.createElement("span");
            emotionLabel.className = "vnccs-sm-label";
            emotionLabel.innerText = "Emotion:";

            const emotionSelect = document.createElement("select");
            emotionSelect.className = "vnccs-sm-select";

            emotionGroup.appendChild(emotionLabel);
            emotionGroup.appendChild(emotionSelect);
            header.appendChild(emotionGroup);

            // Cleanup Group (center)
            const cleanupGroup = document.createElement("div");
            cleanupGroup.className = "vnccs-sm-cleanup-group";

            const btnCleanup = document.createElement("button");
            btnCleanup.className = "vnccs-sm-btn-cleanup";
            btnCleanup.innerText = "🧹 Remove empty folders";

            const helpIcon = document.createElement("div");
            helpIcon.className = "vnccs-sm-help-icon";
            helpIcon.innerText = "?";

            const tooltip = document.createElement("div");
            tooltip.className = "vnccs-sm-tooltip";
            tooltip.innerText = "Safe cleanup: removes only empty folders. Your images won't be deleted.";
            helpIcon.appendChild(tooltip);

            cleanupGroup.appendChild(btnCleanup);
            cleanupGroup.appendChild(helpIcon);
            header.insertBefore(cleanupGroup, emotionGroup);

            container.appendChild(header);

            // === COSTUMES AREA ===
            const costumesArea = document.createElement("div");
            costumesArea.className = "vnccs-sm-costumes-area";

            // Left Scroll Button
            const btnScrollLeft = document.createElement("button");
            btnScrollLeft.className = "vnccs-sm-scroll-btn";
            btnScrollLeft.innerText = "◀";

            // Scroll Container
            const costumesScroll = document.createElement("div");
            costumesScroll.className = "vnccs-sm-costumes-scroll";

            // Right Scroll Button
            const btnScrollRight = document.createElement("button");
            btnScrollRight.className = "vnccs-sm-scroll-btn";
            btnScrollRight.innerText = "▶";

            costumesArea.appendChild(btnScrollLeft);
            costumesArea.appendChild(costumesScroll);
            costumesArea.appendChild(btnScrollRight);
            container.appendChild(costumesArea);

            // === FOOTER ROW ===
            const footer = document.createElement("div");
            footer.className = "vnccs-sm-footer";

            const btnCreate = document.createElement("button");
            btnCreate.className = "vnccs-sm-btn-create";
            btnCreate.innerText = "CREATE SPRITES";

            footer.appendChild(btnCreate);
            container.appendChild(footer);

            // === FUNCTIONS ===
            function saveState() {
                const data = {
                    character: state.character,
                    costumes: state.costumes,
                    emotion: state.emotions[state.currentEmotionIndex] || "neutral"
                };
                if (dataWidget) {
                    dataWidget.value = JSON.stringify(data);
                }
            }

            function updateScrollButtons() {
                // Use requestAnimationFrame to ensure DOM is ready
                requestAnimationFrame(() => {
                    const { scrollLeft, scrollWidth, clientWidth } = costumesScroll;
                    btnScrollLeft.disabled = scrollLeft <= 0;
                    // Enable right button if content is wider than container
                    btnScrollRight.disabled = scrollWidth <= clientWidth || scrollLeft + clientWidth >= scrollWidth - 5;
                });
            }

            function renderCostumes() {
                costumesScroll.innerHTML = "";

                if (state.costumes.length === 0) {
                    const empty = document.createElement("div");
                    empty.className = "vnccs-sm-empty";
                    empty.innerText = state.character ? "No costumes found" : "Select a character";
                    costumesScroll.appendChild(empty);
                    updateScrollButtons();
                    return;
                }

                const emotion = state.emotions[state.currentEmotionIndex] || "neutral";

                state.costumes.forEach(costume => {
                    const card = document.createElement("div");
                    card.className = "vnccs-sm-costume-card";
                    // Calculate card width based on container height to maintain aspect ratio
                    // Assuming sprites are roughly 1:2 (width:height)
                    card.style.width = "auto";
                    card.style.minWidth = "120px";
                    card.style.maxWidth = "200px";

                    const imgContainer = document.createElement("div");
                    imgContainer.className = "vnccs-sm-costume-img-container";

                    const img = document.createElement("img");
                    img.className = "vnccs-sm-costume-img";
                    img.src = `/vnccs/get_sheet_preview?character=${encodeURIComponent(state.character)}&costume=${encodeURIComponent(costume)}&emotion=${encodeURIComponent(emotion)}&ts=${Date.now()}`;
                    img.onerror = () => { img.style.opacity = "0.3"; };

                    imgContainer.appendChild(img);

                    const label = document.createElement("div");
                    label.className = "vnccs-sm-costume-label";
                    label.innerText = costume;
                    label.title = costume;

                    card.appendChild(imgContainer);
                    card.appendChild(label);

                    costumesScroll.appendChild(card);
                });

                // Update scroll buttons after render (multiple times to handle image loading)
                setTimeout(updateScrollButtons, 100);
                setTimeout(updateScrollButtons, 300);
                setTimeout(updateScrollButtons, 500);
            }

            function renderEmotionSelect() {
                emotionSelect.innerHTML = "";
                state.emotions.forEach((em, idx) => {
                    const opt = document.createElement("option");
                    opt.value = idx;
                    opt.innerText = em;
                    if (idx === state.currentEmotionIndex) opt.selected = true;
                    emotionSelect.appendChild(opt);
                });
            }

            async function fetchCharacterData(charName) {
                if (!charName) {
                    state.costumes = [];
                    renderCostumes();
                    return;
                }

                state.character = charName;

                // First fetch emotions to know what's available
                try {
                    const res = await fetch(`/vnccs/get_character_emotions?character=${encodeURIComponent(charName)}`);
                    const emotions = await res.json();
                    state.emotions = emotions.length > 0 ? emotions : ["neutral"];
                    if (state.currentEmotionIndex >= state.emotions.length) {
                        state.currentEmotionIndex = 0;
                    }
                    renderEmotionSelect();
                } catch (e) {
                    state.emotions = ["neutral"];
                    renderEmotionSelect();
                }

                // Then fetch costumes for the current emotion
                await refreshCostumes();
            }

            async function refreshCostumes() {
                const emotion = state.emotions[state.currentEmotionIndex] || "neutral";

                try {
                    const res = await fetch(`/vnccs/get_costumes_by_emotion?character=${encodeURIComponent(state.character)}&emotion=${encodeURIComponent(emotion)}`);
                    const costumes = await res.json();
                    state.costumes = costumes || [];
                } catch (e) {
                    console.error("[SpriteManager] Error fetching costumes:", e);
                    state.costumes = [];
                }

                saveState();
                renderCostumes();
            }

            async function fetchCharacterList() {
                try {
                    const res = await fetch("/vnccs/list_characters");
                    const characters = await res.json();
                    charSelect.innerHTML = "";

                    if (characters.length === 0) {
                        const opt = document.createElement("option");
                        opt.value = "";
                        opt.innerText = "No characters";
                        charSelect.appendChild(opt);
                        return;
                    }

                    characters.forEach(c => {
                        const opt = document.createElement("option");
                        opt.value = c;
                        opt.innerText = c;
                        charSelect.appendChild(opt);
                    });

                    // Auto-select first character
                    if (characters.length > 0) {
                        state.character = characters[0];
                        charSelect.value = state.character;
                        fetchCharacterData(state.character);
                    }
                } catch (e) {
                    console.error("[SpriteManager] Error fetching characters:", e);
                }
            }

            // === EVENT HANDLERS ===
            charSelect.onchange = () => {
                fetchCharacterData(charSelect.value);
            };

            emotionSelect.onchange = async () => {
                state.currentEmotionIndex = parseInt(emotionSelect.value) || 0;
                await refreshCostumes();
            };

            const scrollAmount = 200;
            btnScrollLeft.onclick = () => {
                costumesScroll.scrollBy({ left: -scrollAmount, behavior: "smooth" });
            };

            btnScrollRight.onclick = () => {
                costumesScroll.scrollBy({ left: scrollAmount, behavior: "smooth" });
            };

            costumesScroll.addEventListener("scroll", updateScrollButtons);

            // Cleanup button handler
            btnCleanup.onclick = async () => {
                if (!state.character) {
                    alert("Please select a character first");
                    return;
                }

                try {
                    const res = await fetch(`/vnccs/find_empty_folders?character=${encodeURIComponent(state.character)}`);
                    const data = await res.json();

                    if (data.error) {
                        alert("Error: " + data.error);
                        return;
                    }

                    const emptyFolders = data.empty_folders || [];

                    // Create modal
                    const overlay = document.createElement("div");
                    overlay.className = "vnccs-sm-modal-overlay";

                    const modal = document.createElement("div");
                    modal.className = "vnccs-sm-modal";

                    const title = document.createElement("div");
                    title.className = "vnccs-sm-modal-title";

                    if (emptyFolders.length === 0) {
                        title.innerText = "✅ No empty folders found";

                        const content = document.createElement("div");
                        content.className = "vnccs-sm-no-cleanup";
                        content.innerText = "All folders contain images. Nothing to clean up!";

                        const buttons = document.createElement("div");
                        buttons.className = "vnccs-sm-modal-buttons";

                        const btnClose = document.createElement("button");
                        btnClose.className = "vnccs-sm-modal-btn vnccs-sm-modal-btn-cancel";
                        btnClose.innerText = "Close";
                        btnClose.onclick = () => overlay.remove();

                        buttons.appendChild(btnClose);
                        modal.appendChild(title);
                        modal.appendChild(content);
                        modal.appendChild(buttons);
                    } else {
                        title.innerText = `🗑️ Delete ${emptyFolders.length} empty folders?`;

                        const content = document.createElement("div");
                        content.className = "vnccs-sm-modal-content";

                        const list = document.createElement("ul");
                        list.className = "vnccs-sm-modal-list";

                        emptyFolders.forEach(folder => {
                            const li = document.createElement("li");
                            const pathSpan = document.createElement("span");
                            pathSpan.innerText = folder.path;
                            const reasonSpan = document.createElement("span");
                            reasonSpan.className = "reason";
                            reasonSpan.innerText = folder.reason;
                            li.appendChild(pathSpan);
                            li.appendChild(reasonSpan);
                            list.appendChild(li);
                        });

                        content.appendChild(list);

                        const buttons = document.createElement("div");
                        buttons.className = "vnccs-sm-modal-buttons";

                        const btnCancel = document.createElement("button");
                        btnCancel.className = "vnccs-sm-modal-btn vnccs-sm-modal-btn-cancel";
                        btnCancel.innerText = "Cancel";
                        btnCancel.onclick = () => overlay.remove();

                        const btnConfirm = document.createElement("button");
                        btnConfirm.className = "vnccs-sm-modal-btn vnccs-sm-modal-btn-confirm";
                        btnConfirm.innerText = "Delete";
                        btnConfirm.onclick = async () => {
                            try {
                                const foldersToDelete = emptyFolders.map(f => f.full_path);
                                const deleteRes = await fetch("/vnccs/delete_empty_folders", {
                                    method: "POST",
                                    headers: { "Content-Type": "application/json" },
                                    body: JSON.stringify({ folders: foldersToDelete })
                                });
                                const deleteData = await deleteRes.json();

                                overlay.remove();

                                // Refresh costumes
                                fetchCharacterData(state.character);

                                if (deleteData.deleted_count > 0) {
                                    console.log(`[SpriteManager] Deleted ${deleteData.deleted_count} folders`);
                                }
                                if (deleteData.errors && deleteData.errors.length > 0) {
                                    console.warn("[SpriteManager] Some errors:", deleteData.errors);
                                }
                            } catch (e) {
                                console.error("[SpriteManager] Delete error:", e);
                                overlay.remove();
                            }
                        };

                        buttons.appendChild(btnCancel);
                        buttons.appendChild(btnConfirm);
                        modal.appendChild(title);
                        modal.appendChild(content);
                        modal.appendChild(buttons);
                    }

                    overlay.appendChild(modal);
                    container.appendChild(overlay);

                    overlay.onclick = (e) => {
                        if (e.target === overlay) overlay.remove();
                    };
                } catch (e) {
                    console.error("[SpriteManager] Error finding empty folders:", e);
                }
            };

            btnCreate.onclick = () => {
                if (btnCreate.classList.contains('loading')) return;

                saveState();

                // Show loading state
                btnCreate.classList.add('loading');
                const originalText = btnCreate.innerText;
                btnCreate.innerText = 'CREATING...';

                // Show loading overlay
                const loadingOverlay = document.createElement('div');
                loadingOverlay.className = 'vnccs-sm-loading-overlay';

                const spinner = document.createElement('div');
                spinner.className = 'vnccs-sm-spinner';

                const loadingText = document.createElement('div');
                loadingText.className = 'vnccs-sm-loading-text';
                loadingText.innerHTML = 'Creating sprites<span class="vnccs-sm-loading-dots"></span>';

                loadingOverlay.appendChild(spinner);
                loadingOverlay.appendChild(loadingText);
                container.appendChild(loadingOverlay);

                // Queue prompt
                app.queuePrompt(0);

                // Listen for execution complete
                const onExecuted = (event) => {
                    // Check if this execution was for our node
                    if (event.detail && event.detail.node === node.id) {
                        cleanup();
                    }
                };

                const onError = () => {
                    cleanup();
                };

                const cleanup = () => {
                    btnCreate.classList.remove('loading');
                    btnCreate.innerText = originalText;
                    loadingOverlay.remove();
                    api.removeEventListener('executed', onExecuted);
                    api.removeEventListener('execution_error', onError);
                };

                api.addEventListener('executed', onExecuted);
                api.addEventListener('execution_error', onError);

                // Fallback timeout (60 seconds)
                setTimeout(() => {
                    if (loadingOverlay.parentNode) {
                        cleanup();
                    }
                }, 60000);
            };

            // === WIDGET REGISTRATION ===
            node.addDOMWidget("sprite_manager_ui", "ui", container, {
                serialize: false,
                hideOnZoom: false
            });

            // === INITIALIZATION ===
            fetchCharacterList();
        };
    }
});
