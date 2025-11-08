import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let loadedEmotionsData = {};

/**
 * Fetches emotion data from the custom Python API endpoint.
 * @returns {Promise<object>} The loaded emotion data, grouped by category.
 */
async function fetchEmotionsData() {
    try {
        const response = await api.fetchApi("/vnccs/get_emotions");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        loadedEmotionsData = await response.json();
        return loadedEmotionsData;
    } catch (error) {
        console.error("VNCCS Error fetching emotions.json:", error);
        return {};
    }
}

app.registerExtension({
    name: "VNCCS.EmotionGenerator",

    // 1. Hook to enforce widget type early
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EmotionGenerator") {
            nodeData.input_widgets_names = nodeData.input_widgets_names || {};
            // Enforce 'combo' widget type despite Python saying 'STRING'
            nodeData.input_widgets_names["emotion_selector"] = "combo";

            // Save the original function
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

            // 2. Override the onNodeCreated method
            nodeType.prototype.onNodeCreated = function () {
                originalOnNodeCreated?.apply(this, arguments);

                // --- START OF WIDGET LOGIC ---

                const charNameWidget = this.widgets.find(w => w.name === "character");
                const emotionSelector = this.widgets.find(w => w.name === "emotion_selector"); // Hidden Input Slot
                const selectedEmotionsWidget = this.widgets.find(w => w.name === "selected_emotions");

                selectedEmotionsWidget.inputEl.rows = 5;

                if (!selectedEmotionsWidget.callback) {
                    selectedEmotionsWidget.callback = function() {};
                }

                // 1. HIDE original and create/reorder UI widget (uiDropdown)
                emotionSelector.hidden = true;

                const uiDropdown = this.addWidget(
                    "combo",
                    "ui_emotion_selector_visible", // Unique name for UI
                    "loading...",
                    null,
                    {
                        values: ["loading..."] // Placeholder data
                    }
                );

                // Re-order the uiDropdown
                const uiDropdownIndex = this.widgets.findIndex(w => w.name === "ui_emotion_selector_visible");
                const inputSlotIndex = this.widgets.findIndex(w => w.name === "emotion_selector");

                // Ensure the UI widget is placed right after the hidden input slot
                if (uiDropdownIndex !== -1 && inputSlotIndex !== -1 && uiDropdownIndex > inputSlotIndex) {
                    const uiWidget = this.widgets.splice(uiDropdownIndex, 1)[0];
                    this.widgets.splice(inputSlotIndex + 1, 0, uiWidget);
                }

                // 2. DESCRIPTION WIDGET
                const descWidget = this.addWidget(
                    "text", "Description", "", () => {},
                    { multiline: true, readonly: true }
                );
                if (descWidget.inputEl) {
                    descWidget.inputEl.style.height = "80px";
                    descWidget.inputEl.style.opacity = 0.8;
                    descWidget.inputEl.style.resize = "none";
                }

                // 3. ASYNCHRONOUS DATA LOOKUP FUNCTION
                const updateDescription = (safeName) => {
                    let description = "";
                    for (const category in loadedEmotionsData) {
                        const emotion = loadedEmotionsData[category].find(e => e.safe_name === safeName);
                        if (emotion) {
                            description = emotion.description;
                            break;
                        }
                    }
                    descWidget.value = description;
                };

                // Helper function to extract the clean safe_name from the full display path string
                const getSafeNameFromDisplayPath = (valueString) => {
                    if (!valueString) return null;

                    // The value is the full path string (e.g., "Anger/furious snarl 👹")
                    const parts = valueString.split('/');
                    const fullDisplayKey = parts[parts.length - 1].trim();

                    // Look up the clean safe_name using the full display key
                    for (const category in loadedEmotionsData) {
                        const emotion = loadedEmotionsData[category].find(e => e.key.trim() === fullDisplayKey);
                        if (emotion) {
                            return emotion.safe_name.trim();
                        }
                    }
                    return null;
                };

                // 4. CALLBACK (Handles User Selection)
                uiDropdown.callback = (selectedValue) => {
                    let valueString = (typeof selectedValue === 'object' && selectedValue !== null) ? selectedValue.value : selectedValue;

                    const safeName = getSafeNameFromDisplayPath(valueString);

                    if (safeName) {
                        // Critical sync: Update hidden slot and trigger change
                        emotionSelector.value = safeName;
                        emotionSelector.callback(safeName);
                        emotionSelector.options.forceGet = true;

                        updateDescription(safeName);
                    } else {
                        updateDescription("");
                    }
                };


                // 5. BUTTON LOGIC

                // 5a. Add Emotion to Queue Button
                this.addWidget("button", "➕ Add Emotion to Queue", null, () => {
                    let valueString = uiDropdown.value;
                    if (typeof valueString === 'object' && valueString !== null) {
                        valueString = valueString.value;
                    }

                    const selectedSafeName = getSafeNameFromDisplayPath(valueString);
                    const charName = charNameWidget.value;

                    if (!selectedSafeName || selectedSafeName === "loading...") {
                        console.warn("VNCCS: No valid emotion selected to add to queue.");
                        return;
                    }

                    const newEntry = `${charName}: ${selectedSafeName}`;
                    const currentQueue = selectedEmotionsWidget.value.split('\n').filter(line => line.trim() !== "");

                    if (!currentQueue.includes(newEntry)) {
                         selectedEmotionsWidget.value += (selectedEmotionsWidget.value.trim() ? "\n" : "") + newEntry;
                    } else {
                        console.warn(`VNCCS: Emotion '${newEntry}' already in queue.`);
                    }
                    selectedEmotionsWidget.options.forceGet = true;
                    selectedEmotionsWidget.callback(selectedEmotionsWidget.value);
                });

                // 5b. Clear Queue Button
                this.addWidget("button", "❌ Clear Queue", null, () => {
                    selectedEmotionsWidget.value = "";
                    selectedEmotionsWidget.options.forceGet = true;
                    selectedEmotionsWidget.callback(selectedEmotionsWidget.value);
                });


                // 6. DATA LOAD (Asynchronous)
                fetchEmotionsData().then(data => {
                    const options = [];
                    let firstSafeName = "loading...";

                    for (const category in data) {
                        const groupName = category.charAt(0).toUpperCase() + category.slice(1).trim();

                        data[category].forEach(emotion => {
                            const emotionKey = emotion.key.trim();
                            const safeName = emotion.safe_name.trim();

                            // Use the simple string path for grouping (e.g., "Group/Emotion Name")
                            const displayPath = `${groupName}/${emotionKey}`;

                            options.push(displayPath); // Push the hierarchical string

                            if (firstSafeName === "loading...") {
                                firstSafeName = safeName; // Store the clean safe_name as the default value
                            }
                        });
                    }

                    uiDropdown.options.values = options;

                    // Find the initial display path using the safe_name
                    const defaultDisplayPath = options.find(p => getSafeNameFromDisplayPath(p) === firstSafeName) || options[0];

                    uiDropdown.value = defaultDisplayPath;

                    // Set the initial value on the hidden input slot
                    emotionSelector.value = firstSafeName;
                    emotionSelector.callback(firstSafeName);

                    updateDescription(firstSafeName);
                });
                // --- END OF WIDGET LOGIC ---
            };
        }
    }
});
