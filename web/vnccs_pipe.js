import { app } from "../../scripts/app.js";

function _applyOptions(node, options) {
    const loraWidget  = node.widgets?.find(w => w.name === "lora_name");
    const cacheWidget = node.widgets?.find(w => w.name === "lora_options_json");
    if (!loraWidget) return;

    const full = ["none", ...options];
    loraWidget.options.values = full;
    if (!full.includes(loraWidget.value)) loraWidget.value = "none";
    if (cacheWidget) cacheWidget.value = JSON.stringify(full);
    node.setDirtyCanvas(true, true);
}

function _restoreFromCache(node) {
    const loraWidget  = node.widgets?.find(w => w.name === "lora_name");
    const cacheWidget = node.widgets?.find(w => w.name === "lora_options_json");
    if (!loraWidget || !cacheWidget) return;

    try {
        const cached = JSON.parse(cacheWidget.value);
        if (Array.isArray(cached) && cached.length > 0) {
            loraWidget.options.values = cached;
            // Don't reset value — let ComfyUI restore it from the workflow JSON
        }
    } catch { /* corrupt cache — leave defaults */ }
}

app.registerExtension({
    name: "VNCCS.Pipe",

    async nodeCreated(node) {
        if (node.comfyClass !== "VNCCS_Pipe") return;

        // Hide cache widget immediately, then restore options from it
        const cacheWidget = node.widgets?.find(w => w.name === "lora_options_json");
        if (cacheWidget) {
            cacheWidget.type = "hidden";
            cacheWidget.computeSize = () => [0, -4];
            cacheWidget.draw = () => {};
            if (cacheWidget.element) cacheWidget.element.style.display = "none";
        }

        // Restore persisted options on page load before the CC fires any events
        setTimeout(() => _restoreFromCache(node), 100);

        const onLoraOptions = (e) => {
            const options = e.detail?.options;
            if (Array.isArray(options)) _applyOptions(node, options);
        };
        window.addEventListener("vnccs-lora-options-updated", onLoraOptions);

        const origOnRemoved = node.onRemoved;
        node.onRemoved = function (...args) {
            window.removeEventListener("vnccs-lora-options-updated", onLoraOptions);
            origOnRemoved?.apply(this, args);
        };
    },
});
