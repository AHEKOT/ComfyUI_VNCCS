import { app } from "../../scripts/app.js";

function applyOptions(node, options) {
    const widget = node.widgets?.find(w => w.name === "lora_name");
    if (!widget) return;
    const values = [...new Set(["none", ...options.filter(value => typeof value === "string")])];
    if (widget.value && !values.includes(widget.value)) values.push(widget.value);
    widget.options.values = values;
    node.setDirtyCanvas(true, true);
}

function restoreOptions(node) {
    const cache = node.widgets?.find(w => w.name === "lora_options_json");
    try {
        const options = JSON.parse(cache?.value);
        if (Array.isArray(options)) applyOptions(node, options);
    } catch { /* Preserve the selected LoRA if the options cache is unreadable. */ }
}

app.registerExtension({
    name: "VNCCS.Pipe",

    nodeCreated(node) {
        if (node.comfyClass !== "VNCCS_Pipe") return;
        const cache = node.widgets?.find(w => w.name === "lora_options_json");
        if (cache) {
            cache.type = "hidden";
            cache.computeSize = () => [0, -4];
            cache.draw = () => {};
            if (cache.element) cache.element.style.display = "none";
        }
        const onConfigure = node.onConfigure;
        node.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            restoreOptions(this);
            return result;
        };

        const onSerialize = node.onSerialize;
        node.onSerialize = function (output) {
            const result = onSerialize?.apply(this, arguments);
            const widget = this.widgets?.find(w => w.name === "lora_name");
            const cache = this.widgets?.find(w => w.name === "lora_options_json");
            if (widget && cache) {
                cache.value = JSON.stringify(widget.options.values);
                const index = this.widgets.indexOf(cache);
                if (Array.isArray(output.widgets_values)) output.widgets_values[index] = cache.value;
            }
            return result;
        };

        const onOptions = event => {
            if (Array.isArray(event.detail?.options)) applyOptions(node, event.detail.options);
        };
        window.addEventListener("vnccs-lora-options-updated", onOptions);
        const onRemoved = node.onRemoved;
        node.onRemoved = function () {
            window.removeEventListener("vnccs-lora-options-updated", onOptions);
            return onRemoved?.apply(this, arguments);
        };
    },
});
