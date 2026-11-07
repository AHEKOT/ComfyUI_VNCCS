import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { mountManager } from "./pose_manager_ui.mjs?v=pose-manager-17";

function finishAnalysis(widget, token = widget?.container?.dataset.vnccsPmAnalysis) {
    if (!token || widget?.container?.dataset.vnccsPmAnalysis !== token) return;
    delete widget.container.dataset.vnccsPmAnalysis;
    widget.managerGrid.removeAttribute("aria-busy");
    if (widget._poseImageProgress?.token === token) widget._poseImageProgress.close();
}

app.registerExtension({
    name: "VNCCS.PoseManager",
    setup() {
        api.addEventListener("vnccs_pose_image_progress", ({ detail }) => {
            const token = detail?.sync_token;
            if (!token || detail.node_id == null) return;
            let graph = app.rootGraph ?? app.graph;
            let node;
            for (const id of String(detail.node_id).split(":")) {
                node = graph?.getNodeById?.(id);
                if (!node) return;
                graph = node.subgraph;
            }
            const widget = node.studioWidget;
            if (node.type !== "VNCCS_PoseManager" || !widget || widget._disposed) return;
            if (!detail.active) return finishAnalysis(widget, token);
            if (widget.interfaceMode !== "manager" || widget.exportParams.manager_auto_analyze_proportions === false) return;
            widget.container.dataset.vnccsPmAnalysis = token;
            widget.managerGrid.setAttribute("aria-busy", "true");
        });
    },
    async beforeRegisterNodeDef(type, data) {
        if (data.name !== "VNCCS_PoseManager") return;
        const studio = app.extensions.find(extension => extension.name === "VNCCS.PoseStudio");
        const disabled = app.extensionManager?.setting?.get("Comfy.Extension.Disabled") || [];
        if (!studio?.beforeRegisterNodeDef || disabled.includes(studio.name)) {
            const original = type.prototype.onNodeCreated;
            type.prototype.onNodeCreated = function () {
                const result = original?.apply(this, arguments);
                for (const input of this.widgets || []) {
                    if (!["pose_data", "animation_image_batch"].includes(input.name)) continue;
                    input.type = "hidden";
                    input.hidden = true;
                    input.computeSize = () => [0, -4];
                    if (input.element) input.element.style.display = "none";
                }
                const notice = document.createElement("p");
                notice.setAttribute("role", "alert");
                notice.textContent = "VNCCS Pose Manager requires VNCCS-Utils Pose Studio. Install or enable it and reload ComfyUI.";
                this.addDOMWidget("pose_manager_status", "ui", notice, { serialize: false });
                return result;
            };
            return;
        }
        // Reuse Studio's lifecycle and capture protocol without changing the Studio node type.
        await studio.beforeRegisterNodeDef(type, { ...data, name: "VNCCS_PoseStudio" }, app);
        const created = type.prototype.onNodeCreated;
        type.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            mountManager(this.studioWidget);
            return result;
        };
        for (const hook of ["onConfigure", "onRemoved"]) {
            const original = type.prototype[hook];
            type.prototype[hook] = function () {
                finishAnalysis(this.studioWidget);
                this.studioWidget?.closeSetLibrary?.();
                this.studioWidget?._closeLibraryModal?.();
                return original?.apply(this, arguments);
            };
        }
    },
});
