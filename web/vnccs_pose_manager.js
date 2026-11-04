import { app } from "../../scripts/app.js";
import { mountManager } from "./pose_manager_ui.mjs?v=pose-manager-10";

app.registerExtension({
    name: "VNCCS.PoseManager",
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
                this.studioWidget?.closeSetLibrary?.();
                this.studioWidget?._closeLibraryModal?.();
                return original?.apply(this, arguments);
            };
        }
    },
});
