/**
 * VNCCS Advanced PLY Viewer
 * 
 * Interactive 3D Gaussian Splatting viewer with advanced controls
 * and camera frame selection.
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI_VNCCS";
})();

console.log("[VNCCS.AdvancedPlyViewer] Loading extension...");

app.registerExtension({
    name: "vnccs.advancedplyviewer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VNCCS_AdvancedPlyViewer") {
            console.log("[VNCCS.AdvancedPlyViewer] Registering Advanced Viewer node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#1e1e1e";
                container.style.overflow = "hidden";
                container.style.borderRadius = "4px";

                // Create iframe for advanced viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "400px";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#1e1e1e";

                // Point to new HTML viewer
                iframe.src = `/extensions/${EXTENSION_FOLDER}/advanced_ply_viewer/static/advanced_viewer.html?v=` + Date.now();

                // Create info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#1e1e1e";
                infoPanel.style.borderTop = "1px solid #333";
                infoPanel.style.padding = "6px 12px";
                infoPanel.style.fontSize = "11px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.3";
                infoPanel.style.flexShrink = "0";
                infoPanel.style.overflow = "hidden";
                infoPanel.innerHTML = '<span style="color: #888;">Render result will appear here...</span>';

                // Add elements
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget
                const domWidget = this.addDOMWidget("advanced_viewer", "ADVANCED_VIEWER", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // Find hidden text widgets
                let cameraStateWidget = this.widgets.find(w => w.name === "camera_state");
                let savedCamerasWidget = this.widgets.find(w => w.name === "saved_cameras");

                // Dynamic resize logic
                const node = this;
                let currentNodeSize = [512, 580];
                domWidget.computeSize = () => currentNodeSize;

                this.resizeToAspectRatio = function (imageWidth, imageHeight) {
                    const aspectRatio = imageWidth / imageHeight;
                    const nodeWidth = 768; // Make advanced viewer larger by default
                    const viewerHeight = Math.round(nodeWidth / aspectRatio);
                    const nodeHeight = viewerHeight + 60;

                    currentNodeSize = [nodeWidth, nodeHeight];
                    node.setSize(currentNodeSize);
                    node.setDirtyCanvas(true, true);
                    app.graph.setDirtyCanvas(true, true);
                };

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                    // Send initial saved cameras state if any
                    if (savedCamerasWidget && savedCamerasWidget.value) {
                        iframe.contentWindow.postMessage({
                            type: "SET_SAVED_CAMERAS",
                            data: savedCamerasWidget.value
                        }, "*");
                    }
                });

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Update camera state
                    if (event.data.type === 'CAMERA_STATE_UPDATE') {
                        if (cameraStateWidget) {
                            cameraStateWidget.value = JSON.stringify(event.data.state);
                            // Visual feedback? maybe not needed
                        }
                    }
                    else if (event.data.type === 'SAVED_CAMERAS_UPDATE') {
                        if (savedCamerasWidget) {
                            savedCamerasWidget.value = JSON.stringify(event.data.cameras);
                        }
                    }
                    else if (event.data.type === 'REQUEST_EXECUTE') {
                        // Optional feature: queue prompt directly from iframe
                        app.queuePrompt(0);
                    }
                });

                // Set initial size
                this.setSize([768, 800]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function (message) {
                    onExecuted?.apply(this, arguments);

                    if (message?.ui?.ply_path) {
                        const filename = message.ui.filename?.[0];
                        const rel_path = message.ui.ply_path[0];
                        const type = message.ui.type?.[0] || "output";
                        const state_ack = message.ui.state_ack?.[0];

                        infoPanel.innerHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                                <span style="color: #6c6;">✅ Render Complete</span>
                                <span style="color: #888;">Saved internal view.</span>
                            </div>
                        `;

                        // Re-load the ply into the viewer if not already there,
                        // but only if it actually changed
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=${encodeURIComponent(type)}`;

                        if (iframeLoaded && iframe.contentWindow) {
                            iframe.contentWindow.postMessage({
                                type: "LOAD_MESH_DATA",
                                filepath: filepath, // We pass path now to avoid huge base64 ArrayBuffer transfers if possible
                                filename: filename,
                                timestamp: Date.now()
                            }, "*");
                        }
                    }
                };

                return r;
            };
        }
    }
});
