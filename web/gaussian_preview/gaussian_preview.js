/**
 * VNCCS Background Preview - Gaussian Splat Preview Widget
 * Interactive 3D Gaussian Splatting viewer using gsplat.js
 * 
 * Implements Pattern 1: DOM Embedding (The "Manager" Pattern) from vnccs_widgets skill.
 */

import { app } from "../../../scripts/app.js";

// Path to viewer HTML
const VIEWER_PATH = "/extensions/VNCCS/gaussian_preview/viewer_gaussian.html";

console.log("[VNCCS.BackgroundPreview] Loading extension...");

app.registerExtension({
    name: "VNCCS.BackgroundPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VNCCS_BackgroundPreview") {
            console.log("[VNCCS.BackgroundPreview] Registering node handler");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                // 1. Create container
                const container = document.createElement("div");
                Object.assign(container.style, {
                    width: "100%",
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    backgroundColor: "#1a1a1a",
                    overflow: "hidden"
                });

                // 2. Add to node using addDOMWidget
                this.addDOMWidget("preview_gaussian", "GAUSSIAN_PREVIEW", container, {
                    serialize: false,
                    hideOnZoom: false,
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // 3. Delegate logic to a helper class
                this.widgetLogic = new GaussianPreviewWidget(this, container, app);
            };
        }
    }
});

class GaussianPreviewWidget {
    constructor(node, container, app) {
        this.node = node;
        this.container = container;
        this.app = app;

        this.iframe = null;
        this.infoPanel = null;
        this.previewWidth = 512;
        this.currentNodeSize = [512, 580];
        this.iframeLoaded = false;

        this.initUI();
        this.setupNodeSizing();
        this.setupMessageHandling();
        this.setupExecutionHandler();
    }

    initUI() {
        // Create iframe
        this.iframe = document.createElement("iframe");
        Object.assign(this.iframe.style, {
            width: "100%",
            flex: "1 1 0",
            minHeight: "0",
            border: "none",
            backgroundColor: "#1a1a1a"
        });

        // Auto-detect extension folder for the viewer path is tricky if we want to be 100% robust 
        // without hardcoding, but here we expect VNCCS structure.
        // Using the constant defined above.
        this.iframe.src = VIEWER_PATH + "?v=" + Date.now();
        this.iframe.addEventListener('load', () => { this.iframeLoaded = true; });

        // Create info panel
        this.infoPanel = document.createElement("div");
        Object.assign(this.infoPanel.style, {
            backgroundColor: "#1a1a1a",
            borderTop: "1px solid #444",
            padding: "6px 12px",
            fontSize: "10px",
            fontFamily: "monospace",
            color: "#ccc",
            lineHeight: "1.3",
            flexShrink: "0",
            overflow: "hidden"
        });
        this.infoPanel.innerHTML = '<span style="color: #888;">Gaussian splat info will appear here after execution</span>';

        this.container.appendChild(this.iframe);
        this.container.appendChild(this.infoPanel);
    }

    setupNodeSizing() {
        // Find the widget we listed to hook computeSize, or hook the node directly if needed.
        // The addDOMWidget returns the widget object, but we are inside the class now.
        // We can iterate widgets to find ours or just rely on the node.

        // Actually, in the pattern, we often hook the widget returned by addDOMWidget.
        // But since we are outside `onNodeCreated`, we can find it.
        const widget = this.node.widgets.find(w => w.name === "preview_gaussian");
        if (widget) {
            widget.computeSize = () => this.currentNodeSize;
        }

        this.node.setSize(this.currentNodeSize);
    }

    resizeToAspectRatio(imageWidth, imageHeight) {
        const aspectRatio = imageWidth / imageHeight;
        const nodeWidth = this.previewWidth;
        const viewerHeight = Math.round(nodeWidth / aspectRatio);
        const nodeHeight = viewerHeight + 60; // + info panel space

        this.currentNodeSize = [nodeWidth, nodeHeight];
        this.node.setSize(this.currentNodeSize);
        this.node.setDirtyCanvas(true, true);
        this.app.graph.setDirtyCanvas(true, true);
    }

    setupMessageHandling() {
        window.addEventListener('message', async (event) => {
            if (event.data.type === 'VIDEO_RECORDING' && event.data.video) {
                await this.handleVideoRecording(event.data);
            }
        });
    }

    async handleVideoRecording(data) {
        try {
            const base64Data = data.video.split(',')[1];
            const byteString = atob(base64Data);
            const arrayBuffer = new ArrayBuffer(byteString.length);
            const uint8Array = new Uint8Array(arrayBuffer);
            for (let i = 0; i < byteString.length; i++) uint8Array[i] = byteString.charCodeAt(i);

            const blob = new Blob([uint8Array], { type: 'video/mp4' });
            const filename = `gaussian-recording-${new Date().toISOString().replace(/[:.]/g, '-')}.mp4`;

            const formData = new FormData();
            formData.append('image', blob, filename);
            formData.append('type', 'output');
            formData.append('subfolder', '');

            const response = await fetch('/upload/image', { method: 'POST', body: formData });
            if (response.ok) {
                const result = await response.json();
                this.updateInfoPanel(`<div style="color: #6cc;">Video saved: ${result.name}</div>`);
                setTimeout(() => {
                    this.updateInfoPanel('<span style="color: #888;">Gaussian splat info will appear here after execution</span>');
                }, 5000);

                this.node.setDirtyCanvas(true, true);
            }
        } catch (error) {
            console.error('[VNCCS] Error saving video:', error);
            this.updateInfoPanel(`<div style="color: #ff6b6b;">Error saving video: ${error.message}</div>`);
        }
    }

    updateInfoPanel(html) {
        if (this.infoPanel) this.infoPanel.innerHTML = html;
    }

    setupExecutionHandler() {
        const self = this;
        // Hook into onExecuted
        const originalOnExecuted = this.node.onExecuted;
        this.node.onExecuted = function (message) {
            if (originalOnExecuted) originalOnExecuted.apply(this, arguments);
            self.onNodeExecuted(message);
        };
    }

    onNodeExecuted(message) {
        if (message?.ply_file && message.ply_file[0]) {
            const filename = message.ply_file[0];
            const extrinsics = message.extrinsics?.[0] || null;
            const intrinsics = message.intrinsics?.[0] || null;

            // Update width if provided
            const widthParam = message.preview_width?.[0];
            if (widthParam) this.previewWidth = widthParam;

            // Resize
            if (intrinsics && intrinsics[0]) {
                const cx = intrinsics[0][2] || (intrinsics[0] && intrinsics[0][2]); // Handle both 3x3 and flat
                const cy = intrinsics[1] && intrinsics[1][2] || (intrinsics[0] && intrinsics[0][5]);

                // Fallback if structure is unexpected, but assuming standard 3x3 matrix
                // [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ]
                const w = (intrinsics[0][2]) ? intrinsics[0][2] * 2 : 1024;
                const h = (intrinsics[1][2]) ? intrinsics[1][2] * 2 : 1024;

                this.resizeToAspectRatio(w, h);
            } else {
                const defaultHeight = Math.round(this.previewWidth * 0.75) + 60;
                this.currentNodeSize = [this.previewWidth, defaultHeight];
                this.node.setSize(this.currentNodeSize);
            }

            // Load mesh
            this.loadMesh(filename, extrinsics, intrinsics);
        }
    }

    async loadMesh(filename, extrinsics, intrinsics) {
        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

        const fetchAndSend = async () => {
            if (!this.iframe.contentWindow) return;
            try {
                const res = await fetch(filepath);
                const buf = await res.arrayBuffer();
                this.iframe.contentWindow.postMessage({
                    type: "LOAD_MESH_DATA",
                    data: buf,
                    filename: filename,
                    extrinsics: extrinsics,
                    intrinsics: intrinsics
                }, "*", [buf]);

                const fileSizeMb = (buf.byteLength / (1024 * 1024)).toFixed(2);
                this.updateInfoPanel(`
                    <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                        <span style="color: #888;">File:</span> <span style="color: #6cc;">${filename}</span>
                        <span style="color: #888;">Size:</span> <span>${fileSizeMb} MB</span>
                    </div>
                `);

            } catch (e) {
                this.updateInfoPanel(`<div style="color: #ff6b6b;">Load Error: ${e.message}</div>`);
            }
        };

        if (this.iframeLoaded) {
            fetchAndSend();
        } else {
            // Retry a few times if iframe not ready
            setTimeout(fetchAndSend, 500);
        }
    }
}
