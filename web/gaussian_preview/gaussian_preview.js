/**
 * ComfyUI Gaussian Preview - Gaussian Splat Preview Widget
 * Interactive 3D Gaussian Splatting viewer using gsplat.js
 */

import { app } from "../../../scripts/app.js";

// Auto-detect extension folder name (handles ComfyUI-gaussian_preview or comfyui-gaussian-preview)
const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "ComfyUI-gaussian_preview";
})();

console.log("[YCGaussianPreview] Loading extension...");

app.registerExtension({
    name: "ycgaussianpreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "YCGaussianPreview") {
            console.log("[YCGaussianPreview] Registering Preview Gaussian node");

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Create container for viewer + info panel
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.backgroundColor = "#1a1a1a";
                container.style.overflow = "hidden";

                // Create iframe for gsplat.js viewer
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.flex = "1 1 0";
                iframe.style.minHeight = "0";
                iframe.style.border = "none";
                iframe.style.backgroundColor = "#1a1a1a";

                // Point to gsplat.js HTML viewer (with cache buster)
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_gaussian.html?v=` + Date.now();

                // Create info panel
                const infoPanel = document.createElement("div");
                infoPanel.style.backgroundColor = "#1a1a1a";
                infoPanel.style.borderTop = "1px solid #444";
                infoPanel.style.padding = "6px 12px";
                infoPanel.style.fontSize = "10px";
                infoPanel.style.fontFamily = "monospace";
                infoPanel.style.color = "#ccc";
                infoPanel.style.lineHeight = "1.3";
                infoPanel.style.flexShrink = "0";
                infoPanel.style.overflow = "hidden";
                infoPanel.innerHTML = '<span style="color: #888;">Gaussian splat info will appear here after execution</span>';

                // Add iframe and info panel to container
                container.appendChild(iframe);
                container.appendChild(infoPanel);

                // Add widget with required options
                const widget = this.addDOMWidget("preview_gaussian", "GAUSSIAN_PREVIEW", container, {
                    getValue() { return ""; },
                    setValue(v) { }
                });

                // Store reference to node for dynamic resizing
                const node = this;
                let previewWidth = 512;  // Default width, will be updated from parameter
                let currentNodeSize = [512, 580];

                widget.computeSize = () => currentNodeSize;

                // Store references
                this.gaussianViewerIframe = iframe;
                this.gaussianInfoPanel = infoPanel;

                // Function to resize node dynamically
                this.resizeToAspectRatio = function(imageWidth, imageHeight) {
                    const aspectRatio = imageWidth / imageHeight;
                    const nodeWidth = previewWidth;
                    const viewerHeight = Math.round(nodeWidth / aspectRatio);
                    const nodeHeight = viewerHeight + 60;  // Add space for info panel

                    currentNodeSize = [nodeWidth, nodeHeight];
                    node.setSize(currentNodeSize);
                    node.setDirtyCanvas(true, true);
                    app.graph.setDirtyCanvas(true, true);

                    console.log("[YCGaussianPreview] Resized node to:", nodeWidth, "x", nodeHeight, "(aspect ratio:", aspectRatio.toFixed(2), ")");
                };

                // Track iframe load state
                let iframeLoaded = false;
                iframe.addEventListener('load', () => {
                    iframeLoaded = true;
                });

                // Store node reference for message handler
                const currentNode = this;

                // Listen for messages from iframe
                window.addEventListener('message', async (event) => {
                    // Handle screenshot messages - 已删除截图功能
                    // if (event.data.type === 'SCREENSHOT' && event.data.image) {
                    //     try {
                    //         // Convert base64 data URL to blob
                    //         const base64Data = event.data.image.split(',')[1];
                    //         const byteString = atob(base64Data);
                    //         const arrayBuffer = new ArrayBuffer(byteString.length);
                    //         const uint8Array = new Uint8Array(arrayBuffer);
                    //
                    //         for (let i = 0; i < byteString.length; i++) {
                    //             uint8Array[i] = byteString.charCodeAt(i);
                    //         }
                    //
                    //         const blob = new Blob([uint8Array], { type: 'image/png' });
                    //
                    //         // Generate filename with timestamp
                    //         const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                    //         const filename = `gaussian-screenshot-${timestamp}.png`;
                    //
                    //         // Create FormData for upload
                    //         const formData = new FormData();
                    //         formData.append('image', blob, filename);
                    //         formData.append('type', 'output');
                    //         formData.append('subfolder', '');
                    //
                    //         // Upload to ComfyUI backend
                    //         const response = await fetch('/upload/image', {
                    //             method: 'POST',
                    //             body: formData
                    //         });
                    //
                    //         if (response.ok) {
                    //             const result = await response.json();
                    //             console.log('[YCGaussianPreview] Screenshot saved:', result.name);
                    //         } else {
                    //             throw new Error(`Upload failed: ${response.status}`);
                    //         }
                    //
                    //     } catch (error) {
                    //         console.error('[YCGaussianPreview] Error saving screenshot:', error);
                    //     }
                    // }
                    // Handle video recording messages
                    if (event.data.type === 'VIDEO_RECORDING' && event.data.video) {
                        try {
                            console.log('[YCGaussianPreview] Received video recording, MIME type:', event.data.mimeType);
                            
                            // Convert base64 data URL to blob
                            const base64Data = event.data.video.split(',')[1];
                            const byteString = atob(base64Data);
                            const arrayBuffer = new ArrayBuffer(byteString.length);
                            const uint8Array = new Uint8Array(arrayBuffer);

                            for (let i = 0; i < byteString.length; i++) {
                                uint8Array[i] = byteString.charCodeAt(i);
                            }

                            // Force MP4 extension (we're using MP4 format now)
                            const extension = 'mp4';

                            const blob = new Blob([uint8Array], { type: 'video/mp4' });

                            // Generate filename with timestamp
                            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                            const filename = `gaussian-recording-${timestamp}.${extension}`;

                            // Create FormData for upload
                            const formData = new FormData();
                            formData.append('image', blob, filename);
                            formData.append('type', 'output');
                            formData.append('subfolder', '');

                            // Upload to ComfyUI backend
                            const response = await fetch('/upload/image', {
                                method: 'POST',
                                body: formData
                            });

                            if (response.ok) {
                                const result = await response.json();
                                console.log('[YCGaussianPreview] Video saved:', result.name);
                                
                                // Show success message in info panel
                                if (infoPanel) {
                                    const originalContent = infoPanel.innerHTML;
                                    infoPanel.innerHTML = `<div style="color: #6cc;">Video saved: ${result.name}</div><div style="color: #888; font-size: 9px; margin-top: 2px;">Click "Run" to update video output</div>`;
                                    setTimeout(() => {
                                        infoPanel.innerHTML = originalContent;
                                    }, 5000);
                                }
                                
                                // Mark the node as dirty to trigger re-execution check
                                // This will show the node as needing re-execution in the UI
                                if (currentNode) {
                                    currentNode.setDirtyCanvas(true, true);
                                    app.graph.setDirtyCanvas(true, true);
                                    console.log('[YCGaussianPreview] Node marked as dirty - new video available');
                                }
                            } else {
                                throw new Error(`Upload failed: ${response.status}`);
                            }

                        } catch (error) {
                            console.error('[YCGaussianPreview] Error saving video:', error);
                            
                            // Show error message in info panel
                            if (infoPanel) {
                                infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error saving video: ${error.message}</div>`;
                            }
                        }
                    }
                    // Handle error messages from iframe
                    if (event.data.type === 'MESH_ERROR' && event.data.error) {
                        console.error('[YCGaussianPreview] Error from viewer:', event.data.error);
                        if (infoPanel) {
                            infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${event.data.error}</div>`;
                        }
                    }
                });

                // Set initial node size (will be updated when preview_width parameter is received)
                this.setSize([512, 580]);

                // Handle execution
                const onExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("[YCGaussianPreview] onExecuted called with:", message);
                    onExecuted?.apply(this, arguments);

                    // Check for errors
                    if (message?.error && message.error[0]) {
                        infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error: ${message.error[0]}</div>`;
                        return;
                    }

                    // The message IS the UI data (not message.ui)
                    if (message?.ply_file && message.ply_file[0]) {
                        const filename = message.ply_file[0];
                        const displayName = message.filename?.[0] || filename;
                        const fileSizeMb = message.file_size_mb?.[0] || 'N/A';

                        // Extract camera parameters if provided
                        const extrinsics = message.extrinsics?.[0] || null;
                        const intrinsics = message.intrinsics?.[0] || null;
                        
                        // Extract preview width parameter (default: 512)
                        const widthParam = message.preview_width?.[0];
                        if (widthParam && widthParam >= 256 && widthParam <= 4096) {
                            previewWidth = widthParam;
                            console.log("[YCGaussianPreview] Preview width set to:", previewWidth);
                        }

                        // Resize node to match image aspect ratio from intrinsics
                        if (intrinsics && intrinsics[0] && intrinsics[1]) {
                            const imageWidth = intrinsics[0][2] * 2;   // cx * 2
                            const imageHeight = intrinsics[1][2] * 2;  // cy * 2
                            this.resizeToAspectRatio(imageWidth, imageHeight);
                        } else {
                            // If no intrinsics, use default height based on preview width
                            const defaultHeight = Math.round(previewWidth * 0.75) + 60;  // 4:3 aspect ratio + info panel
                            currentNodeSize = [previewWidth, defaultHeight];
                            this.setSize(currentNodeSize);
                            this.setDirtyCanvas(true, true);
                            app.graph.setDirtyCanvas(true, true);
                        }

                        // Update info panel
                        infoPanel.innerHTML = `
                            <div style="display: grid; grid-template-columns: auto 1fr; gap: 2px 8px;">
                                <span style="color: #888;">File:</span>
                                <span style="color: #6cc;">${displayName}</span>
                                <span style="color: #888;">Size:</span>
                                <span>${fileSizeMb} MB</span>
                            </div>
                        `;

                        // ComfyUI serves output files via /view API endpoint
                        const filepath = `/view?filename=${encodeURIComponent(filename)}&type=output&subfolder=`;

                        // Function to fetch and send data to iframe
                        const fetchAndSend = async () => {
                            if (!iframe.contentWindow) {
                                console.error("[YCGaussianPreview] Iframe contentWindow not available");
                                return;
                            }

                            try {
                                // Fetch the PLY file from parent context (authenticated)
                                console.log("[YCGaussianPreview] Fetching PLY file:", filepath);
                                const response = await fetch(filepath);
                                if (!response.ok) {
                                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                                }
                                const arrayBuffer = await response.arrayBuffer();
                                console.log("[YCGaussianPreview] Fetched PLY file, size:", arrayBuffer.byteLength);

                                // Send the data to iframe with camera parameters
                                iframe.contentWindow.postMessage({
                                    type: "LOAD_MESH_DATA",
                                    data: arrayBuffer,
                                    filename: filename,
                                    extrinsics: extrinsics,
                                    intrinsics: intrinsics,
                                    timestamp: Date.now()
                                }, "*", [arrayBuffer]);
                            } catch (error) {
                                console.error("[YCGaussianPreview] Error fetching PLY:", error);
                                infoPanel.innerHTML = `<div style="color: #ff6b6b;">Error loading PLY: ${error.message}</div>`;
                            }
                        };

                        // Fetch and send when iframe is ready
                        if (iframeLoaded) {
                            fetchAndSend();
                        } else {
                            setTimeout(fetchAndSend, 500);
                        }
                    }
                };

                return r;
            };
        }
    }
});
