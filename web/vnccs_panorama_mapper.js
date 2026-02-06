import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE = `
.vnccs-pm-container {
    display: flex;
    flex-direction: column;
    background: #1a1a1a;
    color: #eee;
    padding: 10px;
    gap: 10px;
    height: 100%;
}
.vnccs-pm-canvas-wrap {
    flex: 1;
    background: #000;
    position: relative;
    overflow: hidden;
    border: 1px solid #333;
    cursor: crosshair;
}
.vnccs-pm-canvas {
    width: 100%;
    height: 100%;
    object-fit: contain;
}
.vnccs-pm-controls {
    display: flex;
    gap: 10px;
    align-items: center;
}
.vnccs-pm-btn {
    padding: 6px 12px;
    background: #3558c7;
    border: none;
    border-radius: 4px;
    color: white;
    cursor: pointer;
    font-weight: bold;
}
.vnccs-pm-btn:hover { background: #4264d9; }
.vnccs-pm-info {
    font-size: 11px;
    color: #888;
}
`;

app.registerExtension({
    name: "VNCCS.PanoramaMapper",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "VNCCS_PanoramaMapper") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                const node = this;
                node.setSize([800, 600]);

                let state = {
                    image_path: "",
                    corners: [0.125, 0.375, 0.625, 0.875], // Normalized [0..1]
                    pitch: 0.0,
                    roll: 0.0,
                    output_size: 1024
                };

                // Hidden data widget
                let dataWidget = node.widgets.find(w => w.name === "widget_data");
                if (!dataWidget) {
                    dataWidget = node.addWidget("text", "widget_data", "{}", (v) => { }, { serialize: true });
                }
                dataWidget.hidden = true;

                const syncState = () => {
                    if (dataWidget.value && dataWidget.value !== "{}") {
                        try {
                            const parsed = JSON.parse(dataWidget.value);
                            Object.assign(state, parsed);
                            // Normalize if loaded as degrees
                            if (state.corners[0] > 1.1) {
                                state.corners = state.corners.map(c => c / 360);
                            }
                            loadPreview();
                        } catch (e) { console.error("[VNCCS] PanoramaMapper: Error syncing state", e); }
                    }
                };

                const save = () => {
                    // Normalize corners to degrees [0..360] for the Python backend
                    const degCorners = state.corners.map(c => c * 360);
                    const persist = { ...state, corners: degCorners };
                    dataWidget.value = JSON.stringify(persist);
                };

                // UI setup
                const container = document.createElement("div");
                container.className = "vnccs-pm-container";

                const style = document.createElement("style");
                style.innerHTML = STYLE;
                document.head.appendChild(style);

                const canvasWrap = document.createElement("div");
                canvasWrap.className = "vnccs-pm-canvas-wrap";

                const canvas = document.createElement("canvas");
                canvas.className = "vnccs-pm-canvas";
                canvasWrap.appendChild(canvas);
                container.appendChild(canvasWrap);

                const previewRow = document.createElement("div");
                previewRow.style.display = "flex";
                previewRow.style.gap = "5px";
                previewRow.style.height = "120px";
                previewRow.style.width = "100%";

                const wallPreviews = [];
                for (let i = 0; i < 4; i++) {
                    const pw = document.createElement("div");
                    pw.style.background = "#000";
                    pw.style.border = "1px solid #444";
                    pw.style.position = "relative";
                    pw.style.overflow = "hidden";
                    pw.style.flex = "1 1 0%";
                    pw.innerHTML = `<div style="position:absolute; top:2px; left:4px; font-size:9px; color:#888; z-index:5;">Wall ${i + 1}</div>`;
                    const pimg = document.createElement("canvas");
                    pimg.style.width = "100%";
                    pimg.style.height = "100%";
                    pw.appendChild(pimg);
                    previewRow.appendChild(pw);
                    wallPreviews.push(pimg);
                }
                container.appendChild(previewRow);

                const controls = document.createElement("div");
                controls.className = "vnccs-pm-controls";

                const btnUpload = document.createElement("button");
                btnUpload.className = "vnccs-pm-btn";
                btnUpload.innerText = "UPLOAD PANORAMA";

                const fileInput = document.createElement("input");
                fileInput.type = "file";
                fileInput.accept = "image/*";
                fileInput.style.display = "none";

                btnUpload.onclick = () => fileInput.click();
                fileInput.onchange = async (e) => {
                    const file = e.target.files[0];
                    if (!file) return;

                    const formData = new FormData();
                    formData.append("image", file);
                    formData.append("overwrite", "true");

                    try {
                        const resp = await api.fetchApi("/upload/image", {
                            method: "POST",
                            body: formData
                        });
                        const result = await resp.json();
                        state.image_path = result.name;
                        loadPreview();
                        save();
                    } catch (err) {
                        alert("Upload failed: " + err);
                    }
                };

                const info = document.createElement("div");
                info.className = "vnccs-pm-info";
                info.innerText = "Drag RED: corners. Drag CYAN: pitch. Hold SHIFT + Drag CYAN: roll.";

                controls.appendChild(btnUpload);
                controls.appendChild(info);
                container.appendChild(controls);

                this.addDOMWidget("ui", "ui", container, { serialize: false });

                // Interaction state
                let dragging = null;
                let img = new Image();

                const loadPreview = () => {
                    if (!state.image_path) return;
                    img.src = `/view?filename=${encodeURIComponent(state.image_path)}&type=input`;
                    img.onload = () => {
                        updateWallPreviews();
                        draw();
                    };
                };

                const updateWallPreviews = () => {
                    if (!img.complete || !img.width) return;
                    wallPreviews.forEach((pcanvas, i) => {
                        const pctx = pcanvas.getContext("2d");

                        const c1 = state.corners[i];
                        const c2 = state.corners[(i + 1) % 4];

                        let w_norm = (c2 >= c1) ? (c2 - c1) : (1 - c1 + c2);
                        pcanvas.parentElement.style.flex = `${w_norm} 1 0%`;

                        pcanvas.width = 512; pcanvas.height = 256; // Fixed ratio for linear preview

                        let x = c1 * img.width;
                        let w = w_norm * img.width;

                        pctx.fillStyle = "#000";
                        pctx.fillRect(0, 0, 512, 256);

                        const stripH = img.height / 3;
                        const stripY = (img.height - stripH) / 2;

                        if (c2 >= c1) {
                            pctx.drawImage(img, x, stripY, w, stripH, 0, 0, 512, 256);
                        } else {
                            const w1 = (1 - c1) * img.width;
                            const w2 = c2 * img.width;
                            const pw1 = (w1 / w) * 512;
                            pctx.drawImage(img, x, stripY, w1, stripH, 0, 0, pw1, 256);
                            pctx.drawImage(img, 0, stripY, w2, stripH, pw1, 0, 512 - pw1, 256);
                        }
                    });
                };
                const draw = () => {
                    const ctx = canvas.getContext("2d");
                    // Ensure internal resolution matches rendered size for crisp hit detection
                    canvas.width = canvas.clientWidth;
                    canvas.height = canvas.clientHeight;
                    const w = canvas.width, h = canvas.height;

                    ctx.clearRect(0, 0, w, h);

                    if (!img.complete) return;

                    const aspect = img.width / img.height;
                    let iw, ih, ix, iy;
                    if (w / h > aspect) {
                        ih = h; iw = h * aspect; ix = (w - iw) / 2; iy = 0;
                    } else {
                        iw = w; ih = w / aspect; ix = 0; iy = (h - ih) / 2;
                    }
                    ctx.drawImage(img, ix, iy, iw, ih);

                    // Shade wall segments
                    for (let i = 0; i < 4; i++) {
                        const c1 = state.corners[i];
                        const c2 = state.corners[(i + 1) % 4];
                        const x1 = ix + c1 * iw;
                        const x2 = ix + c2 * iw;

                        ctx.fillStyle = i % 2 === 0 ? "rgba(255, 0, 0, 0.1)" : "rgba(0, 255, 0, 0.1)";
                        if (c2 >= c1) {
                            ctx.fillRect(x1, iy, x2 - x1, ih);
                        } else {
                            ctx.fillRect(x1, iy, ix + iw - x1, ih);
                            ctx.fillRect(ix, iy, x2 - ix, ih);
                        }

                        // Label
                        ctx.fillStyle = "#fff";
                        ctx.font = "bold 12px Arial";
                        ctx.textAlign = "center";
                        if (c2 >= c1) {
                            ctx.fillText(`Wall ${i + 1}`, (x1 + x2) / 2, iy + 20);
                        } else {
                            // Split wall - draw label in the larger part or middle
                            const midX = (x1 + ix + iw) / 2;
                            ctx.fillText(`Wall ${i + 1}`, midX, iy + 20);
                        }
                    }

                    // Draw Corner Lines
                    state.corners.forEach((c, i) => {
                        const lx = ix + c * iw;
                        ctx.strokeStyle = "red";
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(lx, iy);
                        ctx.lineTo(lx, iy + ih);
                        ctx.stroke();

                        ctx.fillStyle = "red";
                        ctx.beginPath(); ctx.arc(lx, iy + ih / 2, 5, 0, Math.PI * 2); ctx.fill();
                    });

                    // Horizon & Pitch/Roll Visualization
                    const horizonY = iy + ih / 2 + (state.pitch / 90) * (ih / 2);
                    ctx.save();
                    ctx.translate(ix + iw / 2, horizonY);
                    ctx.rotate(state.roll * Math.PI / 180);

                    ctx.strokeStyle = "cyan";
                    ctx.lineWidth = 1;
                    ctx.setLineDash([5, 5]);
                    ctx.beginPath(); ctx.moveTo(-iw / 2, 0); ctx.lineTo(iw / 2, 0); ctx.stroke();

                    // Show "Level" guides for floor/ceiling boundaries
                    ctx.strokeStyle = "rgba(0, 255, 255, 0.3)";
                    ctx.beginPath(); ctx.moveTo(-iw / 2, -ih / 4); ctx.lineTo(iw / 2, -ih / 4); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(-iw / 2, ih / 4); ctx.lineTo(iw / 2, ih / 4); ctx.stroke();

                    ctx.restore();
                };

                const getMousePos = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    return {
                        x: (e.clientX - rect.left) * (canvas.width / rect.width),
                        y: (e.clientY - rect.top) * (canvas.height / rect.height)
                    };
                };

                // Mouse Handling
                canvas.onmousedown = (e) => {
                    const pos = getMousePos(e);
                    const mx = pos.x, my = pos.y;

                    const aspect = img.width / img.height;
                    const w = canvas.width, h = canvas.height;
                    let iw, ih, ix;
                    if (w / h > aspect) {
                        ih = h; iw = h * aspect; ix = (w - iw) / 2;
                    } else {
                        iw = w; ih = w / aspect; ix = 0;
                    }

                    // CHECK CORNERS FIRST - and increase hit box
                    for (let i = 0; i < state.corners.length; i++) {
                        const lx = ix + state.corners[i] * iw;
                        if (Math.abs(mx - lx) < 20) {
                            dragging = { type: 'corner', idx: i };
                            return;
                        }
                    }

                    const horizonY = (h - ih) / 2 + ih / 2 + (state.pitch / 90) * (ih / 2);
                    if (Math.abs(my - horizonY) < 30) {
                        dragging = { type: 'horizon', shift: e.shiftKey };
                    }
                };

                window.onmousemove = (e) => {
                    if (!dragging) return;
                    const pos = getMousePos(e);
                    const mx = pos.x, my = pos.y;

                    const aspect = img.width / img.height;
                    const w = canvas.width, h = canvas.height;
                    let iw, ix, ih, iy;
                    if (w / h > aspect) {
                        ih = h; iw = h * aspect; ix = (w - iw) / 2; iy = 0;
                    } else {
                        iw = w; ih = w / aspect; ix = 0; iy = (h - ih) / 2;
                    }

                    if (dragging.type === 'corner') {
                        let val = (mx - ix) / iw;
                        val = Math.max(0, Math.min(1, val));

                        state.corners[dragging.idx] = val;

                        updateWallPreviews();
                    } else if (dragging.type === 'horizon') {
                        if (dragging.shift || e.shiftKey) {
                            const dx = (mx - (ix + iw / 2)) / (iw / 2);
                            state.roll = dx * 45;
                        } else {
                            // Pitch
                            let val = (my - (iy + ih / 2)) / (ih / 2);
                            state.pitch = val * 90;
                        }
                    }

                    draw();
                };

                window.onmouseup = () => {
                    if (dragging) {
                        dragging = null;
                        save();
                    }
                };

                node.onConfigure = function () {
                    syncState();
                };

                // Init
                syncState();
            };
        }
    }
});
