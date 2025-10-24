// web/pose_manager.js
// ComfyUI extension for VNCCS Pose Manager widget

import { app } from "../../scripts/app.js";

// Debug: indicate module loaded
console.debug("VNCCS.PoseManager: module loaded");
console.log("VNCCS.PoseManager: module loaded (log)");

const POSE_KEYPOINTS = 25;
const POSE_CONNECTIONS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [10, 11], [8, 12], [12, 13], [13, 14],
    [1, 0], [0, 15], [15, 17], [0, 16], [16, 18], [14, 19], [19, 20], [14, 21], [11, 22], [22, 23], [11, 24]
];

class PoseEditor {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.points = [];
        this.dragging = false;
        this.dragIndex = -1;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;

        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('wheel', this.onWheel.bind(this));

        this.initPoints();
        this.draw();
    }

    initPoints() {
        this.points = [];
        for (let i = 0; i < POSE_KEYPOINTS; i++) {
            this.points.push({ x: 256, y: 100 + i * 40, visible: true });
        }
    }

    loadPose(jsonStr) {
        try {
            const data = JSON.parse(jsonStr);
            if (data.people && data.people[0] && data.people[0].pose_keypoints_2d) {
                const keypoints = data.people[0].pose_keypoints_2d;
                this.points = [];
                for (let i = 0; i < POSE_KEYPOINTS; i++) {
                    const x = (typeof keypoints[i * 3] === 'number') ? keypoints[i * 3] : 256;
                    const y = (typeof keypoints[i * 3 + 1] === 'number') ? keypoints[i * 3 + 1] : 100 + i * 40;
                    const conf = keypoints[i * 3 + 2] || 0;
                    this.points.push({ x, y, visible: conf > 0 });
                }
            }
        } catch (e) {
            // invalid JSON -> keep defaults
            console.warn('PoseEditor.loadPose: invalid JSON', e);
        }
        this.draw();
    }

    getPoseJSON() {
        const keypoints = [];
        for (const point of this.points) {
            keypoints.push(point.x, point.y, point.visible ? 1 : 0);
        }
        const data = {
            version: 1.3,
            people: [{
                pose_keypoints_2d: keypoints,
                face_keypoints_2d: [],
                hand_left_keypoints_2d: [],
                hand_right_keypoints_2d: []
            }]
        };
        return JSON.stringify(data, null, 2);
    }

    draw() {
        const cw = this.canvas.width;
        const ch = this.canvas.height;
        this.ctx.clearRect(0, 0, cw, ch);
        this.ctx.save();
        this.ctx.translate(this.offsetX, this.offsetY);
        this.ctx.scale(this.scale, this.scale);

        // Draw connections
        this.ctx.strokeStyle = 'blue';
        this.ctx.lineWidth = 2 / this.scale;
        for (const [a, b] of POSE_CONNECTIONS) {
            if (!this.points[a] || !this.points[b]) continue;
            if (this.points[a].visible && this.points[b].visible) {
                this.ctx.beginPath();
                this.ctx.moveTo(this.points[a].x, this.points[a].y);
                this.ctx.lineTo(this.points[b].x, this.points[b].y);
                this.ctx.stroke();
            }
        }

        // Draw points
        for (let i = 0; i < this.points.length; i++) {
            const point = this.points[i];
            if (!point) continue;
            if (!point.visible) continue;
            this.ctx.fillStyle = this.dragIndex === i ? 'red' : 'green';
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, 6 / this.scale, 0, 2 * Math.PI);
            this.ctx.fill();
            // Label
            this.ctx.fillStyle = 'black';
            this.ctx.font = `${12 / this.scale}px Arial`;
            this.ctx.fillText(i.toString(), point.x + 10 / this.scale, point.y - 10 / this.scale);
        }

        this.ctx.restore();
    }

    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.offsetX) / this.scale;
        const y = (e.clientY - rect.top - this.offsetY) / this.scale;
        for (let i = 0; i < this.points.length; i++) {
            const point = this.points[i];
            if (!point || !point.visible) continue;
            const dx = x - point.x;
            const dy = y - point.y;
            const dist = Math.sqrt(dx * dx + dy * dy);
            if (dist < 12 / this.scale) {
                this.dragging = true;
                this.dragIndex = i;
                return;
            }
        }
    }

    onMouseMove(e) {
        if (!this.dragging || this.dragIndex < 0) return;
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left - this.offsetX) / this.scale;
        const y = (e.clientY - rect.top - this.offsetY) / this.scale;
        this.points[this.dragIndex].x = x;
        this.points[this.dragIndex].y = y;
        this.draw();
    }

    onMouseUp() {
        this.dragging = false;
        this.dragIndex = -1;
    }

    onWheel(e) {
        e.preventDefault();
        const zoom = e.deltaY > 0 ? 0.9 : 1.1;
        this.scale *= zoom;
        this.scale = Math.max(0.1, Math.min(4, this.scale));
        this.draw();
    }
}

// Register extension and attach canvas to existing "pose_json" widget if present.
app.registerExtension({
    name: "VNCCS.PoseManager",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Debug: printing node registration info
    console.debug("VNCCS.PoseManager: beforeRegisterNodeDef for", nodeData && nodeData.name);
    console.log("VNCCS.PoseManager: beforeRegisterNodeDef for", nodeData && nodeData.name);
        if (nodeData.name !== "VNCCS_PoseManager") return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            // Call original handler first to let Comfy create widgets
            const result = originalOnNodeCreated ? originalOnNodeCreated.apply(this, arguments) : undefined;
            try {
                console.debug("VNCCS.PoseManager: onNodeCreated - widgets count", (this.widgets && this.widgets.length) || 0);
                console.log("VNCCS.PoseManager: onNodeCreated - widgets count", (this.widgets && this.widgets.length) || 0);
                if (this.widgets && this.widgets.length) {
                    this.widgets.forEach((w, idx) => {
                        try { console.debug(`VNCCS.PoseManager: widget[${idx}] name=${w.name} type=${w.type}`); console.log(`VNCCS.PoseManager: widget[${idx}] name=${w.name} type=${w.type}`); } catch (e) {}
                    });
                }

                const poseWidget = this.widgets && this.widgets.find(w => w.name === "pose_json");
                console.debug("VNCCS.PoseManager: poseWidget found?", !!poseWidget);
                console.log("VNCCS.PoseManager: poseWidget found?", !!poseWidget);
                if (!poseWidget) {
                    // nothing to do
                    return result;
                }

                // Create canvas and a wrapper so CSS sizing works inside the node
                const canvas = document.createElement('canvas');
                canvas.width = 512;
                canvas.height = 1536;
                canvas.style.display = 'block';
                canvas.style.width = '100%';
                canvas.style.height = '300px'; // keep visible height reasonable in node UI
                canvas.style.border = '1px solid rgba(0,0,0,0.2)';

                const wrapper = document.createElement('div');
                wrapper.style.width = '100%';
                wrapper.appendChild(canvas);

                // Attach the DOM element to the widget so Comfy will render it
                poseWidget.inputEl = wrapper;
                console.debug('VNCCS.PoseManager: attached wrapper to poseWidget.inputEl');
                console.log('VNCCS.PoseManager: attached wrapper to poseWidget.inputEl');

                // Try to force Comfy to re-render this node so it picks up inputEl.
                try {
                    if (typeof this.setDirtyCanvas === 'function') {
                        this.setDirtyCanvas(true, true);
                        console.debug('VNCCS.PoseManager: called setDirtyCanvas to force re-render');
                    }
                } catch (e) {
                    console.warn('VNCCS.PoseManager: setDirtyCanvas call failed', e);
                }

                // MutationObserver: in some Comfy versions the node DOM is created later
                // than onNodeCreated; watch the document for the node element and insert
                // the wrapper once it appears. This targets in-node insertion instead of the float.
                try {
                    const observer = new MutationObserver((mutations, obs) => {
                        try {
                            const selectors = [
                                `.node[data-node-id="${this.id}"]`,
                                `#node-${this.id}`,
                                `.node[node-id="${this.id}"]`,
                                `.node[data-id="${this.id}"]`
                            ];
                            for (const s of selectors) {
                                const el = document.querySelector(s);
                                if (el) {
                                    const content = el.querySelector('.node_content') || el.querySelector('.content') || el;
                                    if (content && !content.contains(wrapper)) {
                                        content.appendChild(wrapper);
                                        console.log('VNCCS.PoseManager: MutationObserver appended wrapper into node DOM via selector', s);
                                    }
                                    obs.disconnect();
                                    return;
                                }
                            }
                            // Fallback: try to find node by title text (less reliable)
                            const nodes = Array.from(document.querySelectorAll('.node, .litegraph .node'));
                            for (const n of nodes) {
                                if (n.innerText && n.innerText.includes('VNCCS Pose Manager')) {
                                    const content = n.querySelector('.node_content') || n.querySelector('.content') || n;
                                    if (content && !content.contains(wrapper)) {
                                        content.appendChild(wrapper);
                                        console.log('VNCCS.PoseManager: MutationObserver appended wrapper into node DOM by title match');
                                    }
                                    obs.disconnect();
                                    return;
                                }
                            }
                        } catch (e) {
                            console.warn('VNCCS.PoseManager: MutationObserver handler error', e);
                        }
                    });

                    observer.observe(document.body, { childList: true, subtree: true });

                    // Safety: stop observing after a short timeout to avoid leaking observers.
                    setTimeout(() => {
                        try { observer.disconnect(); } catch (e) {}
                    }, 6000);
                } catch (e) {
                    console.warn('VNCCS.PoseManager: failed to create MutationObserver', e);
                }

                // Fallback: attempt to append wrapper directly into the node DOM if
                // Comfy doesn't render our inputEl where we expect. Retry a few times.
                (function tryAttachToNodeDom(nodeObj, wrapperEl) {
                    const selectors = [
                        `.node[data-node-id="${nodeObj.id}"]`,
                        `#node-${nodeObj.id}`,
                        `.node[node-id="${nodeObj.id}"]`,
                        `.node[data-id="${nodeObj.id}"]`
                    ];

                    let attempts = 0;
                    const maxAttempts = 6;
                    const interval = 200; // ms

                    const idInfo = nodeObj.id;

                    const timer = setInterval(() => {
                        attempts++;
                        let attached = false;
                        for (const s of selectors) {
                            try {
                                const el = document.querySelector(s);
                                if (el) {
                                    // find a content area to append to if present
                                    const content = el.querySelector('.node_content') || el.querySelector('.content') || el;
                                    // Avoid duplicate appends
                                    if (!content.contains(wrapperEl)) {
                                        content.appendChild(wrapperEl);
                                        console.log(`VNCCS.PoseManager: fallback appended wrapper to DOM selector ${s}`);
                                    } else {
                                        console.log(`VNCCS.PoseManager: wrapper already present in DOM selector ${s}`);
                                    }
                                    attached = true;
                                    break;
                                }
                            } catch (e) {
                                console.warn('VNCCS.PoseManager: selector check error', s, e);
                            }
                        }

                        if (attached || attempts >= maxAttempts) {
                            if (!attached) console.log('VNCCS.PoseManager: fallback attach failed after attempts', attempts, 'nodeId', idInfo);
                            clearInterval(timer);
                        }
                    }, interval);
                })(this, wrapper);

                // If after the retry window the wrapper still isn't attached, create
                // a persistent global docked container so the editor remains visible.
                (function ensureGlobalDock(nodeObj, wrapperEl) {
                    const timeout = 6 * 200 + 200; // maxAttempts*interval + slack
                    setTimeout(() => {
                        try {
                            if (wrapperEl.parentNode) {
                                console.debug('VNCCS.PoseManager: wrapper already attached to DOM, no global dock needed');
                                return;
                            }

                            const globalId = `vnccs-pose-manager-global-${nodeObj.id}`;
                            if (document.getElementById(globalId)) {
                                console.debug('VNCCS.PoseManager: global dock already exists for node', nodeObj.id);
                                // move our wrapper into existing container if it's not there
                                const existing = document.getElementById(globalId).querySelector('.vnccs-pose-wrapper');
                                if (existing && existing !== wrapperEl) {
                                    document.getElementById(globalId).querySelector('.vnccs-pose-wrapper')?.replaceWith(wrapperEl);
                                } else if (!existing) {
                                    document.getElementById(globalId).appendChild(wrapperEl);
                                }
                                return;
                            }

                            // Create container
                            const container = document.createElement('div');
                            container.id = globalId;
                            container.className = 'vnccs-pose-global-dock';
                            Object.assign(container.style, {
                                width: '480px',
                                position: 'fixed',
                                right: '20px',
                                top: '20px',
                                zIndex: 99999,
                                height: '360px',
                                background: 'white',
                                boxShadow: 'rgba(0, 0, 0, 0.25) 0px 6px 18px',
                                overflow: 'auto',
                                pointerEvents: 'auto',
                                borderRadius: '6px',
                                display: 'flex',
                                flexDirection: 'column'
                            });

                            // Header with controls
                            const header = document.createElement('div');
                            header.style.display = 'flex';
                            header.style.alignItems = 'center';
                            header.style.justifyContent = 'space-between';
                            header.style.padding = '6px 8px';
                            header.style.borderBottom = '1px solid rgba(0,0,0,0.06)';

                            const title = document.createElement('div');
                            title.textContent = `VNCCS Pose Manager — node ${nodeObj.id}`;
                            title.style.fontSize = '13px';
                            title.style.fontWeight = '600';

                            const controls = document.createElement('div');

                            const dockBtn = document.createElement('button');
                            dockBtn.textContent = 'Dock';
                            dockBtn.title = 'Try to dock back into node UI';
                            dockBtn.style.marginRight = '6px';

                            const closeBtn = document.createElement('button');
                            closeBtn.textContent = 'Close';
                            closeBtn.title = 'Close this floating editor';

                            controls.appendChild(dockBtn);
                            controls.appendChild(closeBtn);

                            header.appendChild(title);
                            header.appendChild(controls);

                            // Body area for the wrapper
                            const body = document.createElement('div');
                            body.style.flex = '1 1 auto';
                            body.style.padding = '8px';
                            body.className = 'vnccs-pose-body';
                            wrapperEl.classList.add('vnccs-pose-wrapper');
                            wrapperEl.style.width = '100%';
                            wrapperEl.style.height = '100%';

                            body.appendChild(wrapperEl);

                            container.appendChild(header);
                            container.appendChild(body);

                            document.body.appendChild(container);
                            console.log('VNCCS.PoseManager: created global dock container for visibility', globalId);

                            // Close handler
                            closeBtn.addEventListener('click', () => {
                                try { container.remove(); console.log('VNCCS.PoseManager: global dock closed by user'); } catch (e) { }
                            });

                            // Dock handler: attempt to find node DOM and move wrapper back
                            dockBtn.addEventListener('click', () => {
                                try {
                                    const selectors = [
                                        `.node[data-node-id="${nodeObj.id}"]`,
                                        `#node-${nodeObj.id}`,
                                        `.node[node-id="${nodeObj.id}"]`,
                                        `.node[data-id="${nodeObj.id}"]`
                                    ];
                                    let moved = false;
                                    for (const s of selectors) {
                                        const el = document.querySelector(s);
                                        if (!el) continue;
                                        const content = el.querySelector('.node_content') || el.querySelector('.content') || el;
                                        if (content && !content.contains(wrapperEl)) {
                                            content.appendChild(wrapperEl);
                                            moved = true;
                                            break;
                                        }
                                    }
                                    if (moved) {
                                        container.remove();
                                        console.log('VNCCS.PoseManager: docked global editor back into node', nodeObj.id);
                                    } else {
                                        console.log('VNCCS.PoseManager: dock attempt failed; node DOM not found');
                                    }
                                } catch (e) { console.warn('VNCCS.PoseManager: dock error', e); }
                            });

                        } catch (e) {
                            console.warn('VNCCS.PoseManager: ensureGlobalDock error', e);
                        }
                    }, timeout);
                })(this, wrapper);

                // Create editor and wire up serialization/callbacks
                const editor = new PoseEditor(canvas);
                poseWidget.editor = editor;

                // Load initial value if present
                if (poseWidget.value) {
                    try { editor.loadPose(poseWidget.value); console.debug('VNCCS.PoseManager: loaded initial pose value'); console.log('VNCCS.PoseManager: loaded initial pose value'); } catch (e) { console.warn('loadPose error', e); }
                }

                // When Comfy serializes the widget value, return our JSON
                poseWidget.serializeValue = function() {
                    try { const v = editor.getPoseJSON(); console.debug('VNCCS.PoseManager: serializeValue called'); console.log('VNCCS.PoseManager: serializeValue called'); return v; } catch (e) { console.warn('serializeValue error', e); return '{}'; }
                };

                // When external code updates the widget value, load it into the editor
                poseWidget.callback = function(value) {
                    try { editor.loadPose(value); console.debug('VNCCS.PoseManager: callback loaded new value'); console.log('VNCCS.PoseManager: callback loaded new value'); } catch (e) { console.warn('Pose widget callback load error', e); }
                };

            } catch (e) {
                console.error('VNCCS.PoseManager onNodeCreated error', e);
            }

            return result;
        };
    }
});