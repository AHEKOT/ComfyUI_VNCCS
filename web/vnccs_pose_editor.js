/**
 * VNCCS Mannequin Pose Editor - Embedded in ComfyUI Node Widget
 * 
 * This embeds a Three.js mannequin directly in the node widget,
 * using the same approach as vnccs_debug2.js.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Three.js CDN URL
const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";

// Bone name mapping: mannequin.js style -> MakeHuman
const BONE_MAPPING = {
    'body': 'root',
    'pelvis': 'pelvis',
    'torso': 'spine03',
    'neck': 'neck01',
    'head': 'head',
    'l_arm': 'upperarm01.L',
    'l_elbow': 'lowerarm01.L',
    'l_wrist': 'wrist.L',
    'r_arm': 'upperarm01.R',
    'r_elbow': 'lowerarm01.R',
    'r_wrist': 'wrist.R',
    'l_leg': 'upperleg01.L',
    'l_knee': 'lowerleg01.L',
    'l_ankle': 'foot.L',
    'r_leg': 'upperleg01.R',
    'r_knee': 'lowerleg01.R',
    'r_ankle': 'foot.R',
};

// Simple mannequin joint structure
const JOINTS = [
    { name: 'body', parent: null, pos: [0, 0, 0], size: [2, 1, 1] },
    { name: 'pelvis', parent: 'body', pos: [0, 1, 0], size: [2, 1, 1] },
    { name: 'torso', parent: 'pelvis', pos: [0, 2, 0], size: [2, 3, 1] },
    { name: 'neck', parent: 'torso', pos: [0, 2, 0], size: [0.5, 0.5, 0.5] },
    { name: 'head', parent: 'neck', pos: [0, 1, 0], size: [1.2, 1.5, 1.2] },
    // Left arm
    { name: 'l_arm', parent: 'torso', pos: [1.8, 1.5, 0], size: [2, 0.5, 0.5] },
    { name: 'l_elbow', parent: 'l_arm', pos: [2, 0, 0], size: [1.8, 0.4, 0.4] },
    { name: 'l_wrist', parent: 'l_elbow', pos: [1.8, 0, 0], size: [0.8, 0.3, 0.5] },
    // Right arm
    { name: 'r_arm', parent: 'torso', pos: [-1.8, 1.5, 0], size: [2, 0.5, 0.5] },
    { name: 'r_elbow', parent: 'r_arm', pos: [-2, 0, 0], size: [1.8, 0.4, 0.4] },
    { name: 'r_wrist', parent: 'r_elbow', pos: [-1.8, 0, 0], size: [0.8, 0.3, 0.5] },
    // Left leg
    { name: 'l_leg', parent: 'pelvis', pos: [0.8, -0.5, 0], size: [0.6, 3, 0.6] },
    { name: 'l_knee', parent: 'l_leg', pos: [0, -3, 0], size: [0.5, 3, 0.5] },
    { name: 'l_ankle', parent: 'l_knee', pos: [0, -3, 0], size: [0.4, 0.3, 0.8] },
    // Right leg
    { name: 'r_leg', parent: 'pelvis', pos: [-0.8, -0.5, 0], size: [0.6, 3, 0.6] },
    { name: 'r_knee', parent: 'r_leg', pos: [0, -3, 0], size: [0.5, 3, 0.5] },
    { name: 'r_ankle', parent: 'r_knee', pos: [0, -3, 0], size: [0.4, 0.3, 0.8] },
];

class MannequinViewer {
    constructor(width, height) {
        this.width = width;
        this.height = height;
        this.canvas = document.createElement('canvas');
        this.canvas.width = width;
        this.canvas.height = height;

        this.THREE = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.joints = {};
        this.selectedJoint = null;
        this.isDragging = false;
        this.mouse = { x: 0, y: 0 };
        this.lastMouse = { x: 0, y: 0 };
        this.cameraRotation = { x: 0.3, y: 0 };
        this.cameraDistance = 25;

        this.initialized = false;
        this.dirty = true;

        this.init();
    }

    async init() {
        try {
            // Dynamic import of Three.js
            this.THREE = await import(THREE_CDN);
            this.setupScene();
            this.createMannequin();
            this.initialized = true;
            this.render();
            console.log('VNCCS Mannequin: Initialized');
        } catch (e) {
            console.error('VNCCS Mannequin: Failed to init Three.js', e);
        }
    }

    setupScene() {
        const THREE = this.THREE;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.1, 1000);
        this.updateCamera();

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setSize(this.width, this.height);

        // Lights
        const ambient = new THREE.AmbientLight(0x404040, 2);
        this.scene.add(ambient);

        const directional = new THREE.DirectionalLight(0xffffff, 1.5);
        directional.position.set(10, 20, 10);
        this.scene.add(directional);

        // Grid
        const grid = new THREE.GridHelper(20, 20, 0x0f3460, 0x0f3460);
        this.scene.add(grid);
    }

    updateCamera() {
        const x = this.cameraDistance * Math.sin(this.cameraRotation.y) * Math.cos(this.cameraRotation.x);
        const y = this.cameraDistance * Math.sin(this.cameraRotation.x) + 8;
        const z = this.cameraDistance * Math.cos(this.cameraRotation.y) * Math.cos(this.cameraRotation.x);
        this.camera.position.set(x, y, z);
        this.camera.lookAt(0, 8, 0);
    }

    createMannequin() {
        const THREE = this.THREE;

        const jointMaterial = new THREE.MeshPhongMaterial({ color: 0x4a90d9 });
        const selectedMaterial = new THREE.MeshPhongMaterial({ color: 0xe94560, emissive: 0x440000 });

        for (const jdef of JOINTS) {
            const geometry = new THREE.BoxGeometry(...jdef.size);
            const mesh = new THREE.Mesh(geometry, jointMaterial.clone());
            mesh.position.set(...jdef.pos);
            mesh.userData = {
                name: jdef.name,
                rotation: [0, 0, 0],
                defaultMaterial: mesh.material,
                selectedMaterial: selectedMaterial.clone()
            };

            if (jdef.parent && this.joints[jdef.parent]) {
                this.joints[jdef.parent].add(mesh);
            } else {
                mesh.position.y += 8; // Raise the mannequin
                this.scene.add(mesh);
            }

            this.joints[jdef.name] = mesh;
        }
    }

    render() {
        if (!this.initialized || !this.renderer) return;
        this.renderer.render(this.scene, this.camera);
    }

    selectJoint(name) {
        // Deselect previous
        if (this.selectedJoint && this.joints[this.selectedJoint]) {
            const j = this.joints[this.selectedJoint];
            j.material = j.userData.defaultMaterial;
        }

        this.selectedJoint = name;

        // Select new
        if (name && this.joints[name]) {
            const j = this.joints[name];
            j.material = j.userData.selectedMaterial;
        }

        this.render();
    }

    pickJoint(screenX, screenY) {
        if (!this.THREE || !this.camera) return null;

        const THREE = this.THREE;
        const mouse = new THREE.Vector2(
            (screenX / this.width) * 2 - 1,
            -(screenY / this.height) * 2 + 1
        );

        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);

        const meshes = Object.values(this.joints);
        const intersects = raycaster.intersectObjects(meshes, true);

        if (intersects.length > 0) {
            let obj = intersects[0].object;
            while (obj && !obj.userData.name) {
                obj = obj.parent;
            }
            return obj?.userData.name || null;
        }
        return null;
    }

    rotateSelectedJoint(axis, angleDeg) {
        if (!this.selectedJoint || !this.joints[this.selectedJoint]) return;

        const joint = this.joints[this.selectedJoint];
        const rad = angleDeg * Math.PI / 180;

        if (axis === 'x') joint.rotation.x += rad;
        if (axis === 'y') joint.rotation.y += rad;
        if (axis === 'z') joint.rotation.z += rad;

        // Store for export
        joint.userData.rotation = [
            joint.rotation.x * 180 / Math.PI,
            joint.rotation.y * 180 / Math.PI,
            joint.rotation.z * 180 / Math.PI
        ];

        this.render();
    }

    // Screen-space IK: try +/- rotation, keep if closer to mouse
    screenSpaceIK(mouseX, mouseY) {
        if (!this.selectedJoint || !this.joints[this.selectedJoint]) return;

        const joint = this.joints[this.selectedJoint];
        const THREE = this.THREE;

        // Get joint world position
        const worldPos = new THREE.Vector3();
        joint.getWorldPosition(worldPos);

        // Project to screen
        const projected = worldPos.clone().project(this.camera);
        const screenX = (projected.x + 1) / 2 * this.width;
        const screenY = (1 - projected.y) / 2 * this.height;

        const distBefore = Math.sqrt((screenX - mouseX) ** 2 + (screenY - mouseY) ** 2);

        const step = 3 * Math.PI / 180; // 3 degrees
        let bestDist = distBefore;
        let bestAxis = null;
        let bestDir = 0;

        for (const axis of ['x', 'y', 'z']) {
            for (const dir of [1, -1]) {
                // Try rotation
                joint.rotation[axis] += step * dir;
                joint.updateMatrixWorld(true);

                // Measure new distance
                joint.getWorldPosition(worldPos);
                const newProj = worldPos.clone().project(this.camera);
                const newX = (newProj.x + 1) / 2 * this.width;
                const newY = (1 - newProj.y) / 2 * this.height;
                const distAfter = Math.sqrt((newX - mouseX) ** 2 + (newY - mouseY) ** 2);

                // Revert
                joint.rotation[axis] -= step * dir;
                joint.updateMatrixWorld(true);

                if (distAfter < bestDist - 0.5) { // threshold to avoid jitter
                    bestDist = distAfter;
                    bestAxis = axis;
                    bestDir = dir;
                }
            }
        }

        // Apply best rotation if found
        if (bestAxis) {
            joint.rotation[bestAxis] += step * bestDir;
            joint.userData.rotation = [
                joint.rotation.x * 180 / Math.PI,
                joint.rotation.y * 180 / Math.PI,
                joint.rotation.z * 180 / Math.PI
            ];
            this.render();
        }
    }

    getPosture() {
        const result = {};
        for (const [name, joint] of Object.entries(this.joints)) {
            result[name] = joint.userData.rotation || [0, 0, 0];
        }
        return result;
    }

    getMakeHumanPose() {
        const posture = this.getPosture();
        const result = {};

        for (const [mannequinName, mhName] of Object.entries(BONE_MAPPING)) {
            if (posture[mannequinName]) {
                result[mhName] = posture[mannequinName];
            }
        }

        return result;
    }

    resetPose() {
        for (const joint of Object.values(this.joints)) {
            joint.rotation.set(0, 0, 0);
            joint.userData.rotation = [0, 0, 0];
        }
        this.render();
    }
}

// Register ComfyUI Extension
app.registerExtension({
    name: "VNCCS.MannequinPoseEditor",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "VNCCS_PoseEditor") {
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onCreated) onCreated.apply(this, arguments);

                this.setSize([600, 650]);
                this.viewer = new MannequinViewer(580, 500);
                this.viewerRegion = null;
                this.lastPos = [0, 0];
                this.dragMode = null; // 'joint', 'camera'

                // Buttons
                this.addWidget("button", "Apply to MakeHuman", null, () => {
                    if (!this.viewer.initialized) return;
                    const pose = this.viewer.getMakeHumanPose();
                    console.log('VNCCS: Applying pose:', pose);
                    api.fetchApi("/vnccs/character_studio/update_preview", {
                        method: "POST",
                        body: JSON.stringify({ manual_pose: pose, relative: false })
                    }).then(r => r.json()).then(d => {
                        console.log('VNCCS: Pose applied', d);
                    });
                });

                this.addWidget("button", "Reset Pose", null, () => {
                    this.viewer.resetPose();
                });
            };

            nodeType.prototype.onDrawForeground = function (ctx) {
                if (!this.viewer || !this.viewer.canvas) return;

                const margin = 10;
                const y = 80; // Below widgets
                const w = this.size[0] - margin * 2;
                const h = this.size[1] - y - margin;

                this.viewerRegion = { x: margin, y: y, w: w, h: h };

                if (this.viewer.initialized && this.viewer.canvas) {
                    ctx.drawImage(this.viewer.canvas, margin, y, w, h);
                } else {
                    ctx.fillStyle = "#1a1a2e";
                    ctx.fillRect(margin, y, w, h);
                    ctx.fillStyle = "#888";
                    ctx.font = "14px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("Loading Three.js...", this.size[0] / 2, y + h / 2);
                }

                // Status
                if (this.viewer.selectedJoint) {
                    ctx.fillStyle = "#e94560";
                    ctx.textAlign = "left";
                    ctx.fillText("Selected: " + this.viewer.selectedJoint, margin + 5, y + 15);
                }

                // Instructions
                ctx.fillStyle = "#666";
                ctx.font = "11px Arial";
                ctx.textAlign = "center";
                ctx.fillText("Left-click: select & rotate | Right-click: camera", this.size[0] / 2, this.size[1] - 5);
            };

            nodeType.prototype.onMouseDown = function (e, pos) {
                if (!this.viewerRegion || !this.viewer.initialized) return false;

                const lx = pos[0] - this.viewerRegion.x;
                const ly = pos[1] - this.viewerRegion.y;

                if (lx < 0 || ly < 0 || lx > this.viewerRegion.w || ly > this.viewerRegion.h) return false;

                // Scale to viewer coords
                const vx = lx / this.viewerRegion.w * this.viewer.width;
                const vy = ly / this.viewerRegion.h * this.viewer.height;

                this.lastPos = [pos[0], pos[1]];
                this.viewer.mouse = { x: vx, y: vy };

                if (e.button === 0) { // Left click
                    const joint = this.viewer.pickJoint(vx, vy);
                    if (joint) {
                        this.viewer.selectJoint(joint);
                        this.dragMode = 'joint';
                    } else {
                        this.viewer.selectJoint(null);
                        this.dragMode = 'camera';
                    }
                } else {
                    this.dragMode = 'camera';
                }

                this.viewer.dirty = true;
                return true;
            };

            nodeType.prototype.onMouseMove = function (e, pos) {
                if (!this.dragMode || !this.viewer.initialized) return;

                const dx = pos[0] - this.lastPos[0];
                const dy = pos[1] - this.lastPos[1];

                if (this.dragMode === 'joint' && this.viewer.selectedJoint) {
                    // Scale to viewer coords
                    const lx = pos[0] - this.viewerRegion.x;
                    const ly = pos[1] - this.viewerRegion.y;
                    const vx = lx / this.viewerRegion.w * this.viewer.width;
                    const vy = ly / this.viewerRegion.h * this.viewer.height;

                    this.viewer.screenSpaceIK(vx, vy);

                } else if (this.dragMode === 'camera') {
                    this.viewer.cameraRotation.y += dx * 0.01;
                    this.viewer.cameraRotation.x += dy * 0.01;
                    this.viewer.cameraRotation.x = Math.max(-1.5, Math.min(1.5, this.viewer.cameraRotation.x));
                    this.viewer.updateCamera();
                    this.viewer.render();
                }

                this.lastPos = [pos[0], pos[1]];
                this.viewer.dirty = true;
            };

            nodeType.prototype.onMouseUp = function () {
                this.dragMode = null;
            };
        }
    }
});
