/**
 * VNCCS Debug3 - DOM-Based Three.js Widget
 * 
 * Implementation adapted from web/pose_editor_3d.js
 * Uses SkinnedMesh instead of stick figure, but same Gizmo/Orbit logic.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// --- From pose_editor_3d.js ---
const THREE_VERSION = "0.160.0";
const THREE_SOURCES = {
    core: `https://esm.sh/three@${THREE_VERSION}?dev`,
    orbit: `https://esm.sh/three@${THREE_VERSION}/examples/jsm/controls/OrbitControls?dev`,
    transform: `https://esm.sh/three@${THREE_VERSION}/examples/jsm/controls/TransformControls?dev`
};

const ThreeModuleLoader = {
    promise: null,
    async load() {
        if (!this.promise) {
            this.promise = Promise.all([
                import(THREE_SOURCES.core),
                import(THREE_SOURCES.orbit),
                import(THREE_SOURCES.transform)
            ]).then(([core, orbit, transform]) => ({
                THREE: core,
                OrbitControls: orbit.OrbitControls,
                TransformControls: transform.TransformControls
            }));
        }
        return this.promise;
    }
};

const STYLE = `
.vnccs-debug3-container {
    width: 100%;
    height: 100%;
    background: #1a1a2e;
    overflow: hidden;
    position: relative;
    outline: none;
}
.vnccs-hint {
    position: absolute;
    bottom: 5px;
    left: 5px;
    color: #888;
    font-size: 10px;
    pointer-events: none;
    font-family: sans-serif;
    opacity: 0.7;
}
`;
const styleEl = document.createElement("style");
styleEl.textContent = STYLE;
document.head.appendChild(styleEl);

class MakeHumanViewer {
    constructor(canvas) {
        this.canvas = canvas;
        this.width = canvas.width || 680;
        this.height = canvas.height || 550;

        this.THREE = null;
        this.OrbitControls = null;
        this.TransformControls = null;

        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.orbit = null;
        this.transform = null;

        this.skinnedMesh = null;
        this.skeleton = null;
        this.boneList = [];
        this.selectedBone = null;

        // Highlighting
        this.jointMarkers = [];       // Spheres at bone heads
        this.jointMarkersGroup = null;
        this.vertexWeightsPerBone = {}; // {boneIdx: Float32Array of weights per vertex}
        this.baseColor = new Float32Array(0); // Original vertex colors

        this.initialized = false;
        this.init();
    }

    async init() {
        try {
            const modules = await ThreeModuleLoader.load();
            this.THREE = modules.THREE;
            this.OrbitControls = modules.OrbitControls;
            this.TransformControls = modules.TransformControls;

            this.setupScene();
            this.initialized = true;
            console.log('VNCCS Debug3: Initialized (pose_editor style)');

            this.animate();

            if (this.requestModelLoad) {
                this.requestModelLoad();
            }
        } catch (e) {
            console.error('VNCCS Debug3: Init failed', e);
        }
    }

    setupScene() {
        const THREE = this.THREE;

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e); // Keep Debug theme

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.1, 1000);
        this.camera.position.set(0, 10, 30);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Orbit Controls (Configs from pose_editor)
        this.orbit = new this.OrbitControls(this.camera, this.canvas);
        this.orbit.target.set(0, 10, 0);
        this.orbit.enableDamping = true;
        this.orbit.dampingFactor = 0.12;
        this.orbit.rotateSpeed = 0.95;
        this.orbit.update();

        // Transform Controls (Gizmo)
        this.transform = new this.TransformControls(this.camera, this.canvas);
        this.transform.setMode("rotate"); // Default to rotate for bones
        this.transform.setSpace("local"); // Bones rotate locally
        this.transform.setSize(0.8);
        this.scene.add(this.transform);

        // Gizmo Logic
        this.transform.addEventListener("dragging-changed", (event) => {
            this.orbit.enabled = !event.value;
        });

        // Lights
        const light = new THREE.DirectionalLight(0xffffff, 2);
        light.position.set(10, 20, 30);
        this.scene.add(light);
        this.scene.add(new THREE.AmbientLight(0x505050));

        this.scene.add(new THREE.GridHelper(20, 20, 0x0f3460, 0x0f3460)); // Debug grid

        // Events
        this.canvas.addEventListener("pointerdown", (e) => this.handlePointerDown(e));
        // Note: OrbitControls/TransformControls handle their own events too
    }

    animate() {
        if (!this.initialized) return;
        requestAnimationFrame(() => this.animate());
        this.orbit.update();
        if (this.renderer) this.renderer.render(this.scene, this.camera);
    }

    handlePointerDown(e) {
        if (!this.initialized || !this.skinnedMesh) return;
        if (e.button !== 0) return; // Left click only

        // CRITICAL: Don't select bones if gizmo is being used
        if (this.transform.dragging) return;

        // Also check if we're clicking on the gizmo itself
        // TransformControls has its own raycaster, but we can check axis property
        if (this.transform.axis) return; // Gizmo is hovered

        const rect = this.canvas.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

        const raycaster = new this.THREE.Raycaster();
        raycaster.setFromCamera(new this.THREE.Vector2(x, y), this.camera);

        const intersects = raycaster.intersectObject(this.skinnedMesh, true);

        if (intersects.length > 0) {
            const point = intersects[0].point;
            let nearest = null;
            let minD = Infinity;

            const wPos = new this.THREE.Vector3();
            for (const b of this.boneList) {
                b.getWorldPosition(wPos);
                const d = point.distanceTo(wPos);
                if (d < minD) { minD = d; nearest = b; }
            }

            if (nearest && minD < 2.0) { // Click near bone head
                this.selectBone(nearest);
            }
        } else {
            // Click on background - deselect
            this.deselectBone();
        }
    }

    deselectBone() {
        if (!this.selectedBone) return;
        this.selectedBone = null;
        this.transform.detach();
        this.updateHighlight();
        console.log("Deselected");
    }

    selectBone(bone) {
        if (this.selectedBone === bone) return;
        this.selectedBone = bone;
        console.log("Selected:", bone.name);
        this.transform.attach(bone);
        this.updateHighlight();
    }

    updateHighlight() {
        const THREE = this.THREE;
        if (!this.skinnedMesh) return;

        // Update joint markers only (mesh highlighting disabled)
        const boneIdx = this.selectedBone ? this.boneList.indexOf(this.selectedBone) : -1;
        for (let i = 0; i < this.jointMarkers.length; i++) {
            const marker = this.jointMarkers[i];
            if (i === boneIdx) {
                marker.material.color.setHex(0x00ffff); // Cyan for selected
                marker.scale.setScalar(1.8); // Bigger when selected
            } else {
                marker.material.color.setHex(0xffaa00); // Orange for others
                marker.scale.setScalar(1.0);
            }
        }
    }

    resize(w, h) {
        this.width = w;
        this.height = h;
        this.canvas.width = w;
        this.canvas.height = h;
        if (this.renderer) this.renderer.setSize(w, h);
        if (this.camera) {
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
        }
    }

    loadData(data) {
        if (!this.initialized || !data || !data.vertices || !data.bones) return;
        const THREE = this.THREE;

        // Clean
        if (this.skinnedMesh) {
            this.scene.remove(this.skinnedMesh);
            this.skinnedMesh.geometry.dispose();
            this.skinnedMesh.material.dispose();
            if (this.skeletonHelper) this.scene.remove(this.skeletonHelper);
        }
        if (this.jointMarkersGroup) {
            this.scene.remove(this.jointMarkersGroup);
            this.jointMarkers = [];
        }

        // Geom
        const vertices = new Float32Array(data.vertices);
        const indices = new Uint32Array(data.indices);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();

        // Center Cam
        geometry.computeBoundingBox();
        const center = geometry.boundingBox.getCenter(new THREE.Vector3());
        const size = geometry.boundingBox.getSize(new THREE.Vector3());
        if (size.length() > 0.1 && this.orbit) {
            this.orbit.target.copy(center);
            const dist = size.length() * 1.5;
            // Move cam back
            const dir = this.camera.position.clone().sub(this.orbit.target).normalize();
            if (dir.lengthSq() < 0.001) dir.set(0, 0, 1);
            this.camera.position.copy(this.orbit.target).add(dir.multiplyScalar(dist));
            this.orbit.update();
        }

        // Bones
        this.bones = {};
        this.boneList = [];
        const rootBones = [];

        for (const bData of data.bones) {
            const bone = new THREE.Bone();
            bone.name = bData.name;
            bone.userData = { headPos: bData.headPos, parentName: bData.parent };
            bone.position.set(bData.headPos[0], bData.headPos[1], bData.headPos[2]);
            this.bones[bone.name] = bone;
            this.boneList.push(bone);
        }

        for (const bone of this.boneList) {
            const pName = bone.userData.parentName;
            if (pName && this.bones[pName]) {
                const parent = this.bones[pName];
                parent.add(bone);
                const pHead = parent.userData.headPos;
                const cHead = bone.userData.headPos;
                bone.position.set(cHead[0] - pHead[0], cHead[1] - pHead[1], cHead[2] - pHead[2]);
            } else {
                rootBones.push(bone);
            }
        }

        this.skeleton = new THREE.Skeleton(this.boneList);

        // Weights
        const vCount = vertices.length / 3;
        const skinInds = new Float32Array(vCount * 4);
        const skinWgts = new Float32Array(vCount * 4);

        // Build bone head positions array for orphan vertex fallback
        const boneHeads = this.boneList.map(b => b.userData.headPos);

        if (data.weights) {
            const vWeights = new Array(vCount).fill(null).map(() => []);
            const boneMap = {};
            this.boneList.forEach((b, i) => boneMap[b.name] = i);

            for (const [bName, wData] of Object.entries(data.weights)) {
                if (boneMap[bName] === undefined) continue;
                const bIdx = boneMap[bName];
                const wInds = wData.indices;
                const wVals = wData.weights;
                for (let i = 0; i < wInds.length; i++) {
                    const vi = wInds[i];
                    if (vi < vCount) vWeights[vi].push({ b: bIdx, w: wVals[i] });
                }
            }

            for (let v = 0; v < vCount; v++) {
                const vw = vWeights[v];
                vw.sort((a, b) => b.w - a.w);
                let tot = 0;
                for (let i = 0; i < 4 && i < vw.length; i++) {
                    skinInds[v * 4 + i] = vw[i].b;
                    skinWgts[v * 4 + i] = vw[i].w;
                    tot += vw[i].w;
                }
                if (tot > 0) {
                    // Normalize
                    for (let i = 0; i < 4; i++) skinWgts[v * 4 + i] /= tot;
                } else {
                    // Orphan vertex: find nearest bone
                    const vx = vertices[v * 3];
                    const vy = vertices[v * 3 + 1];
                    const vz = vertices[v * 3 + 2];
                    let nearestIdx = 0;
                    let minDistSq = Infinity;
                    for (let bi = 0; bi < boneHeads.length; bi++) {
                        const h = boneHeads[bi];
                        const dx = vx - h[0], dy = vy - h[1], dz = vz - h[2];
                        const dSq = dx * dx + dy * dy + dz * dz;
                        if (dSq < minDistSq) { minDistSq = dSq; nearestIdx = bi; }
                    }
                    skinInds[v * 4] = nearestIdx;
                    skinWgts[v * 4] = 1;
                }
            }
        }

        geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinInds, 4));
        geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWgts, 4));

        // Store per-bone vertex weights for highlighting
        this.vertexWeightsPerBone = {};
        for (let bi = 0; bi < this.boneList.length; bi++) {
            this.vertexWeightsPerBone[bi] = new Float32Array(vCount);
        }
        for (let v = 0; v < vCount; v++) {
            for (let slot = 0; slot < 4; slot++) {
                const bIdx = skinInds[v * 4 + slot];
                const w = skinWgts[v * 4 + slot];
                if (w > 0 && this.vertexWeightsPerBone[bIdx]) {
                    this.vertexWeightsPerBone[bIdx][v] += w;
                }
            }
        }

        // Skin-colored Phong material (matching character_studio.js)
        const material = new THREE.MeshPhongMaterial({
            color: 0xd4a574,        // Warm skin tone base
            specular: 0x332211,     // Subtle warm specular
            shininess: 15,          // Soft skin-like reflection
            side: THREE.DoubleSide
        });

        this.skinnedMesh = new THREE.SkinnedMesh(geometry, material);
        rootBones.forEach(b => this.skinnedMesh.add(b));
        this.skinnedMesh.bind(this.skeleton);
        this.scene.add(this.skinnedMesh);

        this.skeletonHelper = new THREE.SkeletonHelper(this.skinnedMesh);
        this.scene.add(this.skeletonHelper);

        // Create Joint Markers (spheres at bone heads)
        // Two sizes: normal (0.12) and smaller for fingers (0.06)
        this.jointMarkersGroup = new THREE.Group();
        const sphereGeoNormal = new THREE.SphereGeometry(0.12, 8, 6);
        const sphereGeoFinger = new THREE.SphereGeometry(0.06, 6, 4); // 2x smaller for fingers

        const fingerPatterns = ['finger', 'thumb', 'index', 'middle', 'ring', 'pinky', 'f_'];

        for (let i = 0; i < this.boneList.length; i++) {
            const bone = this.boneList[i];
            const boneName = bone.name.toLowerCase();

            // Check if this is a finger bone
            const isFinger = fingerPatterns.some(p => boneName.includes(p));
            const geo = isFinger ? sphereGeoFinger : sphereGeoNormal;

            const mat = new THREE.MeshBasicMaterial({
                color: 0xffaa00,
                transparent: true,
                opacity: 0.9,
                depthTest: false  // SHOW THROUGH MESH
            });
            const sphere = new THREE.Mesh(geo, mat);
            sphere.userData.boneIndex = i;
            sphere.renderOrder = 999; // Render on top
            // Attach to bone so it follows transforms
            bone.add(sphere);
            sphere.position.set(0, 0, 0); // At bone origin (head)
            this.jointMarkers.push(sphere);
        }
        this.scene.add(this.jointMarkersGroup);
    }

    getPostures() {
        const res = {};
        for (const b of this.boneList) {
            const rot = b.rotation;
            if (Math.abs(rot.x) > 1e-4 || Math.abs(rot.y) > 1e-4 || Math.abs(rot.z) > 1e-4) {
                res[b.name] = [rot.x * 180 / Math.PI, rot.y * 180 / Math.PI, rot.z * 180 / Math.PI];
            }
        }
        return res;
    }
}

app.registerExtension({
    name: "VNCCS.Debug3",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "VNCCS_Debug3") {
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onCreated) onCreated.apply(this, arguments);

                this.setSize([700, 650]);

                const container = document.createElement("div");
                container.className = "vnccs-debug3-container";

                const canvas = document.createElement("canvas");
                canvas.tabIndex = 1;
                container.appendChild(canvas);

                const hint = document.createElement("div");
                hint.className = "vnccs-hint";
                hint.innerText = "Select Bone | Rotate Gizmo | Orbit Camera (Right Click)";
                container.appendChild(hint);

                this.addDOMWidget("debug3_ui", "ui", container, {
                    serialize: false,
                    hideOnZoom: false
                });

                this.viewer = new MakeHumanViewer(canvas);

                const load = () => {
                    api.fetchApi("/vnccs/character_studio/update_preview", {
                        method: "POST", body: "{}"
                    }).then(r => r.json()).then(d => this.viewer.loadData(d));
                };

                this.viewer.requestModelLoad = load;

                this.addWidget("button", "Load Model", null, load);
                this.addWidget("button", "Apply Pose", null, () => {
                    const pose = this.viewer.getPostures();
                    api.fetchApi("/vnccs/character_studio/update_preview", {
                        method: "POST",
                        body: JSON.stringify({ manual_pose: pose, relative: false })
                    }).then(r => r.json()).then(d => console.log("Applied", d));
                });

                this._container = container;
            };

            nodeType.prototype.onResize = function (size) {
                if (this.viewer && this._container) {
                    const w = Math.max(200, size[0] - 20);
                    const h = Math.max(200, size[1] - 80);
                    this._container.style.width = w + "px";
                    this._container.style.height = h + "px";
                    this.viewer.resize(w, h);
                }
            };
        }
    }
});
