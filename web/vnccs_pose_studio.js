/**
 * VNCCS Pose Studio - Combined mesh editor and multi-pose generator
 * 
 * Combines Character Studio sliders, dynamic pose tabs, and Debug3 gizmo controls.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// === Three.js Module Loader (from Debug3) ===
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

// === Styles ===
const STYLES = `
.vnccs-pose-studio {
    display: flex;
    flex-direction: row;
    width: 100%;
    height: 100%;
    background: #1a1a2e;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    color: #ccc;
    overflow: hidden;
}

/* Left Panel - Sliders */
.vnccs-ps-left {
    width: 200px;
    min-width: 200px;
    padding: 10px;
    border-right: 1px solid #333;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.vnccs-ps-slider-group {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.vnccs-ps-slider-label {
    font-size: 11px;
    color: #888;
}

.vnccs-ps-slider {
    width: 100%;
    height: 20px;
    -webkit-appearance: none;
    background: #333;
    border-radius: 4px;
    cursor: pointer;
}

.vnccs-ps-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 14px;
    height: 14px;
    background: #4a9;
    border-radius: 50%;
    cursor: pointer;
}

/* Right Panel - 3D Editor */
.vnccs-ps-right {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Tab Bar */
.vnccs-ps-tabs {
    display: flex;
    align-items: center;
    padding: 5px 10px;
    background: #222;
    gap: 5px;
    border-bottom: 1px solid #333;
    overflow-x: auto;
}

.vnccs-ps-tab {
    padding: 6px 16px;
    background: #333;
    border: none;
    border-radius: 4px 4px 0 0;
    color: #888;
    cursor: pointer;
    font-size: 12px;
    white-space: nowrap;
}

.vnccs-ps-tab.active {
    background: #1a1a2e;
    color: #4a9;
}

.vnccs-ps-tab-add {
    padding: 6px 10px;
    background: #2a2a3e;
    border: 1px dashed #444;
    border-radius: 4px;
    color: #666;
    cursor: pointer;
    font-size: 14px;
}

.vnccs-ps-tab-add:hover {
    background: #3a3a4e;
    color: #888;
}

/* 3D Canvas Container */
.vnccs-ps-canvas-container {
    flex: 1;
    position: relative;
    overflow: hidden;
}

.vnccs-ps-canvas-container canvas {
    width: 100% !important;
    height: 100% !important;
}

/* Action Bar */
.vnccs-ps-actions {
    display: flex;
    gap: 8px;
    padding: 8px 10px;
    background: #222;
    border-top: 1px solid #333;
}

.vnccs-ps-btn {
    padding: 6px 12px;
    background: #3a3a4e;
    border: none;
    border-radius: 4px;
    color: #aaa;
    cursor: pointer;
    font-size: 11px;
}

.vnccs-ps-btn:hover {
    background: #4a4a5e;
    color: #fff;
}

.vnccs-ps-btn.danger {
    background: #4a2a2a;
}

.vnccs-ps-btn.danger:hover {
    background: #5a3a3a;
}
`;

// Inject styles
const styleEl = document.createElement("style");
styleEl.textContent = STYLES;
document.head.appendChild(styleEl);


// === 3D Viewer (from Debug3) ===
class PoseViewer {
    constructor(canvas) {
        this.canvas = canvas;
        this.width = 500;
        this.height = 500;

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
        this.bones = {};
        this.selectedBone = null;

        this.jointMarkers = [];

        // Pose state
        this.modelRotation = { x: 0, y: 0, z: 0 };

        // Pose state
        this.modelRotation = { x: 0, y: 0, z: 0 };

        this.syncCallback = null;

        this.initialized = false;
    }

    async init() {
        try {
            const modules = await ThreeModuleLoader.load();
            this.THREE = modules.THREE;
            this.OrbitControls = modules.OrbitControls;
            this.TransformControls = modules.TransformControls;

            this.setupScene();
            this.initialized = true;
            console.log('Pose Studio: 3D Viewer initialized');

            this.animate();
        } catch (e) {
            console.error('Pose Studio: Init failed', e);
        }
    }

    setupScene() {
        const THREE = this.THREE;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.1, 1000);
        this.camera.position.set(0, 10, 30);

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setSize(this.width, this.height);
        this.renderer.setPixelRatio(window.devicePixelRatio);

        // Orbit Controls
        this.orbit = new this.OrbitControls(this.camera, this.canvas);
        this.orbit.target.set(0, 10, 0);
        this.orbit.enableDamping = true;
        this.orbit.dampingFactor = 0.12;
        this.orbit.rotateSpeed = 0.95;
        this.orbit.update();

        // Transform Controls (Gizmo)
        this.transform = new this.TransformControls(this.camera, this.canvas);
        this.transform.setMode("rotate");
        this.transform.setSpace("local");
        this.transform.setSize(0.8);
        this.scene.add(this.transform);

        this.transform.addEventListener("dragging-changed", (e) => {
            this.orbit.enabled = !e.value;
            // Sync when drag ends
            if (!e.value && this.syncCallback) {
                this.syncCallback();
            }
        });

        // Lights
        const light = new THREE.DirectionalLight(0xffffff, 2);
        light.position.set(10, 20, 30);
        this.scene.add(light);
        this.scene.add(new THREE.AmbientLight(0x505050));

        this.gridHelper = new THREE.GridHelper(20, 20, 0x0f3460, 0x0f3460);
        this.scene.add(this.gridHelper);

        // Events
        this.canvas.addEventListener("pointerdown", (e) => this.handlePointerDown(e));
    }

    animate() {
        if (!this.initialized) return;
        requestAnimationFrame(() => this.animate());
        this.orbit.update();
        if (this.renderer) this.renderer.render(this.scene, this.camera);
    }

    handlePointerDown(e) {
        if (!this.initialized || !this.skinnedMesh) return;
        if (e.button !== 0) return;

        if (this.transform.dragging) return;
        if (this.transform.axis) return;

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

            if (nearest && minD < 2.0) {
                this.selectBone(nearest);
            }
        } else {
            this.deselectBone();
        }
    }

    selectBone(bone) {
        if (this.selectedBone === bone) return;
        this.selectedBone = bone;
        this.transform.attach(bone);
        this.updateMarkers();
    }

    deselectBone() {
        if (!this.selectedBone) return;
        this.selectedBone = null;
        this.transform.detach();
        this.updateMarkers();
    }

    updateMarkers() {
        const boneIdx = this.selectedBone ? this.boneList.indexOf(this.selectedBone) : -1;
        for (let i = 0; i < this.jointMarkers.length; i++) {
            const marker = this.jointMarkers[i];
            if (i === boneIdx) {
                marker.material.color.setHex(0x00ffff);
                marker.scale.setScalar(1.8);
            } else {
                marker.material.color.setHex(0xffaa00);
                marker.scale.setScalar(1.0);
            }
        }
    }

    resize(w, h) {
        this.width = w;
        this.height = h;
        if (this.renderer) this.renderer.setSize(w, h);
        if (this.camera) {
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
        }
    }

    loadData(data, keepCamera = false) {
        if (!this.initialized || !data || !data.vertices || !data.bones) return;
        const THREE = this.THREE;

        // Clean previous
        if (this.skinnedMesh) {
            this.scene.remove(this.skinnedMesh);
            this.skinnedMesh.geometry.dispose();
            this.skinnedMesh.material.dispose();
            if (this.skeletonHelper) this.scene.remove(this.skeletonHelper);
        }
        this.jointMarkers.forEach(m => m.parent?.remove(m));
        this.jointMarkers = [];

        // Geometry
        const vertices = new Float32Array(data.vertices);
        const indices = new Uint32Array(data.indices);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();

        // Center camera
        geometry.computeBoundingBox();
        const center = geometry.boundingBox.getCenter(new THREE.Vector3());
        this.meshCenter = center.clone();
        const size = geometry.boundingBox.getSize(new THREE.Vector3());
        if (!keepCamera && size.length() > 0.1 && this.orbit) {
            this.orbit.target.copy(center);
            const dist = size.length() * 1.5;
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

        // Skin material
        const material = new THREE.MeshPhongMaterial({
            color: 0xd4a574,
            specular: 0x332211,
            shininess: 15,
            side: THREE.DoubleSide
        });

        this.skinnedMesh = new THREE.SkinnedMesh(geometry, material);
        rootBones.forEach(b => this.skinnedMesh.add(b));
        this.skinnedMesh.bind(this.skeleton);
        this.scene.add(this.skinnedMesh);

        this.skeletonHelper = new THREE.SkeletonHelper(this.skinnedMesh);
        this.scene.add(this.skeletonHelper);

        // Joint Markers
        const sphereGeoNormal = new THREE.SphereGeometry(0.12, 8, 6);
        const sphereGeoFinger = new THREE.SphereGeometry(0.06, 6, 4);
        const fingerPatterns = ['finger', 'thumb', 'index', 'middle', 'ring', 'pinky', 'f_'];

        for (let i = 0; i < this.boneList.length; i++) {
            const bone = this.boneList[i];
            const boneName = bone.name.toLowerCase();
            const isFinger = fingerPatterns.some(p => boneName.includes(p));
            const geo = isFinger ? sphereGeoFinger : sphereGeoNormal;

            const mat = new THREE.MeshBasicMaterial({
                color: 0xffaa00,
                transparent: true,
                opacity: 0.9,
                depthTest: false
            });
            const sphere = new THREE.Mesh(geo, mat);
            sphere.userData.boneIndex = i;
            sphere.renderOrder = 999;
            bone.add(sphere);
            sphere.position.set(0, 0, 0);
            this.jointMarkers.push(sphere);
        }
    }

    // === Pose State Management ===

    getPose() {
        const bones = {};
        for (const b of this.boneList) {
            const rot = b.rotation;
            if (Math.abs(rot.x) > 1e-4 || Math.abs(rot.y) > 1e-4 || Math.abs(rot.z) > 1e-4) {
                bones[b.name] = [
                    rot.x * 180 / Math.PI,
                    rot.y * 180 / Math.PI,
                    rot.z * 180 / Math.PI
                ];
            }
        }
        return {
            bones,
            modelRotation: [this.modelRotation.x, this.modelRotation.y, this.modelRotation.z]
        };
    }

    setPose(pose) {
        if (!pose) return;

        const bones = pose.bones || {};
        const modelRot = pose.modelRotation || [0, 0, 0];

        // Reset all bones
        for (const b of this.boneList) {
            b.rotation.set(0, 0, 0);
        }

        // Apply bone rotations
        for (const [bName, rot] of Object.entries(bones)) {
            const bone = this.bones[bName];
            if (bone && Array.isArray(rot) && rot.length >= 3) {
                bone.rotation.set(
                    rot[0] * Math.PI / 180,
                    rot[1] * Math.PI / 180,
                    rot[2] * Math.PI / 180
                );
            }
        }

        // Apply model rotation
        this.modelRotation.x = modelRot[0] || 0;
        this.modelRotation.y = modelRot[1] || 0;
        this.modelRotation.z = modelRot[2] || 0;

        // Apply global rotation to root node (skinnedMesh)
        if (this.skinnedMesh) {
            this.skinnedMesh.rotation.set(
                this.modelRotation.x * Math.PI / 180,
                this.modelRotation.y * Math.PI / 180,
                this.modelRotation.z * Math.PI / 180
            );
        }
    }

    resetPose() {
        for (const b of this.boneList) {
            b.rotation.set(0, 0, 0);
        }
        this.modelRotation = { x: 0, y: 0, z: 0 };
        if (this.skinnedMesh) {
            this.skinnedMesh.rotation.set(0, 0, 0);
        }
    }

    updateCaptureCamera(width, height, zoom = 1.0) {
        const target = this.meshCenter || new this.THREE.Vector3(0, 10, 0);
        const dist = 45;

        if (!this.captureCamera) {
            this.captureCamera = new this.THREE.PerspectiveCamera(30, width / height, 0.1, 100);

            // Visual Helper
            this.captureHelper = new this.THREE.CameraHelper(this.captureCamera);
            this.scene.add(this.captureHelper);
        }

        // Positioning relative to mesh center
        this.captureCamera.aspect = width / height;
        this.captureCamera.zoom = zoom;
        this.captureCamera.updateProjectionMatrix();
        this.captureCamera.position.set(target.x, target.y, target.z + dist);
        this.captureCamera.lookAt(target);

        if (this.captureHelper) {
            this.captureHelper.update();
            this.captureHelper.visible = true;
        }
    }

    snapToCaptureCamera(width, height, zoom = 1.0) {
        this.updateCaptureCamera(width, height, zoom);

        // Disable damping for hard reset
        const prevDamping = this.orbit.enableDamping;
        this.orbit.enableDamping = false;

        // Copy capture camera to viewport camera
        this.camera.position.copy(this.captureCamera.position);
        this.camera.zoom = zoom;
        this.camera.updateProjectionMatrix();

        const target = this.meshCenter || new this.THREE.Vector3(0, 10, 0);
        this.orbit.target.copy(target);
        this.orbit.update();

        this.orbit.enableDamping = prevDamping;
    }

    capture(width, height, zoom, bgColor) {
        if (!this.initialized) return null;

        // Ensure camera is setup
        this.updateCaptureCamera(width, height, zoom);

        // Hide UI elements
        const markersVisible = this.jointMarkers[0]?.visible ?? true;
        const transformVisible = this.transform ? this.transform.visible : true;

        // Hide Helpers
        if (this.transform) this.transform.visible = false;
        if (this.skeletonHelper) this.skeletonHelper.visible = false;
        if (this.gridHelper) this.gridHelper.visible = false;
        if (this.captureHelper) this.captureHelper.visible = false; // Hide frame from capture
        this.jointMarkers.forEach(m => m.visible = false);

        // Background Override
        const oldBg = this.scene.background;
        if (bgColor && Array.isArray(bgColor) && bgColor.length === 3) {
            this.scene.background = new this.THREE.Color(
                bgColor[0] / 255, bgColor[1] / 255, bgColor[2] / 255
            );
        }

        let dataURL = null;
        try {
            // Resize renderer to output size
            const originalSize = new this.THREE.Vector2();
            this.renderer.getSize(originalSize);

            this.renderer.setSize(width, height);

            // Render with Fixed Camera
            this.renderer.render(this.scene, this.captureCamera);
            dataURL = this.canvas.toDataURL("image/png");

            // Restore renderer
            this.renderer.setSize(originalSize.x, originalSize.y);

        } catch (e) {
            console.error("Capture failed:", e);
        } finally {
            // Restore state
            this.scene.background = oldBg;

            this.jointMarkers.forEach(m => m.visible = true);
            if (this.transform) this.transform.visible = transformVisible;
            if (this.skeletonHelper) this.skeletonHelper.visible = true;
            if (this.gridHelper) this.gridHelper.visible = true;
            if (this.captureHelper) this.captureHelper.visible = true; // Show frame in editor

            // Re-render viewport
            this.renderer.render(this.scene, this.camera);
        }
        return dataURL;
    }
}


// === Pose Studio Widget ===
class PoseStudioWidget {
    constructor(node) {
        this.node = node;
        this.container = null;
        this.viewer = null;

        this.poses = [{}];  // Array of pose data
        this.activeTab = 0;
        this.poseCaptures = []; // Cache for captured images

        // Slider values
        this.meshParams = {
            age: 25, gender: 0.5, weight: 0.5,
            muscle: 0.5, height: 0.5,
            breast_size: 0.5, genital_size: 0.5
        };

        // Export settings
        this.exportParams = {
            view_width: 512,
            view_height: 512,
            cam_zoom: 1.0,
            output_mode: "LIST",
            grid_columns: 2,
            bg_color: [40, 40, 40]
        };

        this.sliders = {};
        this.exportWidgets = {};
        this.tabsContainer = null;
        this.canvasContainer = null;

        this.createUI();
    }

    createUI() {
        // Main container
        this.container = document.createElement("div");
        this.container.className = "vnccs-pose-studio";

        // Left Panel (Sliders)
        const leftPanel = document.createElement("div");
        leftPanel.className = "vnccs-ps-left";

        // Gender Switch
        const genderGroup = document.createElement("div");
        genderGroup.className = "vnccs-ps-slider-group";
        genderGroup.style.marginBottom = "8px";

        const genderLabel = document.createElement("div");
        genderLabel.className = "vnccs-ps-slider-label";
        genderLabel.innerText = "Gender";
        genderGroup.appendChild(genderLabel);

        const genderSwitch = document.createElement("div");
        genderSwitch.style.display = "flex";
        genderSwitch.style.gap = "2px";
        genderSwitch.style.background = "#222";
        genderSwitch.style.borderRadius = "4px";
        genderSwitch.style.padding = "2px";

        const btnMale = document.createElement("button");
        btnMale.innerText = "Male";
        btnMale.style.flex = "1";
        btnMale.style.border = "none";
        btnMale.style.padding = "4px";
        btnMale.style.cursor = "pointer";
        btnMale.style.borderRadius = "3px";

        const btnFemale = document.createElement("button");
        btnFemale.innerText = "Female";
        btnFemale.style.flex = "1";
        btnFemale.style.border = "none";
        btnFemale.style.padding = "4px";
        btnFemale.style.cursor = "pointer";
        btnFemale.style.borderRadius = "3px";

        this.updateGenderUI = () => {
            // User requested swap: Male=1.0, Female=0.0
            const isFemale = this.meshParams.gender < 0.5;
            btnMale.style.background = isFemale ? "transparent" : "#4a90e2";
            btnMale.style.color = isFemale ? "#888" : "white";
            btnFemale.style.background = isFemale ? "#e24a90" : "transparent";
            btnFemale.style.color = isFemale ? "white" : "#888";
        };

        btnMale.addEventListener("click", () => {
            this.meshParams.gender = 1.0; // Male
            this.updateGenderUI();
            this.onMeshParamsChanged();
        });

        btnFemale.addEventListener("click", () => {
            this.meshParams.gender = 0.0; // Female
            this.updateGenderUI();
            this.onMeshParamsChanged();
        });

        // Initial state
        this.updateGenderUI();

        genderSwitch.appendChild(btnMale);
        genderSwitch.appendChild(btnFemale);
        genderGroup.appendChild(genderSwitch);
        leftPanel.appendChild(genderGroup);

        const sliderDefs = [
            { key: "age", label: "Age", min: 1, max: 90, step: 1 },
            // Gender handled separately as switch
            { key: "weight", label: "Weight", min: 0, max: 1, step: 0.01 },
            { key: "muscle", label: "Muscle", min: 0, max: 1, step: 0.01 },
            { key: "height", label: "Height", min: 0, max: 2, step: 0.01 },
            { key: "breast_size", label: "Breast Size", min: 0, max: 2, step: 0.01 },
            { key: "genital_size", label: "Genital Size", min: 0, max: 1, step: 0.01 }
        ];

        for (const s of sliderDefs) {
            const group = document.createElement("div");
            group.className = "vnccs-ps-slider-group";

            const label = document.createElement("div");
            label.className = "vnccs-ps-slider-label";
            label.innerText = `${s.label}: ${this.meshParams[s.key]}`;

            const slider = document.createElement("input");
            slider.type = "range";
            slider.className = "vnccs-ps-slider";
            slider.min = s.min;
            slider.max = s.max;
            slider.step = s.step;
            slider.value = this.meshParams[s.key];

            slider.addEventListener("input", () => {
                this.meshParams[s.key] = parseFloat(slider.value);
                label.innerText = `${s.label}: ${s.key === 'age' ? Math.round(slider.value) : parseFloat(slider.value).toFixed(2)}`;
                this.onMeshParamsChanged();
            });

            group.appendChild(label);
            group.appendChild(slider);
            leftPanel.appendChild(group);

            this.sliders[s.key] = { slider, label, def: s };
        }

        // Model Rotation Controls
        const rotGroup = document.createElement("div");
        rotGroup.style.marginTop = "10px";
        rotGroup.style.borderTop = "1px solid #333";
        rotGroup.style.paddingTop = "5px";

        const rotTitle = document.createElement("div");
        rotTitle.innerText = "Model Rotation";
        rotTitle.style.fontSize = "12px";
        rotTitle.style.color = "#aaa";
        rotGroup.appendChild(rotTitle);

        ['x', 'y', 'z'].forEach(axis => {
            const row = document.createElement("div");
            row.className = "vnccs-ps-slider-group";

            const label = document.createElement("div");
            label.className = "vnccs-ps-slider-label";
            label.innerText = `${axis.toUpperCase()}: 0`;

            const slider = document.createElement("input");
            slider.type = "range";
            slider.className = "vnccs-ps-slider";
            slider.min = -180;
            slider.max = 180;
            slider.step = 1;
            slider.value = 0;

            slider.addEventListener("input", () => {
                const val = parseFloat(slider.value);
                label.innerText = `${axis.toUpperCase()}: ${val}`;
                if (this.viewer) {
                    this.viewer.modelRotation[axis] = val;
                    // Apply immediately
                    if (this.viewer.skinnedMesh) {
                        const r = this.viewer.modelRotation;
                        this.viewer.skinnedMesh.rotation.set(
                            r.x * Math.PI / 180,
                            r.y * Math.PI / 180,
                            r.z * Math.PI / 180
                        );
                    }
                    this.syncToNode();
                }
            });

            // Store ref to update later
            this.sliders[`rot_${axis}`] = { slider, label };

            row.appendChild(label);
            row.appendChild(slider);
            rotGroup.appendChild(row);
        });
        leftPanel.appendChild(rotGroup);



        // Export Settings
        const exportTitle = document.createElement("div");
        exportTitle.innerText = "Export Settings";
        exportTitle.style.fontSize = "12px";
        exportTitle.style.color = "#aaa";
        exportTitle.style.marginBottom = "5px";
        leftPanel.appendChild(exportTitle);

        // Dimensions
        const dimGroup = document.createElement("div");
        dimGroup.style.display = "flex";
        dimGroup.style.gap = "5px";

        const wDiv = document.createElement("div"); wDiv.style.flex = "1";
        this.createExportInput(wDiv, "Width", "view_width", "number", { min: 64, max: 4096, step: 8 });

        const hDiv = document.createElement("div"); hDiv.style.flex = "1";
        this.createExportInput(hDiv, "Height", "view_height", "number", { min: 64, max: 4096, step: 8 });

        dimGroup.appendChild(wDiv);
        dimGroup.appendChild(hDiv);
        leftPanel.appendChild(dimGroup);

        // Zoom
        this.createExportInput(leftPanel, "Camera Zoom", "cam_zoom", "range", { min: 0.1, max: 5.0, step: 0.1 });

        // Output Mode
        this.createExportSelect(leftPanel, "Output Mode", "output_mode", ["LIST", "GRID"]);

        // Grid Columns
        this.createExportInput(leftPanel, "Grid Columns", "grid_columns", "number", { min: 1, max: 6, step: 1 });

        // BG Color
        this.createColorInput(leftPanel, "BG Color", "bg_color");

        this.container.appendChild(leftPanel);

        // Right Panel
        const rightPanel = document.createElement("div");
        rightPanel.className = "vnccs-ps-right";

        // Tab Bar
        this.tabsContainer = document.createElement("div");
        this.tabsContainer.className = "vnccs-ps-tabs";
        this.updateTabs();
        rightPanel.appendChild(this.tabsContainer);

        // Canvas Container
        this.canvasContainer = document.createElement("div");
        this.canvasContainer.className = "vnccs-ps-canvas-container";

        const canvas = document.createElement("canvas");
        this.canvasContainer.appendChild(canvas);
        rightPanel.appendChild(this.canvasContainer);

        // Action Bar
        const actions = document.createElement("div");
        actions.className = "vnccs-ps-actions";

        const resetBtn = document.createElement("button");
        resetBtn.className = "vnccs-ps-btn";
        resetBtn.innerText = "Reset Pose";
        resetBtn.addEventListener("click", () => this.resetCurrentPose());

        const snapBtn = document.createElement("button");
        snapBtn.className = "vnccs-ps-btn";
        snapBtn.innerText = "Preview Output";
        snapBtn.title = "Snap viewport camera to output camera";
        snapBtn.addEventListener("click", () => {
            if (this.viewer) this.viewer.snapToCaptureCamera(
                this.exportParams.view_width,
                this.exportParams.view_height,
                this.exportParams.cam_zoom || 1.0
            );
        });

        const copyBtn = document.createElement("button");
        copyBtn.className = "vnccs-ps-btn";
        copyBtn.innerText = "Copy";
        copyBtn.addEventListener("click", () => this.copyPose());

        const pasteBtn = document.createElement("button");
        pasteBtn.className = "vnccs-ps-btn";
        pasteBtn.innerText = "Paste";
        pasteBtn.addEventListener("click", () => this.pastePose());

        actions.appendChild(resetBtn);
        actions.appendChild(snapBtn);
        actions.appendChild(copyBtn);
        actions.appendChild(pasteBtn);

        rightPanel.appendChild(actions);

        this.container.appendChild(rightPanel);

        // Initialize viewer
        this.viewer = new PoseViewer(canvas);
        this.viewer.syncCallback = () => this.syncToNode(); // Bind sync callback
        this.viewer.init();
    }

    updateTabs() {
        this.tabsContainer.innerHTML = "";

        for (let i = 0; i < this.poses.length; i++) {
            const tab = document.createElement("button");
            tab.className = "vnccs-ps-tab" + (i === this.activeTab ? " active" : "");
            tab.style.display = "flex";
            tab.style.alignItems = "center";
            tab.style.gap = "6px";
            tab.style.paddingRight = "6px"; // Extra padding for close btn

            const text = document.createElement("span");
            text.innerText = `Pose ${i + 1}`;
            tab.appendChild(text);

            if (this.poses.length > 1) {
                const close = document.createElement("span");
                close.innerText = "×";
                close.style.fontSize = "16px";
                close.style.lineHeight = "12px";
                close.style.color = "#888";
                close.style.cursor = "pointer";
                close.onmouseenter = () => close.style.color = "#ff4444";
                close.onmouseleave = () => close.style.color = "#888";

                close.onclick = (e) => {
                    e.stopPropagation();
                    this.deleteTab(i);
                };
                tab.appendChild(close);
            }

            tab.addEventListener("click", () => this.switchTab(i));
            this.tabsContainer.appendChild(tab);
        }

        // Add button (max 12)
        if (this.poses.length < 12) {
            const addBtn = document.createElement("button");
            addBtn.className = "vnccs-ps-tab-add";
            addBtn.innerText = "+";
            addBtn.addEventListener("click", () => this.addTab());
            this.tabsContainer.appendChild(addBtn);
        }
    }

    switchTab(index) {
        if (index === this.activeTab) return;

        // Save current pose & capture
        if (this.viewer && this.viewer.initialized) {
            this.poses[this.activeTab] = this.viewer.getPose();
            this.syncToNode(false);
        }

        this.activeTab = index;
        this.updateTabs();

        // Load new pose
        if (this.viewer && this.viewer.initialized) {
            this.viewer.setPose(this.poses[this.activeTab] || {});
            this.updateRotationSliders();
        }

        this.syncToNode(false);
    }

    addTab() {
        if (this.poses.length >= 12) return;

        // Save current & capture
        if (this.viewer && this.viewer.initialized) {
            this.poses[this.activeTab] = this.viewer.getPose();
            this.syncToNode(false);
        }

        this.poses.push({});
        this.activeTab = this.poses.length - 1;
        this.updateTabs();

        if (this.viewer && this.viewer.initialized) {
            this.viewer.resetPose();
        }

        this.syncToNode(false);
    }

    deleteTab(targetIndex = -1) {
        if (this.poses.length <= 1) return;
        const idx = targetIndex === -1 ? this.activeTab : targetIndex;

        // Remove capture
        if (this.poseCaptures && this.poseCaptures.length > idx) {
            this.poseCaptures.splice(idx, 1);
        }

        this.poses.splice(idx, 1);

        // Adjust active tab logic
        if (idx < this.activeTab) {
            this.activeTab--;
        } else if (idx === this.activeTab) {
            if (this.activeTab >= this.poses.length) {
                this.activeTab = this.poses.length - 1;
            }
            // Load new pose since active was deleted
            if (this.viewer && this.viewer.initialized) {
                this.viewer.setPose(this.poses[this.activeTab] || {});
                this.updateRotationSliders();
            }
        }

        this.updateTabs();
        this.syncToNode(false);
    }



    resetCurrentPose() {
        if (this.viewer) {
            this.viewer.resetPose();
            this.updateRotationSliders();
        }
        this.poses[this.activeTab] = {};
        this.syncToNode(false);
    }

    copyPose() {
        if (this.viewer && this.viewer.initialized) {
            this.poses[this.activeTab] = this.viewer.getPose();
        }
        this._clipboard = JSON.parse(JSON.stringify(this.poses[this.activeTab]));
    }

    pastePose() {
        if (!this._clipboard) return;
        this.poses[this.activeTab] = JSON.parse(JSON.stringify(this._clipboard));
        if (this.viewer && this.viewer.initialized) {
            this.viewer.setPose(this.poses[this.activeTab]);
        }
        this.syncToNode();
    }

    loadModel() {
        return api.fetchApi("/vnccs/character_studio/update_preview", {
            method: "POST",
            body: JSON.stringify(this.meshParams)
        }).then(r => r.json()).then(d => {
            if (this.viewer) {
                // Keep camera during updates
                this.viewer.loadData(d, true);

                // Apply pose immediately (no timeout/flicker)
                if (this.viewer.initialized) {
                    this.viewer.setPose(this.poses[this.activeTab] || {});
                    this.updateRotationSliders();
                    // Full recapture needed because mesh changed
                    this.syncToNode(true);
                }
            }
        });
    }

    processMeshUpdate() {
        if (this.isMeshUpdating) return;
        this.isMeshUpdating = true;
        this.pendingMeshUpdate = false;

        this.loadModel().finally(() => {
            this.isMeshUpdating = false;
            if (this.pendingMeshUpdate) {
                this.processMeshUpdate();
            }
        });
    }

    updateRotationSliders() {
        if (!this.viewer) return;
        const r = this.viewer.modelRotation;
        ['x', 'y', 'z'].forEach(axis => {
            const info = this.sliders[`rot_${axis}`];
            if (info) {
                info.slider.value = r[axis];
                info.label.innerText = `${axis.toUpperCase()}: ${r[axis]}`;
            }
        });
    }

    onMeshParamsChanged() {
        // Update node widgets
        for (const [key, value] of Object.entries(this.meshParams)) {
            const widget = this.node.widgets?.find(w => w.name === key);
            if (widget) {
                widget.value = value;
            }
        }

        // Async Queue update
        this.pendingMeshUpdate = true;
        this.processMeshUpdate();
    }

    resize(w, h) {
        if (this.viewer && this.canvasContainer) {
            const rect = this.canvasContainer.getBoundingClientRect();
            this.viewer.resize(rect.width || 500, rect.height || 500);
        }
    }

    syncToNode(fullCapture = false) {
        // Save current pose before syncing
        if (this.viewer && this.viewer.initialized) {
            this.poses[this.activeTab] = this.viewer.getPose();
        }

        // Cache Handling
        if (!this.poseCaptures) this.poseCaptures = [];
        // Ensure size
        while (this.poseCaptures.length < this.poses.length) this.poseCaptures.push(null);
        while (this.poseCaptures.length > this.poses.length) this.poseCaptures.pop();

        // Capture Image (CSR)
        if (this.viewer && this.viewer.initialized) {
            const w = this.exportParams.view_width || 512;
            const h = this.exportParams.view_height || 512;
            const z = this.exportParams.cam_zoom || 1.0;
            const bg = this.exportParams.bg_color || [40, 40, 40];

            if (fullCapture) {
                // Determine original pose index to restore
                const originalTab = this.activeTab;

                // Capture ALL
                for (let i = 0; i < this.poses.length; i++) {
                    this.viewer.setPose(this.poses[i]);
                    this.poseCaptures[i] = this.viewer.capture(w, h, z, bg);
                }

                // Restore active pose
                if (this.activeTab !== originalTab) { // Just in case
                    this.activeTab = originalTab;
                }
                this.viewer.setPose(this.poses[this.activeTab]);

            } else {
                // Capture only ACTIVE
                this.poseCaptures[this.activeTab] = this.viewer.capture(w, h, z, bg);
            }
        }

        // Update hidden pose_data widget
        const data = {
            mesh: this.meshParams,
            export: this.exportParams,
            poses: this.poses,
            activeTab: this.activeTab,
            captured_images: this.poseCaptures
        };

        const widget = this.node.widgets?.find(w => w.name === "pose_data");
        if (widget) {
            widget.value = JSON.stringify(data);
        }
    }

    loadFromNode() {
        // Load from pose_data widget
        const widget = this.node.widgets?.find(w => w.name === "pose_data");
        if (!widget || !widget.value) return;

        try {
            const data = JSON.parse(widget.value);

            if (data.mesh) {
                this.meshParams = { ...this.meshParams, ...data.mesh };
                // Update sliders
                for (const [key, info] of Object.entries(this.sliders)) {
                    if (this.meshParams[key] !== undefined) {
                        info.slider.value = this.meshParams[key];
                        info.label.innerText = `${info.def.label}: ${key === 'age' ? Math.round(this.meshParams[key]) : parseFloat(this.meshParams[key]).toFixed(2)}`;
                    }
                }
                // Update gender switch
                if (this.updateGenderUI) this.updateGenderUI();
            }

            if (data.export) {
                this.exportParams = { ...this.exportParams, ...data.export };
                // Update export widgets
                for (const [key, widget] of Object.entries(this.exportWidgets)) {
                    if (key === 'bg_color') {
                        const rgb = this.exportParams.bg_color;
                        const hex = "#" + ((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1);
                        widget.value = hex;
                    } else if (this.exportParams[key] !== undefined) {
                        widget.value = this.exportParams[key];
                    }
                }
            }

            if (data.poses && Array.isArray(data.poses)) {
                this.poses = data.poses;
            }

            if (typeof data.activeTab === 'number') {
                this.activeTab = Math.min(data.activeTab, this.poses.length - 1);
            }

            if (data.captured_images && Array.isArray(data.captured_images)) {
                this.poseCaptures = data.captured_images;
            }

            this.updateTabs();

            // Auto-load model
            this.loadModel();

        } catch (e) {
            console.error("Failed to parse pose_data:", e);
        }
    }

    createExportInput(parent, labelText, key, type, attrs = {}) {
        const group = document.createElement("div");
        group.className = "vnccs-ps-slider-group";

        const label = document.createElement("div");
        label.className = "vnccs-ps-slider-label";
        label.innerText = labelText;

        const input = document.createElement("input");
        input.type = type;
        input.className = "vnccs-ps-slider"; // Reuse style
        input.value = this.exportParams[key];

        // Style adjustments based on type
        if (type === "number") {
            input.style.cursor = "text";
            input.style.height = "24px";
            input.style.padding = "2px 5px";
            input.style.background = "#222";
            input.style.border = "1px solid #444";
            input.style.color = "#ccc";
        }

        Object.entries(attrs).forEach(([k, v]) => input[k] = v);

        // Initial label
        if (type === "range") {
            label.innerText = `${labelText}: ${parseFloat(input.value).toFixed(1)}`;
        }

        input.addEventListener("input", () => {
            if (type === "range") {
                const val = parseFloat(input.value);
                label.innerText = `${labelText}: ${val.toFixed(1)}`;

                // Live update for Zoom
                if (key === 'cam_zoom' && this.viewer) {
                    this.exportParams[key] = val;
                    this.viewer.updateCaptureCamera(
                        this.exportParams.view_width,
                        this.exportParams.view_height,
                        val
                    );
                }
            }
        });

        input.addEventListener("change", () => {
            let val = (type === "number" || type === "range") ? parseFloat(input.value) : input.value;
            if (type === "number" && isNaN(val)) val = this.exportParams[key];
            this.exportParams[key] = val;
            // If dimensions, zoom or bg_color change, we need full recapture
            const needsFull = (key === 'view_width' || key === 'view_height' || key === 'cam_zoom' || key === 'bg_color');
            this.syncToNode(needsFull);
        });

        group.appendChild(label);
        group.appendChild(input);
        parent.appendChild(group);
        this.exportWidgets[key] = input;
    }

    createExportSelect(parent, labelText, key, options) {
        const group = document.createElement("div");
        group.className = "vnccs-ps-slider-group";

        const label = document.createElement("div");
        label.className = "vnccs-ps-slider-label";
        label.innerText = labelText;

        const select = document.createElement("select");
        select.style.width = "100%";
        select.style.background = "#222";
        select.style.color = "#ccc";
        select.style.border = "1px solid #444";
        select.style.padding = "4px";

        options.forEach(opt => {
            const el = document.createElement("option");
            el.value = opt;
            el.innerText = opt;
            el.selected = this.exportParams[key] === opt;
            select.appendChild(el);
        });

        select.addEventListener("change", () => {
            this.exportParams[key] = select.value;
            this.syncToNode();
        });

        group.appendChild(label);
        group.appendChild(select);
        parent.appendChild(group);
        this.exportWidgets[key] = select;
    }

    createColorInput(parent, labelText, key) {
        const group = document.createElement("div");
        group.className = "vnccs-ps-slider-group";

        const label = document.createElement("div");
        label.className = "vnccs-ps-slider-label";
        label.innerText = labelText;

        const container = document.createElement("div");
        container.style.display = "flex";
        container.style.gap = "5px";

        const input = document.createElement("input");
        input.type = "color";
        input.style.width = "40px";
        input.style.height = "24px";
        input.style.border = "none";
        input.style.padding = "0";
        input.style.background = "none";

        // Convert initial RGB to Hex
        const rgb = this.exportParams[key]; // [r,g,b]
        const hex = "#" + ((1 << 24) + (rgb[0] << 16) + (rgb[1] << 8) + rgb[2]).toString(16).slice(1);
        input.value = hex;

        input.addEventListener("input", () => {
            const hex = input.value;
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            this.exportParams[key] = [r, g, b];
            // No live preview of BG color to avoid eye strain
        });

        input.addEventListener("change", () => {
            // Full render on commit
            this.syncToNode(true);
        });

        container.appendChild(input);
        group.appendChild(label);
        group.appendChild(container);
        parent.appendChild(group);
        this.exportWidgets[key] = input;
    }
}


// === ComfyUI Extension Registration ===
app.registerExtension({
    name: "VNCCS.PoseStudio",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== "VNCCS_PoseStudio") return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onCreated) onCreated.apply(this, arguments);

            this.setSize([900, 700]);

            // Create widget
            this.studioWidget = new PoseStudioWidget(this);

            this.addDOMWidget("pose_studio_ui", "ui", this.studioWidget.container, {
                serialize: false,
                hideOnZoom: false
            });

            // Hide pose_data widget
            const poseWidget = this.widgets?.find(w => w.name === "pose_data");
            if (poseWidget) {
                poseWidget.type = "hidden";
                poseWidget.computeSize = () => [0, -4];
            }

            // Load model after initialization
            setTimeout(() => {
                this.studioWidget.loadFromNode();
                this.studioWidget.loadModel();
            }, 500);
        };

        nodeType.prototype.onResize = function (size) {
            if (this.studioWidget) {
                const w = Math.max(600, size[0] - 20);
                const h = Math.max(400, size[1] - 100);
                this.studioWidget.container.style.width = w + "px";
                this.studioWidget.container.style.height = h + "px";

                setTimeout(() => this.studioWidget.resize(w, h), 50);
            }
        };

        // Save state on configure
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);

            if (this.studioWidget) {
                setTimeout(() => {
                    this.studioWidget.loadFromNode();
                    this.studioWidget.loadModel();
                }, 200);
            }
        };
    }
});
