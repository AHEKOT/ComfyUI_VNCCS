/**
 * VNCCS Debug3 - DOM-Based Three.js Widget
 * 
 * Uses addDOMWidget to embed a REAL canvas in the DOM.
 * Native mouse events for better precision.
 * Same IK logic as Debug2.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";

// Inject styles once
const STYLE = `
.vnccs-debug3-container {
    width: 100%;
    height: 100%;
    background: #1a1a2e;
    overflow: hidden;
}
.vnccs-debug3-container canvas {
    display: block;
    width: 100%;
    height: 100%;
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
        this.scene = null;
        this.camera = null;
        this.renderer = null;

        this.skinnedMesh = null;
        this.skeleton = null;
        this.bones = {};
        this.boneList = [];
        this.selectedBone = null;
        this.dragPoint = null;

        // Interaction
        this.isDragging = false;
        this.dragMode = null;
        this.mouse = { x: 0, y: 0 };
        this.lastX = 0;
        this.lastY = 0;
        this.initialized = false;

        // Camera State
        this.camState = {
            dist: 30,
            rotX: 0,
            rotY: 0,
            center: { x: 0, y: 10, z: 0 }
        };

        this.init();
    }

    async init() {
        try {
            this.THREE = await import(THREE_CDN);
            this.camState.center = new this.THREE.Vector3(0, 10, 0);

            this.setupScene();
            this.initialized = true;
            console.log('VNCCS Debug3: Three.js initialized');

            // Auto-load
            if (this.requestModelLoad) {
                this.requestModelLoad();
            }
        } catch (e) {
            console.error('VNCCS Debug3: Failed to init', e);
        }
    }

    resize(w, h) {
        this.width = w;
        this.height = h;
        this.canvas.width = w;
        this.canvas.height = h;
        if (this.renderer) {
            this.renderer.setSize(w, h);
        }
        if (this.camera) {
            this.camera.aspect = w / h;
            this.camera.updateProjectionMatrix();
        }
        this.render();
    }

    setupScene() {
        const THREE = this.THREE;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.1, 1000);
        this.updateCamera();

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true });
        this.renderer.setSize(this.width, this.height);

        const light = new THREE.DirectionalLight(0xffffff, 2);
        light.position.set(10, 20, 30);
        this.scene.add(light);
        this.scene.add(new THREE.AmbientLight(0x505050));

        this.scene.add(new THREE.GridHelper(20, 20, 0x0f3460, 0x0f3460));
    }

    updateCamera() {
        if (!this.camera || !this.camState || !this.THREE) return;
        const s = this.camState;

        const y = Math.sin(s.rotX) * s.dist;
        const hDist = Math.cos(s.rotX) * s.dist;
        const x = Math.sin(s.rotY) * hDist;
        const z = Math.cos(s.rotY) * hDist;

        this.camera.position.copy(s.center).add(new this.THREE.Vector3(x, y, z));
        this.camera.lookAt(s.center);
    }

    loadData(data) {
        if (!this.initialized || !data || !data.vertices || !data.bones) return;
        console.log(`VNCCS Debug3: Loading. Verts: ${data.vertices.length}, Bones: ${data.bones.length}`);

        const THREE = this.THREE;

        // Cleanup
        if (this.skinnedMesh) {
            this.scene.remove(this.skinnedMesh);
            this.skinnedMesh.geometry.dispose();
            this.skinnedMesh.material.dispose();
            if (this.skeletonHelper) this.scene.remove(this.skeletonHelper);
        }

        // Geometry
        const vertices = new Float32Array(data.vertices);
        const indices = new Uint32Array(data.indices);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();

        geometry.computeBoundingBox();
        const center = geometry.boundingBox.getCenter(new THREE.Vector3());
        const size = geometry.boundingBox.getSize(new THREE.Vector3());

        if (size.length() > 0.1) {
            this.camState.center.copy(center);
            this.camState.dist = size.length() * 1.5;
            this.updateCamera();
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

        // Skins
        const vCount = vertices.length / 3;
        const skinInds = new Float32Array(vCount * 4);
        const skinWgts = new Float32Array(vCount * 4);

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
                if (tot > 0) for (let i = 0; i < 4; i++) skinWgts[v * 4 + i] /= tot;
                else skinWgts[v * 4] = 1;
            }
        }

        geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinInds, 4));
        geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWgts, 4));

        const material = new THREE.MeshPhongMaterial({
            color: 0x4a90d9,
            side: THREE.DoubleSide
        });

        this.skinnedMesh = new THREE.SkinnedMesh(geometry, material);
        rootBones.forEach(b => this.skinnedMesh.add(b));
        this.skinnedMesh.bind(this.skeleton);
        this.scene.add(this.skinnedMesh);

        this.skeletonHelper = new THREE.SkeletonHelper(this.skinnedMesh);
        this.scene.add(this.skeletonHelper);

        this.render();
    }

    // IK Logic (copy from Debug2)
    kinematic2D(bone, axis, angle, ignoreIfPositive = false) {
        if (!bone) return 0;
        const THREE = this.THREE;
        const wPos = new THREE.Vector3();

        const target = (this.dragPoint && this.dragPoint.parent === bone) ? this.dragPoint : bone;
        target.getWorldPosition(wPos);

        const scrBefore = wPos.clone().project(this.camera);
        const distBefore = Math.sqrt((scrBefore.x - this.mouse.x) ** 2 + (scrBefore.y - this.mouse.y) ** 2);

        const oldRot = bone.rotation[axis];
        bone.rotation[axis] += angle;
        bone.updateMatrixWorld(true);

        target.getWorldPosition(wPos);
        const scrAfter = wPos.clone().project(this.camera);
        const distAfter = Math.sqrt((scrAfter.x - this.mouse.x) ** 2 + (scrAfter.y - this.mouse.y) ** 2);

        const improvement = distBefore - distAfter;

        if (ignoreIfPositive && improvement > 0) return improvement;

        bone.rotation[axis] = oldRot;
        bone.updateMatrixWorld(true);
        return improvement;
    }

    inverseKinematics(bone, axis, step) {
        const kPos = this.kinematic2D(bone, axis, 0.001);
        const kNeg = this.kinematic2D(bone, axis, -0.001);
        if (kPos > 0 || kNeg > 0) {
            if (kPos < kNeg) step = -step;
            this.kinematic2D(bone, axis, step, true);
        }
    }

    onPointerDown(e) {
        if (!this.initialized || !this.skinnedMesh) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        this.lastX = e.clientX;
        this.lastY = e.clientY;

        this.isDragging = true;
        this.dragMode = 'camera';

        if (e.button === 0) {
            const THREE = this.THREE;
            const mouse = new THREE.Vector2((x / rect.width) * 2 - 1, -(y / rect.height) * 2 + 1);
            const ray = new THREE.Raycaster();
            ray.setFromCamera(mouse, this.camera);
            const intersects = ray.intersectObject(this.skinnedMesh, true);

            if (intersects.length > 0) {
                const point = intersects[0].point;
                let nearest = null;
                let minD = Infinity;

                for (const b of this.boneList) {
                    const wPos = new THREE.Vector3();
                    b.getWorldPosition(wPos);
                    const d = point.distanceTo(wPos);
                    if (d < minD) { minD = d; nearest = b; }
                }

                if (nearest && minD < 2.0) {
                    this.selectedBone = nearest;
                    this.dragMode = 'bone';
                    console.log("Debug3 Selected:", nearest.name);

                    if (!this.dragPoint) this.dragPoint = new THREE.Object3D();
                    nearest.attach(this.dragPoint);
                    this.dragPoint.position.copy(nearest.worldToLocal(point.clone()));
                }
            }
        }
    }

    onPointerMove(e) {
        if (!this.initialized) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        this.mouse.x = (x / rect.width) * 2 - 1;
        this.mouse.y = -(y / rect.height) * 2 + 1;

        const dx = e.clientX - this.lastX;
        const dy = e.clientY - this.lastY;
        this.lastX = e.clientX;
        this.lastY = e.clientY;

        if (!this.isDragging) return;

        if (this.dragMode === 'bone' && this.selectedBone) {
            for (let step = 5 * Math.PI / 180; step > 0.1 * Math.PI / 180; step *= 0.75) {
                this.inverseKinematics(this.selectedBone, 'x', step);
                this.inverseKinematics(this.selectedBone, 'y', step);
                this.inverseKinematics(this.selectedBone, 'z', step);
            }
        } else if (this.dragMode === 'camera') {
            const sensitivity = 0.01;
            this.camState.rotY -= dx * sensitivity;
            this.camState.rotX += dy * sensitivity;
            this.camState.rotX = Math.max(-1.5, Math.min(1.5, this.camState.rotX));
            this.updateCamera();
        }
        this.render();
    }

    onPointerUp(e) {
        this.isDragging = false;
    }

    render() {
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
    }
}

// ComfyUI Extension
app.registerExtension({
    name: "VNCCS.Debug3",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "VNCCS_Debug3") {
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onCreated) onCreated.apply(this, arguments);

                this.setSize([700, 650]);

                // Create container
                const container = document.createElement("div");
                container.className = "vnccs-debug3-container";

                // Create canvas
                const canvas = document.createElement("canvas");
                canvas.width = 680;
                canvas.height = 550;
                container.appendChild(canvas);

                // Add DOM widget
                this.addDOMWidget("debug3_viewer", "ui", container, {
                    serialize: false,
                    hideOnZoom: false,
                    getValue() { return undefined; },
                    setValue(v) { }
                });

                // Create viewer
                this.viewer = new MakeHumanViewer(canvas);
                this.viewer.requestModelLoad = () => {
                    api.fetchApi("/vnccs/character_studio/update_preview", {
                        method: "POST", body: "{}"
                    }).then(r => r.json()).then(d => this.viewer.loadData(d));
                };

                // Native event listeners
                canvas.addEventListener("mousedown", e => {
                    e.preventDefault();
                    this.viewer.onPointerDown(e);
                });
                canvas.addEventListener("mousemove", e => {
                    this.viewer.onPointerMove(e);
                });
                canvas.addEventListener("mouseup", e => {
                    this.viewer.onPointerUp(e);
                });
                canvas.addEventListener("contextmenu", e => e.preventDefault());

                // Store for resize
                this._canvas = canvas;
                this._container = container;
            };

            nodeType.prototype.onResize = function (size) {
                if (!this._canvas || !this.viewer) return;
                const w = Math.max(200, size[0] - 20);
                const h = Math.max(200, size[1] - 100);
                this.viewer.resize(w, h);
            };
        }
    }
});
