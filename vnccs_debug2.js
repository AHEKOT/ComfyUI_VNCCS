/**
 * VNCCS Debug2 - Three.js SkinnedMesh Viewer with mannequin.js IK
 * 
 * Uses stable jsdelivr loader pattern for Three.js.
 * Loads MakeHuman model as SkinnedMesh and applies screen-space IK.
 * Implements manual camera control to work on offscreen canvas.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js";

class MakeHumanViewer {
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

        this.skinnedMesh = null;
        this.skeleton = null;
        this.bones = {};
        this.boneList = [];
        this.selectedBone = null;

        // Interaction
        this.isDragging = false;
        this.dragMode = null; // 'bone' or 'camera'
        this.mouse = { x: 0, y: 0 };
        this.initialized = false;

        // Manual Camera State
        this.camState = {
            dist: 30,
            rotX: 0,
            rotY: 0,
            center: { x: 0, y: 10, z: 0 } // simple object to avoid THREE dependency before init
        };

        this.init();
    }

    async init() {
        try {
            // Use same CDN as vnccs_pose_editor.js to avoid "Multiple instances" warning
            this.THREE = await import(THREE_CDN);
            this.camState.center = new this.THREE.Vector3(0, 10, 0);

            this.setupScene();
            this.initialized = true;
            console.log('VNCCS: Three.js initialized via jsdelivr');

            if (!this.dataLoaded && this.requestModelLoad) {
                this.requestModelLoad();
            }
        } catch (e) {
            console.error('VNCCS: Failed to init Three.js', e);
        }
    }

    setupScene() {
        const THREE = this.THREE;

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x1a1a2e);

        this.camera = new THREE.PerspectiveCamera(45, this.width / this.height, 0.1, 1000);
        this.updateCamera();

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
        this.renderer.setSize(this.width, this.height);

        // Lights
        const light = new THREE.DirectionalLight(0xffffff, 2);
        light.position.set(10, 20, 30);
        this.scene.add(light);
        this.scene.add(new THREE.AmbientLight(0x505050));

        // Grid
        this.scene.add(new THREE.GridHelper(20, 20, 0x0f3460, 0x0f3460));
    }

    updateCamera() {
        if (!this.camera || !this.camState || !this.THREE) return;
        const s = this.camState;

        // Spherical to Cartesian
        const y = Math.sin(s.rotX) * s.dist;
        const hDist = Math.cos(s.rotX) * s.dist;
        const x = Math.sin(s.rotY) * hDist;
        const z = Math.cos(s.rotY) * hDist;

        this.camera.position.copy(s.center).add(new this.THREE.Vector3(x, y, z));
        this.camera.lookAt(s.center);
    }

    loadData(data) {
        if (!this.initialized || !data || !data.vertices || !data.bones) return;
        console.log(`VNCCS: Loading Model. Verts: ${data.vertices.length}, Bones: ${data.bones.length}`);

        const THREE = this.THREE;

        // Cleanup
        if (this.skinnedMesh) {
            this.scene.remove(this.skinnedMesh);
            this.skinnedMesh.geometry.dispose();
            this.skinnedMesh.material.dispose();
            if (this.skeletonHelper) this.scene.remove(this.skeletonHelper);
        }

        // 1. Geometry
        const vertices = new Float32Array(data.vertices);
        const indices = new Uint32Array(data.indices);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(indices, 1));
        geometry.computeVertexNormals();

        // Center camera on mesh
        geometry.computeBoundingBox();
        const center = geometry.boundingBox.getCenter(new THREE.Vector3());
        const size = geometry.boundingBox.getSize(new THREE.Vector3());

        if (size.length() > 0.1) {
            this.camState.center.copy(center);
            this.camState.dist = size.length() * 1.5;
            this.updateCamera();
        }

        // 2. Bones
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

        // 3. Skins
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
            // skinning property removed to fix warning
        });

        this.skinnedMesh = new THREE.SkinnedMesh(geometry, material);
        rootBones.forEach(b => this.skinnedMesh.add(b));
        this.skinnedMesh.bind(this.skeleton);
        this.scene.add(this.skinnedMesh);

        this.skeletonHelper = new THREE.SkeletonHelper(this.skinnedMesh);
        this.scene.add(this.skeletonHelper);

        this.render();
    }

    kinematic2D(bone, axis, angle, ignoreIfPositive = false) {
        if (!bone) return 0;
        const THREE = this.THREE;
        const wPos = new THREE.Vector3();
        
        // Measure before
        // Use dragPoint if available, otherwise bone (fallback, though bone origin doesn't move)
        const target = (this.dragPoint && this.dragPoint.parent === bone) ? this.dragPoint : bone;
        target.getWorldPosition(wPos);
        
        const scrBefore = wPos.clone().project(this.camera);
        // Mouse is in NDC [-1, 1]. Project assumes NDC.
        // wait, mouse is NDC, scrBefore is NDC.
        // dist is in NDC units.
        const distBefore = Math.sqrt((scrBefore.x - this.mouse.x)**2 + (scrBefore.y - this.mouse.y)**2);
        
        // Rotate
        const oldRot = bone.rotation[axis];
        bone.rotation[axis] += angle;
        bone.updateMatrixWorld(true);
        
        // Measure after
        target.getWorldPosition(wPos);
        const scrAfter = wPos.clone().project(this.camera);
        const distAfter = Math.sqrt((scrAfter.x - this.mouse.x)**2 + (scrAfter.y - this.mouse.y)**2);
        
        const improvement = distBefore - distAfter;
        
        // Debug Log only for test moves (angle 0.001) to avoid spamming committed moves
        if (Math.abs(angle) < 0.002 && Math.random() < 0.01) {
             console.log(`IK Debug: ${axis} D:${distBefore.toFixed(6)}->${distAfter.toFixed(6)} Imp:${improvement.toFixed(8)}`);
        }

        if(ignoreIfPositive && improvement > 0) return improvement;
        
        // Revert
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

    onPointerDown(x, y, button) {
        if (!this.initialized || !this.skinnedMesh) return false;

        this.isDragging = true;
        this.dragMode = 'camera';

        if (button === 0) { // Left Click
            const THREE = this.THREE;
            const mouse = new THREE.Vector2((x / this.width) * 2 - 1, -(y / this.height) * 2 + 1);
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
                    console.log("Selected:", nearest.name);
                    
                    // Attach dragPoint to the specific click location on the bone
                    if (!this.dragPoint) this.dragPoint = new THREE.Object3D();
                    nearest.attach(this.dragPoint);
                    this.dragPoint.position.copy(nearest.worldToLocal(point.clone()));
                    
                    return true;
                }
            }
        }
        return true;
    }

    onPointerMove(x, y, dx, dy) {
        this.mouse.x = (x / this.width) * 2 - 1;
        this.mouse.y = -(y / this.height) * 2 + 1;

        if (!this.isDragging) return;

        if (this.dragMode === 'bone' && this.selectedBone) {
            // Log random sample to verify IK is running
            if (Math.random() < 0.05) console.log("VNCCS: IK Running for", this.selectedBone.name);

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

    onPointerUp() {
        this.isDragging = false;
    }

    render() {
        if (this.renderer && this.scene && this.camera) {
            this.renderer.render(this.scene, this.camera);
        }
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
    name: "VNCCS.Debug2",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name === "VNCCS_Debug2") {
            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onCreated) onCreated.apply(this, arguments);

                this.setSize([700, 650]);
                this.viewer = new MakeHumanViewer(680, 550);
                this.viewer.requestModelLoad = () => this.loadModel();
                this.viewerRegion = { x: 10, y: 80, w: 680, h: 550 };
                this.lastPos = [0, 0];

                this.addWidget("button", "Load Model", null, () => this.loadModel());
                this.addWidget("button", "Apply Pose", null, () => this.applyPose());
            };

            nodeType.prototype.loadModel = function () {
                api.fetchApi("/vnccs/character_studio/update_preview", {
                    method: "POST", body: "{}"
                }).then(r => r.json()).then(d => this.viewer.loadData(d));
            };

            nodeType.prototype.applyPose = function () {
                const pose = this.viewer.getPostures();
                api.fetchApi("/vnccs/character_studio/update_preview", {
                    method: "POST",
                    body: JSON.stringify({ manual_pose: pose, relative: false })
                }).then(r => r.json()).then(d => console.log("Applied", d));
            };

            nodeType.prototype.onDrawForeground = function (ctx) {
                if (!this.viewer || !this.viewer.canvas) return;
                const r = this.viewerRegion;
                ctx.drawImage(this.viewer.canvas, r.x, r.y, r.w, r.h);
                if (this.viewer.initialized) {
                    this.viewer.render();
                    this.setDirtyCanvas(true);
                }
            };

            nodeType.prototype.onMouseDown = function (e, pos) {
                if (!this.viewer) return;
                const r = this.viewerRegion;
                const lx = pos[0] - r.x;
                const ly = pos[1] - r.y;
                if (lx >= 0 && ly >= 0 && lx <= r.w && ly <= r.h) {
                    this.lastPos = [pos[0], pos[1]];
                    return this.viewer.onPointerDown(lx, ly, e.button);
                }
            };

            nodeType.prototype.onMouseMove = function (e, pos) {
                if (!this.viewer) return;
                const r = this.viewerRegion;
                const dx = pos[0] - this.lastPos[0];
                const dy = pos[1] - this.lastPos[1];
                this.lastPos = [pos[0], pos[1]];
                this.viewer.onPointerMove(pos[0] - r.x, pos[1] - r.y, dx, dy);
            };

            nodeType.prototype.onMouseUp = function (e, pos) {
                if (this.viewer) this.viewer.onPointerUp();
            };
        }
    }
});
