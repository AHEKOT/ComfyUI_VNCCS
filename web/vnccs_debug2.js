/**
 * VNCCS Debug2 - Three.js SkinnedMesh Viewer with Rotation Gizmo (TransformControls)
 * 
 * Uses esm.sh to ensure TransformControls and Three.js core share the same instance.
 * DIRECT CANVAS DISPATCH STRATEGY:
 * Instead of a MockDOM, we pass the real (offscreen) canvas to TransformControls.
 * We then manually dispatch native PointerEvents to this canvas from the ComfyUI widget callbacks.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// CDN Configuration
const THREE_VERSION = "0.160.0";
const THREE_SOURCES = {
    core: `https://esm.sh/three@${THREE_VERSION}?dev`,
    transform: `https://esm.sh/three@${THREE_VERSION}/examples/jsm/controls/TransformControls?dev`
};

class MakeHumanViewer {
    constructor(width, height) {
        this.width = width;
        this.height = height;

        // The "Real" DOM element (though offscreen in ComfyUI context)
        this.canvas = document.createElement('canvas');
        this.canvas.width = width;
        this.canvas.height = height;
        // Important: styling to ensure coordinate mapping works if it were mounted
        this.canvas.style.width = width + "px";
        this.canvas.style.height = height + "px";
        // Prevent default browser actions on this canvas if it ever gets events
        this.canvas.style.touchAction = 'none';

        this.THREE = null;
        this.TransformControls = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.transformControl = null;

        this.skinnedMesh = null;
        this.skeleton = null;
        this.bones = {};
        this.boneList = [];
        this.selectedBone = null;

        // Interaction Flags
        this.isGizmoDragging = false;
        this.isCameraDragging = false;
        this.leftButtonDown = false;

        this.mouse = { x: 0, y: 0 };
        this.initialized = false;

        // Manual Camera State
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
            const [core, transformLib] = await Promise.all([
                import(THREE_SOURCES.core),
                import(THREE_SOURCES.transform)
            ]);

            this.THREE = core;
            this.TransformControls = transformLib.TransformControls;

            this.camState.center = new this.THREE.Vector3(0, 10, 0);

            this.setupScene();
            this.initialized = true;
            console.log('VNCCS: Three.js + Gizmo initialized (Direct Canvas Strategy)');

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

        // TransformGizmo
        // Pass the REAL canvas. TransformControls will look for pointer events on this.
        this.transformControl = new this.TransformControls(this.camera, this.canvas);
        this.transformControl.addEventListener('dragging-changed', (event) => {
            this.isGizmoDragging = event.value;
            if (this.isGizmoDragging) {
                this.startRenderLoop();
            } else {
                this.checkStopRender();
            }
        });
        this.transformControl.addEventListener('change', () => this.render());

        this.scene.add(this.transformControl);

        // Lights
        const light = new THREE.DirectionalLight(0xffffff, 2);
        light.position.set(10, 20, 30);
        this.scene.add(light);
        this.scene.add(new THREE.AmbientLight(0x505050));

        // Grid
        const grid = new THREE.GridHelper(20, 20, 0x0f3460, 0x0f3460);
        this.scene.add(grid);
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

        const THREE = this.THREE;

        if (this.skinnedMesh) {
            this.scene.remove(this.skinnedMesh);
            this.skinnedMesh.geometry.dispose();
            this.skinnedMesh.material.dispose();
            if (this.skeletonHelper) this.scene.remove(this.skeletonHelper);
            this.transformControl.detach();
        }

        // 1. Geometry
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
        this.skeleton.calculateInverses();

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
        });

        this.skinnedMesh = new THREE.SkinnedMesh(geometry, material);
        rootBones.forEach(b => this.skinnedMesh.add(b));
        this.skinnedMesh.bind(this.skeleton);
        this.scene.add(this.skinnedMesh);

        this.skeletonHelper = new THREE.SkeletonHelper(this.skinnedMesh);
        this.scene.add(this.skeletonHelper);

        this.render();
    }

    // Dispatch REAL PointerEvents to the canvas
    dispatchPointerEvent(type, x, y, button) {
        if (!this.canvas) return;

        // Pointer IDs: Mouse is usually 1.
        // We simulate a real PointerEvent.

        const eventInit = {
            bubbles: true,
            cancelable: true,
            clientX: x, // Important: TransformControls uses clientX/Y from event
            clientY: y,
            pointerId: 1,
            pointerType: 'mouse',
            button: button,
            buttons: this.leftButtonDown ? 1 : 0, // Bitmask
            isPrimary: true
            // In a real browser, clientX/Y are global. 
            // TransformControls logic: 
            //   scope.domElement.ownerDocument.addEventListener( 'pointermove', onPointerMove );
            //   
            // It uses event.clientX - rect.left
            // Since our canvas is offscreen, getBoundingClientRect() might return 0,0,0,0.
            // But we can monkey-patch getBoundingClientRect on our canvas instance!
        };

        // Trick: If we dispatch standard events, we need the logic inside TC to work.
        // TC calls rect = domElement.getBoundingClientRect();
        // then ( event.clientX - rect.left ) / rect.width * 2 - 1 ...
        // So we need clientX to be relative to the "rect" we fake.

        // Let's assume we pass "local" widget coordinates as x,y (0..width, 0..height).
        // Then we want rect.left = 0, rect.top = 0.
        // So clientX = x, clientY = y works perfectly.

        const evt = new PointerEvent(type, eventInit);
        this.canvas.dispatchEvent(evt);
    }

    onWidgetMouseDown(x, y, button) {
        if (!this.initialized) return;

        // Monkey-patch rect if not already done, or update it
        if (!this.canvas.getBoundingClientRect._patched) {
            this.canvas.getBoundingClientRect = () => ({
                left: 0, top: 0,
                width: this.width, height: this.height,
                right: this.width, bottom: this.height,
                x: 0, y: 0
            });
            this.canvas.getBoundingClientRect._patched = true;
        }

        if (button === 0) this.leftButtonDown = true;

        // 1. Dispatch to Gizmo first
        this.dispatchPointerEvent('pointerdown', x, y, button);

        // Check if Gizmo handles it (hovering over axis)
        const isGizmoHovered = (this.transformControl && this.transformControl.axis !== null);

        if (this.isGizmoDragging || isGizmoHovered) {
            this.startRenderLoop();
            return true;
        }

        if (button === 0) { // Left Click
            const THREE = this.THREE;
            const mouse = new THREE.Vector2((x / this.width) * 2 - 1, -(y / this.height) * 2 + 1);
            const ray = new THREE.Raycaster();
            ray.setFromCamera(mouse, this.camera);

            const intersects = ray.intersectObject(this.skinnedMesh, true);
            let selectedABone = false;

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

                if (nearest && minD < 3.0) {
                    this.selectedBone = nearest;
                    if (this.transformControl) {
                        this.transformControl.attach(nearest);
                        this.transformControl.setMode('rotate');
                    }
                    console.log("Selected:", nearest.name);
                    selectedABone = true;
                }
            }

            if (!selectedABone) {
                this.isCameraDragging = true;
            }
        } else {
            // Right click or others -> Camera
            this.isCameraDragging = true;
        }

        this.startRenderLoop();
        return true;
    }

    onWidgetMouseMove(x, y, dx, dy) {
        // Dispatch to Gizmo!
        this.dispatchPointerEvent('pointermove', x, y, 0);

        if (this.isCameraDragging) {
            const sensitivity = 0.005;
            this.camState.rotY -= dx * sensitivity;
            this.camState.rotX += dy * sensitivity;
            this.camState.rotX = Math.max(-1.5, Math.min(1.5, this.camState.rotX));
            this.updateCamera();
        }
    }

    onWidgetMouseUp(x, y) {
        this.leftButtonDown = false;
        this.dispatchPointerEvent('pointerup', x, y, 0);
        this.isCameraDragging = false;
        setTimeout(() => this.checkStopRender(), 100);
    }

    startRenderLoop() {
        if (!this.renderLoop) {
            const loop = () => {
                this.render();
                if (this.leftButtonDown || this.isGizmoDragging || this.isCameraDragging) {
                    this.renderLoop = requestAnimationFrame(loop);
                } else {
                    this.renderLoop = null;
                }
            };
            this.renderLoop = requestAnimationFrame(loop);
        }
    }

    checkStopRender() {
        if (!this.leftButtonDown && !this.isGizmoDragging && !this.isCameraDragging) {
            // Loop checks flags itself
        }
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
                    return this.viewer.onWidgetMouseDown(lx, ly, e.button);
                }
            };

            nodeType.prototype.onMouseMove = function (e, pos) {
                if (!this.viewer) return;
                const r = this.viewerRegion;
                const lx = pos[0] - r.x;
                const ly = pos[1] - r.y;
                const dx = pos[0] - this.lastPos[0];
                const dy = pos[1] - this.lastPos[1];
                this.lastPos = [pos[0], pos[1]];
                this.viewer.onWidgetMouseMove(lx, ly, dx, dy);
            };

            nodeType.prototype.onMouseUp = function (e, pos) {
                if (!this.viewer) return;
                const r = this.viewerRegion;
                const lx = pos[0] - r.x;
                const ly = pos[1] - r.y;
                this.viewer.onWidgetMouseUp(lx, ly);
            };
        }
    }
});
