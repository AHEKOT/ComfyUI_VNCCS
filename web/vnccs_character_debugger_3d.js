import { createPoseBody, JOINT_TREE } from "./pose_editor_3d_body.js";
import { api } from "../../scripts/api.js";

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

function defaultDepth() {
    return 0;
}

class Pose3DEditor {
    static async create(container, options) {
        const modules = await ThreeModuleLoader.load();
        return new Pose3DEditor(container, modules, options);
    }

    constructor(container, modules, options) {
        this.container = container;
        this.options = options;
        this.canvasSize = options.canvas;
        this.defaultPose = options.defaultPose;
        this.connections = options.connections;
        this.callbacks = {
            onPoseChanged: options.onPoseChanged || (() => { }),
            onPoseCommitted: options.onPoseCommitted || (() => { }),
            onSelectJoint: options.onSelectJoint || (() => { }),
            onHoverJoint: options.onHoverJoint || (() => { })
        };

        this.THREE = modules.THREE;
        this.OrbitControls = modules.OrbitControls;
        this.TransformControls = modules.TransformControls;

        this.scene = new this.THREE.Scene();
        this.scene.background = new this.THREE.Color(0x0d1424);

        this.renderer = new this.THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(container.clientWidth || 1, container.clientHeight || 1, false);
        this.renderer.outputEncoding = this.THREE.sRGBEncoding;
        container.appendChild(this.renderer.domElement);

        const aspect = (container.clientWidth || 1) / (container.clientHeight || 1);
        this.camera = new this.THREE.PerspectiveCamera(38, aspect, 0.1, 5000);
        this.camera.position.set(0, 220, 900);

        this.orbit = new this.OrbitControls(this.camera, this.renderer.domElement);
        this.orbit.target.set(0, 200, 0);
        this.orbit.enableDamping = true;
        this.orbit.dampingFactor = 0.12;
        this.orbit.rotateSpeed = 0.95;
        this.orbit.update();

        this.transform = new this.TransformControls(this.camera, this.renderer.domElement);
        this.transform.setMode("translate");
        this.transform.setSpace("world");
        this.transform.setSize(0.75);
        this.scene.add(this.transform);

        this.transform.addEventListener("dragging-changed", (event) => {
            this.orbit.enabled = !event.value;
        });

        this.transform.addEventListener("mouseDown", () => {
            this.transforming = true;
        });

        this.transform.addEventListener("mouseUp", () => {
            this.transforming = false;
            this.updateBones();
            this.emitPoseChange();
            this.callbacks.onPoseCommitted();
            this.loadCharacterMesh(); // Update mesh with new pose
        });

        this.transform.addEventListener("objectChange", () => {
            this.updateBones();
            this.schedulePoseEmit();
            this.loadCharacterMesh(); // Real-time update
        });

        const ambient = new this.THREE.AmbientLight(0xffffff, 0.65);
        this.scene.add(ambient);
        const directional = new this.THREE.DirectionalLight(0xffffff, 0.72);
        directional.position.set(260, 420, 380);
        directional.castShadow = false;
        this.scene.add(directional);

        const rimLight = new this.THREE.DirectionalLight(0x70a8ff, 0.35);
        rimLight.position.set(-320, 140, -420);
        this.scene.add(rimLight);

        const grid = new this.THREE.GridHelper(1400, 28, 0x1e2c49, 0x111a2e);
        grid.position.y = -420;
        grid.material.opacity = 0.35;
        grid.material.transparent = true;
        this.scene.add(grid);

        this.currentDepth = {};

        this.bodyHelper = createPoseBody(this.THREE, {
            connections: this.connections,
            defaultDepth
        });

        this.root = this.bodyHelper.root;
        this.jointData = this.bodyHelper.jointData;
        this.jointNames = this.bodyHelper.jointNames;
        this.pickTargets = this.bodyHelper.pickTargets;
        this.scene.add(this.root);

        this.raycaster = new this.THREE.Raycaster();
        this.pointer = new this.THREE.Vector2();

        this.selectedJoint = null;
        this.hoveredJoint = null;
        this.transforming = false;
        this.active = true;
        this.frameHandle = null;
        this.poseEmitHandle = null;
        this.suppressPoseCallback = false;

        this.boundPointerDown = (event) => this.handlePointerDown(event);
        this.boundPointerMove = (event) => this.handlePointerMove(event);
        this.boundPointerUp = () => this.handlePointerUp();

        this.renderer.domElement.addEventListener("pointerdown", this.boundPointerDown);
        this.renderer.domElement.addEventListener("pointermove", this.boundPointerMove);
        window.addEventListener("pointerup", this.boundPointerUp);

        this.resizeObserver = new ResizeObserver(() => this.handleResize());
        this.resizeObserver.observe(container);

        this.animate = this.animate.bind(this);
        this.animate();

        this.setPose(this.defaultPose, true);
        this.frameAll();

        this.loadCharacterMesh();
    }

    setPose(joints, skipEmit = false) {
        if (!joints) {
            return;
        }
        this.suppressPoseCallback = true;
        this.root.updateMatrixWorld(true);

        for (const { name, parent } of JOINT_TREE) {
            const joint = this.jointData[name];
            const reference = joints[name] ?? this.defaultPose[name];
            if (!reference) {
                continue;
            }
            const value = Array.isArray(reference) ? reference : this.defaultPose[name];
            const x = Number(value[0]);
            const y = Number(value[1]);
            const world = new this.THREE.Vector3(
                x - this.canvasSize.width / 2,
                this.canvasSize.height / 2 - y,
                this.currentDepth[name] ?? joint.defaultDepth
            );
            if (Array.isArray(value) && value.length >= 3 && Number.isFinite(value[2])) {
                world.z = value[2];
                this.currentDepth[name] = value[2];
            }

            if (!parent) {
                joint.group.position.copy(world);
                joint.group.rotation.set(0, 0, 0);
                joint.group.scale.set(1, 1, 1);
            } else {
                const parentGroup = this.jointData[parent].group;
                const local = parentGroup.worldToLocal(world.clone());
                joint.group.position.copy(local);
                joint.group.rotation.set(0, 0, 0);
                joint.group.scale.set(1, 1, 1);
            }
            joint.group.updateMatrixWorld(true);
        }

        this.updateBones();
        this.updateJointMaterials();
        this.suppressPoseCallback = false;

        if (!skipEmit) {
            this.emitPoseChange();
        }
        if (this.selectedJoint) {
            const target = this.jointData[this.selectedJoint]?.group;
            if (target) {
                this.transform.attach(target);
            }
        }
        this.renderOnce();
    }

    updateBones() {
        this.bodyHelper.updateBones();
        this.renderOnce();
    }

    emitPoseChange() {
        if (this.suppressPoseCallback) {
            return;
        }
        this.root.updateMatrixWorld(true);
        const result = {};
        const world = new this.THREE.Vector3();
        for (const name of this.jointNames) {
            const joint = this.jointData[name];
            joint.group.getWorldPosition(world);
            this.currentDepth[name] = world.z;
            const x = world.x + this.canvasSize.width / 2;
            const y = this.canvasSize.height / 2 - world.y;
            result[name] = [x, y];
        }
        this.callbacks.onPoseChanged(result);
    }

    schedulePoseEmit() {
        if (this.poseEmitHandle) {
            return;
        }
        this.poseEmitHandle = requestAnimationFrame(() => {
            this.poseEmitHandle = null;
            this.emitPoseChange();
        });
    }

    handlePointerDown(event) {
        if (this.transforming) {
            return;
        }
        if (event.button !== 0) {
            return;
        }
        const name = this.pickJoint(event);
        this.setSelection(name, "3d");
    }

    handlePointerMove(event) {
        if (this.transforming) {
            return;
        }
        const name = this.pickJoint(event);
        this.setHover(name, "3d");
    }

    handlePointerUp() {
        this.transforming = false;
    }

    pickJoint(event) {
        const rect = this.renderer.domElement.getBoundingClientRect();
        if (!rect.width || !rect.height) {
            return null;
        }
        this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        this.raycaster.setFromCamera(this.pointer, this.camera);
        const intersections = this.raycaster.intersectObjects(this.pickTargets, false);
        const hit = intersections.find((i) => i.object?.userData?.jointName);
        return hit ? hit.object.userData.jointName : null;
    }

    setSelection(name, source = "api") {
        if (this.selectedJoint === name) {
            return;
        }
        this.selectedJoint = name;
        if (name) {
            const record = this.jointData[name];
            const joint = record?.group;
            if (joint && record?.sphere) {
                this.transform.attach(joint);
            }
        } else {
            this.transform.detach();
        }
        this.updateJointMaterials();
        this.callbacks.onSelectJoint(name, source);
        this.renderOnce();
    }

    setHover(name, source = "api") {
        if (this.hoveredJoint === name) {
            return;
        }
        this.hoveredJoint = name;
        this.updateJointMaterials();
        this.callbacks.onHoverJoint(name, source);
        this.renderOnce();
    }

    updateJointMaterials() {
        for (const data of Object.values(this.jointData)) {
            const material = data.material;
            if (!material) {
                continue;
            }
            if (data.name === this.selectedJoint) {
                material.color.setHex(0x8bd5ff);
                material.emissive.setHex(0x244c7a);
            } else if (data.name === this.hoveredJoint) {
                material.color.setHex(0xffc680);
                material.emissive.setHex(0x7a4b22);
            } else {
                material.color.setHex(data.baseColor);
                material.emissive.setHex(data.baseEmissive);
            }
            material.needsUpdate = true;
        }
    }

    setTransformMode(mode) {
        if (!mode) {
            return;
        }
        this.transform.setMode(mode);
        if (mode === "translate") {
            this.transform.setSpace("world");
        } else {
            this.transform.setSpace("local");
        }
    }

    resetView() {
        this.orbit.target.set(0, 200, 0);
        this.camera.position.set(0, 220, 900);
        this.camera.up.set(0, 1, 0);
        this.orbit.update();
        this.renderOnce();
    }

    focusSelection() {
        if (!this.selectedJoint) {
            this.frameAll();
            return;
        }
        const joint = this.jointData[this.selectedJoint];
        if (!joint) {
            return;
        }
        const world = new this.THREE.Vector3();
        joint.group.getWorldPosition(world);
        this.orbit.target.copy(world);
        const offset = this.camera.position.clone().sub(world);
        if (offset.length() < 160) {
            offset.setLength(160);
        }
        this.camera.position.copy(world.clone().add(offset));
        this.orbit.update();
        this.renderOnce();
    }

    frameAll(padding = 1.22) {
        this.root.updateMatrixWorld(true);
        const bounds = new this.THREE.Box3();
        const scratch = new this.THREE.Vector3();
        for (const name of this.jointNames) {
            const joint = this.jointData[name];
            if (!joint) {
                continue;
            }
            joint.group.getWorldPosition(scratch);
            bounds.expandByPoint(scratch);
        }

        if (bounds.isEmpty()) {
            this.resetView();
            return;
        }

        const size = new this.THREE.Vector3();
        bounds.getSize(size);
        size.multiplyScalar(padding);

        const center = new this.THREE.Vector3();
        bounds.getCenter(center);

        const halfVerticalFov = this.THREE.MathUtils.degToRad(this.camera.fov * 0.5);
        const halfHeight = Math.max(size.y * 0.5, 1);
        let distance = halfHeight / Math.tan(halfVerticalFov);

        const halfHorizontalFov = Math.atan(Math.tan(halfVerticalFov) * this.camera.aspect);
        const halfWidth = Math.max(size.x * 0.5, 1);
        distance = Math.max(distance, halfWidth / Math.tan(halfHorizontalFov));

        distance = Math.max(distance, 160);

        const currentDirection = this.camera.position.clone().sub(this.orbit.target);
        if (currentDirection.lengthSq() === 0) {
            currentDirection.set(0, 0, 1);
        }
        currentDirection.normalize();

        this.orbit.target.copy(center);
        this.camera.position.copy(center.clone().add(currentDirection.multiplyScalar(distance)));
        this.camera.updateProjectionMatrix();
        this.orbit.update();
        this.renderOnce();
    }

    handleResize() {
        const rect = this.container.getBoundingClientRect();
        const width = Math.max(rect.width, 1);
        const height = Math.max(rect.height, 1);
        this.renderer.setSize(width, height, false);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderOnce();
    }

    renderOnce() {
        if (!this.active) {
            return;
        }
        this.renderer.render(this.scene, this.camera);
    }

    animate() {
        if (!this.active) {
            return;
        }
        this.frameHandle = requestAnimationFrame(this.animate);
        this.orbit.update();
        this.renderer.render(this.scene, this.camera);
    }

    setActive(state) {
        if (this.active === state) {
            return;
        }
        this.active = state;
        if (state) {
            this.animate();
            this.renderOnce();
        } else if (this.frameHandle) {
            cancelAnimationFrame(this.frameHandle);
            this.frameHandle = null;
        }
    }

    dispose() {
        this.setActive(false);
        this.renderer.domElement.removeEventListener("pointerdown", this.boundPointerDown);
        this.renderer.domElement.removeEventListener("pointermove", this.boundPointerMove);
        window.removeEventListener("pointerup", this.boundPointerUp);
        this.resizeObserver.disconnect();
        this.renderer.dispose();
    }
    async loadCharacterMesh() {
        if (this.isMeshLoading) return; // Prevent concurrent requests
        this.isMeshLoading = true;

        try {
            const poseData = {};
            for (const name of this.jointNames) {
                const joint = this.jointData[name];
                const world = new this.THREE.Vector3();
                joint.group.getWorldPosition(world);
                const x = world.x + this.canvasSize.width / 2;
                const y = this.canvasSize.height / 2 - world.y;
                poseData[name] = { x: x, y: y, z: world.z };
            }

            const response = await api.fetchApi("/vnccs/character_studio/update_preview", {
                method: "POST",
                body: JSON.stringify({
                    // Defaults
                    age: 25.0,
                    gender: 0.5,
                    weight: 0.5,
                    muscle: 0.5,
                    height: 0.5,
                    breast_size: 0.5,
                    genital_size: 0.0,
                    pose: poseData
                })
            });

            if (!response.ok) {
                console.error("Failed to fetch character mesh:", response.status);
                return;
            }

            const data = await response.json();
            // Expected: { vertices: [x,y,z...], indices: [i1,i2,i3...], normals: [...] }

            const geometry = new this.THREE.BufferGeometry();

            // Vertices
            const vertices = new Float32Array(data.vertices);
            // CharacterStudio usually returns (N,3) list or flat list? 
            // The API code in __init__.py returns "new_verts.tolist()".
            // new_verts is (N,3). JSON will be [[x,y,z], [x,y,z]].

            // We need to flatten if it's nested
            const flatVertices = [];
            if (Array.isArray(data.vertices[0])) {
                for (let i = 0; i < data.vertices.length; i++) {
                    flatVertices.push(data.vertices[i][0], data.vertices[i][1], data.vertices[i][2]);
                }
            } else {
                // assume flat or let Float32Array handle it if it supports compatible types
                // But Float32Array([1,2,3]) works. Float32Array([[1,2]]) does NOT.
                // We should flatten.
                // Actually raw json is array of arrays from numpy tolist().
                flatVertices.push(...data.vertices.flat());
                // .flat() might be slow for big arrays? 
                // Better manual loop if huge, but let's trust .flat() or flat arrays from server.
                // Python tolist() creates list of lists for 2D array.
            }

            // Actually, let's optimize the flattening.
            // If data.vertices is array of arrays:
            let vData = data.vertices;
            if (vData.length > 0 && Array.isArray(vData[0])) {
                vData = vData.flat();
            }

            geometry.setAttribute('position', new this.THREE.Float32BufferAttribute(vData, 3));

            // Indices
            if (data.indices) {
                // Indices might be flattened already or not?
                // The API code: tri_indices is a flat list [f0, f1, f2, ...]
                geometry.setIndex(data.indices);
            }

            // Normals
            if (data.normals) {
                let nData = data.normals;
                if (nData.length > 0 && Array.isArray(nData[0])) {
                    nData = nData.flat();
                }
                geometry.setAttribute('normal', new this.THREE.Float32BufferAttribute(nData, 3));
            } else {
                geometry.computeVertexNormals();
            }

            // Material
            const material = new this.THREE.MeshStandardMaterial({
                color: 0xaaaaaa,
                metalness: 0.1,
                roughness: 0.6,
                flatShading: false,
                side: this.THREE.DoubleSide
            });

            const mesh = new this.THREE.Mesh(geometry, material);

            // Center the mesh? User said "strictly in center".
            // We assume the received mesh IS in standard MakeHuman coordinates (feet on ground, center 0,0).
            // We'll just add it at 0,0,0.

            // Scale? OpenPose Editor uses pixels (CANVAS_WIDTH ~512).
            // MakeHuman usually uses decimeters or meters. 
            // 1 unit in OpenPose editor = 1 pixel.
            // A character is ~170cm = 17dm = 1.7m.
            // If we render at scale 1, it will be tiny (1.7 pixels high).
            // We need to scale it UP to match the editor canvas.
            // Editor canvas is 512x1536.
            // A standing character should probably fill ~80% of height?
            // Say ~1200 pixels height.
            // If character is ~18 units tall (dm), we need scale ~ 66.

            // Let's create a scale control or guess standard scale.
            // In character_studio.py, MH_VisualDesigner used "scale" parameter.
            // Standard MakeHuman is unit = decimeter (10 units = 1 meter).
            // Typical height 17 units.
            // We want 17 units -> 1200 units (pixels).
            // 1200 / 17 ~ 70.
            // Let's try scale 10 first (if it's cm then 170 -> 1200 is scale 7).

            // The user said: "Я сам маплю точки... помещаем 3д модель строго по центру".
            // I'll set a default scale but keep it editable or just fixed.
            // I'll set scale to match roughly the OpenPose Skeleton visual.
            // The default OpenPose skeleton is spread across the 512x1536 canvas.

            mesh.scale.set(60, 60, 60); // Guess based on dm -> pixels
            mesh.position.set(0, -600, 0); // Move down so feet are at bottom?
            // Actually, OpenPose editor coordinates:
            // Y increases downwards (0 is top).
            // But Three.js scene in pose_editor_3d.js:
            // The camera is at (0, 220, 900).
            // The joint positions are converted:
            // world.y = this.canvasSize.height / 2 - y
            // So Y+ is UP. Center of canvas (height/2) is Y=0.
            // If character height is ~1700 px (full canvas), and center is waist (0),
            // head is at +850, feet at -850.
            // MakeHuman: Feet at 0, Head at 17.
            // To center it vertically: Move Y down by half height.
            // Or just put feet at approximately where skeleton feet are.
            // Skeleton feet (default): Y=1320 (in pixels from top).
            // Center is 1536/2 = 768.
            // Feet Y in world = 768 - 1320 = -552.
            // So if mesh feet are at 0, we need to move mesh to Y = -552.
            // And scale: 17 units -> ~1100 pixels (head to toe). Scale ~ 65.

            // Adjusted based on visual feedback (skeleton head ~ +560, feet ~ -530)
            // Character height ~1100 (Scale 65).
            // Position (0, -400, 0) puts feet at -400, Head at +700.
            // Let's try to match feet: Skeleton feet at -530.
            // So if we put mesh feet at -530: Mesh Position = (0, -530, 0).
            // Head will be at -530 + 1100 = +570.
            // Skeleton head is at +568.
            // This suggests Position Y = -530 is close to ideal alignment.
            // User complained it was "bottomed" at -550?
            // Wait, previous setting was -550.
            // If -550 was "too low" (skeleton above), then I need to move it UP.
            // wait, -550 (current) < -530 (proposed).
            // So moving from -550 to -530 is moving UP (positive direction).
            // But 20 units is tiny.
            // Screenshot showed a HUGE gap.
            // This implies my scaling/unit assumption is wrong.
            // Maybe scaling is closer to 10?
            // If Scale 65 -> Height 1100.
            // If Scale 10 -> Height 170.
            // 170 pixels is tiny on 1536 canvas.

            // Maybe the skeleton is NOT filling the canvas?
            // Default skeleton in `skeleton_512x1536.py`:
            // Head (Nose) ~ (256, 170). In 3D World Y: 768 - 170 = 598.
            // Ankle ~ (256, 1370). In 3D World Y: 768 - 1370 = -602.
            // Height = 1200 units.

            // So Scale 65 (Height ~1100) is roughly correct size.
            // Why gap?
            // Maybe the mesh was NOT at -550?
            // Ah, I set `mesh.position.set(0, -550, 0)`.
            // Maybe the mesh origin is NOT at feet.
            // If mesh origin is at CENTER (navel), and I put navel at -550...
            // Then head is at -550 + 550 = 0.
            // Feet at -550 - 550 = -1100.
            // Skeleton is +600 to -600.
            // My mesh is 0 to -1100.
            // Gap is 0 to 600 (top half of skeleton empty).
            // This MATCHES the screenshot! Skeleton floating above.

            // CONCLUSION: Mesh Origin is likely CENTER OF MASS (Navel/Hips).
            // To align Navel (Mesh Origin) with Skeleton Hips:
            // Skeleton Hips (MidHip) ~ (256, 768).
            // In 3D World Y: 768 - 768 = 0.
            // So Skeleton Hips are at Y=0.
            // If Mesh Origin is Hips, we should put Mesh at Y=0.

            mesh.scale.set(65, 65, 65);
            mesh.position.set(0, 0, 0); // Center at origin

            // Add to scene
            this.scene.add(mesh);
            this.renderOnce();

        } catch (e) {
            console.error("Error loading debug mesh:", e);
        } finally {
            this.isMeshLoading = false;
        }
    }

    exportDebugPoses() {
        const debugData = {};
        const world = new this.THREE.Vector3();

        // Collect all joint world positions
        for (const name of this.jointNames) {
            const joint = this.jointData[name];
            if (joint && joint.group) {
                joint.group.getWorldPosition(world);
                debugData[name] = {
                    x: world.x,
                    y: world.y,
                    z: world.z
                };
            }
        }

        const json = JSON.stringify(debugData, null, 2);
        console.log("Debug Poses:", json);

        // Use a prompt or alert to let user copy
        // Or create a temporary text area
        // prompt("Copy Debug Data:", json); 
        // Prompt text box length is limited in some browsers?

        // Better: Copy to clipboard API
        navigator.clipboard.writeText(json).then(() => {
            alert("Debug poses copied to clipboard!");
        }).catch(err => {
            console.error("Clipboard failed", err);
            alert("Check console for debug data");
        });
    }
}

export { Pose3DEditor };
