import { createPoseBody, JOINT_TREE } from "./pose_editor_3d_body.js";

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
            onPoseChanged: options.onPoseChanged || (() => {}),
            onPoseCommitted: options.onPoseCommitted || (() => {}),
            onSelectJoint: options.onSelectJoint || (() => {}),
            onHoverJoint: options.onHoverJoint || (() => {})
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
        });

        this.transform.addEventListener("objectChange", () => {
            this.updateBones();
            this.schedulePoseEmit();
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
}

export { Pose3DEditor };
