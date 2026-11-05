const vector = value => Array.isArray(value) && value.length === 3 && value.every(n => Number.isFinite(n) && Math.abs(n) <= 1e6);

export const MESH_DEFAULTS = Object.freeze({ age: 25, gender: .5, weight: .5, muscle: .5, height: .5 });
export const LIGHT_DEFAULTS = Object.freeze({ x: 10, y: 20, z: 30, intensity: 1.25, color: "#ffffff" });

export function validateCharacters(value) {
    if (!Array.isArray(value) || value.length > 16) throw new Error("Invalid CG characters.");
    const ids = new Set();
    for (const item of value) {
        if (!item || !/^[a-f0-9]{32}$/.test(item.id) || ids.has(item.id) || !vector(item.position)
            || !Number.isFinite(item.scale) || item.scale < .001 || item.scale > 1000 || !item.pose || Array.isArray(item.pose))
            throw new Error("Invalid CG character transform.");
        ids.add(item.id);
        for (const key of ["bones", "bonePositions"]) {
            const bones = item.pose[key];
            if (!bones || typeof bones !== "object" || Array.isArray(bones) || Object.keys(bones).length > 256
                || Object.entries(bones).some(([name, value]) => !/^[A-Za-z0-9_.-]{1,96}$/.test(name) || Object.hasOwn(Object.prototype, name) || !vector(value)))
                throw new Error("Invalid CG character pose.");
        }
        if (item.mesh != null && (!item.mesh || typeof item.mesh !== "object" || Array.isArray(item.mesh)
            || Object.entries(item.mesh).some(([key, n]) => !Object.hasOwn(MESH_DEFAULTS, key) || !Number.isFinite(n)
                || n < (key === "age" ? 1 : 0) || n > (key === "age" ? 90 : key === "height" ? 2 : 1))))
            throw new Error("Invalid CG character mesh.");
        if (item.light != null && (!item.light || typeof item.light !== "object" || Array.isArray(item.light)
            || Object.entries(item.light).some(([key, n]) => key === "color" ? !/^#[0-9a-f]{6}$/i.test(n)
                : !["x", "y", "z", "intensity"].includes(key) || !Number.isFinite(n) || Math.abs(n) > 100
                    || key === "intensity" && (n < 0 || n > 5)))) throw new Error("Invalid CG character light.");
        if (item.source != null && (!item.source || typeof item.source !== "object" || Array.isArray(item.source)
            || Object.entries(item.source).some(([key, value]) => !["character", "costume", "sprite", "emotion"].includes(key)
                || typeof value !== "string" || value.length > 255 || /[\\/\x00-\x1f]/.test(value) || value === "." || value === "..")))
            throw new Error("Invalid CG character source.");
        if (item.origin != null && !vector(item.origin) || item.baseScale != null && (!Number.isFinite(item.baseScale)
            || item.baseScale < .001 || item.baseScale > 1000)) throw new Error("Invalid CG character placement.");
        if (!vector(item.pose.modelRotation)) throw new Error("Invalid CG character rotation.");
    }
    return value;
}

export async function loadCharacterRuntime(viewerURL) {
    const [core, morph, THREE, controls] = await Promise.all([
        import(new URL("../pose_studio/core.js", viewerURL).href),
        import(new URL("../pose_studio/morph_runtime.mjs", viewerURL).href),
        import(new URL("../vendor/spark/three.module.js", viewerURL).href),
        import(new URL("../vendor/spark/TransformControls.js", viewerURL).href),
    ]);
    const pack = await morph.loadMorphPack(), staticData = morph.buildStaticModelData(pack, false);
    const modelData = params => {
        const solved = morph.solveMorph(pack, { ...params, show_genitals: false });
        return { ...staticData, vertices: solved.vertices,
        bones: pack.bones.map((bone, index) => {
            const headPos = Array.from(solved.bonePositions.subarray(index * 6, index * 6 + 3));
            const tailPos = Array.from(solved.bonePositions.subarray(index * 6 + 3, index * 6 + 6));
            return { ...bone, headPos, tailPos, length: Math.hypot(...tailPos.map((n, i) => n - headPos[i])) };
        }) };
    };
    const data = modelData(MESH_DEFAULTS);
    // Studio supplies the rig and IK; Factory owns the only renderer and camera.
    class ScenePose extends core.PoseViewerCore {
        constructor(viewer) {
            super(viewer.canvas, { useHandControlPopover: false, enableTextureSkinning: false, showSkeletonHelper: false });
            this.viewer = viewer; this.THREE = THREE; this.scene = new THREE.Scene();
            this.camera = viewer.camera; this.orbit = { target: viewer.controls.target, enabled: true, update() {}, dispose() {} };
            this._raycaster = new THREE.Raycaster(); this._raycaster.layers.enable(1);
            // Route pointer input ourselves so Factory cannot also start a camera drag.
            this.transform = new controls.TransformControls(this.camera, document.createElement("div"));
            this.transform.setMode("rotate"); this.transform.setSpace("local"); this.transform.setSize(.7);
            this.scene.add(this.transform.getHelper());
            this.directDrag = { active: false };
            this.initialized = true; this.loadData(data, true);
            this.styleMesh();
            this.mesh = { ...MESH_DEFAULTS };
            // Dedicated lights keep the neutral mannequin visible with raw-splat lighting off.
            const ambient = new THREE.AmbientLight(0xffffff, .55), key = new THREE.DirectionalLight(0xffffff, 1.25);
            key.position.set(10, 20, 30); ambient.layers.set(1); key.layers.set(1); this.scene.add(ambient, key);
            this.keyLight = key; this.light = { ...LIGHT_DEFAULTS };
            viewer.camera.layers.enable(1); viewer.captureCamera.layers.enable(1);
            viewer.scene.add(this.scene);
        }
        styleMesh() {
            this.skinnedMesh.material.color.set("#c8c3ce");
            this.skinnedMesh.traverse(object => object.layers.set(1));
            this.skeletonHelper.visible = false;
        }
        setMesh(params) {
            const next = { ...MESH_DEFAULTS, ...params };
            if (JSON.stringify(next) === JSON.stringify(this.mesh)) return;
            const solved = morph.solveMorph(pack, { ...next, show_genitals: false });
            this.updateBodyVertices(solved.vertices, solved.bonePositions, solved.landmarks, solved.landmarkIndices);
            this.mesh = next; this.updateIKEffectorPositions();
        }
        setLight(params) {
            this.light = { ...LIGHT_DEFAULTS, ...params };
            this.keyLight.position.set(this.light.x, this.light.y, this.light.z);
            this.keyLight.intensity = this.light.intensity; this.keyLight.color.set(this.light.color);
            this.requestRender();
        }
        requestRender() { if (!this._disposed) this.viewer?.invalidate(); }
        recordState() {} // CG Creator owns one history for the camera and every character.
        dispose() {
            this.scene.removeFromParent();
            super.dispose();
            this.markerGeoNormal?.dispose(); this.markerGeoFinger?.dispose();
            this.markerMatNormal?.dispose(); this.markerMatSelected?.dispose(); this.markerMatHandHover?.dispose();
        }
    }
    let minY = Infinity, maxY = -Infinity;
    for (let i = 1; i < data.vertices.length; i += 3) { minY = Math.min(minY, data.vertices[i]); maxY = Math.max(maxY, data.vertices[i]); }
    return { THREE, ScenePose, minY, maxY };
}

export class CGCharacterEditor {
    constructor(viewer, runtime, { canEdit, onStart, onChange, onEnd }) {
        Object.assign(this, { viewer, runtime, canEdit, onStart, onChange, onEnd });
        this.entries = new Map(); this.selected = ""; this.mode = "pose"; this.events = new AbortController();
        const canvas = viewer.canvas;
        for (const type of ["pointerdown", "pointermove", "pointerup", "pointercancel", "lostpointercapture"])
            canvas.addEventListener(type, event => this.pointer(event), { capture: true, signal: this.events.signal });
    }
    get active() { return this.entries.get(this.selected); }
    snapshot() {
        return [...this.entries].map(([id, core]) => {
            const { bones, bonePositions } = core.getPose(), mesh = core.skinnedMesh;
            return { id, position: mesh.position.toArray(), scale: mesh.scale.x,
                mesh: { ...core.mesh }, light: { ...core.light }, source: { ...core.source },
                origin: [...core.origin], baseScale: core.baseScale,
                pose: { bones, bonePositions, modelRotation: [mesh.rotation.x, mesh.rotation.y, mesh.rotation.z].map(n => n * 180 / Math.PI) } };
        });
    }
    restore(characters) {
        validateCharacters(characters);
        const wanted = new Set(characters.map(item => item.id));
        for (const [id, core] of this.entries) if (!wanted.has(id)) { core.dispose(); this.entries.delete(id); }
        for (const item of characters) {
            let core = this.entries.get(item.id);
            if (!core) {
                core = new this.runtime.ScenePose(this.viewer); this.entries.set(item.id, core);
                core.transform.addEventListener("objectChange", () => {
                    if (this.mode === "pose" && core.selectedIKEffector) core.solveIKForEffector();
                    else if (this.mode === "pose" && core.selectedPoleTarget) core.solveIKForPoleTarget();
                    core.skinnedMesh.updateMatrixWorld(true); core.updateIKEffectorPositions();
                    core.requestRender(); this.onChange();
                });
            }
            core.transform.setMode(this.mode === "move" ? "translate" : "rotate");
            core.source = { ...item.source }; core.origin = [...(item.origin || item.position)]; core.baseScale = item.baseScale || item.scale;
            core.setMesh(item.mesh); core.setLight(item.light);
            core.skinnedMesh.position.fromArray(item.position); core.skinnedMesh.scale.setScalar(item.scale);
            core.setPose(item.pose, true); core.updateIKEffectorPositions();
        }
        this.entries = new Map(characters.map(item => [item.id, this.entries.get(item.id)]));
        this.select(wanted.has(this.selected) ? this.selected : characters.at(-1)?.id || "");
    }
    add(id) {
        if (this.entries.size) return;
        const THREE = this.runtime.THREE, camera = this.viewer.camera;
        const target = this.viewer.controls.target.clone(), distance = camera.position.distanceTo(target);
        const height = Math.max(.1, distance * Math.tan(camera.fov * Math.PI / 360));
        const scale = Math.max(.001, Math.min(1000, height / (this.runtime.maxY - this.runtime.minY)));
        // Center the new figure in the current view; subsequent placement is authored.
        target.add(new THREE.Vector3(0, -(this.runtime.minY + this.runtime.maxY) / 2 * scale, 0));
        this.restore([...this.snapshot(), { id, position: target.toArray(), scale,
            pose: { bones: {}, bonePositions: {}, modelRotation: [0, 0, 0] } }]);
        this.select(id); this.onChange();
    }
    select(id) {
        this.selected = this.entries.has(id) ? id : "";
        for (const core of this.entries.values()) {
            core.transform.detach(); core.selectedBone = null; core.selectedIKEffector = null; core.selectedPoleTarget = null;
        }
        if (this.mode === "move" && this.active) this.active.transform.attach(this.active.skinnedMesh);
        this.showHelpers(); this.viewer.invalidate();
    }
    setMode(mode) {
        this.mode = mode === "move" ? "move" : "pose";
        for (const core of this.entries.values()) core.transform.setMode(this.mode === "move" ? "translate" : "rotate");
        this.select(this.selected || this.entries.keys().next().value);
    }
    edit(change) {
        const core = this.entries.values().next().value;
        if (!core || !this.canEdit()) return;
        this.onStart(); change(core);
        core.skinnedMesh.updateMatrixWorld(true); core.updateIKEffectorPositions();
        this.showHelpers(); this.onChange();
    }
    reset() {
        this.edit(core => core.setPose({ bones: {}, bonePositions: {}, modelRotation: [0, 0, 0] }, true));
        this.select(this.entries.keys().next().value); this.onEnd();
    }
    showHelpers(hidden = false) {
        for (const [id, core] of this.entries) {
            const visible = !hidden && id === this.selected;
            core.jointMarkers.forEach(marker => { marker.visible = visible && this.mode === "pose"; });
            for (const helper of [...Object.values(core.ikController?.effectors || {}), ...Object.values(core.ikController?.poleTargets || {})]) helper.visible = false;
            core.transform.getHelper().visible = !hidden && id === this.selected && Boolean(core.transform.object);
        }
        this.viewer.invalidate();
    }
    async capture(options) {
        this.showHelpers(true);
        try { return await this.viewer.capturePreview(options); }
        finally { if (!this.disposed) this.showHelpers(); }
    }
    pointer(event) {
        if (!this.canEdit() || !this.entries.size) return;
        const canvas = this.viewer.canvas, rect = canvas.getBoundingClientRect();
        const pointer = { x: (event.clientX - rect.left) / rect.width * 2 - 1, y: 1 - (event.clientY - rect.top) / rect.height * 2, button: event.button };
        let core = this.active;
        if (event.type === "pointerdown") {
            if (event.button !== 0) return;
            core?.transform.pointerHover(pointer);
            if (!core?.transform.axis) {
                const ray = new this.runtime.THREE.Raycaster(); ray.layers.enable(1); ray.setFromCamera(pointer, this.viewer.camera);
                const meshes = [...this.entries.values()].flatMap(item => [item.skinnedMesh, ...item.jointMarkers.filter(marker => marker.visible)]);
                const hit = ray.intersectObjects(meshes, false)[0]?.object;
                const selected = [...this.entries].find(([, item]) => item.skinnedMesh === hit || item.jointMarkers.includes(hit));
                if (!selected) {
                    core?.transform.detach();
                    if (core) { core.selectedBone = core.selectedIKEffector = core.selectedPoleTarget = null; core.updateMarkers(); }
                    this.showHelpers(); return;
                }
                if (selected[0] !== this.selected) this.select(selected[0]);
                core = this.active;
            }
            this.onStart(); this.pointerId = event.pointerId;
            canvas.focus({ preventScroll: true }); canvas.setPointerCapture(event.pointerId);
            if (core.transform.axis) core.transform.pointerDown(pointer);
            else if (this.mode === "pose") core.handlePointerDown(event);
            else {
                this.select(this.selected);
                this.moveStart = core.skinnedMesh.position.clone();
                this.movePlane = new this.runtime.THREE.Plane().setFromNormalAndCoplanarPoint(
                    this.viewer.camera.getWorldDirection(new this.runtime.THREE.Vector3()), this.moveStart);
                const ray = new this.runtime.THREE.Raycaster(); ray.setFromCamera(pointer, this.viewer.camera);
                this.moveHit = ray.ray.intersectPlane(this.movePlane, new this.runtime.THREE.Vector3());
            }
        } else if (event.type === "pointermove") {
            if (this.pointerId == null) { core?.transform.pointerHover(pointer); return; }
            if (event.pointerId !== this.pointerId) return;
            if (core.transform.dragging) core.transform.pointerMove(pointer);
            else if (this.mode === "move" && this.moveHit) {
                const ray = new this.runtime.THREE.Raycaster(); ray.setFromCamera(pointer, this.viewer.camera);
                const hit = ray.ray.intersectPlane(this.movePlane, new this.runtime.THREE.Vector3());
                if (hit) { core.skinnedMesh.position.copy(this.moveStart).add(hit.sub(this.moveHit));
                    core.skinnedMesh.updateMatrixWorld(true); core.updateIKEffectorPositions(); core.requestRender(); this.onChange(); }
            } else if (this.mode === "pose") { core.handlePointerMove(event); this.onChange(); }
        } else {
            if (event.pointerId !== this.pointerId) return;
            this.pointerId = null;
            core.transform.pointerUp(pointer);
            if (this.mode === "pose") core.handlePointerUp(event, { cancelled: event.type !== "pointerup" });
            this.moveHit = null;
            if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
            this.onChange(); this.onEnd(); this.showHelpers();
        }
        event.preventDefault(); event.stopImmediatePropagation();
    }
    dispose() {
        this.disposed = true; this.events.abort();
        for (const core of this.entries.values()) core.dispose();
        this.entries.clear();
    }
}
