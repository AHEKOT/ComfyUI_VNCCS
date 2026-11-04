// Studio owns the scene and scheduling. Manager previews only change the capture destination.
export function mountLivePreviews(w) {
    const nativeRefresh = w.refreshAllManagerPreviews;
    const nativeReady = w.awaitReadyForCompositeCapture;
    const nativeUpdate = w.updatePoseManagerPreviewImage;
    const nativeLayout = w.layoutPoseManager;
    const nativeSchedule = w.scheduleAllManagerPreviewRefresh;
    const nativeCommit = w.commitViewerPoseToCurrentEditor;
    let exporting = false, pending = null, encodedGeneration = -1;
    let partialGeneration = 0, singleIndex = null;
    let live = null, home = null;
    const tile = i => w.managerGrid?.children[i]?.querySelector(".vnccs-ps-pose-preview");
    const canvas = i => tile(i)?.querySelector("canvas.vnccs-pm-live");
    const snapshot = source => {
        const copy = document.createElement("canvas");
        copy.className = "vnccs-pm-live";
        copy.width = source.width;
        copy.height = source.height;
        copy.getContext("2d").drawImage(source, 0, 0);
        return copy;
    };
    const release = (keep = true) => {
        if (live) {
            if (keep && live.parentElement) live.replaceWith(snapshot(live));
            home.append(live);
            live.classList.remove("vnccs-pm-live");
            live = home = null;
        }
        w.viewer?.endCaptureBatch?.();
    };
    const decorate = i => {
        const source = canvas(i);
        if (!source) return;
        tile(i).classList.add("vnccs-pm-live-preview");
        const detail = w.interfaceMode === "managerDetail"
            ? w.managerDetailStrip?.children[i]?.querySelector(".vnccs-ps-detail-card-preview") : null;
        if (detail) {
            detail.querySelector("canvas.vnccs-pm-live")?.remove();
            detail.append(snapshot(source));
            detail.classList.add("vnccs-pm-live-preview");
        }
    };
    for (const name of ["applyInterfaceMode", "dispose"]) {
        const original = w[name];
        w[name] = function () {
            release(name !== "dispose");
            return original.apply(this, arguments);
        };
    }
    const animate = w.viewer.animate;
    w.viewer.animate = function () {
        if (w.interfaceMode !== "manager" || exporting) return animate.apply(this, arguments);
        this._renderFrame = null;
        this._needsRender = false;
        if (w._managerPreviewRefreshFrame) cancelAnimationFrame(w._managerPreviewRefreshFrame);
        w._managerPreviewRefreshFrame = null;
        w.refreshAllManagerPreviews();
    };
    w.updatePoseManagerPreviewImage = function (i) {
        if (canvas(i)) decorate(i);
        else nativeUpdate.call(this, i);
    };
    w.layoutPoseManager = function () {
        const previous = this._lastManagerLayoutKey;
        nativeLayout.apply(this, arguments);
        if (this.interfaceMode === "manager" && previous !== this._lastManagerLayoutKey) this.scheduleAllManagerPreviewRefresh();
    };
    w.scheduleAllManagerPreviewRefresh = function () {
        singleIndex = null;
        const next = this._managerPreviewRefreshNextIndex || 0;
        const generation = nativeSchedule.apply(this, arguments);
        partialGeneration = 0;
        if (generation && !exporting && this._poseGestureActive && next > 0 && next < this.poses.length) {
            // Keep later cards moving during a drag; a complete pass still gates export.
            this._managerPreviewRefreshNextIndex = next;
            partialGeneration = generation;
        }
        return generation;
    };
    w.commitViewerPoseToCurrentEditor = function () {
        const isolated = this._poseCardLibraryOpen && this.interfaceMode === "manager" && !exporting
            && this._managerPreviewRefreshGeneration === this._managerPreviewRefreshCompletedGeneration;
        if (!isolated) return nativeCommit.apply(this, arguments);
        const refresh = this.refreshManagerPreviewsIfNeeded;
        // Persist through Studio, then invalidate only the card changed by the picker.
        this.refreshManagerPreviewsIfNeeded = () => this._managerPreviewRefreshGeneration;
        try { nativeCommit.apply(this, arguments); }
        finally { this.refreshManagerPreviewsIfNeeded = refresh; }
        this.scheduleAllManagerPreviewRefresh();
        singleIndex = this.activeTab;
    };
    w.refreshAllManagerPreviews = function (...args) {
        if (exporting) return nativeRefresh.apply(this, args);
        if (this._managerPreviewRefreshGeneration
            && this._managerPreviewRefreshGeneration === this._managerPreviewRefreshCompletedGeneration) return;
        if (singleIndex !== null && singleIndex !== this.activeTab) this.scheduleAllManagerPreviewRefresh();
        const viewer = this.viewer;
        const capture = viewer.capture, begin = viewer.beginCaptureBatch;
        const end = viewer.endCaptureBatch, requestRender = viewer.requestRender;
        const set = this.setPoseCapture, fit = this.computePoseManagerCaptureFraming;
        let index = singleIndex ?? (this._managerPreviewRefreshNextIndex || 0), capturing = false;
        const size = (width, height) => {
            const pixelRatio = globalThis.devicePixelRatio || 1;
            const scale = Math.min(1, (tile(index)?.clientWidth || 320) * pixelRatio / width,
                (tile(index)?.clientHeight || 480) * pixelRatio / height);
            return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
        };
        viewer.beginCaptureBatch = function (width, height) {
            return begin.apply(this, capturing ? [width, height] : size(width, height));
        };
        if (this.interfaceMode === "manager") {
            viewer.endCaptureBatch = () => {};
            viewer.requestRender = () => {};
            if (viewer._renderFrame) cancelAnimationFrame(viewer._renderFrame);
            viewer._renderFrame = null;
        }
        viewer.capture = function (...params) {
            if (singleIndex !== null) {
                const framing = w.computePoseManagerCaptureFraming(params[0], params[1], {
                    yaw_deg: params[6], pitch_deg: params[7],
                });
                if (!framing) throw new Error(`Pose ${index + 1} framing is not ready.`);
                params[2] = framing.zoom;
                params[4] = framing.offsetX;
                params[5] = framing.offsetY;
            }
            const preview = tile(index);
            if (!preview) throw new Error("Pose preview is not ready");
            let target = canvas(index);
            if (w.interfaceMode === "manager" && w.poses.length === 1) {
                if (!live) {
                    home = this.canvas.parentElement;
                    live = this.canvas;
                    target?.remove();
                    live.classList.add("vnccs-pm-live");
                    preview.append(live);
                }
                target = live;
            }
            if (!target) {
                target = document.createElement("canvas");
                target.className = "vnccs-pm-live";
                preview.append(target);
            }
            target.setAttribute("role", "img");
            target.setAttribute("aria-label", `Pose ${index + 1}`);
            [params[0], params[1]] = size(params[0], params[1]);
            if (params[3]?.every(channel => channel === 255)) params[3] = [3, 3, 5];
            params[8] = { ...params[8] };
            const encode = this.canvas.toDataURL;
            if (target === live) this.canvas.toDataURL = () => target;
            else params[8].targetCanvas = target;
            capturing = true;
            try { return capture.apply(this, params); }
            finally {
                capturing = false;
                if (target === live) this.canvas.toDataURL = encode;
            }
        };
        this.computePoseManagerCaptureFraming = function (...params) {
            const mesh = viewer.skinnedMesh, transform = mesh?.applyBoneTransform;
            if (!transform) return fit.apply(this, params);
            const owned = Object.hasOwn(mesh, "applyBoneTransform"), points = new Map();
            // Studio probes four cameras against the same mesh; skin each vertex once per fit.
            mesh.applyBoneTransform = function (i, point) {
                const cached = points.get(i);
                if (cached) return point.copy(cached);
                const result = transform.call(this, i, point);
                points.set(i, point.clone());
                return result;
            };
            try { return fit.apply(this, params); }
            finally {
                if (owned) mesh.applyBoneTransform = transform;
                else delete mesh.applyBoneTransform;
            }
        };
        this.setPoseCapture = function (i, value) {
            if (!value) throw new Error(`Pose ${i + 1} preview failed`);
            if (encodedGeneration !== this._managerPreviewRefreshGeneration) {
                this.forgetPoseManagerImageMetrics(this.poseCaptures[i]);
                this.poseCaptures[i] = null;
            }
            this.managerPoseMetrics[i] = { width: this.exportParams.view_width, height: this.exportParams.view_height, loading: false };
            index = i + 1;
            return true;
        };
        const completed = this._managerPreviewRefreshCompletedGeneration || 0;
        try {
            if (singleIndex !== null && viewer.isCaptureReady?.() !== false) {
                const i = singleIndex, generation = this._managerPreviewRefreshGeneration;
                const width = this.exportParams.view_width || 1024, height = this.exportParams.view_height || 1024;
                const camera = this.cameraParamsForPose(this.poses[i], this.getActiveCharacter());
                try {
                    this.syncToNode(false, { skipCapture: false });
                    this.updatePoseManagerPreviewImage(i);
                    if (generation === this._managerPreviewRefreshGeneration) {
                        this._managerPreviewRefreshCompletedGeneration = generation;
                        singleIndex = null;
                    }
                } catch (error) {
                    this._managerPreviewRefreshError = { generation, message: String(error?.message || error) };
                    this.showMessage?.(this._managerPreviewRefreshError.message, true);
                } finally {
                    viewer.updateLights(this.effectiveLights());
                    viewer.updateCaptureCamera?.(width, height, 1, 0, 0, camera.yaw_deg, camera.pitch_deg);
                }
                return;
            }
            const result = nativeRefresh.apply(this, args);
            if (partialGeneration && this._managerPreviewRefreshCompletedGeneration === partialGeneration) {
                this._managerPreviewRefreshCompletedGeneration = completed;
                partialGeneration = 0;
                this._managerPreviewRefreshFrame = requestAnimationFrame(() => {
                    this._managerPreviewRefreshFrame = null;
                    this.refreshAllManagerPreviews();
                });
            }
            return result;
        }
        finally {
            viewer.capture = capture;
            viewer.beginCaptureBatch = begin;
            viewer.endCaptureBatch = end;
            viewer.requestRender = requestRender;
            this.setPoseCapture = set;
            this.computePoseManagerCaptureFraming = fit;
        }
    };
    w.awaitReadyForCompositeCapture = function (...args) {
        if (pending) return pending;
        const prepare = async () => {
            await nativeReady.apply(this, args);
            if (!["manager", "managerDetail"].includes(this.interfaceMode)) return true;
            if (this.poseCaptures.length === this.poses.length && this.poseCaptures.every(value => typeof value === "string" && value)) return true;
            exporting = true;
            try {
                release();
                this.scheduleAllManagerPreviewRefresh();
                const result = await nativeReady.apply(this, args);
                encodedGeneration = this._managerPreviewRefreshCompletedGeneration;
                return result;
            } finally {
                exporting = false;
                if (!this._disposed) {
                    if (this._managerPreviewRefreshError) this.scheduleAllManagerPreviewRefresh();
                    else this.refreshAllManagerPreviews();
                }
            }
        };
        pending = prepare();
        const clear = () => { pending = null; };
        pending.then(clear, clear);
        return pending;
    };
    return { decorate, get: i => canvas(i) || w.poseCaptures[i],
        beforeRender: () => { if (live && w.poses.length !== 1) release(); } };
}
