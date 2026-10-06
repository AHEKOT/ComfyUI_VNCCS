/**
 * VNCCS Common Utilities — shared patterns for all VNCCS widgets.
 * Import: import { debounce, showModal, ... } from "./vnccs_common.js";
 */
import { vnccsApi as api, mediaURL, storage, serverRegistry } from "./vnccs_transport.js";
import { app } from "../../scripts/app.js";

// ── Shared Input Normalization ───────────────────────────────────────────────
export const SAKURA_THEME_CSS = `
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-elevated: #1a1a26;
    --bg-surface: #22222e;
    --bg-hover: #2a2a38;
    --text-primary: #e8e8f0;
    --text-secondary: #9898a8;
    --text-muted: #5e5e70;
    --accent: #ff8fa3;
    --accent-hover: #ffb6c8;
    --accent-glow: rgba(255, 143, 163, 0.3);
    --accent-subtle: rgba(255, 143, 163, 0.1);
    --accent-border: rgba(255, 143, 163, 0.22);
    --accent-lavender: #b8a9e8;
    --success: #00d68f;
    --error: #ff4757;
    --border: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(255, 255, 255, 0.12);
    --font: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --transition: 0.2s ease;
`;

const RESOLUTION_SCALE_BASE = 1024;
export const RESOLUTION_SCALE_MIN_MP = 1;
export const RESOLUTION_SCALE_MAX_MP = 4;
export const RESOLUTION_SCALE_STEP_MP = 0.1;
const RESOLUTION_SCALE_PRESETS = new Map([
    [1.3, 1344],
    [1.5, 1536],
]);
export const resolutionScaleMegapixels = value => {
    const numeric = Number(value);
    const megapixels = Number.isFinite(numeric) ? numeric / RESOLUTION_SCALE_BASE : RESOLUTION_SCALE_MIN_MP;
    return Math.max(RESOLUTION_SCALE_MIN_MP, Math.min(RESOLUTION_SCALE_MAX_MP, megapixels));
};
export const resolutionScaleValue = megapixels => {
    const clamped = Math.max(
        RESOLUTION_SCALE_MIN_MP,
        Math.min(RESOLUTION_SCALE_MAX_MP, Number(megapixels) || RESOLUTION_SCALE_MIN_MP)
    );
    const stepped = Number((Math.round(clamped / RESOLUTION_SCALE_STEP_MP) * RESOLUTION_SCALE_STEP_MP).toFixed(1));
    return RESOLUTION_SCALE_PRESETS.get(stepped) ?? Math.round(stepped * RESOLUTION_SCALE_BASE);
};
export const resolutionScaleText = value => `${resolutionScaleMegapixels(value).toFixed(1)} MP`;

// Generator restoration falls back to the minimum for non-finite values.
export function finiteResolutionScaleValue(megapixels) {
    const numeric = Number(megapixels);
    return resolutionScaleValue(Number.isFinite(numeric) ? numeric : RESOLUTION_SCALE_MIN_MP);
}

export function normalizeUploadFile(file, prefix = "vnccs_upload") {
    const originalName = String(file?.name || "").trim();
    const extMatch = originalName.match(/(\.[A-Za-z0-9]{1,8})$/);
    const ext = extMatch ? extMatch[1] : ".png";
    let name = originalName.replace(/[\\/]/g, "_").trim();
    name = name.replace(/^[\s.-]+/, "");
    name = name.replace(/\s+/g, "_");
    name = name.replace(/[^A-Za-z0-9._-]/g, "_");
    if (!name || !/[A-Za-z0-9]/.test(name)) {
        name = `${prefix}_${Date.now()}${ext}`;
    }
    if (name !== originalName) {
        name = `${prefix}_${name}`;
    }
    return name === file.name ? file : new File([file], name, {
        type: file.type,
        lastModified: file.lastModified,
    });
}

export function normalizeAgeValue(value) {
    const parsed = parseFloat(value);
    if (!Number.isFinite(parsed)) return 18;
    return Math.max(1, Math.min(100, parsed));
}

// Each widget keeps its own view state and polling cleanup.
export function createControlCenterClient(repoId, hasRequiredFamilies, state, render) {
    const cacheKey = `vnccs_cc_cache_${repoId}`;
    let interval = null;
    const fetchConfig = async (force = false) => {
        if (!force && serverRegistry("VNCCS_CC_REGISTRY")?.[repoId] && hasRequiredFamilies(serverRegistry("VNCCS_CC_REGISTRY")[repoId])) {
            state.config = serverRegistry("VNCCS_CC_REGISTRY")[repoId];
            render();
            return state.config;
        }
        if (!force && state.config && hasRequiredFamilies(state.config)) return state.config;
        if (!force) {
            try {
                const cached = storage.getItem(cacheKey);
                if (cached) {
                    state.config = JSON.parse(cached);
                    if (hasRequiredFamilies(state.config)) render();
                    else state.config = null;
                }
            } catch (_) {}
        }

        const url = `/vnccs/control_center/check?repo_id=${encodeURIComponent(repoId)}${force ? "&force_refresh=true" : ""}`;
        const response = await api.fetchApi(url);
        const payload = await response.json();
        if (!response.ok || payload.error) throw new Error(payload.error || "Failed to load Control Center config");
        state.config = payload;

        serverRegistry("VNCCS_CC_REGISTRY")[repoId] = payload;
        storage.setItem(cacheKey, JSON.stringify(payload));
        render();
        return payload;
    };

    const refreshDownloadStatus = async () => {
        try {
            const response = await api.fetchApi("/vnccs/manager/status");
            if (!response.ok) return;
            state.downloadStatus = await response.json();
            const active = Object.values(state.downloadStatus || {}).some(item => ["queued", "downloading"].includes(item?.status));
            if (!active) {
                stopPolling();
                await fetchConfig(true);
            } else {
                render();
            }
        } catch (_) {}
    };

    const startPolling = () => {
        if (interval) return;
        interval = setInterval(refreshDownloadStatus, 2000);
    };

    const stopPolling = () => {
        if (!interval) return;
        clearInterval(interval);
        interval = null;
    };
    return { fetchConfig, startPolling, stopPolling };
}

// Each widget owns its dialog styling; model preparation and polling are shared.
export function createQwenVLModelLoader(node, showModal, createProgressModal, vision = true) {
    let disposed = false;
    const cleanups = new Set();
    registerCleanup(node, () => {
        disposed = true;
        for (const cleanup of cleanups) cleanup();
        cleanups.clear();
    });
    return async () => {
        if (disposed) return false;
        let overlay, timer, resolvePending, closing = false;
        const cancel = () => {
            clearTimeout(timer);
            overlay?.remove();
            resolvePending?.(false);
        };
        cleanups.add(cancel);
        try {
            const suffix = vision ? "" : "?vision=false";
            const response = await api.fetchApi(`/vnccs/qwen_vl_model_status${suffix}`);
            if (disposed) return false;
            if (!response.ok) throw new Error("Failed to check Qwen3.5 model files.");
            const modelStatus = await response.json();
            if (disposed) return false;
            if (modelStatus.ready) return true;
            const approved = await new Promise(resolve => {
                resolvePending = resolve;
                const dialog = showModal("Qwen3.5 Model Required", () => {
                    const text = document.createElement("div");
                    text.textContent = `${modelStatus.message || modelStatus.model_name} Download the required files from Hugging Face now?`;
                    return text;
                }, [
                    { text: "Cancel", action: () => { resolve(false); return false; } },
                    { text: "DOWNLOAD & INSTALL", class: "primary", action: () => { resolve(true); return false; } },
                ]);
                overlay = dialog.overlay;
                dialog.modal.addEventListener("keydown", event => {
                    if (event.key === "Escape") resolve(false);
                }, true);
            });
            if (!approved || disposed) return false;
            const start = await api.fetchApi(`/vnccs/qwen_vl_download_model${suffix}`, { method: "POST" });
            if (disposed) return false;
            if (!start.ok && start.status !== 409) {
                let error;
                try { error = await start.json(); } catch (_) { error = { error: await start.text() }; }
                throw new Error(error?.error || error?.message || "Failed to start QwenVL download.");
            }
            const progressModal = createProgressModal();
            overlay = progressModal.overlay;
            const { statusEl, barEl, pctEl, completedMessage } = progressModal;
            return await new Promise((resolve, reject) => {
                resolvePending = resolve;
                const poll = async () => {
                    try {
                        const response = await api.fetchApi("/vnccs/qwen_vl_download_status");
                        if (disposed) return;
                        if (!response.ok) throw new Error(await response.text());
                        const data = await response.json();
                        if (disposed) return;
                        const progress = Math.max(0, Math.min(100, Number(data.progress) || 0));
                        statusEl.innerText = data.current_file ? `Downloading ${data.current_file}...` : "Preparing model files...";
                        barEl.style.width = `${progress}%`;
                        pctEl.innerText = `${progress}%`;
                        if (data.status === "completed") {
                            statusEl.innerText = completedMessage;
                            barEl.style.width = "100%";
                            pctEl.innerText = "100%";
                            closing = true;
                            timer = setTimeout(() => { overlay.remove(); cleanups.delete(cancel); }, 450);
                            resolve(true);
                            return;
                        }
                        if (data.status === "error") throw new Error(data.error || "QwenVL download failed.");
                        timer = setTimeout(poll, 700);
                    } catch (error) {
                        overlay.remove();
                        reject(error);
                    }
                };
                poll();
            });
        } finally {
            if (!closing) cleanups.delete(cancel);
        }
    };
}

export function createTraitInput({ prefix, label: lbl, key, target: targetObj, save, choose }) {
    const wrap = document.createElement("div");
    wrap.className = `${prefix}-trait-row`;
    const label = document.createElement("span");
    label.className = `${prefix}-trait-label`;
    label.textContent = lbl;
    const editor = document.createElement("div");
    editor.className = `${prefix}-trait-editor`;
    const values = document.createElement("button");
    values.type = "button";
    values.className = `${prefix}-trait-values`;
    const inp = document.createElement("input");
    inp.type = "text";
    inp.className = `${prefix}-input ${prefix}-trait-input`;
    inp.setAttribute("aria-label", lbl);
    inp.placeholder = "Add tags";
    inp.hidden = true;
    const renderTags = () => {
        values.replaceChildren();
        const tokens = inp.value.split(",").map(token => token.trim()).filter(Boolean);
        for (const token of tokens.length ? tokens : ["Add tags"]) {
            const chip = document.createElement("span");
            chip.className = tokens.length ? `${prefix}-trait-token` : `${prefix}-trait-empty`;
            chip.textContent = token;
            values.appendChild(chip);
        }
        values.setAttribute("aria-label", `Edit ${lbl.toLowerCase()} tags: ${inp.value || "Add tags"}`);
    };
    inp.setValue = value => {
        inp.value = value ?? "";
        renderTags();
    };
    inp.setValue(targetObj[key]);
    inp.oninput = (e) => {
        targetObj[key] = e.target.value;
        renderTags();
        save();
    };
    inp.startEditing = () => {
        values.hidden = true;
        inp.hidden = false;
        inp.focus({ preventScroll: true });
    };
    values.onclick = inp.startEditing;
    inp.onblur = () => {
        inp.hidden = true;
        values.hidden = false;
    };
    inp.onkeydown = e => {
        if (e.key === "Enter") {
            e.preventDefault();
            inp.blur();
            values.focus({ preventScroll: true });
        }
    };
    const add = document.createElement("button");
    add.type = "button";
    add.className = `${prefix}-trait-add`;
    add.textContent = "+";
    add.setAttribute("aria-label", `Choose ${lbl.toLowerCase()} presets`);
    add.title = "Choose Presets";
    add.onclick = () => choose(inp);
    editor.append(values, inp);
    wrap.append(label, editor, add);
    return { element: wrap, input: inp };
}

// ── Debounce ──────────────────────────────────────────────────────────────────
export function debounce(fn, delay = 300) {
    let timer;
    return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), delay);
    };
}

// ── Node Cleanup Registry ─────────────────────────────────────────────────────
// Accumulates cleanup functions per node; hooks into node.onRemoved once.
export function registerCleanup(node, cleanupFn) {
    if (!node._vnccsCleanups) {
        node._vnccsCleanups = [];
        const orig = node.onRemoved;
        node.onRemoved = function () {
            for (const fn of node._vnccsCleanups) {
                try { fn(); } catch (e) { console.warn("[VNCCS] cleanup error:", e); }
            }
            node._vnccsCleanups = null;
            if (orig) orig.apply(this, arguments);
        };
    }
    node._vnccsCleanups.push(cleanupFn);
}

// Each loader owns a guard. Starting a newer request or removing the node
// invalidates every older response, including requests for the same selection.
export function createRequestGuard(node) {
    let sequence = 0;
    let removed = false;
    registerCleanup(node, () => { removed = true; sequence += 1; });
    return () => {
        const request = ++sequence;
        return () => !removed && request === sequence;
    };
}

// ── DOM Widget Width Sync ────────────────────────────────────────────────────
// ComfyUI/LiteGraph can restore stale DOM widget widths from older layouts.
// Keep DOM widgets tied to the current node width instead.
export function syncDOMWidgetWidth(node, widgetName) {
    const widget = node?.widgets?.find(w => w.name === widgetName);
    const nodeWidth = Number(node?.size?.[0]);

    if (widget && Number.isFinite(nodeWidth) && nodeWidth > 0) {
        if (!widget._vnccsWidthBound) {
            Object.defineProperty(widget, "width", {
                configurable: true,
                get() {
                    const width = Number(this._node?.size?.[0] ?? node?.size?.[0]);
                    return Number.isFinite(width) && width > 0 ? width : undefined;
                },
                set(_value) {
                    // Ignore stale widths restored by ComfyUI/LiteGraph.
                }
            });
            widget._vnccsWidthBound = true;
        }

        if (typeof widget.triggerDraw === "function") {
            widget.triggerDraw();
        }
    }
}

export function syncDOMWidgetWidthSoon(node, widgetName, delay = 100) {
    syncDOMWidgetWidth(node, widgetName);
    requestAnimationFrame(() => syncDOMWidgetWidth(node, widgetName));
    setTimeout(() => syncDOMWidgetWidth(node, widgetName), delay);
}

// ── Character Sprite Preview Navigation ──────────────────────────────────────
export function createSpritePreviewNavigator({
    node,
    isSelectionCurrent = () => true,
    image,
    placeholder,
    loading,
    nav,
    prevButton,
    nextButton,
    countLabel,
    onLoaded,
    onMissing,
    onError,
} = {}) {
    const state = {
        character: "",
        costume: "",
        count: 0,
        index: 0,
        cacheBust: "",
        requestId: 0,
    };
    let disposed = false;
    const isCurrent = (requestId) => !disposed && requestId === state.requestId && isSelectionCurrent(state);
    const invalidate = () => { state.requestId += 1; };
    if (node) registerCleanup(node, () => { disposed = true; invalidate(); });

    const setLoading = (value) => {
        loading?.classList.toggle("is-visible", !!value);
        if (prevButton) prevButton.disabled = !!value;
        if (nextButton) nextButton.disabled = !!value;
    };

    const updateNav = () => {
        const visible = state.count > 1;
        nav?.classList.toggle("is-visible", visible);
        if (countLabel) countLabel.textContent = visible ? `${state.index + 1}/${state.count}` : "";
    };

    const hideNav = () => {
        state.count = 0;
        state.index = 0;
        updateNav();
    };

    const spriteUrl = (character, index, costume = "") => {
        const params = new URLSearchParams({
            character,
            index: String(index),
            v: state.cacheBust || "current",
        });
        if (costume) params.set("costume", costume);
        return mediaURL(`/vnccs/get_character_pose_preview?${params.toString()}`);
    };

    const applyImage = (url) => {
        if (image) {
            image.src = url;
            image.style.display = "block";
        }
        if (placeholder) placeholder.style.display = "none";
        onLoaded?.(url, { ...state });
    };

    const applyMissing = () => {
        if (image) image.style.display = "none";
        if (placeholder) placeholder.style.display = "flex";
        hideNav();
        onMissing?.({ ...state });
    };

    const prefetch = (index) => {
        if (!state.character || state.count <= 1) return;
        const normalized = ((Number(index || 0) % state.count) + state.count) % state.count;
        const img = new Image();
        img.src = spriteUrl(state.character, normalized, state.costume);
    };

    const showFallback = (url, { character = state.character, costume = state.costume } = {}) => {
        url = mediaURL(url);
        if (disposed || !isSelectionCurrent({ ...state, character, costume })) return;
        state.character = character;
        state.costume = costume;
        const requestId = state.requestId + 1;
        state.requestId = requestId;
        if (!url) {
            applyMissing();
            return;
        }
        hideNav();
        setLoading(true);
        const loader = new Image();
        loader.onerror = () => {
            if (!isCurrent(requestId)) return;
            setLoading(false);
            applyMissing();
        };
        loader.onload = () => {
            if (!isCurrent(requestId)) return;
            setLoading(false);
            applyImage(url);
        };
        loader.src = url;
    };

    const show = (index) => {
        if (disposed || !isSelectionCurrent(state)) return;
        if (!state.character || state.count <= 0) return;
        const normalized = ((Number(index || 0) % state.count) + state.count) % state.count;
        const requestId = state.requestId + 1;
        const url = spriteUrl(state.character, normalized, state.costume);
        state.requestId = requestId;
        state.index = normalized;
        setLoading(true);
        updateNav();

        const loader = new Image();
        loader.onerror = () => {
            if (!isCurrent(requestId)) return;
            setLoading(false);
            showFallback(state.fallbackUrl);
            onError?.({ ...state });
        };
        loader.onload = () => {
            if (!isCurrent(requestId)) return;
            setLoading(false);
            applyImage(url);
            updateNav();
            prefetch(normalized - 1);
            prefetch(normalized + 1);
        };
        loader.src = url;
    };

    const load = async (character, { costume = "", random = true, index = 0, fallbackUrl = "" } = {}) => {
        if (disposed) return;
        state.character = character || "";
        state.costume = costume || "";
        state.fallbackUrl = mediaURL(fallbackUrl || "");
        state.cacheBust = `${state.character}:${state.costume}:${Date.now()}`;
        const loadRequestId = state.requestId + 1;
        state.requestId = loadRequestId;

        if (!state.character) {
            setLoading(false);
            applyMissing();
            return;
        }

        setLoading(true);
        try {
            const params = new URLSearchParams({ character: state.character, t: String(Date.now()) });
            if (state.costume) params.set("costume", state.costume);
            const response = await api.fetchApi(`/vnccs/get_character_pose_preview_meta?${params.toString()}`);
            if (!isCurrent(loadRequestId)) return;
            const meta = response.ok ? await response.json() : {};
            if (!isCurrent(loadRequestId)) return;
            state.count = Number(meta.count || 0);
            if (state.count <= 0) {
                setLoading(false);
                showFallback(state.fallbackUrl);
                return;
            }
            const nextIndex = random ? Math.floor(Math.random() * state.count) : index;
            show(nextIndex);
        } catch (error) {
            if (!isCurrent(loadRequestId)) return;
            console.warn("[VNCCS] Failed to load sprite preview metadata:", error);
            setLoading(false);
            showFallback(state.fallbackUrl);
            onError?.({ ...state, error });
        }
    };

    prevButton?.addEventListener("click", () => show(state.index - 1));
    nextButton?.addEventListener("click", () => show(state.index + 1));
    updateNav();

    return {
        load,
        invalidate,
        show,
        showFallback,
        hideNav,
        setLoading,
        state,
    };
}

// ── DOM Widget Canvas Navigation ──────────────────────────────────────────────
// DOM widgets sit above LiteGraph's canvas, so MMB events can never reach the
// canvas naturally. Forward canvas navigation gestures while leaving normal
// widget interaction and real scroll containers untouched.
export function enableMiddleMouseCanvasPan(root, node = root?._node) {
    if (!root || root._vnccsMiddleMouseCanvasPan) return;
    root._vnccsMiddleMouseCanvasPan = true;

    const canvas = () => app.canvasEl || app.canvas?.canvas || document.querySelector("canvas.litegraph");
    const pointerEvents = typeof window.PointerEvent === "function";
    const downEvent = pointerEvents ? "pointerdown" : "mousedown";
    const moveEvent = pointerEvents ? "pointermove" : "mousemove";
    const upEvent = pointerEvents ? "pointerup" : "mouseup";
    let panning = false;
    let pointerId = null;
    let lastEvent = null;
    let panCanvas = null;

    const markForwarded = (event) => {
        Object.defineProperty(event, "_vnccsForwardedCanvasInput", { value: true });
        return event;
    };

    const cloneMouseEvent = (type, source, buttons = source.buttons) => markForwarded(new MouseEvent(type, {
        bubbles: true,
        cancelable: true,
        view: window,
        detail: source.detail,
        screenX: source.screenX,
        screenY: source.screenY,
        clientX: source.clientX,
        clientY: source.clientY,
        ctrlKey: source.ctrlKey,
        altKey: source.altKey,
        shiftKey: source.shiftKey,
        metaKey: source.metaKey,
        button: type === "mousemove" ? 0 : 1,
        buttons,
    }));

    const clonePointerEvent = (type, source, buttons = source.buttons) => {
        const EventCtor = window.PointerEvent || window.MouseEvent;
        return markForwarded(new EventCtor(type, {
            bubbles: true,
            cancelable: true,
            view: window,
            detail: source.detail,
            screenX: source.screenX,
            screenY: source.screenY,
            clientX: source.clientX,
            clientY: source.clientY,
            ctrlKey: source.ctrlKey,
            altKey: source.altKey,
            shiftKey: source.shiftKey,
            metaKey: source.metaKey,
            button: type === "pointermove" ? -1 : 1,
            buttons,
            pointerId: source.pointerId ?? 1,
            pointerType: source.pointerType || "mouse",
            isPrimary: true,
        }));
    };

    const forward = (type, event, buttons) => {
        const canvasEl = panCanvas;
        if (!canvasEl) return;
        const pointerType = type === "mousedown" ? "pointerdown" : type === "mousemove" ? "pointermove" : "pointerup";
        if (pointerEvents) canvasEl.dispatchEvent(clonePointerEvent(pointerType, event, buttons));
        canvasEl.dispatchEvent(cloneMouseEvent(type, event, buttons));
    };

    const cloneWheelEvent = (source) => markForwarded(new WheelEvent("wheel", {
        bubbles: true,
        cancelable: true,
        view: window,
        detail: source.detail,
        screenX: source.screenX,
        screenY: source.screenY,
        clientX: source.clientX,
        clientY: source.clientY,
        ctrlKey: source.ctrlKey,
        altKey: source.altKey,
        shiftKey: source.shiftKey,
        metaKey: source.metaKey,
        deltaX: source.deltaX,
        deltaY: source.deltaY,
        deltaZ: source.deltaZ,
        deltaMode: source.deltaMode,
    }));

    const hasOwnWheelHandler = (target) => {
        for (let el = target; el && el !== root; el = el.parentElement) {
            if (typeof el.onwheel === "function") return true;
        }
        return false;
    };

    const hasScrollableAncestor = (target) => {
        for (let el = target; el && el !== root; el = el.parentElement) {
            if (!(el instanceof HTMLElement)) continue;
            const style = getComputedStyle(el);
            const scrollY = /(auto|scroll|overlay)/.test(style.overflowY) && el.scrollHeight > el.clientHeight + 1;
            const scrollX = /(auto|scroll|overlay)/.test(style.overflowX) && el.scrollWidth > el.clientWidth + 1;
            if (scrollY || scrollX) return true;
        }
        return false;
    };

    const stopPan = () => {
        if (!panning) return;
        panning = false;
        window.removeEventListener(moveEvent, movePan, true);
        window.removeEventListener(upEvent, finishPan, true);
        window.removeEventListener("pointercancel", cancelPan, true);
        panCanvas?.removeEventListener("lostpointercapture", cancelPan);
        window.removeEventListener("blur", stopPan);
        document.removeEventListener("visibilitychange", onVisibilityChange);
        if (pointerEvents) {
            window.removeEventListener("mousedown", suppressCompatibilityMouse, true);
            window.removeEventListener("mousemove", suppressCompatibilityMouse, true);
            window.removeEventListener("mouseup", suppressCompatibilityMouse, true);
        }
        forward("mouseup", lastEvent, 0);
        if (pointerId !== null && panCanvas?.hasPointerCapture?.(pointerId)) {
            panCanvas.releasePointerCapture(pointerId);
        }
        pointerId = null;
        lastEvent = null;
        panCanvas = null;
    };

    const ownsEvent = (event) => !event._vnccsForwardedCanvasInput
        && panning && (pointerId === null || event.pointerId === pointerId);

    const finishPan = (event) => {
        if (!ownsEvent(event) || (event.buttons & 4)) return;
        event.preventDefault();
        event.stopPropagation();
        lastEvent = event;
        stopPan();
    };

    const cancelPan = (event) => {
        if (!ownsEvent(event)) return;
        event.stopPropagation();
        stopPan();
    };

    const onVisibilityChange = () => {
        if (document.hidden) stopPan();
    };

    const suppressCompatibilityMouse = (event) => {
        if (!panning || event._vnccsForwardedCanvasInput) return;
        event.preventDefault();
        event.stopPropagation();
    };

    const movePan = (event) => {
        if (!ownsEvent(event)) return;
        event.preventDefault();
        event.stopPropagation();
        lastEvent = event;
        if (!(event.buttons & 4) || !root.isConnected) {
            stopPan();
            return;
        }
        forward("mousemove", event, event.buttons);
    };

    const startPan = (event) => {
        if (event._vnccsForwardedCanvasInput) return;
        if (event.button !== 1) return;
        if (panning) stopPan();
        panCanvas = canvas();
        if (!panCanvas) return;
        panning = true;
        pointerId = pointerEvents ? event.pointerId : null;
        lastEvent = event;
        event.preventDefault();
        event.stopPropagation();
        window.addEventListener(moveEvent, movePan, true);
        window.addEventListener(upEvent, finishPan, true);
        window.addEventListener("pointercancel", cancelPan, true);
        panCanvas.addEventListener("lostpointercapture", cancelPan);
        window.addEventListener("blur", stopPan);
        document.addEventListener("visibilitychange", onVisibilityChange);
        if (pointerEvents) {
            window.addEventListener("mousedown", suppressCompatibilityMouse, true);
            window.addEventListener("mousemove", suppressCompatibilityMouse, true);
            window.addEventListener("mouseup", suppressCompatibilityMouse, true);
        }
        // Claim the pointer before an embedded viewer can start its own drag.
        forward("mousedown", event, event.buttons);
    };
    root.addEventListener(downEvent, startPan, true);

    const suppressAuxClick = (event) => {
        if (event.button !== 1) return;
        event.preventDefault();
        event.stopPropagation();
    };
    root.addEventListener("auxclick", suppressAuxClick, true);

    const forwardWheel = (event) => {
        if (event._vnccsForwardedCanvasInput) return;
        if (hasOwnWheelHandler(event.target) || hasScrollableAncestor(event.target)) return;
        const canvasEl = canvas();
        if (!canvasEl) return;
        canvasEl.dispatchEvent(cloneWheelEvent(event));
        event.preventDefault();
        event.stopPropagation();
    };
    root.addEventListener("wheel", forwardWheel, { capture: true, passive: false });

    if (node) registerCleanup(node, () => {
        stopPan();
        root.removeEventListener(downEvent, startPan, true);
        root.removeEventListener("auxclick", suppressAuxClick, true);
        root.removeEventListener("wheel", forwardWheel, true);
        delete root._vnccsMiddleMouseCanvasPan;
    });
}

// ── CSS Injection (once per class prefix) ─────────────────────────────────────
const _injectedStyles = new Set();

export function injectStyles(css, id) {
    if (_injectedStyles.has(id)) return;
    _injectedStyles.add(id);
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);
}

// ── Delayed Field Help Tooltips ──────────────────────────────────────────────
const HELP_TOOLTIP_CSS = `
.vnccs-help-tooltip {
    position: fixed;
    z-index: 100000;
    max-width: min(320px, calc(100vw - 24px));
    padding: 10px 12px;
    border: 1px solid rgba(255, 143, 163, 0.28);
    border-radius: 8px;
    background: rgba(18, 18, 26, 0.96);
    color: #e8e8f0;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.45);
    font-family: 'Sora', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 12px;
    line-height: 1.45;
    pointer-events: none;
    opacity: 0;
    transform: translateY(4px);
    transition: opacity 0.16s ease, transform 0.16s ease;
    white-space: normal;
}
.vnccs-help-tooltip.is-visible {
    opacity: 1;
    transform: translateY(0);
}
`;

let _helpTooltipEl = null;

function getHelpTooltip() {
    injectStyles(HELP_TOOLTIP_CSS, "vnccs-help-tooltip");
    if (_helpTooltipEl?.isConnected) return _helpTooltipEl;
    _helpTooltipEl = document.createElement("div");
    _helpTooltipEl.className = "vnccs-help-tooltip";
    _helpTooltipEl.id = "vnccs-field-help-tooltip";
    _helpTooltipEl.setAttribute("role", "tooltip");
    document.body.appendChild(_helpTooltipEl);
    return _helpTooltipEl;
}

function positionHelpTooltip(anchor, tooltip) {
    const rect = anchor.getBoundingClientRect();
    const margin = 12;
    tooltip.style.left = "0px";
    tooltip.style.top = "0px";
    const width = tooltip.offsetWidth || 280;
    const height = tooltip.offsetHeight || 80;
    let left = rect.left + Math.min(24, rect.width * 0.5);
    let top = rect.bottom + 8;

    if (left + width > window.innerWidth - margin) left = window.innerWidth - width - margin;
    if (left < margin) left = margin;
    if (top + height > window.innerHeight - margin) top = rect.top - height - 8;
    if (top < margin) top = margin;

    tooltip.style.left = `${Math.round(left)}px`;
    tooltip.style.top = `${Math.round(top)}px`;
}

export function setHelpText(element, text) {
    if (!element || !text) return element;
    element.dataset.vnccsHelp = text;
    element.setAttribute("aria-describedby", "vnccs-field-help-tooltip");
    return element;
}

export function attachHelpTooltips(root, { delay = 1800 } = {}) {
    if (!root || root._vnccsHelpTooltipsAttached) return;
    root._vnccsHelpTooltipsAttached = true;

    let timer = null;
    let activeAnchor = null;

    const clear = () => {
        if (timer) clearTimeout(timer);
        timer = null;
        activeAnchor = null;
        if (_helpTooltipEl) _helpTooltipEl.classList.remove("is-visible");
    };

    const schedule = (anchor) => {
        const text = anchor?.dataset?.vnccsHelp;
        if (!text) return;
        clear();
        activeAnchor = anchor;
        timer = setTimeout(() => {
            if (!activeAnchor?.isConnected) return;
            const tooltip = getHelpTooltip();
            tooltip.id = "vnccs-field-help-tooltip";
            tooltip.textContent = text;
            tooltip.classList.add("is-visible");
            positionHelpTooltip(activeAnchor, tooltip);
        }, delay);
    };

    root.addEventListener("mouseover", (event) => {
        const anchor = event.target?.closest?.("[data-vnccs-help]");
        if (!anchor || !root.contains(anchor)) return;
        schedule(anchor);
    }, true);

    root.addEventListener("focusin", (event) => {
        const anchor = event.target?.closest?.("[data-vnccs-help]");
        if (!anchor || !root.contains(anchor)) return;
        schedule(anchor);
    }, true);

    root.addEventListener("mouseout", (event) => {
        if (!activeAnchor) return;
        if (event.relatedTarget && activeAnchor.contains(event.relatedTarget)) return;
        clear();
    }, true);

    root.addEventListener("focusout", clear, true);
    root.addEventListener("mousedown", clear, true);
    root.addEventListener("wheel", clear, true);
    window.addEventListener("blur", clear);
    registerCleanup(root._node || root, () => window.removeEventListener("blur", clear));
}

// Shared CSS for modal/loading overlay (injected once)
const COMMON_CSS = `
.vnccs-common-modal-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.8); display: flex; align-items: center; justify-content: center;
    z-index: 1000; pointer-events: auto;
}
.vnccs-common-modal {
    background: #252525; border: 1px solid #444; padding: 20px; border-radius: 8px;
    width: 300px; display: flex; flex-direction: column; gap: 15px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5); max-height: 80vh;
    color: #e0e0e0; font-family: 'Consolas', 'Monaco', monospace; font-size: 14px;
}
.vnccs-common-modal-title {
    font-weight: bold; font-size: 16px; border-bottom: 1px solid #444; padding-bottom: 8px;
}
.vnccs-common-modal-btn-row {
    display: flex; gap: 8px; justify-content: flex-end;
}
.vnccs-common-modal-btn {
    padding: 6px 16px; border: 1px solid #555; border-radius: 4px;
    cursor: pointer; font-size: 13px; background: #333; color: #e0e0e0;
}
.vnccs-common-modal-btn:hover { background: #444; }
.vnccs-common-modal-btn:focus,
.vnccs-common-modal-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.35);
}
.vnccs-common-modal-btn-primary {
    appearance: none;
    -webkit-appearance: none;
    background: linear-gradient(135deg, #ff8fa3 0%, #ffb6c8 100%) !important;
    background-color: #ff8fa3 !important;
    background-image: linear-gradient(135deg, #ff8fa3 0%, #ffb6c8 100%) !important;
    color: #1a1525 !important;
    border-color: transparent !important;
    -webkit-tap-highlight-color: rgba(255,143,163,0.22);
}
.vnccs-common-modal-btn-primary:hover,
.vnccs-common-modal-btn-primary:focus,
.vnccs-common-modal-btn-primary:focus-visible,
.vnccs-common-modal-btn-primary:active {
    background: linear-gradient(135deg, #ff8fa3 0%, #ffb6c8 100%) !important;
    background-color: #ff8fa3 !important;
    background-image: linear-gradient(135deg, #ff8fa3 0%, #ffb6c8 100%) !important;
    color: #1a1525 !important;
}
.vnccs-common-modal-btn-danger { background: #d32f2f; color: #fff; border-color: #d32f2f; }
.vnccs-common-modal-btn-danger:hover { background: #e33f3f; }

.vnccs-common-loading-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.9);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    z-index: 1000; pointer-events: auto; gap: 20px;
}
.vnccs-common-spinner {
    width: 50px; height: 50px;
    border: 4px solid #333; border-top-color: #5b96f5;
    border-radius: 50%;
    animation: vnccs-common-spin 1s linear infinite;
}
@keyframes vnccs-common-spin { to { transform: rotate(360deg); } }
.vnccs-common-loading-text {
    color: #fff; font-size: 16px; font-weight: bold;
}
.vnccs-common-loading-dots::after {
    content: '';
    animation: vnccs-common-dots 1.5s steps(4, end) infinite;
}
@keyframes vnccs-common-dots {
    0%, 20% { content: ''; }
    40% { content: '.'; }
    60% { content: '..'; }
    80%, 100% { content: '...'; }
}

.vnccs-common-message-overlay {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.6); display: flex; align-items: center; justify-content: center;
    z-index: 1001; pointer-events: auto;
    backdrop-filter: blur(4px);
}
.vnccs-common-message-box {
    background: #252525; border: 1px solid #444; padding: 20px; border-radius: 8px;
    max-width: 320px; text-align: center; display: flex; flex-direction: column; gap: 12px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    color: #e0e0e0; font-family: 'Consolas', 'Monaco', monospace; font-size: 14px;
}
.vnccs-common-message-box.error { border-color: #d32f2f; }
.vnccs-common-message-box .msg-icon { font-size: 28px; }
`;

// ── Modal Dialog ──────────────────────────────────────────────────────────────
// showModal(container, title, contentFunc, buttons)
// buttons: [{ text, class?: "primary"|"danger", autofocus?, action?: async (overlay, btn) => keepOpen? }]
// Returns { overlay, modal, content, title, actions, buttons }
let _modalSequence = 0;
const _modalStack = [];
const _modalOpeners = new WeakMap();

export function showModal(container, title, contentFunc, buttons) {
    injectStyles(COMMON_CSS, "vnccs-common");
    const parentModal = _modalStack.at(-1);
    const openers = [document.activeElement, ...(_modalOpeners.get(parentModal) || [])];

    const overlay = document.createElement("div");
    overlay.className = "vnccs-common-modal-overlay";

    const m = document.createElement("div");
    m.className = "vnccs-common-modal";
    m.tabIndex = -1;
    m.setAttribute("role", "dialog");
    m.setAttribute("aria-modal", "true");
    _modalOpeners.set(m, openers);

    const titleEl = document.createElement("div");
    titleEl.className = "vnccs-common-modal-title";
    titleEl.textContent = title;
    titleEl.id = `vnccs-modal-title-${++_modalSequence}`;
    m.setAttribute("aria-labelledby", titleEl.id);
    m.appendChild(titleEl);

    const content = contentFunc(m);
    if (content) m.appendChild(content);

    const row = document.createElement("div");
    row.className = "vnccs-common-modal-btn-row";
    const buttonEls = [];
    buttons.forEach(b => {
        const btn = document.createElement("button");
        let cls = "vnccs-common-modal-btn";
        if (b.class === "primary" || b.class?.includes("primary")) cls += " vnccs-common-modal-btn-primary";
        if (b.class === "danger" || b.class?.includes("danger")) cls += " vnccs-common-modal-btn-danger";
        btn.className = cls;
        btn.type = "button";
        btn.autofocus = !!b.autofocus;
        btn.innerText = b.text;
        btn.onclick = async () => {
            if (b.action) {
                if (btn.disabled) return;
                btn.disabled = true;
                try {
                    const keepOpen = await b.action(overlay, btn);
                    if (!keepOpen) overlay.remove();
                } catch (error) {
                    showMessage(m, error.message || String(error), true);
                } finally { btn.disabled = false; }
            } else {
                overlay.remove();
            }
        };
        row.appendChild(btn);
        buttonEls.push({ button: btn, config: b });
    });
    m.appendChild(row);

    const focusable = () => [...m.querySelectorAll("button, input:not([type='hidden']), textarea, select, a[href], [tabindex]:not([tabindex='-1'])")]
        .filter(element => !element.disabled && !element.hidden && !element.closest("[inert]")
            && element.getClientRects().length > 0);
    const focusFirst = () => {
        const fields = focusable();
        (fields.find(element => element.autofocus) || fields[0] || m).focus({ preventScroll: true });
    };
    const containFocus = event => {
        if (_modalStack.at(-1) === m && !m.contains(event.target)) focusFirst();
    };
    let closed = false;
    let observer = null;
    const remove = overlay.remove;
    const cleanup = () => {
        if (closed) return;
        closed = true;
        const wasTop = _modalStack.at(-1) === m;
        const index = _modalStack.indexOf(m);
        if (index >= 0) _modalStack.splice(index, 1);
        document.removeEventListener("focusin", containFocus, true);
        observer?.disconnect();
        if (wasTop) openers.find(element => element?.isConnected)?.focus({ preventScroll: true });
    };
    overlay.remove = function () {
        remove.apply(this, arguments);
        cleanup();
    };

    m.addEventListener("keydown", (event) => {
        if (_modalStack.at(-1) !== m) return;
        if (event.key === "Tab") {
            const fields = focusable();
            const index = fields.indexOf(document.activeElement);
            if (!fields.length || (event.shiftKey && index <= 0) || (!event.shiftKey && (index < 0 || index === fields.length - 1))) {
                event.preventDefault();
                event.stopPropagation();
                (event.shiftKey ? fields.at(-1) || m : fields[0] || m).focus({ preventScroll: true });
            }
            return;
        }
        if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            overlay.remove();
            return;
        }
        if (event.key !== "Enter" || event.shiftKey || event.ctrlKey || event.metaKey || event.altKey) return;
        const target = event.target;
        if (target?.tagName === "BUTTON") return;
        const confirm = buttonEls.find(item => item.config.class === "primary" || item.config.class?.includes("primary"))
            || buttonEls[buttonEls.length - 1];
        if (!confirm?.button) return;
        event.preventDefault();
        event.stopPropagation();
        confirm.button.click();
    }, true);

    overlay.appendChild(m);
    container.appendChild(overlay);
    _modalStack.push(m);
    document.addEventListener("focusin", containFocus, true);
    if (typeof MutationObserver !== "undefined") {
        observer = new MutationObserver(() => { if (!overlay.isConnected) cleanup(); });
        observer.observe(document.body, { childList: true, subtree: true });
    }
    requestAnimationFrame(() => {
        if (closed || _modalStack.at(-1) !== m) return;
        const field = focusable().find(element => ["INPUT", "TEXTAREA", "SELECT"].includes(element.tagName));
        if (!field) { focusFirst(); return; }
        field.focus({ preventScroll: true });
        if (typeof field.select === "function" && (field.tagName === "INPUT" || field.tagName === "TEXTAREA")) {
            field.select();
        }
    });
    return { overlay, modal: m, content, title: titleEl, actions: row, buttons: buttonEls.map(item => item.button) };
}

// ── Info/Error Message ────────────────────────────────────────────────────────
// Auto-dismissing notification box. Returns the overlay element.
export function showMessage(container, text, isError = false) {
    injectStyles(COMMON_CSS, "vnccs-common");

    const overlay = document.createElement("div");
    overlay.className = "vnccs-common-message-overlay";

    const box = document.createElement("div");
    box.className = "vnccs-common-message-box" + (isError ? " error" : "");

    const icon = document.createElement("div");
    icon.className = "msg-icon";
    icon.textContent = isError ? "⚠️" : "✅";
    box.appendChild(icon);

    const msg = document.createElement("div");
    msg.textContent = text;
    box.appendChild(msg);

    const btn = document.createElement("button");
    btn.className = "vnccs-common-modal-btn vnccs-common-modal-btn-primary";
    btn.textContent = "OK";
    btn.onclick = () => overlay.remove();
    box.appendChild(btn);

    overlay.appendChild(box);
    container.appendChild(overlay);
    return overlay;
}

// ── Loading Overlay ───────────────────────────────────────────────────────────
// Returns { overlay, remove() }
export function createLoadingOverlay(container, message = "Generating preview") {
    injectStyles(COMMON_CSS, "vnccs-common");

    const overlay = document.createElement("div");
    overlay.className = "vnccs-common-loading-overlay";
    overlay.innerHTML = `
        <div class="vnccs-common-spinner"></div>
        <div class="vnccs-common-loading-text">${message}<span class="vnccs-common-loading-dots"></span></div>
    `;
    container.appendChild(overlay);
    return {
        overlay,
        remove() { if (overlay.parentNode) overlay.remove(); }
    };
}

// ── Generate Random Seed ──────────────────────────────────────────────────────
export function generateRandomSeed() {
    return Math.floor(Math.random() * 10000000000000);
}
