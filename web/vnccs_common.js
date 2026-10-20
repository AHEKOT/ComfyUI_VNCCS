/**
 * VNCCS Common Utilities — shared patterns for all VNCCS widgets.
 * Import: import { debounce, showModal, ... } from "./vnccs_common.js";
 */
import { vnccsApi as api, mediaURL, storage, serverRegistry } from "./vnccs_transport.js";
import { app } from "../../scripts/app.js";
import { mountSakuraDialogMotion } from "./sakura_interactions.mjs";

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

function copyWizardSnapshotValues(source, clone) {
    const sourceControls = [];
    const cloneControls = [];
    if (source?.matches?.("input, textarea, select")) sourceControls.push(source);
    if (clone?.matches?.("input, textarea, select")) cloneControls.push(clone);
    sourceControls.push(...(source?.querySelectorAll?.("input, textarea, select") || []));
    cloneControls.push(...(clone?.querySelectorAll?.("input, textarea, select") || []));
    sourceControls.forEach((control, index) => {
        const snapshot = cloneControls[index];
        if (!snapshot) return;
        snapshot.value = control.value;
        if (snapshot.tagName === "TEXTAREA") snapshot.textContent = control.value;
        if ("checked" in control) snapshot.checked = control.checked;
        if (control.tagName === "SELECT") {
            [...snapshot.options].forEach((option, optionIndex) => {
                option.selected = optionIndex === control.selectedIndex;
            });
        }
    });
}

function wizardSnapshotBackground(element) {
    for (let current = element; current; current = current.parentElement) {
        const value = globalThis.getComputedStyle?.(current)?.backgroundColor;
        if (!value || value === "transparent") continue;
        const rgba = value.match(/^rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$/i);
        if (rgba) {
            const alpha = Number(rgba[4]);
            if (alpha <= 0) continue;
            if (alpha >= 0.98) return value;
            // Wizard columns sit on the shared #0a0a0f canvas. Flatten their
            // translucent fill so the new value cannot ghost through the old one.
            const base = [10, 10, 15];
            const rgb = [1, 2, 3].map((part, index) =>
                Math.round(Number(rgba[part]) * alpha + base[index] * (1 - alpha)));
            return `rgb(${rgb.join(", ")})`;
        }
        return value;
    }
    return "#14101e";
}

function wizardRevealTarget(field) {
    const explicit = field?._vnccsWizardRevealTarget;
    if (explicit && explicit.hidden !== true) return explicit;
    if (field?.matches?.("textarea, input[type='text'], input:not([type])")) return field;
    return field?.querySelector?.([
        "[data-vnccs-wizard-reveal-target]",
        ".vnccs-creator-trait-values:not([hidden])",
        ".vnccs-cloner-trait-values:not([hidden])",
        ".vnccs-clothes-textarea",
        ".vnccs-clothes-input",
        "textarea:not([hidden])",
        "input[type='text']:not([hidden])",
        "input:not([type]):not([hidden])",
    ].join(",")) || null;
}

function positionWizardReveal(entry) {
    const { field, target, layer } = entry;
    const fieldRect = field.getBoundingClientRect?.();
    const targetRect = target.getBoundingClientRect?.();
    if (!fieldRect || !targetRect) return;
    const width = Math.max(1, targetRect.width || target.offsetWidth || 1);
    const height = Math.max(1, targetRect.height || target.offsetHeight || 1);
    layer.style.left = `${targetRect.left - fieldRect.left - (field.clientLeft || 0)}px`;
    layer.style.top = `${targetRect.top - fieldRect.top - (field.clientTop || 0)}px`;
    layer.style.width = `${width}px`;
    layer.style.height = `${height}px`;
    layer.style.borderRadius = globalThis.getComputedStyle?.(target)?.borderRadius || "8px";
}

// Reveals wizard-populated controls behind their previous values, one row at a time.
// `apply` is deliberately synchronous so persistence tests and widget serialization
// observe the result immediately; only the visual handoff is deferred.
export function animateWizardRows(rows, apply, options = {}) {
    const fields = [...new Set((rows || []).filter(field => field && wizardRevealTarget(field)))];
    const runToken = {};
    const reduced = globalThis.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;
    const canAnimate = fields.length && !reduced && fields.every(field =>
        typeof field.cloneNode === "function" && typeof field.appendChild === "function");
    if (!canAnimate) {
        apply();
        return Promise.resolve();
    }

    const snapshots = fields.map(field => {
        field.querySelectorAll?.(".vnccs-common-wizard-reveal")?.forEach(layer => layer.remove());
        const baseStyle = field._vnccsWizardRevealBaseStyle || {
            position: field.style.position,
            isolation: field.style.isolation,
        };
        field._vnccsWizardRevealBaseStyle = baseStyle;
        field._vnccsWizardRevealRun = runToken;
        if (globalThis.getComputedStyle?.(field)?.position === "static") field.style.position = "relative";
        field.style.isolation = "isolate";

        const target = wizardRevealTarget(field);
        const clone = target.cloneNode(true);
        copyWizardSnapshotValues(target, clone);
        clone.querySelectorAll?.("[id]")?.forEach(element => element.removeAttribute("id"));
        clone.removeAttribute?.("id");
        clone.classList?.add("vnccs-common-wizard-reveal-snapshot");

        const layer = document.createElement("div");
        layer.className = "vnccs-common-wizard-reveal";
        if (options.accent) layer.style.setProperty("--vnccs-common-ui-accent", options.accent);
        layer.setAttribute("aria-hidden", "true");
        const old = document.createElement("div");
        old.className = "vnccs-common-wizard-reveal-old";
        old.style.background = wizardSnapshotBackground(target);
        old.appendChild(clone);
        const scan = document.createElement("span");
        scan.className = "vnccs-common-wizard-reveal-scan";
        layer.append(old, scan);
        field.appendChild(layer);

        const entry = { field, target, layer, old, scan, baseStyle };
        positionWizardReveal(entry);
        return entry;
    });
    try {
        apply();
        // Updating the value can resize a textarea or rebuild a tag list. Re-measure
        // before paint so the old snapshot still covers the exact live control.
        snapshots.forEach(positionWizardReveal);
    } catch (error) {
        snapshots.forEach(({ field, layer, baseStyle }) => {
            layer.remove();
            field.style.position = baseStyle.position;
            field.style.isolation = baseStyle.isolation;
            delete field._vnccsWizardRevealRun;
            delete field._vnccsWizardRevealBaseStyle;
        });
        throw error;
    }

    const duration = Math.max(120, Number(options.duration) || 240);
    const gap = Math.max(0, Number(options.gap) || 30);
    const delay = Math.max(0, Number(options.delay) || 90);
    const easing = "cubic-bezier(0.4, 0, 0.2, 1)";
    const schedule = globalThis.setTimeout || (callback => callback());

    return new Promise(resolve => {
        let remaining = snapshots.length;
        const finish = () => { if (--remaining === 0) resolve(); };
        snapshots.forEach((entry, index) => {
            const { field, layer, old, scan, baseStyle } = entry;
            const start = delay + index * (duration + gap);
            schedule(() => {
                if (field._vnccsWizardRevealRun !== runToken) {
                    finish();
                    return;
                }
                if (!field.isConnected || typeof field.animate !== "function") {
                    layer.remove();
                    field.style.position = baseStyle.position;
                    field.style.isolation = baseStyle.isolation;
                    delete field._vnccsWizardRevealRun;
                    delete field._vnccsWizardRevealBaseStyle;
                    finish();
                    return;
                }
                positionWizardReveal(entry);
                layer.classList.add("is-active");
                const width = Math.max(1, layer.getBoundingClientRect?.().width || layer.offsetWidth || 1);
                const oldAnimation = old.animate([
                    { clipPath: "inset(0 0 0 0)" },
                    { clipPath: "inset(0 0 0 100%)" },
                ], { duration, easing, fill: "forwards" });
                const scanAnimation = scan.animate([
                    { opacity: 1, transform: "translate3d(0, -50%, 0)" },
                    { opacity: 1, transform: `translate3d(${width}px, -50%, 0)` },
                ], { duration, easing, fill: "forwards" });
                Promise.allSettled([oldAnimation.finished, scanAnimation.finished]).finally(() => {
                    layer.remove();
                    if (field._vnccsWizardRevealRun === runToken) {
                        field.style.position = baseStyle.position;
                        field.style.isolation = baseStyle.isolation;
                        delete field._vnccsWizardRevealRun;
                        delete field._vnccsWizardRevealBaseStyle;
                    }
                    finish();
                });
            }, start);
        });
    });
}

export function installResolvedSeedTransport(app, nodeTypes, inputName = "widget_data") {
    if (typeof app.graphToPrompt !== "function") return;
    if (!app._vnccsResolvedSeedInputs) {
        const inputs = app._vnccsResolvedSeedInputs = new Map();
        const graphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function (...args) {
            const result = await graphToPrompt.apply(this, args);
            const output = { ...result.output };
            for (const [id, entry] of Object.entries(output)) {
                const name = inputs.get(entry.class_type);
                const node = app.graph?._nodes?.find(node => String(node.id) === id);
                if (!name || [2, 4].includes(node?.mode) || typeof entry.inputs?.[name] !== "string") continue;
                try {
                    const state = JSON.parse(entry.inputs[name]);
                    if ((state?.gen_settings || state?.editor?.generation || state)?.seed_mode !== "randomize") continue;
                    // Queue inputs carry the UI seed; workflow state keeps its randomize preference.
                    output[id] = { ...entry, inputs: { ...entry.inputs,
                        [name]: JSON.stringify({ ...state, _vnccs_seed_resolved: true }) } };
                } catch (_) {}
            }
            return { ...result, output };
        };
    }
    for (const type of nodeTypes) app._vnccsResolvedSeedInputs.set(type, inputName);
}

// Each widget keeps its own view state and polling cleanup.
export function createControlCenterClient(repoId, hasRequiredFamilies, state, render, node) {
    const cacheKey = `vnccs_cc_cache_${repoId}`;
    const isMounted = node ? createRequestGuard(node)() : () => true;
    let interval = null;
    let configRequest = 0;
    let polling = false;
    let pollingRevision = 0;
    const fetchConfig = async (force = false) => {
        if (!isMounted()) return state.config;
        const request = ++configRequest;
        const isCurrent = () => isMounted() && request === configRequest;
        const promises = serverRegistry("VNCCS_CC_FETCH_PROMISES");
        const previousPromise = promises[repoId];
        if (!force && !previousPromise && serverRegistry("VNCCS_CC_REGISTRY")?.[repoId] && hasRequiredFamilies(serverRegistry("VNCCS_CC_REGISTRY")[repoId])) {
            state.config = serverRegistry("VNCCS_CC_REGISTRY")[repoId];
            render();
            return state.config;
        }
        if (!force && !previousPromise && state.config && hasRequiredFamilies(state.config)) return state.config;
        if (!force && !previousPromise) {
            try {
                const cached = storage.getItem(cacheKey);
                if (cached) {
                    state.config = JSON.parse(cached);
                    if (hasRequiredFamilies(state.config)) render();
                    else state.config = null;
                }
            } catch (_) {}
        }

        let fetchPromise = !force && previousPromise;
        if (!fetchPromise) {
            fetchPromise = (async () => {
                try {
                    const url = `/vnccs/control_center/check?repo_id=${encodeURIComponent(repoId)}${force ? "&force_refresh=true" : ""}`;
                    const response = await api.fetchApi(url);
                    const payload = await response.json();
                    if (!response.ok || payload.error) throw new Error(payload.error || "Failed to load Control Center config");
                    if (promises[repoId] === fetchPromise) serverRegistry("VNCCS_CC_REGISTRY")[repoId] = payload;
                    return payload;
                } finally {
                    if (promises[repoId] === fetchPromise) delete promises[repoId];
                }
            })();
            promises[repoId] = fetchPromise;
            if (previousPromise) previousPromise.replacement = fetchPromise;
        }
        let payload;
        while (true) {
            try { payload = await fetchPromise; }
            catch (error) {
                if (!isCurrent()) return state.config;
                if (!fetchPromise.replacement) throw error;
            }
            if (!isCurrent()) return state.config;
            if (!fetchPromise.replacement) break;
            fetchPromise = fetchPromise.replacement;
        }
        if (serverRegistry("VNCCS_CC_REGISTRY")[repoId] !== payload) return state.config;
        state.config = payload;

        storage.setItem(cacheKey, JSON.stringify(payload));
        render();
        return payload;
    };

    const refreshDownloadStatus = async () => {
        if (!isMounted() || polling) return;
        polling = true;
        const revision = pollingRevision;
        try {
            const response = await api.fetchApi(`/vnccs/manager/status?repo_id=${encodeURIComponent(repoId)}`);
            if (!response.ok) return;
            const status = await response.json();
            if (!isMounted() || revision !== pollingRevision) return;
            state.downloadStatus = status;
            const active = Object.values(state.downloadStatus || {}).some(item => ["queued", "downloading"].includes(item?.status));
            if (!active) {
                stopPolling();
                await fetchConfig(true);
            } else {
                render();
            }
        } catch (_) {}
        finally { polling = false; }
    };

    const startPolling = () => {
        if (!isMounted()) return;
        pollingRevision++;
        if (interval) return;
        interval = setInterval(refreshDownloadStatus, 2000);
    };

    const stopPolling = () => {
        if (!interval) return;
        clearInterval(interval);
        interval = null;
    };
    const onRegistryUpdate = event => {
        if (!isMounted() || event.detail?.repo_id !== repoId) return;
        const config = serverRegistry("VNCCS_CC_REGISTRY")[repoId];
        if (config) state.config = config;
        if (event.detail.download_status) state.downloadStatus = event.detail.download_status;
        startPolling();
        render();
    };
    window.addEventListener("vnccs-cc-registry-updated", onRegistryUpdate);
    if (node) registerCleanup(node, () => {
        stopPolling();
        window.removeEventListener("vnccs-cc-registry-updated", onRegistryUpdate);
    });
    startPolling();
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

export function createTraitInput({ prefix, label: lbl, key, target: targetObj, save, choose, removable = false, motion }) {
    const wrap = document.createElement("div");
    wrap.className = `${prefix}-trait-row`;
    const label = document.createElement("span");
    label.className = `${prefix}-trait-label`;
    label.textContent = lbl;
    const editor = document.createElement("div");
    editor.className = `${prefix}-trait-editor`;
    const values = document.createElement(removable ? "div" : "button");
    if (removable) { values.setAttribute("role", "group"); values.tabIndex = -1; }
    else values.type = "button";
    values.className = `${prefix}-trait-values`;
    const inp = document.createElement("input");
    inp.type = "text";
    inp.className = `${prefix}-input ${prefix}-trait-input`;
    inp.setAttribute("aria-label", lbl);
    inp.placeholder = "Add tags";
    inp.hidden = true;
    let chips = [];
    const renderTags = (animate = false) => {
        const previous = [...chips];
        const entering = [];
        const tokens = inp.value.split(",").map((text, index) => ({ text: text.trim(), index })).filter(token => token.text);
        chips = tokens.map(({ text, index }, position) => {
            const existing = previous.findIndex(chip => chip.tag === text);
            const chip = existing >= 0 ? previous.splice(existing, 1)[0] : document.createElement("span");
            if (existing < 0) entering.push(chip);
            chip.tag = text;
            chip.className = `${prefix}-trait-token`;
            if (!removable) chip.textContent = text;
            else {
                chip.setAttribute("data-tag", text);
                if (!chip.children.length) {
                    const textLabel = document.createElement("span");
                    textLabel.className = `${prefix}-trait-text`; textLabel.textContent = text;
                    textLabel.setAttribute("aria-hidden", "true");
                    const edit = document.createElement("button");
                    edit.type = "button"; edit.className = `${prefix}-trait-edit`;
                    edit.setAttribute("aria-label", `Edit ${lbl.toLowerCase()} tag: ${text}`);
                    edit.onclick = event => { event.stopPropagation(); inp.startEditing(); };
                    const remove = document.createElement("button");
                    remove.type = "button"; remove.className = `${prefix}-trait-remove`;
                    remove.innerHTML = '<svg viewBox="0 0 12 12" width="10" height="10" aria-hidden="true"><path d="m3 3 6 6m0-6-6 6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
                    remove.setAttribute("aria-label", `Remove ${lbl.toLowerCase()} tag: ${text}`);
                    remove.title = `Remove ${text}`;
                    chip.append(edit, remove, textLabel);
                }
                chip.children[1].onclick = event => {
                    event.stopPropagation();
                    const parts = inp.value.split(",");
                    parts.splice(index, 1);
                    inp.value = parts.some(part => part.trim()) ? parts.join(",") : "";
                    targetObj[key] = inp.value;
                    save();
                    renderTags();
                    (chips[Math.min(position, chips.length - 1)]?.children[1] || add).focus({ preventScroll: true });
                };
            }
            return chip;
        });
        values.replaceChildren(...chips);
        if (!tokens.length) {
            const empty = document.createElement(removable ? "button" : "span");
            empty.className = `${prefix}-trait-empty`;
            empty.textContent = "Add tags";
            if (removable) { empty.type = "button"; empty.onclick = () => inp.startEditing(); }
            values.appendChild(empty);
        }
        if (animate && !values.hidden) for (const chip of entering) {
            motion?.(chip, [{ opacity: 0, transform: "translateY(6px) scale(0.96)" }, { opacity: 1, transform: "none" }], 240);
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
        renderTags(true);
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
    return { element: wrap, input: inp, display: values };
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
    interactions,
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
        if (image && interactions) {
            interactions.presentPreview(image, placeholder, url, state.character);
        } else if (image) {
            image.src = url;
            image.style.display = "block";
        }
        if (placeholder && !interactions) placeholder.style.display = "none";
        onLoaded?.(url, { ...state });
    };

    const applyMissing = () => {
        if (image && interactions) interactions.emptyPreview(image, placeholder, state.character, "No preview available");
        else {
            if (image) image.style.display = "none";
            if (placeholder) placeholder.style.display = "flex";
        }
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

    const load = async (character, { costume = "", index, fallbackUrl = "" } = {}) => {
        if (disposed) return;
        const sameSelection = character === state.character && costume === state.costume;
        const currentIndex = sameSelection ? state.index : 0;
        if (interactions && !sameSelection) {
            interactions.prepareCharacter(image, placeholder, character);
        }
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
            show(index ?? currentIndex);
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
    const motion = {};

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
        for (const animation of [...(motion._motionAnimations || [])]) animation.cancel();
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
    mountSakuraDialogMotion(motion, overlay, m);
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
    mountSakuraDialogMotion({}, overlay, box);
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


export const SAKURA_INTERACTION_CSS = `
.vnccs-common-ui-host { --vnccs-common-ui-accent: var(--accent, #ff8fa3); }
.vnccs-common-ui-host select[hidden] { display: none !important; }
.vnccs-common-ui-host .vnccs-common-ui-select-trigger { min-width: 0; }
.vnccs-common-ui-host .vnccs-common-ui-segmented {
    position: relative; display: grid; grid-template-columns: repeat(var(--segments, 2), minmax(0, 1fr));
    gap: 4px; padding: 4px; min-height: var(--setup-control-height, 48px); box-sizing: border-box;
    border: 1px solid var(--border); border-radius: var(--radius-md, 12px);
    background: rgba(0,0,0,0.18);
}
.vnccs-common-ui-segmented::before {
    content: ""; position: absolute; top: 4px; bottom: 4px; left: 4px;
    width: calc((100% - 8px - (var(--segments, 2) - 1) * 4px) / var(--segments, 2)); border-radius: var(--radius-md, 12px);
    background: linear-gradient(180deg, #ff9bad 0%, #ff87a0 100%); box-shadow: 0 10px 22px rgba(255,143,163,0.22);
    transform: translateX(calc(var(--segment-index, 0) * (100% + 4px))); pointer-events: none;
    transition: transform 280ms cubic-bezier(0.22, 1, 0.36, 1);
}
.vnccs-common-ui-host .vnccs-common-ui-segmented > .vnccs-common-ui-segment {
    position: relative; z-index: 1; min-width: 0; min-height: 0; height: 100%;
    border: 0; background: transparent; box-shadow: none;
    transition: color 180ms ease; transform: none;
}
.vnccs-common-ui-host .vnccs-common-ui-segmented > .vnccs-common-ui-segment[aria-pressed="true"] { color: #20141a; }
.vnccs-common-ui-preview-aura {
    position: absolute; inset: 0; opacity: 0; pointer-events: none;
    background: radial-gradient(ellipse at center, rgba(255,143,163,0.35), rgba(184,169,232,0.15) 45%, transparent 75%);
}
.vnccs-common-ui-preview-host.is-male-character .vnccs-common-ui-preview-aura {
    background: radial-gradient(ellipse at center, rgba(99,199,255,0.38), rgba(91,159,224,0.16) 45%, transparent 75%);
}
.vnccs-common-ui-select-trigger {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    cursor: pointer;
    text-align: left;
    text-transform: none;
    letter-spacing: normal;
    background-image: none !important;
}
.vnccs-common-ui-select-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.vnccs-common-ui-select-trigger::after {
    content: "";
    flex-shrink: 0;
    width: 5px;
    height: 5px;
    border-right: 1.5px solid currentColor;
    border-bottom: 1.5px solid currentColor;
    transform: rotate(45deg);
    transition: transform 180ms ease;
}
.vnccs-common-ui-select-trigger[aria-expanded="true"] { border-color: var(--vnccs-common-ui-accent); box-shadow: 0 0 0 2px rgba(255,143,163,0.08); }
.vnccs-common-ui-select-trigger[aria-expanded="true"]::after { transform: rotate(225deg); }
.vnccs-common-ui-dropdown {
    pointer-events: auto;
    position: absolute;
    z-index: 2000;
    box-sizing: border-box;
    overflow: auto;
    padding: 4px;
    border: 1px solid rgba(255,143,163,0.28);
    border-radius: 9px;
    background: #17121d;
    box-shadow: 0 12px 28px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,143,163,0.04);
    scrollbar-width: thin;
    scrollbar-color: rgba(255,143,163,0.3) transparent;
}
.vnccs-common-ui-dropdown-option {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    gap: 8px;
    min-height: 28px;
    padding: 4px 8px;
    box-sizing: border-box;
    border-radius: 5px;
    color: #e8e8f0;
    font-size: 11px;
    line-height: 1.4;
    overflow-wrap: anywhere;
    cursor: pointer;
    transform: none !important;
    translate: none !important;
    scale: none !important;
    background: transparent;
    transition: color 120ms ease;
}
.vnccs-common-ui-dropdown::before {
    content: ""; position: absolute; top: 0; left: 4px; right: 4px;
    height: var(--highlight-height, 28px); border-radius: 5px; background: rgba(255,143,163,0.14);
    transform: translateY(var(--highlight-y, 4px)); pointer-events: none;
    transition: transform 140ms cubic-bezier(0.22, 1, 0.36, 1);
}
.vnccs-common-ui-dropdown-option::before { content: ""; width: 10px; flex-shrink: 0; }
.vnccs-common-ui-dropdown-option[aria-selected="true"]::before { content: "✓"; color: var(--vnccs-common-ui-accent); }
.vnccs-common-ui-dropdown-option--active { color: #ffe8ef; }
.vnccs-common-ui-host > .vnccs-common-ui-motion-ghost,
.vnccs-common-ui-preview-host > .vnccs-common-ui-motion-ghost { position: absolute; pointer-events: none; z-index: 3; margin: 0 !important; }
.vnccs-common-ui-tab-glide {
    border: 1px solid rgba(255,143,163,0.55);
    border-radius: 8px;
    background: rgba(255,143,163,0.12);
    transform-origin: top left;
}
.vnccs-common-wizard-reveal {
    position: absolute;
    z-index: 40;
    overflow: hidden;
    pointer-events: none;
    contain: paint;
}
.vnccs-common-wizard-reveal-old {
    position: absolute;
    z-index: 1;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    will-change: clip-path;
}
.vnccs-common-wizard-reveal-snapshot {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    max-width: none !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    pointer-events: none !important;
}
.vnccs-common-wizard-reveal-snapshot,
.vnccs-common-wizard-reveal-snapshot * {
    animation: none !important;
    transition: none !important;
}
.vnccs-common-wizard-reveal-scan {
    position: absolute;
    z-index: 2;
    top: 50%;
    left: 0;
    width: 3px;
    height: calc(100% - 8px);
    margin-left: -1.5px;
    border-radius: 99px;
    opacity: 0;
    background: linear-gradient(180deg, transparent, rgba(255,255,255,0.98) 18%, var(--vnccs-common-ui-accent) 50%, rgba(255,255,255,0.98) 82%, transparent);
    box-shadow: 0 0 7px var(--vnccs-common-ui-accent), 0 0 16px var(--vnccs-common-ui-accent), 0 0 28px color-mix(in srgb, var(--vnccs-common-ui-accent) 45%, transparent);
    will-change: transform, opacity;
}
.is-male-character .vnccs-common-wizard-reveal-scan,
.vnccs-common-ui-host.is-male-character .vnccs-common-wizard-reveal-scan {
    --vnccs-common-ui-accent: #63c7ff;
    box-shadow: 0 0 8px #63c7ff, 0 0 18px #63c7ff;
}
.vnccs-common-ui-host .vnccs-common-ui-select-trigger:focus-visible { outline: 2px solid #ff8fa3; outline-offset: -2px; }
@media (prefers-reduced-motion: reduce) {
    .vnccs-common-ui-segmented::before { transition: none !important; }
    .vnccs-common-ui-host .vnccs-common-ui-select-trigger, .vnccs-common-ui-host .vnccs-common-ui-select-trigger::after,
    .vnccs-common-ui-dropdown-option, .vnccs-common-ui-dropdown::before { transition: none !important; }
    .vnccs-common-wizard-reveal { display: none !important; }
}`;


// Creator control declarations also used by Background Creator.
export const SAKURA_CREATOR_CONTROL_CSS = `
.vnccs-common-creator-col {
    display: flex;
    flex-direction: column;
    background: rgba(20, 16, 30, 0.88);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-lg);
    padding: 16px;
    gap: 10px;
    overflow-y: auto;
    height: 100%;
    box-sizing: border-box;
    pointer-events: auto;
    position: relative;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}
.vnccs-common-creator-col::before {
    content: '';
    position: absolute;
    top: 0; left: 18%; right: 18%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,143,163,0.5), transparent);
    border-radius: 1px;
}
.vnccs-common-creator-col::-webkit-scrollbar { width: 4px; }
.vnccs-common-creator-col::-webkit-scrollbar-thumb { background: var(--accent-border); border-radius: 2px; }

/* Section titles */
.vnccs-common-creator-section-title {
    font-size: 10px;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
    flex-shrink: 0;
    pointer-events: auto;
    display: flex;
    align-items: center;
    gap: 8px;
}
.vnccs-common-creator-section-title::before {
    content: '';
    width: 3px;
    height: 12px;
    background: linear-gradient(180deg, var(--accent), var(--accent-lavender));
    border-radius: 2px;
    box-shadow: 0 0 8px var(--accent-glow);
    flex-shrink: 0;
}

/* Interactive elements */
.vnccs-common-creator-field,
.vnccs-common-creator-btn-row > *,
.vnccs-common-creator-preview-container,
.vnccs-common-creator-lora-item,
.vnccs-common-creator-textarea-wrapper,
.vnccs-common-creator-slider-container,
.vnccs-common-creator-input,
.vnccs-common-creator-select,
.vnccs-common-creator-textarea {
    pointer-events: auto;
}

/* Fields */
.vnccs-common-creator-field { display: flex; flex-direction: column; gap: 5px; margin-bottom: 6px; flex-shrink: 0; }
.vnccs-common-creator-framing-age-row {
    display: grid; grid-template-columns: 132px minmax(0, 1fr); gap: 10px; align-items: start;
}
.vnccs-common-creator-framing-age-row > .vnccs-common-creator-field { min-width: 0; }
.vnccs-common-creator-framing-field .vnccs-common-creator-select { width: 100%; }
.vnccs-common-creator-label {
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Inputs */
.vnccs-common-creator-input, .vnccs-common-creator-textarea {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    color: var(--text-primary);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    font-family: var(--font);
    font-size: 12px;
    width: 100%;
    box-sizing: border-box;
    transition: all var(--transition);
}
.vnccs-common-creator-select {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    color: var(--text-primary);
    border-radius: var(--radius-md);
    padding: 8px 12px;
    font-family: var(--font);
    font-size: 12px;
    width: 100%;
    box-sizing: border-box;
    min-height: 34px;
    transition: all var(--transition);
    color-scheme: dark;
}
.vnccs-common-creator-select option {
    background: #1e1e2e;
    color: #e8e8f0;
}

.vnccs-common-creator-input,
.vnccs-common-creator-select {
    height: 34px;
    min-height: 34px;
}
.vnccs-common-creator-input:focus, .vnccs-common-creator-select:focus, .vnccs-common-creator-textarea:focus {
    outline: none;
    border-color: var(--accent-border);
    background: rgba(255,143,163,0.04);
    box-shadow: 0 0 0 3px rgba(255,143,163,0.06);
}

/* Slider */
.vnccs-common-creator-slider-container {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 6px 10px;
}
.vnccs-common-creator-slider {
    flex: 1;
    accent-color: var(--accent);
    cursor: pointer;
    height: 3px;
}
.vnccs-common-creator-slider-val {
    width: 42px;
    text-align: right;
    font-size: 11px;
    font-family: var(--font-mono);
    color: var(--text-primary);
    background: transparent;
    border: none;
}
.vnccs-common-creator-slider-val:focus { outline: none; border-bottom: 1px solid var(--accent-border); }

.vnccs-common-creator-preview-container {
    --vnccs-preview-accent: #ff8fa3;
    --vnccs-preview-accent-rgb: 255, 143, 163;
    --vnccs-preview-highlight-rgb: 255, 238, 246;
    --vnccs-preview-aura-rgb: 184, 169, 232;
    flex: 1;
    background: radial-gradient(circle, rgba(var(--vnccs-preview-accent-rgb), 0.04) 1px, transparent 1px), rgba(10,10,15,0.7);
    background-size: 20px 20px, 100% 100%;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    position: relative;
    min-height: 0;
}

.vnccs-common-creator-placeholder-title { color: var(--text-primary); font-size: 15px; font-weight: 600; }
.vnccs-common-creator-placeholder-hint { color: var(--text-secondary); line-height: 1.5; max-width: 240px; }

/* LoRA stack */
.vnccs-common-creator-lora-stack {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 8px;
    border-top: 1px solid var(--border);
    padding-top: 10px;
}
.vnccs-common-creator-lora-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
    background: rgba(255,255,255,0.02);
    padding: 6px 8px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    transition: border-color var(--transition);
}
.vnccs-common-creator-lora-item:hover { border-color: var(--border-hover); }
.vnccs-common-creator-lora-row { display: flex; gap: 6px; align-items: center; }

/* Buttons */
.vnccs-common-creator-btn-row { display: flex; gap: 8px; margin-top: auto; min-height: 48px; flex-shrink: 0; }
.vnccs-common-creator-btn {
    flex: 1;
    padding: 10px;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 11px;
    font-family: var(--font);
    color: white;
    text-align: center;
    transition: all var(--transition);
    position: relative;
    overflow: hidden;
}
.vnccs-common-creator-btn-primary {
    appearance: none;
    -webkit-appearance: none;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    background-color: var(--accent) !important;
    background-image: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525;
    box-shadow: 0 4px 16px rgba(255,143,163,0.25);
    -webkit-tap-highlight-color: rgba(255,143,163,0.22);
}
.vnccs-common-creator-btn-primary::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%);
    transform: translateX(-120%) skewX(-15deg);
    animation: vnccs-common-creator-shimmer 3.5s ease-in-out infinite;
    pointer-events: none;
}
@keyframes vnccs-common-creator-shimmer {
    0% { transform: translateX(-120%) skewX(-15deg); opacity: 1; }
    35% { transform: translateX(120%) skewX(-15deg); opacity: 1; }
    100% { transform: translateX(120%) skewX(-15deg); opacity: 0; }
}
.vnccs-common-creator-btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(255,143,163,0.4);
}
.vnccs-common-creator-container button.vnccs-common-creator-btn.vnccs-common-creator-btn-primary:not(:disabled),
.vnccs-common-creator-container button.vnccs-common-creator-btn.vnccs-common-creator-btn-primary:not(:disabled):hover,
.vnccs-common-creator-container button.vnccs-common-creator-btn.vnccs-common-creator-btn-primary:not(:disabled):focus,
.vnccs-common-creator-container button.vnccs-common-creator-btn.vnccs-common-creator-btn-primary:not(:disabled):focus-visible,
.vnccs-common-creator-container button.vnccs-common-creator-btn.vnccs-common-creator-btn-primary:not(:disabled):active {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    background-color: var(--accent) !important;
    background-image: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    outline: none;
}
.vnccs-common-creator-btn-success {
    background: rgba(0,214,143,0.15);
    color: var(--success);
    border: 1px solid rgba(0,214,143,0.3);
}
.vnccs-common-creator-btn-success:hover:not(:disabled) {
    background: rgba(0,214,143,0.25);
    transform: translateY(-1px);
}
.vnccs-common-creator-btn-danger {
    background: rgba(255,71,87,0.15);
    color: var(--error);
    border: 1px solid rgba(255,71,87,0.3);
}
.vnccs-common-creator-btn-danger:hover:not(:disabled) {
    background: rgba(255,71,87,0.25);
    transform: translateY(-1px);
}
.vnccs-common-creator-btn-disabled, .vnccs-common-creator-btn:disabled {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text-muted) !important;
    cursor: not-allowed;
    box-shadow: none !important;
    transform: none !important;
}
.vnccs-common-creator-btn:focus,
.vnccs-common-creator-btn:focus-visible,
.vnccs-common-creator-segmented-btn:focus,
.vnccs-common-creator-segmented-btn:focus-visible,
.vnccs-common-creator-seed-dice-btn:focus,
.vnccs-common-creator-seed-dice-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.28);
}
.vnccs-common-creator-btn-primary:focus:not(:disabled),
.vnccs-common-creator-btn-primary:focus-visible:not(:disabled),
.vnccs-common-creator-btn-primary:active:not(:disabled) {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    box-shadow: 0 8px 28px rgba(255,143,163,0.4), 0 0 0 2px rgba(255,143,163,0.28);
}

/* Bottom textareas */
.vnccs-common-creator-textarea-wrapper {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    background: rgba(20,16,30,0.88);
    padding: 8px 10px;
    border-radius: var(--radius-md);
    border: 1px solid var(--accent-border);
    position: relative;
}
.vnccs-common-creator-textarea-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 15%; right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,143,163,0.4), transparent);
}
.vnccs-common-creator-textarea-wrapper textarea {
    flex: 1;
    resize: none;
    border: none;
    background: transparent;
    padding: 4px;
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 11px;
}
.vnccs-common-creator-textarea-wrapper textarea:focus { outline: none; }
.vnccs-common-creator-textarea-label {
    font-size: 9px;
    color: var(--accent);
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0 2px 4px;
}

.vnccs-common-creator-trait-list {
    border-top: 1px solid var(--accent-border);
    flex-shrink: 0;
}

.vnccs-common-creator-trait-row {
    display: grid;
    grid-template-columns: minmax(56px, 18%) minmax(0, 1fr) 44px;
    align-items: center;
    gap: 12px;
    min-height: 64px;
    padding: 10px 0;
    border-bottom: 1px solid var(--accent-border);
    box-sizing: border-box;
}
.vnccs-common-creator-trait-label {
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 400;
}
.vnccs-common-creator-trait-editor { min-width: 0; }
.vnccs-common-creator-trait-values {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    width: 100%;
    min-height: 44px;
    padding: 0;
    border: none;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-primary);
    font: inherit;
    font-size: 13px;
    text-align: left;
    cursor: pointer;
}
.vnccs-common-creator-trait-token {
    position: relative;
    min-width: 0;
    box-sizing: border-box;
    max-width: 100%;
    padding: 5px 8px;
    border: 1px solid rgba(255,182,200,0.22);
    border-radius: 7px;
    background: rgba(184,169,232,0.14);
    line-height: 1.4;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.vnccs-common-creator-trait-token:has(.vnccs-common-creator-trait-remove) { overflow: visible; }
.vnccs-common-creator-trait-text { display: block; overflow: hidden; text-overflow: ellipsis; }
.vnccs-common-creator-trait-edit, .vnccs-common-creator-trait-remove {
    appearance: none; min-width: 0; min-height: 0; margin: 0; padding: 0;
    border: 0 !important; background: transparent !important; box-shadow: none !important;
    transform: none !important; font: inherit; line-height: inherit; cursor: pointer;
}
.vnccs-common-creator-trait-edit { position: absolute; inset: 0; width: 100%; height: 100%; border-radius: inherit; }
.vnccs-common-creator-trait-remove {
    position: absolute; top: -5px; right: -5px; width: 12px; height: 12px; z-index: 2;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%; background: var(--accent) !important; color: var(--bg-primary);
    box-shadow: 0 0 0 2px var(--bg-secondary) !important;
    transition: background-color 180ms ease, box-shadow 180ms ease;
}
.vnccs-common-creator-trait-remove::after { content: ""; position: absolute; inset: -4px; border-radius: inherit; }
.vnccs-common-creator-trait-remove svg { width: 12px; height: 12px; pointer-events: none; }
.vnccs-common-creator-trait-remove:hover { background: var(--accent-hover) !important; }
.vnccs-common-creator-trait-empty { border: 0; padding: 0; min-height: 0; background: transparent; color: inherit; font: inherit; cursor: pointer; }
.vnccs-common-creator-trait-token:focus-within { border-color: var(--accent); }
.vnccs-common-creator-trait-edit:focus-visible, .vnccs-common-creator-trait-remove:focus-visible, button.vnccs-common-creator-trait-empty:focus-visible {
    outline: 2px solid var(--accent); outline-offset: -2px;
}
.vnccs-common-creator-trait-remove:focus-visible { outline: 2px solid var(--text-primary); outline-offset: 2px; }
.vnccs-common-creator-trait-empty { color: var(--text-secondary); }
.vnccs-common-creator-trait-input { min-height: 44px; font-size: 13px; }
.vnccs-common-creator-trait-values[hidden], .vnccs-common-creator-trait-input[hidden] { display: none; }
.vnccs-common-creator-trait-add {
    width: 44px;
    height: 44px;
    padding: 0;
    background: rgba(255,255,255,0.06);
    color: var(--text-primary);
    border: 1px solid rgba(255,182,200,0.22);
    border-radius: 10px;
    font: inherit;
    font-size: 20px;
    cursor: pointer;
    transition: background var(--transition), border-color var(--transition);
}
.vnccs-common-creator-trait-add:hover { background: var(--bg-hover); border-color: var(--accent); }
.vnccs-common-creator-trait-values:hover .vnccs-common-creator-trait-token { border-color: var(--accent-border); }

/* Custom toggle checkbox */
.vnccs-common-creator-toggle-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    padding: 6px 0;
    user-select: none;
}
.vnccs-common-creator-toggle {
    position: relative;
    width: 36px;
    height: 20px;
    flex-shrink: 0;
}
.vnccs-common-creator-toggle input {
    opacity: 0;
    width: 0; height: 0;
    position: absolute;
}
.vnccs-common-creator-toggle-track {
    position: absolute;
    inset: 0;
    border-radius: 10px;
    background: rgba(255,255,255,0.08);
    border: 1px solid var(--border);
    transition: all 0.25s ease;
}
.vnccs-common-creator-toggle-thumb {
    position: absolute;
    top: 3px; left: 3px;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--text-muted);
    transition: all 0.25s ease;
}
.vnccs-common-creator-toggle input:checked ~ .vnccs-common-creator-toggle-track {
    background: rgba(255,143,163,0.2);
    border-color: var(--accent);
    box-shadow: 0 0 8px var(--accent-glow);
}
.vnccs-common-creator-toggle input:checked ~ .vnccs-common-creator-toggle-thumb {
    transform: translateX(16px);
    background: var(--accent);
}
.vnccs-common-creator-toggle input:focus-visible ~ .vnccs-common-creator-toggle-track { outline: 2px solid var(--accent); outline-offset: 2px; }
.vnccs-common-creator-toggle-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
    transition: color var(--transition);
}
.vnccs-common-creator-toggle-wrap:has(input:checked) .vnccs-common-creator-toggle-label {
    color: var(--accent-hover);
}

.vnccs-common-creator-segmented-field {
    position: relative;
    isolation: isolate;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 4px;
    padding: 4px;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border);
    background: rgba(0,0,0,0.18);
    min-height: 48px;
    box-sizing: border-box;
}
.vnccs-common-creator-segmented-field::before {
    content: ""; position: absolute; top: 4px; bottom: 4px; left: 4px; z-index: -1;
    width: calc((100% - 8px - (var(--segments, 2) - 1) * 4px) / var(--segments, 2));
    border-radius: var(--radius-md); background: linear-gradient(180deg, #ff9bad 0%, #ff87a0 100%);
    box-shadow: 0 10px 22px rgba(255,143,163,0.22); pointer-events: none;
    transform: translateX(calc(var(--selected, 0) * (100% + 4px)));
    transition: transform 280ms cubic-bezier(0.22, 1, 0.36, 1);
}
.vnccs-common-creator-segmented-field.is-three {
    grid-template-columns: repeat(3, minmax(0, 1fr));
}

.vnccs-common-creator-segmented-btn {
    border: 0;
    border-radius: var(--radius-md);
    background: transparent;
    color: var(--text-secondary);
    font-family: var(--font);
    font-size: 14px;
    font-weight: 800;
    cursor: pointer;
    transition: color 180ms ease;
}

.vnccs-common-creator-segmented-btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
}
.vnccs-common-creator-segmented-btn:hover:not(:disabled):not(.is-active) {
    color: var(--text-primary);
    background: rgba(255,255,255,0.045);
}

.vnccs-common-creator-segmented-btn.is-active {
    color: #20141a;
    background: transparent;
    box-shadow: none;
}

/* Filled input highlight */
.vnccs-common-creator-input:not(:placeholder-shown):not([value=""]),
.vnccs-common-creator-input.has-value {
    border-color: rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.05);
}

/* Preview placeholder with icon */
.vnccs-common-creator-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    color: var(--text-muted);
    font-size: 11px;
    letter-spacing: 0.05em;
}
.vnccs-common-creator-placeholder-icon {
    width: 48px;
    height: 48px;
    opacity: 0.25;
}

/* LoRA slot collapsed state */
.vnccs-common-creator-lora-item.is-empty {
    opacity: 0.45;
}
.vnccs-common-creator-lora-item.is-empty:hover {
    opacity: 1;
}

.vnccs-common-creator-placeholder { color: var(--text-muted); text-align: center; font-size: 11px; }
.vnccs-common-creator-container :is(.vnccs-common-creator-segmented-btn, .vnccs-common-creator-tab, .vnccs-common-creator-model-card) {
    transition: border-color 180ms ease, background-color 180ms ease, color 180ms ease, box-shadow 180ms ease;
}
.vnccs-common-creator-trait-add:focus-visible, .vnccs-common-creator-trait-values:focus-visible, .vnccs-common-creator-tag-chip:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

.vnccs-common-creator-tag-grid {
    display: flex; flex-wrap: wrap; gap: 5px;
    max-height: 300px; overflow-y: auto;
    padding: 8px;
    background: rgba(10,10,15,0.6);
    border-radius: var(--radius-sm);
}
.vnccs-common-creator-tag-chip {
    font-family: inherit;
    text-align: left;
    padding: 4px 10px;
    background: rgba(255,255,255,0.05);
    border: 1px solid var(--border);
    border-radius: 20px;
    font-size: 11px;
    color: var(--text-secondary);
    cursor: pointer;
    user-select: none;
    transition: all var(--transition);
}
.vnccs-common-creator-tag-chip:hover:not(.selected) { background: rgba(255,143,163,0.1); border-color: var(--accent-border); color: var(--accent-hover); }
.vnccs-common-creator-tag-chip.selected { background: rgba(255,143,163,0.18); color: var(--accent-hover); border-color: var(--accent); }


/* Button hierarchy */
.vnccs-common-creator-btn-generate {
    flex: 2;
}
.vnccs-common-creator-btn-secondary {
    flex: 1;
    font-size: 10px;
}

.vnccs-common-creator-tab-row {
    position: relative;
    isolation: isolate;
    display: grid;
    grid-template-columns: repeat(var(--tabs, 3), minmax(0, 1fr));
    gap: 8px;
    margin-bottom: 6px;
}
.vnccs-common-creator-tab-row::before {
    content: ""; position: absolute; inset: 0 auto 0 0; z-index: -1;
    width: calc((100% - (var(--tabs, 3) - 1) * 8px) / var(--tabs, 3)); box-sizing: border-box;
    border: 1px solid var(--accent); border-radius: 8px; background: rgba(255,143,163,0.12);
    transform: translateX(calc(var(--selected, 0) * (100% + 8px)));
    transition: transform 280ms cubic-bezier(0.22, 1, 0.36, 1); pointer-events: none;
}

.vnccs-common-creator-tab {
    border: 1px solid var(--border);
    background: rgba(255,255,255,0.04);
    color: var(--text-secondary);
    border-radius: var(--radius-md);
    padding: 8px 10px;
    font-family: var(--font);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all var(--transition);
}

.vnccs-common-creator-tab:hover:not(.is-active) {
    border-color: var(--border-hover);
    color: var(--text-primary);
}

.vnccs-common-creator-tab.is-active {
    border-color: transparent;
    color: var(--accent-hover);
    background: transparent;
    box-shadow: none;
}

.vnccs-common-creator-subsection {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.vnccs-common-creator-model-card-list {
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.vnccs-common-creator-model-card {
    display: flex;
    flex-direction: column;
    gap: 5px;
    background: rgba(0,214,143,0.05);
    border: 1px solid rgba(0,214,143,0.25);
    border-radius: 10px;
    padding: 10px 12px 8px;
    cursor: default;
    transition: all var(--transition);
    position: relative;
    overflow: hidden;
}

.vnccs-common-creator-model-card.is-picker-head {
    min-height: 58px;
}

.vnccs-common-creator-model-card.is-installed {
    cursor: pointer;
}

.vnccs-common-creator-model-card.is-installed:hover:not(.is-selected) {
    border-color: rgba(0,214,143,0.42);
    background: rgba(0,214,143,0.08);
}

.vnccs-common-creator-model-card.is-selected {
    border-color: var(--accent);
    background: rgba(255,143,163,0.12);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.12) inset;
}

.vnccs-common-creator-model-card.is-missing {
    opacity: 0.92;
}

.vnccs-common-creator-model-card-top {
    display: flex;
    align-items: center;
    gap: 7px;
    min-width: 0;
}

.vnccs-common-creator-model-card-name {
    flex: 1;
    min-width: 0;
    color: var(--text-primary);
    font-size: 13px;
    font-weight: 700;
    line-height: 1.25;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.vnccs-common-creator-model-card-status {
    flex-shrink: 0;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.vnccs-common-creator-model-card-status.ok { color: var(--success); }
.vnccs-common-creator-model-card-status.missing { color: var(--error); }
.vnccs-common-creator-model-card-status.progress { color: var(--accent-lavender); }

.vnccs-common-creator-model-card-desc {
    color: var(--text-secondary);
    font-size: 11px;
    line-height: 1.4;
}

.vnccs-common-creator-model-card-actions {
    display: flex;
    align-items: center;
    gap: 8px;
}

.vnccs-common-creator-model-card-download {
    width: 100%;
    padding: 7px 9px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--accent-border);
    background: rgba(255,143,163,0.08);
    color: var(--accent-hover);
    font-family: var(--font);
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    cursor: pointer;
}

.vnccs-common-creator-model-card-download:hover {
    background: rgba(255,143,163,0.14);
}

.vnccs-common-creator-model-card-badge {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    background: var(--text-muted);
}

.vnccs-common-creator-model-card-badge.ok { background: var(--success); }
.vnccs-common-creator-model-card-badge.missing { background: var(--error); }
.vnccs-common-creator-model-card-badge.progress { background: var(--accent-lavender); }

.vnccs-common-creator-model-card-toggle {
    margin-left: auto;
    flex-shrink: 0;
}

.vnccs-common-creator-generation-fallback {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.vnccs-common-creator-gen-param-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px 18px;
    padding-top: 2px;
}

.vnccs-common-creator-resolution-field {
    display: block;
    flex-shrink: 0;
    margin: 4px 0 12px;
}

.vnccs-common-creator-resolution-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}

.vnccs-common-creator-resolution-value {
    color: var(--accent-hover);
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 700;
}

.vnccs-common-creator-resolution-slider {
    width: 100%;
    accent-color: var(--accent);
    cursor: pointer;
}

.vnccs-common-creator-qi2-cache {
    display: none;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 12px 18px;
    padding: 10px;
    border: 1px solid rgba(255,143,163,0.18);
    border-radius: var(--radius-md);
    background: rgba(255,143,163,0.045);
}

.vnccs-common-creator-qi2-cache-title {
    grid-column: 1 / -1;
    color: var(--accent);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.vnccs-common-creator-gen-param-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
    min-width: 0;
}

.vnccs-common-creator-gen-param-input,
.vnccs-common-creator-gen-param-select {
    width: 100%;
    height: 48px;
    box-sizing: border-box;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.045);
    color: var(--text-primary);
    font-family: var(--font);
    font-size: 14px;
    font-weight: 600;
    padding: 8px 12px;
    transition: all var(--transition);
    color-scheme: dark;
}

.vnccs-common-creator-gen-param-select {
    zoom: 1;
}

.vnccs-common-creator-gen-param-input:focus,
.vnccs-common-creator-gen-param-select:focus {
    outline: none;
    border-color: var(--accent-border);
    background: rgba(255,143,163,0.045);
    box-shadow: 0 0 0 3px rgba(255,143,163,0.06);
}

.vnccs-common-creator-gen-param-select option {
    background: #1e1e2e;
    color: #e8e8f0;
}

.vnccs-common-creator-seed-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 52px;
    gap: 8px;
    align-items: stretch;
}

.vnccs-common-creator-seed-dice-btn {
    width: 52px;
    height: 48px;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.045);
    color: var(--text-secondary);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all var(--transition);
}

.vnccs-common-creator-seed-dice-btn:hover:not(.is-active) {
    border-color: var(--border-hover);
    color: var(--text-primary);
}

.vnccs-common-creator-seed-dice-btn.is-active {
    border-color: var(--accent);
    background: rgba(255,143,163,0.16);
    color: var(--accent-hover);
    box-shadow: 0 0 0 1px rgba(255,143,163,0.14) inset;
}

.vnccs-common-creator-seed-dice-btn svg {
    width: 24px;
    height: 24px;
    display: block;
}
.vnccs-common-creator-seed-row .vnccs-common-creator-gen-param-input.is-random {
    border-color: var(--accent-border); color: var(--accent-hover); background: var(--accent-subtle);
}

.vnccs-common-creator-character-wizard-btn {
    width: 100%;
    min-height: 40px;
    margin-bottom: 8px;
    flex: 0 0 auto;
}

.vnccs-common-creator-container .vnccs-common-modal {
    width: min(520px, calc(100% - 48px));
    max-width: min(520px, calc(100% - 48px));
    box-sizing: border-box;
    background: rgba(26,26,38,0.96);
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-md);
    color: var(--text-primary);
    font-family: var(--font);
    overflow: hidden;
}
.vnccs-common-creator-container .vnccs-common-modal-title {
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-hover);
    font-family: var(--font);
}
.vnccs-common-creator-container .vnccs-common-modal-btn {
    border: 1px solid var(--border-hover);
    border-radius: var(--radius-sm);
    background: var(--bg-surface);
    color: var(--text-primary);
    font-family: var(--font);
    font-weight: 700;
}
.vnccs-common-creator-container .vnccs-common-modal-btn:focus,
.vnccs-common-creator-container .vnccs-common-modal-btn:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(255,143,163,0.28);
}
.vnccs-common-creator-container .vnccs-common-modal-btn-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
    border-color: transparent !important;
}
.vnccs-common-creator-container .vnccs-common-modal-btn-primary:hover,
.vnccs-common-creator-container .vnccs-common-modal-btn-primary:focus,
.vnccs-common-creator-container .vnccs-common-modal-btn-primary:focus-visible,
.vnccs-common-creator-container .vnccs-common-modal-btn-primary:active {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
    color: #1a1525 !important;
}


.vnccs-common-creator-btn-upload {
    background: rgba(255, 143, 163, 0.08);
    border: 1px dashed var(--accent-border);
    color: var(--accent);
    border-radius: var(--radius-md);
    padding: 12px 24px;
    cursor: pointer;
    font-family: var(--font);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    transition: all var(--transition);
}
.vnccs-common-creator-btn-upload:hover {
    background: rgba(255, 143, 163, 0.15);
    border-color: var(--accent);
    box-shadow: 0 0 16px var(--accent-subtle);
}
.vnccs-common-creator-sprite-nav {
    display: none;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding-top: 7px;
    pointer-events: auto;
}
.vnccs-common-creator-sprite-nav.is-visible {
    display: flex;
}
.vnccs-common-creator-sprite-nav-btn {
    width: 34px;
    height: 26px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--accent-border);
    border-radius: var(--radius-md);
    background: linear-gradient(180deg, rgba(255,143,163,0.14), rgba(255,255,255,0.04));
    color: var(--accent);
    box-shadow: 0 0 12px rgba(255,143,163,0.12);
    cursor: pointer;
    transition: all var(--transition);
}
.vnccs-common-creator-sprite-nav-btn:hover {
    border-color: var(--accent);
    background: linear-gradient(180deg, rgba(255,143,163,0.22), rgba(255,255,255,0.06));
    box-shadow: 0 0 16px rgba(255,143,163,0.22);
}
.vnccs-common-creator-sprite-nav-btn:disabled {
    opacity: 0.55;
    cursor: default;
}
.vnccs-common-creator-sprite-nav-btn svg {
    width: 16px;
    height: 16px;
    stroke: currentColor;
    stroke-width: 2.6;
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.vnccs-common-creator-sprite-nav-count {
    min-width: 46px;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-secondary);
}

.vnccs-common-creator-model-picker {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.vnccs-common-creator-model-picker-menu {
    display: none;
    flex-direction: column;
    gap: 8px;
    padding: 8px;
    border: 1px solid rgba(255,143,163,0.18);
    border-radius: 10px;
    background: rgba(8,8,12,0.48);
}

.vnccs-common-creator-model-picker.is-open .vnccs-common-creator-model-picker-menu {
    display: flex;
}

.vnccs-common-creator-model-picker-group {
    display: flex;
    flex-direction: column;
    gap: 7px;
}

.vnccs-common-creator-model-picker-group-title {
    color: var(--accent-hover);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}


.vnccs-common-creator-generation-layer {
    position: absolute;
    inset: 0;
    z-index: 5;
    overflow: hidden;
    pointer-events: none;
    background: radial-gradient(circle, rgba(var(--vnccs-preview-accent-rgb), 0.04) 1px, transparent 1px), #0f0f16;
    background-size: 20px 20px, 100% 100%;
}
.vnccs-common-creator-generation-layer[hidden] { display: none; }
.vnccs-common-creator-generation-img {
    position: absolute;
    top: 0;
    left: 50%;
    width: auto;
    height: 100%;
    max-width: none;
    object-fit: contain;
    transform: translateX(-50%);
}
.vnccs-common-creator-generation-layer.uses-mannequin .vnccs-common-creator-generation-base {
    opacity: 0.64;
    filter: saturate(0.9) brightness(0.88) drop-shadow(0 0 10px rgba(var(--vnccs-preview-accent-rgb), 0.3));
    animation: vnccs-common-creator-generation-wait 1800ms ease-in-out infinite;
}
.vnccs-common-creator-generation-new {
    z-index: 1;
    clip-path: inset(0 0 100% 0);
}
.vnccs-common-creator-generation-scan {
    position: absolute;
    z-index: 2;
    top: 0;
    left: 0;
    right: 0;
    height: 100%;
    transform: translateY(-100%);
    opacity: 0.78;
    mix-blend-mode: screen;
    will-change: transform, opacity;
    animation: vnccs-common-creator-generation-ready 1500ms ease-in-out infinite;
}
.vnccs-common-creator-generation-scan::after {
    content: "";
    position: absolute;
    left: 0;
    right: 0;
    bottom: 0;
    height: clamp(18px, 6%, 42px);
    background: linear-gradient(180deg, transparent, rgba(var(--vnccs-preview-accent-rgb), 0.24) 30%, rgba(var(--vnccs-preview-highlight-rgb), 0.98) 50%, rgba(var(--vnccs-preview-accent-rgb), 0.34) 68%, transparent);
    box-shadow: 0 0 24px rgba(var(--vnccs-preview-accent-rgb), 0.3);
}
.vnccs-common-creator-generation-layer.is-revealing .vnccs-common-creator-generation-scan { animation: none; }
@keyframes vnccs-common-creator-generation-wait { 0%, 100% { opacity: 0.48; } 50% { opacity: 0.72; } }
@keyframes vnccs-common-creator-generation-ready { from { opacity: 0.52; transform: translateY(-100%); } to { opacity: 0.9; transform: translateY(0); } }

`;
