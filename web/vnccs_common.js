/**
 * VNCCS Common Utilities — shared patterns for all VNCCS widgets.
 * Import: import { debounce, showModal, ... } from "./vnccs_common.js";
 */
import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";

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

// ── Widget Data Sync ──────────────────────────────────────────────────────────
// Sets widget value, triggers callback, marks canvas dirty.
export function syncWidgetData(node, widgetName, data) {
    const widget = node.widgets?.find(w => w.name === widgetName);
    if (!widget) return;
    widget.value = typeof data === "string" ? data : JSON.stringify(data);
    if (widget.callback) widget.callback(widget.value);
    if (app.graph?.setDirtyCanvas) app.graph.setDirtyCanvas(true, true);
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

// ── DOM Widget Canvas Navigation ──────────────────────────────────────────────
// DOM widgets sit above LiteGraph's canvas, so MMB events can never reach the
// canvas naturally. Forward canvas navigation gestures while leaving normal
// widget interaction and real scroll containers untouched.
export function enableMiddleMouseCanvasPan(root) {
    if (!root || root._vnccsMiddleMouseCanvasPan) return;
    root._vnccsMiddleMouseCanvasPan = true;

    const canvas = () => app.canvasEl || app.canvas?.canvas || document.querySelector("canvas.litegraph");
    let panning = false;

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
        button: source.button,
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
            button: 1,
            buttons,
            pointerId: 1,
            pointerType: "mouse",
            isPrimary: true,
        }));
    };

    const forward = (type, event, buttons) => {
        const canvasEl = canvas();
        if (!canvasEl) return;
        const pointerType = type === "mousedown" ? "pointerdown" : type === "mousemove" ? "pointermove" : "pointerup";
        canvasEl.dispatchEvent(clonePointerEvent(pointerType, event, buttons));
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

    const finishPan = (event) => {
        if (event._vnccsForwardedCanvasInput) return;
        if (!panning) return;
        panning = false;
        event.preventDefault();
        event.stopPropagation();
        forward("mouseup", event, 0);
        window.removeEventListener("mousemove", movePan, true);
        window.removeEventListener("mouseup", finishPan, true);
    };

    const movePan = (event) => {
        if (event._vnccsForwardedCanvasInput) return;
        if (!panning) return;
        event.preventDefault();
        event.stopPropagation();
        forward("mousemove", event, event.buttons || 4);
    };

    root.addEventListener("mousedown", (event) => {
        if (event._vnccsForwardedCanvasInput) return;
        if (event.button !== 1) return;
        panning = true;
        event.preventDefault();
        event.stopPropagation();
        forward("mousedown", event, 4);
        window.addEventListener("mousemove", movePan, true);
        window.addEventListener("mouseup", finishPan, true);
    }, true);

    root.addEventListener("auxclick", (event) => {
        if (event.button !== 1) return;
        event.preventDefault();
        event.stopPropagation();
    }, true);

    root.addEventListener("wheel", (event) => {
        if (event._vnccsForwardedCanvasInput) return;
        if (hasOwnWheelHandler(event.target) || hasScrollableAncestor(event.target)) return;
        const canvasEl = canvas();
        if (!canvasEl) return;
        canvasEl.dispatchEvent(cloneWheelEvent(event));
        event.preventDefault();
        event.stopPropagation();
    }, { capture: true, passive: false });
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
// buttons: [{ text, class?: "primary"|"danger", action?: async (overlay, btn) => keepOpen? }]
// Returns { overlay, modal, content }
export function showModal(container, title, contentFunc, buttons) {
    injectStyles(COMMON_CSS, "vnccs-common");

    const overlay = document.createElement("div");
    overlay.className = "vnccs-common-modal-overlay";

    const m = document.createElement("div");
    m.className = "vnccs-common-modal";

    const titleEl = document.createElement("div");
    titleEl.className = "vnccs-common-modal-title";
    titleEl.textContent = title;
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
        btn.innerText = b.text;
        btn.onclick = async () => {
            if (b.action) {
                const keepOpen = await b.action(overlay, btn);
                if (!keepOpen) overlay.remove();
            } else {
                overlay.remove();
            }
        };
        row.appendChild(btn);
        buttonEls.push({ button: btn, config: b });
    });
    m.appendChild(row);

    m.addEventListener("keydown", (event) => {
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
    requestAnimationFrame(() => {
        const field = m.querySelector("input:not([type='hidden']):not([disabled]), textarea:not([disabled]), select:not([disabled])");
        if (!field) return;
        field.focus({ preventScroll: true });
        if (typeof field.select === "function" && (field.tagName === "INPUT" || field.tagName === "TEXTAREA")) {
            field.select();
        }
    });
    return { overlay, modal: m, content };
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

// ── Safe Fetch ────────────────────────────────────────────────────────────────
// Wraps api.fetchApi with error handling. On failure shows message in container (if provided).
// Returns response or null on error.
export async function safeFetch(url, options, container = null) {
    try {
        const r = await api.fetchApi(url, options);
        if (!r.ok) {
            const errText = await r.text().catch(() => "Unknown error");
            const msg = `Server Error (${r.status}): ${errText}`;
            console.warn("[VNCCS]", msg);
            if (container) showMessage(container, msg, true);
            return null;
        }
        return r;
    } catch (e) {
        const msg = `Network Error: ${e.message || e}`;
        console.warn("[VNCCS]", msg);
        if (container) showMessage(container, msg, true);
        return null;
    }
}

// ── Generate Random Seed ──────────────────────────────────────────────────────
export function generateRandomSeed() {
    return Math.floor(Math.random() * 10000000000000);
}
