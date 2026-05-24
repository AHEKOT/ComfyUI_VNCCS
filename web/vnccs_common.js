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

// ── CSS Injection (once per class prefix) ─────────────────────────────────────
const _injectedStyles = new Set();

export function injectStyles(css, id) {
    if (_injectedStyles.has(id)) return;
    _injectedStyles.add(id);
    const style = document.createElement("style");
    style.textContent = css;
    document.head.appendChild(style);
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
.vnccs-common-modal-btn-primary { background: #3558c7; color: #fff; border-color: #3558c7; }
.vnccs-common-modal-btn-primary:hover { background: #4268d7; }
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
    });
    m.appendChild(row);

    overlay.appendChild(m);
    container.appendChild(overlay);
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
