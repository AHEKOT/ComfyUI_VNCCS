import { api } from "../../scripts/api.js";

// Keep extension requests on ComfyUI's transport, including launcher sessions.
export const vnccsApi = new Proxy(api, {
    get(target, property) {
        if (property === "fetchApi") return (route, options = {}) => {
            const headers = new Headers(options.headers || {});
            const method = (options.method || "GET").toUpperCase();
            if (!["GET", "HEAD"].includes(method)) headers.set("X-VNCCS-CSRF", "1");
            if (typeof options.body === "string" && !headers.has("Content-Type")) {
                headers.set("Content-Type", "application/json");
            }
            // Plain records also work with older Comfy clients that spread headers.
            return target.fetchApi(route, { cache: "no-store", ...options, headers: Object.fromEntries(headers.entries()) });
        };
        const value = target[property];
        return typeof value === "function" ? (...args) => value.apply(target, args) : value;
    },
});

export function mediaURL(value) {
    if (typeof value !== "string") return "";
    // Stored workflow URLs may already have been resolved by apiURL.
    if (["/vnccs/", "/view?"].some(route => value.startsWith(api.apiURL(route)))) return value;
    return /^\/(?:vnccs\/|view\?|api\/)/.test(value) ? api.apiURL(value) : value;
}

// Refresh the displayed source without changing the editor's selection or fallback.
export function refreshPreviewImage(img) {
    const source = img?.getAttribute("src");
    if (!source || /^(?:data:|blob:)/.test(source)) return;
    const loader = new Image();
    const url = new URL(source, window.location.href);
    url.searchParams.set("vnccs_refresh", String(Date.now()));
    loader.onload = () => {
        if (img.getAttribute("src") === source && img.isConnected) img.src = url.href;
    };
    loader.src = url.href;
}

export function cacheIdentity() {
    // This is a cache identity, not an authentication token.
    return globalThis.crypto?.randomUUID?.()
        || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}

export async function checkedJSON(route, options = {}) {
    const response = await vnccsApi.fetchApi(route, options);
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`Invalid server response (HTTP ${response.status})`); }
    if (!response.ok || data?.error) throw new Error(data?.error || `HTTP ${response.status}`);
    return data;
}

export function serverScope() {
    return JSON.stringify([api.apiURL("/vnccs/"), api.user || ""]);
}

export function scopedKey(key) {
    return `vnccs:v2:${serverScope()}:${key}`;
}

function scopedStorage(kind) {
    return {
        getItem(key) {
            try { return globalThis[kind].getItem(scopedKey(key)); } catch { return null; }
        },
        setItem(key, value) {
            try { globalThis[kind].setItem(scopedKey(key), value); return true; } catch { return false; }
        },
        removeItem(key) {
            try { globalThis[kind].removeItem(scopedKey(key)); } catch { /* Optional cache. */ }
        },
    };
}

export const storage = scopedStorage("localStorage");
export const sessionStore = scopedStorage("sessionStorage");

const registries = new Map();
export function serverRegistry(name) {
    const key = scopedKey(name);
    if (!registries.has(key)) registries.set(key, Object.create(null));
    return registries.get(key);
}

export function workflowScope(node) {
    const graph = node.graph;
    if (!graph) return null;
    graph.extra ||= {};
    if (!/^[A-Za-z0-9_-]{1,80}$/.test(graph.extra.vnccs_workflow_id || "")) {
        // randomUUID is unavailable on some plain-HTTP ComfyUI installations.
        graph.extra.vnccs_workflow_id = cacheIdentity();
    }
    return `${graph.extra.vnccs_workflow_id}:${node.type || "node"}:${node.id}`;
}

// Re-read server state after reconnect or when a suspended launcher becomes active.
export function watchConnection(node, refresh, registerCleanup) {
    let disposed = false;
    let pending = false;
    let disconnected = false;
    const run = async () => {
        if (disposed || pending) return;
        pending = true;
        try { await refresh(); }
        catch (error) { console.warn("[VNCCS] State refresh failed", error); }
        finally { pending = false; }
    };
    const lost = () => { disconnected = true; };
    const status = () => { if (disconnected) { disconnected = false; run(); } };
    const visible = () => { if (document.visibilityState === "visible") run(); };
    api.addEventListener("reconnecting", lost);
    api.addEventListener("reconnected", run);
    api.addEventListener("status", status);
    window.addEventListener("online", run);
    window.addEventListener("focus", run);
    document.addEventListener("visibilitychange", visible);
    registerCleanup(node, () => {
        disposed = true;
        api.removeEventListener("reconnecting", lost);
        api.removeEventListener("reconnected", run);
        api.removeEventListener("status", status);
        window.removeEventListener("online", run);
        window.removeEventListener("focus", run);
        document.removeEventListener("visibilitychange", visible);
    });
}
