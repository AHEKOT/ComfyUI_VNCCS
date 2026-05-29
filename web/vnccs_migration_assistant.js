import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { syncDOMWidgetWidth, syncDOMWidgetWidthSoon, enableMiddleMouseCanvasPan } from "./vnccs_common.js";

const STYLE_ID = "vnccs-migration-assistant-style";
const STYLE = `
.vnccs-ma {
    height: 520px;
    box-sizing: border-box;
    display: grid;
    grid-template-rows: auto auto 1fr;
    gap: 10px;
    padding: 12px;
    background: #17191f;
    color: #e8e8ee;
    font-family: Inter, system-ui, sans-serif;
    border: 1px solid #343844;
}
.vnccs-ma * { box-sizing: border-box; }
.vnccs-ma-top {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 12px;
    align-items: start;
}
.vnccs-ma-title { font-size: 18px; font-weight: 700; }
.vnccs-ma-paths { margin-top: 4px; color: #aeb4c2; font-size: 11px; line-height: 1.45; overflow-wrap: anywhere; }
.vnccs-ma-actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
.vnccs-ma-btn {
    min-height: 30px;
    padding: 0 11px;
    border: 1px solid #4a5060;
    background: #242936;
    color: #f4f4f6;
    cursor: pointer;
    font-size: 12px;
}
.vnccs-ma-btn.primary { background: #2f6fbd; border-color: #4c88d3; }
.vnccs-ma-btn:disabled { opacity: .45; cursor: default; }
.vnccs-ma-progress {
    display: grid;
    grid-template-columns: 1fr auto;
    gap: 10px;
    align-items: center;
    color: #c9ced8;
    font-size: 12px;
}
.vnccs-ma-bar { height: 8px; background: #252a34; border: 1px solid #3a4050; overflow: hidden; }
.vnccs-ma-fill { height: 100%; width: 0%; background: #59a6ff; transition: width .2s ease; }
.vnccs-ma-main { display: grid; grid-template-columns: minmax(310px, 42%) 1fr; min-height: 0; border-top: 1px solid #303541; }
.vnccs-ma-list { min-height: 0; overflow: auto; border-right: 1px solid #303541; padding: 10px 10px 0 0; }
.vnccs-ma-detail { min-height: 0; display: grid; grid-template-rows: auto 1fr; gap: 10px; padding: 10px 0 0 10px; }
.vnccs-ma-row {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 9px;
    align-items: center;
    min-height: 52px;
    padding: 8px;
    border: 1px solid #303642;
    margin-bottom: 8px;
    background: #1d212b;
}
.vnccs-ma-name { font-size: 13px; font-weight: 650; color: #f1f3f7; }
.vnccs-ma-meta { margin-top: 3px; font-size: 11px; color: #aab2c1; }
.vnccs-ma-pill { font-size: 11px; color: #cdd6e5; background: #2a303c; border: 1px solid #3b4353; padding: 3px 6px; }
.vnccs-ma-summary { display: flex; gap: 8px; flex-wrap: wrap; }
.vnccs-ma-log {
    min-height: 0;
    overflow: auto;
    white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px;
    line-height: 1.45;
    color: #cdd3df;
    background: #111318;
    border: 1px solid #303541;
    padding: 10px;
}
.vnccs-ma-empty { color: #9da6b7; padding: 18px 4px; font-size: 13px; }
`;

function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = STYLE;
    document.head.appendChild(style);
}

function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
}

app.registerExtension({
    name: "VNCCS.MigrationAssistant",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VNCCS_MigrationAssistant") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            injectStyle();

            const node = this;
            node.setSize([920, 600]);

            const state = {
                scan: null,
                selected: new Set(),
                running: false,
                runId: "",
                status: null,
            };

            const root = el("div", "vnccs-ma");
            const top = el("div", "vnccs-ma-top");
            const titleWrap = el("div");
            titleWrap.appendChild(el("div", "vnccs-ma-title", "VNCCS Migration Assistent"));
            const paths = el("div", "vnccs-ma-paths", "Legacy: scanning...\nNew: scanning...");
            titleWrap.appendChild(paths);

            const actions = el("div", "vnccs-ma-actions");
            const scanBtn = el("button", "vnccs-ma-btn", "Scan");
            const selectedBtn = el("button", "vnccs-ma-btn primary", "Migrate Selected");
            const allBtn = el("button", "vnccs-ma-btn", "Migrate All");
            actions.append(scanBtn, selectedBtn, allBtn);
            top.append(titleWrap, actions);

            const progress = el("div", "vnccs-ma-progress");
            const bar = el("div", "vnccs-ma-bar");
            const fill = el("div", "vnccs-ma-fill");
            bar.appendChild(fill);
            const progressText = el("div", "", "Idle");
            progress.append(bar, progressText);

            const main = el("div", "vnccs-ma-main");
            const list = el("div", "vnccs-ma-list");
            const detail = el("div", "vnccs-ma-detail");
            const summary = el("div", "vnccs-ma-summary");
            const log = el("div", "vnccs-ma-log", "Ready.");
            detail.append(summary, log);
            main.append(list, detail);
            root.append(top, progress, main);

            const renderSummary = () => {
                summary.replaceChildren();
                const chars = state.scan?.characters || [];
                const selected = state.selected.size;
                const sheets = chars.reduce((n, c) => n + (c.sheet_count || 0), 0);
                const missing = chars.reduce((n, c) => n + (c.missing_sprite_targets || 0), 0);
                summary.append(
                    el("span", "vnccs-ma-pill", `${chars.length} legacy character(s)`),
                    el("span", "vnccs-ma-pill", `${selected} selected`),
                    el("span", "vnccs-ma-pill", `${sheets} sheet file(s)`),
                    el("span", "vnccs-ma-pill", `${missing} target(s) need sprites`)
                );
            };

            const renderList = () => {
                list.replaceChildren();
                const chars = state.scan?.characters || [];
                if (!chars.length) {
                    list.appendChild(el("div", "vnccs-ma-empty", "No legacy characters found."));
                    return;
                }
                for (const item of chars) {
                    const row = el("label", "vnccs-ma-row");
                    const check = document.createElement("input");
                    check.type = "checkbox";
                    check.checked = state.selected.has(item.legacy_name);
                    check.onchange = () => {
                        if (check.checked) state.selected.add(item.legacy_name);
                        else state.selected.delete(item.legacy_name);
                        renderSummary();
                        setBusy(state.running);
                    };
                    const body = el("div");
                    body.appendChild(el("div", "vnccs-ma-name", item.legacy_name === item.new_name ? item.legacy_name : `${item.legacy_name} -> ${item.new_name}`));
                    body.appendChild(el("div", "vnccs-ma-meta", `${item.sheet_count} sheet(s), ${item.existing_sprite_count} existing sprite(s), ${item.missing_sprite_targets} target(s) missing`));
                    const pill = el("span", "vnccs-ma-pill", item.config_exists ? "config" : "no config");
                    row.append(check, body, pill);
                    list.appendChild(row);
                }
            };

            const renderStatus = () => {
                const status = state.status;
                if (!status) {
                    fill.style.width = "0%";
                    progressText.textContent = "Idle";
                    return;
                }
                const total = status.total || 0;
                const current = status.current || 0;
                const pct = total ? Math.min(100, Math.round((current / total) * 100)) : 0;
                fill.style.width = `${pct}%`;
                progressText.textContent = `${status.status || "idle"} ${current}/${total}`;
                log.textContent = (status.log || []).join("\n") || status.message || "";
            };

            const setBusy = (busy) => {
                state.running = busy;
                selectedBtn.disabled = busy || state.selected.size === 0;
                allBtn.disabled = busy || !(state.scan?.characters || []).length;
                scanBtn.disabled = busy;
            };

            const scan = async () => {
                setBusy(true);
                log.textContent = "Scanning legacy path...";
                try {
                    const response = await api.fetchApi("/vnccs/migration/characters");
                    const data = await response.json();
                    if (!response.ok || data.error) throw new Error(data.error || "Scan failed");
                    state.scan = data;
                    state.selected = new Set((data.characters || []).filter(c => c.missing_sprite_targets > 0).map(c => c.legacy_name));
                    paths.textContent = `Legacy: ${data.legacy_root}\nNew: ${data.new_root}`;
                    log.textContent = "Scan complete.";
                    renderList();
                    renderSummary();
                } catch (error) {
                    log.textContent = `Scan failed: ${error?.message || error}`;
                } finally {
                    setBusy(false);
                }
            };

            const poll = async () => {
                if (!state.runId) return;
                const response = await api.fetchApi(`/vnccs/migration/status/${state.runId}`);
                const data = await response.json();
                state.status = data;
                renderStatus();
                if (data.status === "done" || data.status === "error") {
                    setBusy(false);
                    return;
                }
                setTimeout(poll, 700);
            };

            const start = async (all) => {
                const chars = all ? (state.scan?.characters || []).map(c => c.legacy_name) : Array.from(state.selected);
                if (!chars.length) return;
                setBusy(true);
                log.textContent = "Starting migration...";
                try {
                    const response = await api.fetchApi("/vnccs/migration/start", {
                        method: "POST",
                        headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                        body: JSON.stringify({ characters: chars }),
                    });
                    const data = await response.json();
                    if (!response.ok || data.error) throw new Error(data.error || "Start failed");
                    state.runId = data.run_id;
                    await poll();
                } catch (error) {
                    log.textContent = `Migration failed: ${error?.message || error}`;
                    setBusy(false);
                }
            };

            scanBtn.onclick = scan;
            selectedBtn.onclick = () => start(false);
            allBtn.onclick = () => start(true);

            enableMiddleMouseCanvasPan(root);
            node.addDOMWidget("migration_assistant_ui", "ui", root, { serialize: false, hideOnZoom: false });
            syncDOMWidgetWidthSoon(node, "migration_assistant_ui");
            renderSummary();
            renderList();
            setBusy(false);
            scan();
        };

        const onResize = nodeType.prototype.onResize;
        nodeType.prototype.onResize = function (size) {
            onResize?.apply(this, arguments);
            syncDOMWidgetWidth(this, "migration_assistant_ui");
            requestAnimationFrame(() => syncDOMWidgetWidth(this, "migration_assistant_ui"));
        };
    },
});
