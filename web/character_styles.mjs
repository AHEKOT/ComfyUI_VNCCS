import { mountSakuraDialogMotion } from "./sakura_interactions.mjs";

export const STYLE_PICKER_CSS = `
/* Style library uses the whole node workspace without changing its size. */
.vnccs-creator-container .vnccs-style-summary {
    display: flex; gap: 12px; align-items: center; width: 100%; padding: 10px;
    text-align: left; color: var(--text-primary); background: var(--bg-surface);
    border: 1px solid var(--accent-border); border-radius: 12px; cursor: pointer;
}
.vnccs-creator-container .vnccs-style-placeholder {
    position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: center; aspect-ratio: 1;
    background: linear-gradient(145deg, #292034, #17131f);
    color: var(--text-secondary); font-size: 11px; border-radius: 10px;
}
.vnccs-style-preview-image {
    position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover;
    border-radius: inherit;
}
.vnccs-style-summary > .vnccs-style-placeholder { width: 106.4px; flex-shrink: 0; }
.vnccs-style-selection {
    display: flex; align-items: stretch; gap: 12px; min-width: 0; padding: 10px;
    border: 1px solid var(--accent-border); border-radius: 12px; background: var(--bg-surface);
}
.vnccs-style-selection > .vnccs-style-summary { flex: 1 1 0; min-width: 0; }
.vnccs-style-secondary { position: relative; flex: 1 1 0; min-width: 0; }
.vnccs-style-secondary[hidden], .vnccs-style-secondary-clear[hidden] { display: none; }
.vnccs-creator-container .vnccs-style-selection > .vnccs-style-summary,
.vnccs-creator-container .vnccs-style-secondary-trigger {
    display: flex; align-items: center; gap: 10px; width: 100%; min-height: 88px; padding: 0;
    border: 0; border-radius: 8px; background: transparent;
    text-align: left; color: var(--text-primary); cursor: pointer;
}
.vnccs-style-secondary-trigger { justify-content: flex-end; height: 100%; }
.vnccs-style-selection > .vnccs-style-summary > .vnccs-style-placeholder,
.vnccs-style-secondary-trigger > .vnccs-style-placeholder {
    width: min(88px, 45%); flex: 0 0 auto;
}
.vnccs-style-secondary-trigger > .vnccs-style-placeholder { font-size: 30px; }
.vnccs-style-selection .vnccs-style-name { overflow-wrap: anywhere; }
.vnccs-style-secondary-name {
    flex: 1 1 0; min-width: 0; padding-left: 16px; text-align: right;
    font-size: 14px; font-weight: 600; overflow-wrap: anywhere;
}
.vnccs-style-secondary-name[data-empty="true"] {
    color: var(--text-secondary); font-size: 11px; font-weight: 400; line-height: 1.4;
}
.vnccs-creator-container .vnccs-style-secondary-clear {
    position: absolute; top: 0; right: 0; width: 28px; height: 28px; padding: 0;
    border: 1px solid var(--accent-border); border-radius: 50%; background: var(--bg-primary);
    color: var(--text-primary); font-size: 20px; cursor: pointer;
}
.vnccs-style-secondary-clear::after { content: ""; position: absolute; inset: -8px; }
.vnccs-style-secondary-trigger:hover, .vnccs-style-secondary-clear:hover { border-color: var(--accent); }
.vnccs-style-secondary-trigger:focus-visible, .vnccs-style-secondary-clear:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.vnccs-style-details { display: flex; flex-direction: column; gap: 5px; min-width: 0; }
.vnccs-style-name { font-size: 14px; }
.vnccs-style-description, .vnccs-style-reference {
    display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    font-size: 11px; color: var(--text-secondary);
}
.vnccs-creator-container .vnccs-style-gallery {
    position: absolute; inset: 0; z-index: 100; display: flex; flex-direction: column;
    padding: 20px; gap: 12px; box-sizing: border-box; border-radius: inherit;
    background: var(--bg-primary); color: var(--text-primary); overflow: hidden;
    pointer-events: auto;
}
.vnccs-style-toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; flex-shrink: 0; }
.vnccs-style-toolbar input { flex: 1; min-width: 120px; }
.vnccs-style-toolbar :is(select, .vnccs-common-ui-select-trigger) { width: auto; max-width: 240px; }
.vnccs-style-heading { font-size: 18px; margin-right: auto; }
.vnccs-creator-container .vnccs-style-gallery .vnccs-creator-btn {
    flex: 0 0 auto; background: var(--bg-surface); border: 1px solid var(--accent-border);
}
.vnccs-creator-container .vnccs-style-gallery .vnccs-creator-btn:hover:not(:disabled) {
    background: var(--bg-hover); border-color: var(--accent);
}
.vnccs-style-status { color: var(--text-secondary); font-size: 12px; flex-shrink: 0; }
.vnccs-style-content { flex: 1; min-height: 0; overflow-y: auto; padding: 3px; }
.vnccs-style-size-controls { display: flex; align-items: center; gap: 12px; flex-shrink: 0; font-size: 12px; color: var(--text-secondary); }
.vnccs-style-size-slider { width: 180px; max-width: 50%; accent-color: var(--accent); cursor: pointer; }
.vnccs-style-size-value { min-width: 40px; font-variant-numeric: tabular-nums; }
.vnccs-style-preview-location { font-size: 11px; color: var(--text-secondary); overflow-wrap: anywhere; }
.vnccs-style-preview-location:empty { display: none; }
.vnccs-style-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, var(--vnccs-style-card-size, 182px)), 1fr)); gap: 14px; }
.vnccs-style-tile { position: relative; min-width: 0; }
.vnccs-creator-container .vnccs-style-card {
    position: relative; width: 100%; aspect-ratio: 1; padding: 0; overflow: hidden;
    border-radius: 14px; border: 1px solid var(--accent-border); cursor: pointer;
    background: var(--bg-surface); color: var(--text-primary); text-align: left;
}
.vnccs-style-card > .vnccs-style-placeholder { width: 100%; height: 100%; border-radius: 0; }
.vnccs-style-editor > .vnccs-style-placeholder {
    grid-row: 1 / span 5; width: 100%; max-width: 320px; align-self: start; justify-self: center;
}
.vnccs-style-card-label {
    position: absolute; inset: auto 0 0; padding: 26px 12px 12px; font-size: 13px;
    font-weight: 600; background: linear-gradient(transparent, rgba(0,0,0,.8));
}
.vnccs-creator-container .vnccs-style-card[aria-pressed="true"] { border: 2px solid var(--accent); }
.vnccs-creator-container .vnccs-style-summary:hover,
.vnccs-creator-container .vnccs-style-card:hover { border-color: var(--accent); }
.vnccs-creator-container .vnccs-style-edit,
.vnccs-creator-container .vnccs-style-delete {
    position: absolute; top: 8px; right: 8px; border: 1px solid var(--accent-border);
    border-radius: 6px; background: var(--bg-primary); color: var(--text-primary); cursor: pointer;
}
.vnccs-creator-container .vnccs-style-edit { right: 44px; }
.vnccs-creator-container .vnccs-style-delete {
    width: 28px; height: 28px; padding: 0; font-size: 20px; line-height: 1;
}
.vnccs-creator-container .vnccs-style-delete:hover:not(:disabled) { color: var(--error); }
.vnccs-creator-container .vnccs-style-editor {
    width: min(100%, 1440px); margin: 0 auto; display: grid;
    grid-template-columns: minmax(240px, 320px) minmax(0, 1fr); gap: 16px 24px; align-content: start;
}
.vnccs-style-editor > .vnccs-creator-field {
    display: flex; flex-direction: column; gap: 5px; min-width: 0; margin: 0;
}
.vnccs-style-editor > .vnccs-style-toolbar { grid-column: 2; }
.vnccs-style-editor .vnccs-creator-label {
    color: var(--text-secondary); font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .06em;
}
.vnccs-creator-container .vnccs-style-editor textarea { height: auto; min-height: 160px; resize: vertical; }
.vnccs-creator-container .vnccs-style-gallery [hidden] { display: none; }
.vnccs-creator-container .vnccs-style-gallery :focus-visible,
.vnccs-creator-container .vnccs-style-summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
`;

// Style selection stays in widget_data; the user library lives on the server.
export function createStylePicker({ host, catalog, getInfo, save, fetchApi, cleanup,
    getPreviewPayload, imageURL = value => value, listenPreview, showModal, allowNone = () => false,
    catalogURL = () => "/vnccs/character_styles", getSecondaryInfo, allowSecondary = () => true, showGenerateAll = false }) {
    const make = (tag, className, text) => {
        const element = document.createElement(tag);
        element.className = className;
        if (text !== undefined) element.textContent = text;
        if (tag === "button") element.type = "button";
        return element;
    };
    const libraryURL = () => catalogURL().split("?")[0];
    const allStyles = () => catalog.groups.flatMap(group => group.styles.map(style => ({ ...style, group: group.label })));
    const custom = { id: "custom", label: "Custom style", description: "Describe your own visual style", reference: "Your prompt", image: catalog.custom_preview || "" };
    const noStyle = { id: "none", label: "None", description: "", reference: "", prompt: "" };
    const root = make("div", "vnccs-creator-field");
    root.append(make("div", "vnccs-creator-label", "Style"));
    const trigger = make("button", "vnccs-style-summary");
    trigger.setAttribute("aria-haspopup", "dialog");
    trigger.setAttribute("aria-expanded", "false");
    const thumbnail = make("span", "vnccs-style-placeholder", "Preview");
    thumbnail.setAttribute("aria-hidden", "true");
    const details = make("span", "vnccs-style-details");
    const name = make("strong", "vnccs-style-name");
    const description = make("span", "vnccs-style-description");
    const reference = make("span", "vnccs-style-reference");
    details.append(name, description, reference);
    trigger.append(thumbnail, details);
    const secondarySlot = make("div", "vnccs-style-secondary");
    const secondaryTrigger = make("button", "vnccs-style-secondary-trigger");
    secondaryTrigger.setAttribute("aria-haspopup", "dialog");
    secondaryTrigger.setAttribute("aria-expanded", "false");
    const secondaryThumbnail = make("span", "vnccs-style-placeholder");
    secondaryThumbnail.setAttribute("aria-hidden", "true");
    const secondaryName = make("span", "vnccs-style-secondary-name");
    secondaryTrigger.append(secondaryName, secondaryThumbnail);
    const clearSecondary = make("button", "vnccs-style-secondary-clear", "×");
    clearSecondary.setAttribute("aria-label", "Remove second style");
    clearSecondary.title = "Remove second style";
    secondarySlot.append(secondaryTrigger, clearSecondary);
    const customInput = make("input", "vnccs-creator-input");
    customInput.type = "text";
    customInput.placeholder = "Describe any visual style";
    customInput.setAttribute("aria-label", "Custom style description");
    customInput.maxLength = 16000;
    customInput.hidden = true;
    customInput.oninput = () => { getInfo().custom_style = customInput.value; save(); };
    if (getSecondaryInfo) {
        const selection = make("div", "vnccs-style-selection");
        selection.append(trigger, secondarySlot);
        root.append(selection, customInput);
    } else root.append(trigger, customInput);
    let overlay = null;
    let disposed = false;
    const motion = { get _removed() { return disposed; } };
    let requestId = 0;
    let contextRevision = 0;
    let inertChildren = [];
    let batch = null;
    let deleting = false;
    let confirmation = null;
    let editingSecondary = false;
    const selectedInfo = () => editingSecondary ? getSecondaryInfo() : getInfo();
    let cardScale = 130;
    let progressMessage = "";
    let updateGallery = () => {};
    let refreshLibrary = () => {};
    let generatePreviews = async () => false;
    const hasPrimaryStyle = () => !["", "none"].includes(String(getInfo()?.style || "").trim().toLowerCase());
    const paintPreview = (element, style) => {
        element.replaceChildren();
        const fallback = make("span", "", "Preview");
        element.append(fallback);
        if (style?.image?.startsWith(`${libraryURL()}/preview?`)) {
            const image = make("img", "vnccs-style-preview-image");
            image.alt = "";
            image.loading = "lazy";
            image.decoding = "async";
            fallback.hidden = true;
            image.onerror = () => { image.remove(); fallback.hidden = false; };
            image.src = imageURL(style.image);
            element.append(image);
        }
    };
    listenPreview?.(event => {
        const detail = event.detail;
        if (!batch || disposed || String(detail.node_id) !== String(batch.payload.node_id)
            || detail.request_id !== batch.requestId) return;
        if (["queued", "running"].includes(detail.status)) {
            progressMessage = `${detail.status === "queued" ? "Queued" : "Rendering"} ${batch.index + 1}/${batch.styles.length}: ${batch.styles[batch.index].label}`;
            updateGallery();
        }
    });
    const selectedStyle = (info = getInfo()) => allStyles().find(style => style.id === info.style);
    const setValue = (value, persist = false, secondary = false) => {
        const info = secondary ? getSecondaryInfo() : getInfo();
        info.style = catalog.aliases?.[value] || value || (secondary ? "none" : catalog.default_style || "custom");
        const style = info.style === "none" ? noStyle : info.style === "custom" ? custom : selectedStyle(info);
        if (!secondary) {
            paintPreview(thumbnail, style);
            name.textContent = style?.label || info.style_label || "Unavailable style";
            description.textContent = style ? style.description || "" : info.style_description || "Saved style; restore its library to edit it";
            const styleReference = style ? style.reference || "" : info.style_reference || "";
            reference.textContent = info.style === "none" ? "No style selected" : styleReference ? `Reference: ${styleReference}` : "";
            description.hidden = !description.textContent;
            reference.hidden = !reference.textContent;
            description.title = description.textContent;
            reference.title = reference.textContent;
            customInput.value = info.custom_style || "";
            customInput.style.display = "none";
            trigger.setAttribute("aria-label", `Choose style: ${name.textContent}`);
        }
        if (getSecondaryInfo) {
            const extra = getSecondaryInfo();
            const empty = !extra.style || extra.style === "none";
            secondarySlot.hidden = !allowSecondary() || !hasPrimaryStyle();
            secondaryName.textContent = empty
                ? "Choose a second style to blend, or leave empty"
                : selectedStyle(extra)?.label || extra.style_label || "Unavailable style";
            secondaryName.setAttribute("data-empty", String(empty));
            if (empty) secondaryThumbnail.replaceChildren(make("span", "", "+"));
            else paintPreview(secondaryThumbnail, selectedStyle(extra));
            secondaryTrigger.setAttribute("aria-label", empty ? "Add second style" : `Choose second style: ${secondaryName.textContent}`);
            secondaryTrigger.title = empty ? "Add an optional second style to mix with the first" : secondaryName.textContent;
            clearSecondary.hidden = empty;
        }
        if (persist) {
            info.style_prompt = style?.prompt || "";
            info.style_label = style?.label || "";
            info.style_description = style?.description || "";
            info.style_reference = style?.reference || "";
            save();
        }
    };
    clearSecondary.onclick = () => setValue("none", true, true);
    const close = (restoreFocus = true) => {
        requestId++;
        if (batch) {
            batch.cancelled = true;
            progressMessage = "Stopping after the current image is saved...";
        }
        updateGallery = () => {};
        refreshLibrary = () => {};
        confirmation?.remove();
        confirmation = null;
        overlay?.remove();
        overlay = null;
        for (const [element, previous] of inertChildren) element.inert = previous;
        inertChildren = [];
        trigger.setAttribute("aria-expanded", "false");
        secondaryTrigger.setAttribute("aria-expanded", "false");
        if (restoreFocus && !disposed) (editingSecondary ? secondaryTrigger : trigger).focus({ preventScroll: true });
    };
    cleanup(() => {
        disposed = true; close(false);
        for (const animation of [...(motion._motionAnimations || [])]) animation.cancel();
    });
    const open = async (secondary = false) => {
        if (overlay || disposed || (secondary && (!getSecondaryInfo || !allowSecondary() || !hasPrimaryStyle()))) return;
        editingSecondary = secondary;
        if (!batch) progressMessage = "";
        overlay = make("section", "vnccs-style-gallery");
        overlay.setAttribute("role", "dialog");
        overlay.setAttribute("aria-modal", "true");
        overlay.setAttribute("aria-label", "Style library");
        overlay.onkeydown = event => {
            if (event.key === "Escape") { event.stopPropagation(); event.preventDefault(); close(); }
            if (event.key === "Tab") {
                const controls = [...overlay.querySelectorAll("button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled)")].filter(el => !el.closest("[hidden]"));
                const first = controls[0], last = controls.at(-1);
                if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
                else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
            }
        };
        const header = make("div", "vnccs-style-toolbar");
        header.append(make("strong", "vnccs-style-heading", "Style library"));
        const search = make("input", "vnccs-creator-input");
        search.placeholder = "Search styles...";
        search.setAttribute("aria-label", "Search styles");
        const filter = make("select", "vnccs-creator-select");
        filter.setAttribute("aria-label", "Style category");
        const add = make("button", "vnccs-creator-btn", "New style");
        const generateAll = make("button", "vnccs-creator-btn", "Generate all previews");
        generateAll.title = "Render every library style sequentially using current generation settings and resolution scale. Stop finishes the current image.";
        generateAll.hidden = !showGenerateAll || !getPreviewPayload;
        const dismiss = make("button", "vnccs-creator-btn", "Close");
        dismiss.title = "Close the library and stop after the current preview";
        dismiss.onclick = () => close();
        header.append(search, filter, add, generateAll, dismiss);
        const sizeControls = make("label", "vnccs-style-size-controls", "Card size");
        const sizeSlider = make("input", "vnccs-style-size-slider");
        sizeSlider.type = "range";
        sizeSlider.min = "80";
        sizeSlider.max = "250";
        sizeSlider.step = "10";
        sizeSlider.value = String(cardScale);
        sizeSlider.setAttribute("aria-label", "Style card size");
        const sizeValue = make("output", "vnccs-style-size-value");
        sizeControls.append(sizeSlider, sizeValue);
        const location = make("div", "vnccs-style-preview-location");
        location.textContent = catalog.preview_directory ? `Preview folder: ${catalog.preview_directory}` : "";
        const status = make("div", "vnccs-style-status", "Loading styles...");
        status.setAttribute("role", "status");
        const content = make("div", "vnccs-style-content");
        const grid = make("div", "vnccs-style-grid");
        const updateCardSize = () => {
            const value = Number(sizeSlider.value);
            cardScale = Number.isFinite(value) ? Math.max(80, Math.min(250, value)) : 130;
            grid.style.setProperty("--vnccs-style-card-size", `${140 * cardScale / 100}px`);
            sizeValue.textContent = `${cardScale}%`;
            sizeSlider.setAttribute("aria-valuetext", `${cardScale}%`);
        };
        sizeSlider.oninput = updateCardSize;
        updateCardSize();
        const editor = make("form", "vnccs-style-editor");
        editor.hidden = true;
        content.append(grid, editor);
        overlay.append(header, sizeControls, location, status, content);
        inertChildren = [...host.children].map(element => [element, element.inert]);
        for (const [element] of inertChildren) element.inert = true;
        host.append(overlay);
        mountSakuraDialogMotion(motion, overlay, overlay);
        (editingSecondary ? secondaryTrigger : trigger).setAttribute("aria-expanded", "true");
        search.focus({ preventScroll: true });
        let loadingCatalog = true;
        const previewSlots = new Map();
        updateGallery = (styleId) => {
            add.disabled = !editor.hidden || !!batch || deleting;
            generateAll.textContent = batch ? (batch.cancelled ? "Stopping..." : "Stop after current") : "Generate all previews";
            generateAll.disabled = loadingCatalog || deleting || !editor.hidden || !!batch?.cancelled;
            for (const edit of grid.querySelectorAll(".vnccs-style-edit")) edit.disabled = !!batch || deleting;
            for (const button of grid.querySelectorAll(".vnccs-style-delete")) button.disabled = loadingCatalog || !!batch || deleting;
            if (styleId) {
                const style = allStyles().find(item => item.id === styleId) || (styleId === "custom" ? custom : null);
                const slot = previewSlots.get(styleId);
                if (slot) paintPreview(slot, style);
                if (getInfo().style === styleId) paintPreview(thumbnail, style);
                if (getSecondaryInfo?.().style === styleId) paintPreview(secondaryThumbnail, style);
            }
            if (progressMessage) status.textContent = progressMessage;
        };
        const render = () => {
            grid.replaceChildren();
            previewSlots.clear();
            const query = search.value.trim().toLowerCase();
            const styles = [...(editingSecondary || allowNone() ? [noStyle] : []), custom, ...allStyles()].filter(style =>
                (!filter.value || style.group === filter.value) &&
                [style.label, style.description, style.reference].join(" ").toLowerCase().includes(query));
            status.textContent = styles.length ? `${styles.length} styles` : "No matching styles";
            for (const style of styles) {
                const tile = make("div", "vnccs-style-tile");
                const card = make("button", "vnccs-style-card");
                card.setAttribute("aria-pressed", String(style.id === selectedInfo().style));
                card.title = [style.description, style.reference].filter(Boolean).join("\n");
                const placeholder = make("span", "vnccs-style-placeholder", "Preview");
                placeholder.setAttribute("aria-hidden", "true");
                paintPreview(placeholder, style);
                previewSlots.set(style.id, placeholder);
                card.append(placeholder, make("span", "vnccs-style-card-label", style.label));
                card.onclick = () => {
                    if (style.id === "custom") {
                        showEditor({ label: "Custom style", prompt: selectedInfo().custom_style || "", reference: "Your prompt" });
                    } else { setValue(style.id, true, editingSecondary); close(); }
                };
                tile.append(card);
                if (style.user) {
                    const edit = make("button", "vnccs-style-edit", "Edit");
                    edit.setAttribute("aria-label", `Edit ${style.label}`);
                    edit.onclick = () => showEditor(style);
                    tile.append(edit);
                    if (showModal) {
                        const remove = make("button", "vnccs-style-delete", "×");
                        remove.title = "Delete style";
                        remove.setAttribute("aria-label", `Delete ${style.label}`);
                        remove.onclick = event => {
                            event?.stopPropagation();
                            if (loadingCatalog || batch || deleting || confirmation?.isConnected) return;
                            const currentOverlay = overlay;
                            const message = make("p", "", `Delete “${style.label}”? Its saved style and preview will be permanently removed.`);
                            const errorText = make("p", "vnccs-style-status");
                            const body = make("div", "");
                            body.append(message, errorText);
                            const dialog = showModal(overlay, "Delete style", () => body, [
                                { text: "Cancel" },
                                { text: "Delete", class: "danger", action: async () => {
                                    if (deleting || disposed || overlay !== currentOverlay) return false;
                                    deleting = true;
                                    updateGallery();
                                    const selection = JSON.stringify(getInfo());
                                    const selectionRevision = contextRevision;
                                    try {
                                        const response = await fetchApi(`${libraryURL()}/delete?style=${encodeURIComponent(style.id)}`, {
                                            method: "POST", headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" }, body: JSON.stringify({ id: style.id }),
                                        });
                                        const result = await response.json();
                                        if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
                                        if (result.deleted !== true || result.style_id !== style.id) throw new Error("Invalid style deletion response");
                                        if (disposed) return false;
                                        for (const group of catalog.groups) group.styles = group.styles.filter(item => item.id !== style.id);
                                        catalog.groups = catalog.groups.filter(group => group.label !== "My styles" || group.styles.length);
                                        if (contextRevision === selectionRevision && JSON.stringify(getInfo()) === selection) {
                                            if (getInfo().style === style.id) setValue(catalog.default_style, true);
                                            if (getSecondaryInfo?.().style === style.id) setValue("none", true, true);
                                        }
                                        progressMessage = `Deleted: ${style.label}`;
                                        refreshLibrary();
                                        if (overlay === currentOverlay) search.focus();
                                        return false;
                                    } catch (error) {
                                        errorText.textContent = `Cannot delete style: ${error.message}`;
                                        return true;
                                    } finally {
                                        deleting = false;
                                        if (!disposed) updateGallery();
                                    }
                                } },
                            ]);
                            confirmation = dialog.overlay;
                        };
                        tile.append(remove);
                    }
                }
                grid.append(tile);
            }
            updateGallery();
        };
        const updateFilters = () => {
            const previous = filter.value;
            filter.replaceChildren();
            for (const label of ["", ...catalog.groups.map(group => group.label)]) {
                const option = make("option", "", label || "All styles");
                option.value = label;
                filter.append(option);
            }
            filter.value = previous;
        };
        const showEditor = (style = {}) => {
            if (batch || deleting) return;
            progressMessage = "";
            requestId++;
            loadingCatalog = false;
            editor.replaceChildren();
            editor.hidden = false;
            grid.hidden = true;
            search.disabled = filter.disabled = add.disabled = true;
            status.textContent = style.id ? "Edit your style" : "Create your style";
            updateGallery();
            let draft = { ...style };
            const preview = make("div", "vnccs-style-placeholder");
            paintPreview(preview, draft);
            editor.append(preview);
            const fields = {};
            for (const [key, label, limit] of [["label", "Name", 100], ["description", "Short description", 240], ["reference", "Reference", 500], ["prompt", "Style prompt", 16000]]) {
                const wrapper = make("label", "vnccs-creator-field");
                wrapper.append(make("span", "vnccs-creator-label", label));
                const input = make(key === "prompt" ? "textarea" : "input", key === "prompt" ? "vnccs-creator-textarea" : "vnccs-creator-input");
                input.value = style[key] || "";
                input.maxLength = limit;
                input.required = key === "label" || key === "prompt";
                if (key === "prompt") input.rows = 8;
                wrapper.append(input);
                editor.append(wrapper);
                fields[key] = input;
            }
            const actions = make("div", "vnccs-style-toolbar");
            const submit = make("button", "vnccs-creator-btn", "Save style");
            submit.type = "submit";
            const cancel = make("button", "vnccs-creator-btn", "Back to library");
            const generate = make("button", "vnccs-creator-btn", "Generate preview");
            generate.title = "Save this style and render only its style preview with seed 0 and current generation settings.";
            generate.hidden = !getPreviewPayload;
            cancel.onclick = () => {
                editor.hidden = true; grid.hidden = false;
                search.disabled = filter.disabled = add.disabled = false;
                render(); search.focus();
            };
            actions.append(submit, generate, cancel);
            editor.append(actions);
            fields.label.focus();
            const storeDraft = async () => {
                const token = ++requestId;
                const payload = Object.fromEntries(Object.entries(fields).map(([key, input]) => [key, input.value]));
                if (draft.id) payload.id = draft.id;
                if (draft.revision) payload.revision = draft.revision;
                const response = await fetchApi(libraryURL(), {
                    method: "POST", headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" }, body: JSON.stringify(payload),
                });
                const result = await response.json();
                if (token !== requestId || disposed) return null;
                if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
                let group = catalog.groups.find(item => item.label === "My styles");
                if (!group) { group = { label: "My styles", styles: [] }; catalog.groups.push(group); }
                group.styles = group.styles.filter(item => item.id !== result.style.id);
                draft = result.style;
                group.styles.push(draft);
                setValue(draft.id, true, editingSecondary);
                paintPreview(preview, draft);
                return draft;
            };
            const commit = async renderPreview => {
                if (submit.disabled || (editor.reportValidity && !editor.reportValidity())) return;
                submit.disabled = generate.disabled = cancel.disabled = true;
                for (const field of Object.values(fields)) field.disabled = true;
                const currentOverlay = overlay;
                try {
                    const stored = await storeDraft();
                    if (!stored) return;
                    if (renderPreview) {
                        const payload = JSON.parse(JSON.stringify(getPreviewPayload()));
                        if (payload.gen_settings) payload.gen_settings = { ...payload.gen_settings, seed: 0, seed_mode: "fixed", mode_settings: {} };
                        await generatePreviews([stored], payload);
                        if (!disposed && overlay === currentOverlay) paintPreview(preview, draft);
                    } else close();
                } catch (error) {
                    if (!disposed && overlay === currentOverlay) status.textContent = `Cannot save style: ${error.message}`;
                } finally {
                    submit.disabled = generate.disabled = cancel.disabled = false;
                    for (const field of Object.values(fields)) field.disabled = false;
                }
            };
            editor.onsubmit = event => { event.preventDefault(); return commit(false); };
            generate.onclick = () => commit(true);
        };
        search.oninput = filter.onchange = render;
        add.onclick = () => showEditor();
        generateAll.onclick = () => generatePreviews();
        generatePreviews = async (requestedStyles, requestedPayload) => {
            if (batch) {
                batch.cancelled = true;
                progressMessage = "Stopping after the current image is saved...";
                updateGallery();
                return;
            }
            if (loadingCatalog || deleting || (!editor.hidden && !requestedStyles) || disposed || !overlay) return false;
            let current;
            try {
                const payload = JSON.parse(JSON.stringify(requestedPayload || getPreviewPayload()));
                const styles = requestedStyles || [...allStyles()];
                if (!requestedStyles && getInfo().custom_style?.trim()) styles.push({ ...custom });
                current = { library: libraryURL(), payload, styles, index: 0, cancelled: false, requestId: "", key: `${Date.now()}-${++requestId}` };
                batch = current;
                for (const [index, style] of styles.entries()) {
                    if (current.cancelled || disposed) break;
                    current.index = index;
                    current.requestId = `${current.key}-${index}`;
                    progressMessage = `Rendering ${index + 1}/${styles.length}: ${style.label}`;
                    updateGallery();
                    const response = await fetchApi(`${current.library}/preview`, {
                        method: "POST", headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                        body: JSON.stringify({ ...payload, style_id: style.id, request_id: current.requestId }),
                    });
                    if (!response.ok) throw new Error(await response.text());
                    const result = await response.json();
                    if (result.saved !== true) throw new Error("Server did not confirm saving the WebP file to disk");
                    if (result.style_id !== style.id || !result.image?.startsWith(`${current.library}/preview?`))
                        throw new Error("Invalid style preview response");
                    for (const group of catalog.groups) {
                        const item = group.styles.find(item => item.id === style.id);
                        if (item) item.image = result.image;
                    }
                    if (style.id === "custom") custom.image = result.image;
                    progressMessage = `Saved ${index + 1}/${styles.length}: ${style.label} (${result.width} × ${result.height}, WebP)`;
                    updateGallery(style.id);
                }
                progressMessage = current.cancelled ? "Stopped. Completed previews are saved." : `Saved all ${styles.length} style previews.`;
                return true;
            } catch (error) {
                progressMessage = `Preview generation stopped: ${error.message}. Completed previews are saved.`;
                return false;
            } finally {
                batch = null;
                if (!disposed) updateGallery();
            }
        };
        refreshLibrary = () => { requestId++; loadingCatalog = false; updateFilters(); render(); };
        updateFilters();
        render();
        const token = ++requestId;
        try {
            const response = await fetchApi(catalogURL());
            const result = await response.json();
            if (token !== requestId || disposed) return;
            if (!response.ok || !Array.isArray(result.groups)) throw new Error(result.error || `HTTP ${response.status}`);
            catalog = result;
            location.textContent = result.preview_directory ? `Preview folder: ${result.preview_directory}` : "";
            custom.image = result.custom_preview || "";
            loadingCatalog = false;
            updateFilters(); render(); setValue(getInfo().style);
        } catch (error) {
            if (token === requestId && !disposed) {
                loadingCatalog = false;
                updateGallery();
                status.textContent = `Cannot refresh styles: ${error.message}. Showing cached library.`;
            }
        }
    };
    trigger.onclick = () => open();
    secondaryTrigger.onclick = () => open(true);
    setValue(getInfo().style);
    const setCatalog = nextCatalog => {
        if (!nextCatalog || !Array.isArray(nextCatalog.groups) || catalog === nextCatalog) return;
        contextRevision++;
        close(false);
        catalog = nextCatalog;
        custom.image = catalog.custom_preview || "";
    };
    return { root, setValue, setCatalog, close: (restoreFocus = true) => { contextRevision++; close(restoreFocus); },
        customInput, generatePreviews: (...args) => generatePreviews(...args) };
}
