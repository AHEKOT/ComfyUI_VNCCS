// Style selection stays in widget_data; the user library lives on the server.
export function createStylePicker({ host, catalog, getInfo, save, fetchApi, cleanup,
    getPreviewPayload, imageURL = value => value, listenPreview, showModal }) {
    const make = (tag, className, text) => {
        const element = document.createElement(tag);
        element.className = className;
        if (text !== undefined) element.textContent = text;
        if (tag === "button") element.type = "button";
        return element;
    };
    const allStyles = () => catalog.groups.flatMap(group => group.styles.map(style => ({ ...style, group: group.label })));
    const custom = { id: "custom", label: "Custom style", description: "Describe your own visual style", reference: "Your prompt", image: catalog.custom_preview || "" };
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
    const customInput = make("input", "vnccs-creator-input");
    customInput.type = "text";
    customInput.placeholder = "Describe any visual style";
    customInput.setAttribute("aria-label", "Custom style description");
    customInput.maxLength = 16000;
    customInput.hidden = true;
    customInput.oninput = () => { getInfo().custom_style = customInput.value; save(); };
    root.append(trigger, customInput);
    let overlay = null;
    let disposed = false;
    let requestId = 0;
    let inertChildren = [];
    let batch = null;
    let deleting = false;
    let confirmation = null;
    let cardScale = 130;
    let progressMessage = "";
    let updateGallery = () => {};
    let refreshLibrary = () => {};
    let generatePreviews = async () => false;
    const paintPreview = (element, style) => {
        element.replaceChildren();
        const fallback = make("span", "", "Preview");
        element.append(fallback);
        if (style?.image?.startsWith("/vnccs/character_styles/preview?")) {
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
    const selectedStyle = () => allStyles().find(style => style.id === getInfo().style);
    const setValue = (value, persist = false) => {
        const info = getInfo();
        info.style = catalog.aliases?.[value] || value || catalog.default_style || "custom";
        const style = info.style === "custom" ? custom : selectedStyle();
        paintPreview(thumbnail, style);
        name.textContent = style?.label || info.style_label || "Unavailable style";
        description.textContent = style?.description || info.style_description || "Saved style; restore its library to edit it";
        reference.textContent = `Reference: ${style?.reference || info.style_reference || "Not specified"}`;
        description.title = description.textContent;
        reference.title = reference.textContent;
        customInput.value = info.custom_style || "";
        customInput.style.display = "none";
        trigger.setAttribute("aria-label", `Choose style: ${name.textContent}`);
        if (persist) {
            info.style_prompt = style?.prompt || "";
            info.style_label = style?.label || "";
            info.style_description = style?.description || "";
            info.style_reference = style?.reference || "";
            save();
        }
    };
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
        if (restoreFocus && !disposed) trigger.focus();
    };
    cleanup(() => { disposed = true; close(false); });
    const open = async () => {
        if (overlay || disposed) return;
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
        const dismiss = make("button", "vnccs-creator-btn", "Close");
        dismiss.title = "Close the library and stop after the current preview";
        dismiss.onclick = () => close();
        header.append(search, filter, add, dismiss);
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
        trigger.setAttribute("aria-expanded", "true");
        search.focus();
        let loadingCatalog = true;
        const previewSlots = new Map();
        updateGallery = (styleId) => {
            add.disabled = !editor.hidden || !!batch || deleting;
            for (const edit of grid.querySelectorAll(".vnccs-style-edit")) edit.disabled = !!batch || deleting;
            for (const button of grid.querySelectorAll(".vnccs-style-delete")) button.disabled = loadingCatalog || !!batch || deleting;
            if (styleId) {
                const style = allStyles().find(item => item.id === styleId) || (styleId === "custom" ? custom : null);
                const slot = previewSlots.get(styleId);
                if (slot) paintPreview(slot, style);
                if (getInfo().style === styleId) paintPreview(thumbnail, style);
            }
            if (progressMessage) status.textContent = progressMessage;
        };
        const render = () => {
            grid.replaceChildren();
            previewSlots.clear();
            const query = search.value.trim().toLowerCase();
            const styles = [custom, ...allStyles()].filter(style =>
                (!filter.value || style.group === filter.value) &&
                [style.label, style.description, style.reference].join(" ").toLowerCase().includes(query));
            status.textContent = styles.length ? `${styles.length} styles` : "No matching styles";
            for (const style of styles) {
                const tile = make("div", "vnccs-style-tile");
                const card = make("button", "vnccs-style-card");
                card.setAttribute("aria-pressed", String(style.id === getInfo().style));
                card.title = [style.description, style.reference].filter(Boolean).join("\n");
                const placeholder = make("span", "vnccs-style-placeholder", "Preview");
                placeholder.setAttribute("aria-hidden", "true");
                paintPreview(placeholder, style);
                previewSlots.set(style.id, placeholder);
                card.append(placeholder, make("span", "vnccs-style-card-label", style.label));
                card.onclick = () => {
                    if (style.id === "custom") {
                        showEditor({ label: "Custom style", prompt: getInfo().custom_style || "", reference: "Your prompt" });
                    } else { setValue(style.id, true); close(); }
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
                                    try {
                                        const response = await fetchApi(`/vnccs/character_styles/delete?style=${encodeURIComponent(style.id)}`, {
                                            method: "POST", headers: { "X-VNCCS-CSRF": "1" },
                                        });
                                        const result = await response.json();
                                        if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
                                        if (result.deleted !== true || result.style_id !== style.id) throw new Error("Invalid style deletion response");
                                        if (disposed) return false;
                                        for (const group of catalog.groups) group.styles = group.styles.filter(item => item.id !== style.id);
                                        catalog.groups = catalog.groups.filter(group => group.label !== "My styles" || group.styles.length);
                                        if (getInfo().style === style.id) setValue(catalog.default_style, true);
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
                const wrapper = make("label", "vnccs-creator-field", label);
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
            generate.title = "Save this style and render only its portrait preview with seed 0 and current Creator settings.";
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
                const response = await fetchApi("/vnccs/character_styles", {
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
                setValue(draft.id, true);
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
                        payload.gen_settings = { ...payload.gen_settings, seed: 0, seed_mode: "fixed", mode_settings: {} };
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
        // Retain the sequential renderer for maintenance; the UI generates one user style.
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
                if (!requestedStyles && payload.character_info.custom_style?.trim()) styles.push({ ...custom });
                current = { payload, styles, index: 0, cancelled: false, requestId: "", key: `${Date.now()}-${++requestId}` };
                batch = current;
                for (const [index, style] of styles.entries()) {
                    if (current.cancelled || disposed) break;
                    current.index = index;
                    current.requestId = `${current.key}-${index}`;
                    progressMessage = `Rendering ${index + 1}/${styles.length}: ${style.label}`;
                    updateGallery();
                    const response = await fetchApi("/vnccs/character_styles/preview", {
                        method: "POST", headers: { "Content-Type": "application/json", "X-VNCCS-CSRF": "1" },
                        body: JSON.stringify({ ...payload, style_id: style.id, request_id: current.requestId }),
                    });
                    if (!response.ok) throw new Error(await response.text());
                    const result = await response.json();
                    if (result.saved !== true) throw new Error("Server did not confirm saving the WebP file to disk");
                    if (result.style_id !== style.id || !result.image?.startsWith("/vnccs/character_styles/preview?"))
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
            const response = await fetchApi("/vnccs/character_styles");
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
    trigger.onclick = open;
    setValue(getInfo().style);
    return { root, setValue, customInput, generatePreviews: (...args) => generatePreviews(...args) };
}
