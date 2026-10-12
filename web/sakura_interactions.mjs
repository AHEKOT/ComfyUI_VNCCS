/* Shared Sakura controls and motion for VNCCS DOM widgets. */


export function createSakuraSelect(owner, id, choices, value, onChange, className = "vnccs-cc-select") {
    let options = choices.map(option => typeof option === "string" ? { value: option, label: option } : option);
    const button = document.createElement("button");
    button.type = "button";
    button.id = id;
    button.dataset.focusKey = id;
    button.className = `${className} vnccs-common-ui-select-trigger`;
    button.role = "combobox";
    button.ariaHasPopup = "listbox";
    button.ariaExpanded = "false";
    button.disabled = !options.length;
    button.value = options.find(option => option.value === value)?.value ?? options[0]?.value ?? "";
    const label = document.createElement("span");
    label.className = "vnccs-common-ui-select-text";
    label.textContent = options.find(option => option.value === button.value)?.label || "No options";
    button.appendChild(label);
    button.setChoices = (choices, value) => {
        const next = choices.map(option => typeof option === "string" ? { value: option, label: option } : option);
        if (button.value === value && JSON.stringify(next) === JSON.stringify(options)) return;
        if (owner._dropdown?.trigger === button) owner._closeDropdown(false);
        options = next;
        button.value = options.find(option => option.value === value)?.value ?? options[0]?.value ?? "";
        button.disabled = !options.length;
        const text = options.find(option => option.value === button.value)?.label || "No options";
        if (label.textContent !== text) label.textContent = text;
    };
    button.onchange = () => onChange?.(button.value);
    button.onclick = () => {
        if (owner._dropdown?.trigger === button) owner._closeDropdown();
        else owner._openDropdown(button, options, label);
    };
    button.onkeydown = event => {
        if (owner._dropdown?.trigger === button) {
            owner._dropdown.handleKey(event);
        } else if (["Enter", " ", "ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)
            || (event.key.length === 1 && !event.ctrlKey && !event.altKey && !event.metaKey)) {
            event.preventDefault();
            event.stopPropagation();
            owner._openDropdown(button, options, label);
            if (["Home", "End"].includes(event.key) || (event.key.length === 1 && event.key !== " ")) owner._dropdown?.handleKey(event);
        }
    };
    return button;
}


export function openSakuraDropdown(owner, trigger, options, label) {
    if (owner._removed || trigger.disabled || !trigger.isConnected) return;
    owner._closeDropdown(false);
    const menu = document.createElement("div");
    menu.className = "vnccs-common-ui-dropdown";
    menu.role = "listbox";
    menu.ariaLabel = trigger.ariaLabel || label.textContent;
    owner._dropdownSequence = (owner._dropdownSequence || 0) + 1;
    menu.id = `vnccs-cc-options-${owner.node.id}-${owner._dropdownSequence}`;
    trigger.setAttribute("aria-controls", menu.id);
    trigger.ariaExpanded = "true";
    let active = Math.max(0, options.findIndex(option => option.value === trigger.value));
    const rows = [];
    const highlight = (index, reveal = false) => {
        active = (index + options.length) % options.length;
        rows.forEach((row, rowIndex) => row.classList.toggle("vnccs-common-ui-dropdown-option--active", rowIndex === active));
        trigger.setAttribute("aria-activedescendant", rows[active].id);
        menu.style.setProperty("--highlight-y", `${rows[active].offsetTop || 0}px`);
        menu.style.setProperty("--highlight-height", `${rows[active].offsetHeight || 28}px`);
        if (reveal) {
            const row = rows[active];
            if (row.offsetTop < menu.scrollTop) menu.scrollTop = row.offsetTop;
            else if (row.offsetTop + row.offsetHeight > menu.scrollTop + menu.clientHeight) {
                menu.scrollTop = row.offsetTop + row.offsetHeight - menu.clientHeight;
            }
        }
    };
    const choose = index => {
        const value = options[index].value;
        const changed = value !== trigger.value;
        trigger.value = value;
        label.textContent = options[index].label;
        if (changed) owner._playMotion(label, [{ opacity: 0.35, transform: "translateY(2px)" }, { opacity: 1, transform: "none" }], 160);
        owner._closeDropdown();
        if (changed) trigger.onchange?.();
    };
    options.forEach((option, index) => {
        const row = document.createElement("div");
        row.className = "vnccs-common-ui-dropdown-option";
        row.id = `${menu.id}-${index}`;
        row.role = "option";
        row.ariaSelected = String(option.value === trigger.value);
        row.textContent = option.label;
        row.onpointerenter = () => highlight(index);
        row.onpointerdown = event => event.preventDefault();
        row.onclick = () => choose(index);
        rows.push(row);
        menu.appendChild(row);
    });
    let typed = "", typedAt = 0;
    const handleKey = event => {
        if (event.key === "Tab") { owner._closeDropdown(false); return; }
        if (event.key === "Escape") {
            event.preventDefault(); event.stopPropagation(); owner._closeDropdown(); return;
        }
        if (["Enter", " "].includes(event.key)) {
            event.preventDefault(); event.stopPropagation(); choose(active); return;
        }
        let next;
        if (event.key === "ArrowDown") next = active + 1;
        if (event.key === "ArrowUp") next = active - 1;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = options.length - 1;
        if (event.key === "PageDown") next = Math.min(options.length - 1, active + 8);
        if (event.key === "PageUp") next = Math.max(0, active - 8);
        if (event.key.length === 1 && !event.ctrlKey && !event.altKey && !event.metaKey) {
            typed = (Date.now() - typedAt < 700 ? typed : "") + event.key.toLowerCase();
            typedAt = Date.now();
            const query = [...typed].every(char => char === typed[0]) ? typed[0] : typed;
            for (let offset = 1; offset <= options.length; offset++) {
                const index = (active + offset) % options.length;
                if (options[index].label.toLowerCase().startsWith(query)) { next = index; break; }
            }
        }
        if (next !== undefined) {
            event.preventDefault(); event.stopPropagation(); highlight(next, true);
        }
    };
    const outside = event => {
        if (!menu.contains(event.target) && !trigger.contains(event.target)) owner._closeDropdown(false);
    };
    owner._dropdown = { trigger, menu, handleKey, outside };
    owner.container.appendChild(menu);
    owner.container.addEventListener("pointerdown", outside, true);
    const root = owner.container.getBoundingClientRect();
    const anchor = trigger.getBoundingClientRect();
    const scaleX = root.width / owner.container.clientWidth || 1;
    const scaleY = root.height / owner.container.clientHeight || 1;
    const top = (anchor.top - root.top) / scaleY;
    const bottom = (anchor.bottom - root.top) / scaleY;
    const above = Math.max(0, top - 12);
    const below = Math.max(0, owner.container.clientHeight - bottom - 12);
    const openAbove = below < 140 && above > below;
    const height = Math.min(280, options.length * 32 + 8, openAbove ? above : below);
    const width = Math.min(Math.max(anchor.width / scaleX, 200), owner.container.clientWidth - 16);
    menu.style.width = `${width}px`;
    menu.style.maxHeight = `${height}px`;
    menu.style.left = `${Math.max(8, Math.min((anchor.left - root.left) / scaleX, owner.container.clientWidth - width - 8))}px`;
    menu.style.top = `${openAbove ? top - height - 4 : bottom + 4}px`;
    const savedScroll = owner._dropdownScrolls?.get(trigger.dataset.focusKey);
    if (savedScroll) {
        menu.scrollTop = savedScroll.top;
        menu.scrollLeft = savedScroll.left;
    }
    highlight(active, !savedScroll);
    trigger.focus({ preventScroll: true });
    owner._playMotion(menu, [{ opacity: 0 }, { opacity: 1 }], 180);
}


export function closeSakuraDropdown(owner, restoreFocus = true) {
    const dropdown = owner._dropdown;
    if (!dropdown) return;
    owner._dropdown = null;
    const { trigger, menu, outside } = dropdown;
    owner._dropdownScrolls ??= new Map();
    owner._dropdownScrolls.set(trigger.dataset.focusKey, { top: menu.scrollTop, left: menu.scrollLeft });
    owner.container.removeEventListener("pointerdown", outside, true);
    trigger.ariaExpanded = "false";
    trigger.removeAttribute("aria-activedescendant");
    menu.inert = true;
    menu.style.pointerEvents = "none";
    owner._playMotion(menu, [{ opacity: 1 }, { opacity: 0 }], 120, () => menu.remove());
    if (restoreFocus && !owner._removed && trigger.isConnected) trigger.focus({ preventScroll: true });
}


export function playSakuraMotion(owner, element, keyframes, duration = 240, onFinish) {
    if (owner._removed || !element?.animate || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
        onFinish?.();
        return;
    }
    owner._motionByElement ??= new WeakMap();
    owner._motionAnimations ??= new Set();
    const previous = owner._motionByElement.get(element);
    if (previous) {
        const opacity = window.getComputedStyle?.(element).opacity;
        if (opacity !== undefined && keyframes[0].opacity !== undefined) keyframes[0].opacity = Number(opacity);
        previous.cancel();
    }
    const animation = element.animate(keyframes, { duration, easing: "cubic-bezier(0.22, 1, 0.36, 1)" });
    owner._motionByElement.set(element, animation);
    owner._motionAnimations.add(animation);
    let settled = false;
    animation.onfinish = animation.oncancel = () => {
        if (settled) return;
        settled = true;
        owner._motionAnimations.delete(animation);
        if (owner._motionByElement.get(element) === animation) owner._motionByElement.delete(element);
        onFinish?.();
    };
    return animation;
}


export function stopSakuraMotion(owner) {
    owner._closeDropdown(false);
    for (const animation of [...(owner._motionAnimations || [])]) animation.cancel();
}

export function mountSakuraDialogMotion(owner, overlay, panel) {
    const remove = overlay.remove;
    let closing = false;
    overlay.remove = function () {
        if (closing) return;
        closing = true;
        const parent = overlay.parentNode;
        const ghost = !owner._removed && parent && overlay.animate && !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
            ? overlay.cloneNode(true) : null;
        const opacity = ghost ? window.getComputedStyle?.(overlay).opacity || "1" : "1";
        if (ghost) {
            ghost.inert = true;
            ghost.setAttribute("aria-hidden", "true");
            ghost.style.pointerEvents = "none";
            ghost.removeAttribute("id");
            const originals = [overlay, ...overlay.querySelectorAll("*")];
            [ghost, ...ghost.querySelectorAll("*")].forEach((element, index) => {
                element.removeAttribute("id");
                element.scrollTop = originals[index].scrollTop;
                element.scrollLeft = originals[index].scrollLeft;
                if (originals[index] === panel) {
                    element.style.transform = window.getComputedStyle?.(panel).transform || "none";
                    element.style.opacity = window.getComputedStyle?.(panel).opacity || "1";
                }
            });
        }
        for (const element of [overlay, panel]) owner._motionByElement?.get(element)?.cancel();
        // Close immediately so pending actions cannot commit during the visual exit.
        remove.apply(this, arguments);
        if (ghost) {
            parent.appendChild(ghost);
            playSakuraMotion(owner, ghost, [{ opacity: Number(opacity) }, { opacity: 0 }], 140, () => ghost.remove());
        }
    };
    if (overlay !== panel) playSakuraMotion(owner, overlay, [{ opacity: 0 }, { opacity: 1 }], 180);
    playSakuraMotion(owner, panel, [{ opacity: 0, transform: "translateY(8px) scale(0.98)" }, { opacity: 1, transform: "none" }], 240);
}

export function sakuraRect(owner, element, host = owner.container) {
    const root = host.getBoundingClientRect();
    const rect = element.getBoundingClientRect();
    const scaleX = root.width / host.clientWidth || 1;
    const scaleY = root.height / host.clientHeight || 1;
    return { x: (rect.left - root.left) / scaleX + host.scrollLeft, y: (rect.top - root.top) / scaleY + host.scrollTop,
        width: rect.width / scaleX, height: rect.height / scaleY };
}

function motionGhost(owner, element, rect, host = owner.container) {
    const ghost = element.cloneNode(true);
    ghost.removeAttribute("id");
    for (const child of ghost.querySelectorAll("[id]")) child.removeAttribute("id");
    ghost.classList.add("vnccs-common-ui-motion-ghost");
    ghost.inert = true;
    ghost.ariaHidden = "true";
    Object.assign(ghost.style, { left: `${rect.x}px`, top: `${rect.y}px`, width: `${rect.width}px`, height: `${rect.height}px`, transform: "none" });
    host.appendChild(ghost);
    return ghost;
}

export function animateSakuraIndicator(owner, previous, current) {
    if (!previous || !current || owner._removed || !current.element.animate || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    if (previous.element === current.element && previous.x === current.x && previous.y === current.y) return;
    const marker = document.createElement("div");
    marker.className = "vnccs-common-ui-tab-glide";
    const ghost = motionGhost(owner, marker, previous);
    playSakuraMotion(owner, ghost, [{ opacity: 1, transform: "none" }, {
        opacity: 0, transform: `translate(${current.x - previous.x}px, ${current.y - previous.y}px) scale(${current.width / previous.width}, ${current.height / previous.height})`,
    }], 280, () => ghost.remove());
}

export function createSakuraInteractions(container, node) {
    const owner = { container, node, _removed: false };
    owner._closeDropdown = restore => closeSakuraDropdown(owner, restore);
    owner._openDropdown = (...args) => openSakuraDropdown(owner, ...args);
    owner._playMotion = (...args) => playSakuraMotion(owner, ...args);
    const selects = new Map();
    let selectSequence = 0;
    owner.syncSelects = () => {
        if (owner._removed) return;
        for (const select of container.querySelectorAll("select")) {
            if (select.closest("[inert]")) continue;
            if (!selects.has(select)) {
                const labels = [...(select.labels || [])];
                const label = select.ariaLabel || labels[0]?.textContent || select.parentElement.querySelector(".vnccs-creator-label")?.textContent || "Selection";
                const button = createSakuraSelect(owner, `vnccs-common-ui-select-${node.id}-${++selectSequence}`, [], "", value => {
                    select.value = value;
                    select.dispatchEvent(new Event("change", { bubbles: true }));
                    owner.syncSelects();
                }, select.className);
                button.ariaLabel = label;
                button.style.cssText = select.style.cssText;
                select.hidden = true;
                select.tabIndex = -1;
                select.ariaHidden = "true";
                select.after(button);
                for (const fieldLabel of labels) fieldLabel.htmlFor = button.id;
                selects.set(select, button);
            }
            const choices = [...select.options].map(option => ({ value: option.value, label: option.textContent }));
            if (choices.length && select.value === "" && !choices.some(option => option.value === "")) choices.unshift({ value: "", label: "Select…" });
            selects.get(select).setChoices(choices, select.value);
            selects.get(select).disabled = select.disabled || !choices.length;
        }
        for (const [select, button] of selects) if (!container.contains(select)) { button.remove(); selects.delete(select); }
    };
    const observer = typeof MutationObserver === "function" ? new MutationObserver(owner.syncSelects) : null;
    observer?.observe(container, { childList: true, subtree: true });
    const scroll = event => {
        if (owner._dropdown && !owner._dropdown.menu.contains(event.target)) owner._closeDropdown(false);
    };
    container.addEventListener("scroll", scroll, true);
    owner.captureSelection = elements => {
        const element = elements.find(element => element.classList.contains("is-active") || element.classList.contains("is-selected"));
        return element && container.clientWidth ? { element, ...sakuraRect(owner, element) } : null;
    };
    owner.animateSelection = (previous, elements) => animateSakuraIndicator(owner, previous, owner.captureSelection(elements));
    owner.captureValues = elements => elements.filter(element => element?.cloneNode).map(element => ({ element, value: element.value, clone: element.cloneNode(true) }));
    owner.animateValues = values => {
        if (owner._removed) return;
        for (const previous of values || []) {
            if (previous.element.value === previous.value || !previous.element.animate || !container.clientWidth
                || window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) continue;
            const ghost = motionGhost(owner, previous.clone, sakuraRect(owner, previous.element));
            Object.assign(ghost.style, { background: "transparent", borderColor: "transparent", boxShadow: "none" });
            playSakuraMotion(owner, ghost, [{ opacity: 1 }, { opacity: 0 }], 160, () => ghost.remove());
            const color = window.getComputedStyle?.(previous.element).color || "#e8e8f0";
            playSakuraMotion(owner, previous.element, [{ color: "transparent" }, { color }]);
        }
    };
    owner.captureView = () => {
        const positions = new Map([...container.querySelectorAll("[data-sakura-scroll]")]
            .map(region => [region.dataset.sakuraScroll, { top: region.scrollTop, left: region.scrollLeft }]));
        const active = document.activeElement;
        const focusKey = container.contains(active) ? active.dataset?.focusKey : null;
        const focusGroup = container.contains(active) ? active.closest("[data-sakura-focus-group]")?.dataset.sakuraFocusGroup : null;
        const heads = new Map([...container.querySelectorAll(".vnccs-creator-model-card.is-picker-head")]
            .filter(element => element.getClientRects().length && container.clientWidth)
            .map(element => [element.closest(".vnccs-creator-model-picker").dataset.mode,
                { element, rect: sakuraRect(owner, element), signature: element.textContent }]));
        const toggles = new Map([...container.querySelectorAll("[data-toggle-state]")].map(card => {
            const thumb = card.querySelector(".vnccs-creator-toggle-thumb");
            const track = card.querySelector(".vnccs-creator-toggle-track");
            return [card.dataset.focusKey, { checked: card.dataset.toggleState === "true",
                transform: window.getComputedStyle?.(thumb).transform, background: window.getComputedStyle?.(track).backgroundColor,
                animation: owner._motionByElement?.get(thumb), trackAnimation: owner._motionByElement?.get(track) }];
        }));
        return { positions, focusKey, focusGroup, heads, toggles };
    };
    owner.restoreView = ({ positions, focusKey, focusGroup, heads, toggles }) => {
        if (owner._removed) return;
        const restore = () => {
            if (owner._removed) return;
            for (const region of container.querySelectorAll("[data-sakura-scroll]")) {
                const position = positions.get(region.dataset.sakuraScroll);
                if (position) { region.scrollTop = position.top; region.scrollLeft = position.left; }
            }
        };
        if (focusKey) {
            let replacement = [...container.querySelectorAll("[data-focus-key]")].find(element => element.dataset.focusKey === focusKey);
            if (!replacement?.getClientRects().length && focusGroup) {
                const group = [...container.querySelectorAll("[data-sakura-focus-group]")].find(element => element.dataset.sakuraFocusGroup === focusGroup);
                replacement = group?.querySelector("[data-sakura-focus-default]");
            }
            if (replacement?.getClientRects().length) replacement.focus({ preventScroll: true });
        }
        restore();
        const after = owner.captureView();
        for (const card of container.querySelectorAll("[data-toggle-state]")) {
            const previous = toggles?.get(card.dataset.focusKey);
            const checked = card.dataset.toggleState === "true";
            if (!previous || (previous.checked === checked && !previous.animation)) continue;
            const duration = previous.checked === checked ? Math.max(1, 280 - Number(previous.animation.currentTime || 0)) : 280;
            previous.animation?.cancel(); previous.trackAnimation?.cancel();
            const thumb = card.querySelector(".vnccs-creator-toggle-thumb");
            playSakuraMotion(owner, thumb, [{ transform: previous.transform || (previous.checked ? "translateX(16px)" : "none") },
                { transform: checked ? "translateX(16px)" : "none" }], duration);
            const track = card.querySelector(".vnccs-creator-toggle-track");
            playSakuraMotion(owner, track, [{ backgroundColor: previous.background || (previous.checked ? "rgba(255,143,163,0.2)" : "rgba(255,255,255,0.08)") },
                { backgroundColor: checked ? "rgba(255,143,163,0.2)" : "rgba(255,255,255,0.08)" }], duration);
        }
        for (const [mode, head] of after.heads) {
            const previous = heads.get(mode);
            if (previous?.signature === head.signature) continue;
            if (previous && head.element.animate && !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
                const ghost = motionGhost(owner, previous.element, previous.rect);
                playSakuraMotion(owner, ghost, [{ opacity: 1 }, { opacity: 0, transform: "translateY(-5px)" }], 160, () => ghost.remove());
            }
            playSakuraMotion(owner, head.element, [{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }]);
        }
        const revision = owner._viewRevision = (owner._viewRevision || 0) + 1;
        requestAnimationFrame(() => {
            if (owner._removed || owner._viewRevision !== revision) return;
            for (const region of container.querySelectorAll("[data-sakura-scroll]")) {
                const expected = after.positions.get(region.dataset.sakuraScroll);
                const original = positions.get(region.dataset.sakuraScroll);
                if (expected && original && region.scrollTop === expected.top && region.scrollLeft === expected.left) {
                    region.scrollTop = original.top; region.scrollLeft = original.left;
                }
            }
        });
    };
    owner.emptyPreview = (image, placeholder, name, message, mode = "idle") => {
        if (owner._removed) return;
        owner._previewRevision = (owner._previewRevision || 0) + 1;
        owner._motionByElement?.get(image)?.cancel();
        image.style.display = "none";
        image.style.opacity = "";
        placeholder.style.display = "flex";
        const title = placeholder.querySelector(".vnccs-creator-placeholder-title");
        const hint = placeholder.querySelector(".vnccs-creator-placeholder-hint");
        title.textContent = name || "Create a character";
        hint.textContent = mode === "new" ? "Ready for the first preview" : message;
        placeholder.dataset.phase = mode;
        if (owner._emptyCharacter !== name) {
            playSakuraMotion(owner, placeholder, [{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }]);
            owner._emptyCharacter = name;
            const aura = image.parentElement.querySelector(".vnccs-creator-preview-aura");
            if (mode === "new" && aura) playSakuraMotion(owner, aura, [{ opacity: 0, transform: "scale(0.75)" }, { opacity: 0.5, offset: 0.4 }, { opacity: 0, transform: "scale(1.1)" }], 700);
        }
    };
    owner.prepareCharacter = (image, placeholder, name, mode = "switch") => {
        if (owner._removed) return;
        owner._previewRevision = (owner._previewRevision || 0) + 1;
        owner._characterMode = mode;
        owner._previewGhost?.remove();
        owner._previewGhost = null;
        if (image.style.display !== "none" && image.style.opacity !== "0" && image.getAttribute("src") && image.animate && !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
            const ghost = motionGhost(owner, image, sakuraRect(owner, image, image.parentElement), image.parentElement);
            playSakuraMotion(owner, ghost, [{ opacity: 1 }, { opacity: 0 }], 160, () => ghost.remove());
        }
        owner.emptyPreview(image, placeholder, name, mode === "new" ? "" : "Loading character preview…", mode);
    };
    owner.presentPreview = (image, placeholder, src, name) => {
        if (owner._removed) return;
        const first = !owner._portraitShown;
        const revision = owner._previewRevision = (owner._previewRevision || 0) + 1;
        const previous = image.style.display !== "none" && image.style.opacity !== "0" && image.getAttribute("src")
            ? { clone: image.cloneNode(true), rect: sakuraRect(owner, image, image.parentElement) } : null;
        owner._previewGhost?.remove();
        owner._previewGhost = previous && image.animate && !window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
            ? motionGhost(owner, previous.clone, previous.rect, image.parentElement) : null;
        owner._motionByElement?.get(image)?.cancel();
        image.src = src;
        image.alt = `Preview of ${name}`;
        image.style.display = "block";
        image.style.opacity = "0";
        const reveal = () => {
            if (owner._removed || owner._previewRevision !== revision) return;
            owner._portraitShown = true;
            image.style.opacity = "";
            placeholder.style.display = "none";
            if (owner._previewGhost) {
                const ghost = owner._previewGhost;
                owner._previewGhost = null;
                playSakuraMotion(owner, ghost, [{ opacity: 1 }, { opacity: 0 }], 160, () => ghost.remove());
            }
            playSakuraMotion(owner, image, [{ opacity: 0, filter: first ? "blur(5px)" : "blur(2px)" }, { opacity: 1, filter: "blur(0px)" }], first ? 420 : 240);
            const aura = image.parentElement.querySelector(".vnccs-creator-preview-aura");
            if (first && aura) playSakuraMotion(owner, aura, [{ opacity: 0, transform: "scale(0.75)" }, { opacity: 0.5, offset: 0.4 }, { opacity: 0, transform: "scale(1.1)" }], 700);
        };
        if (image.decode) image.decode().then(reveal, reveal);
        else reveal();
    };
    owner.dispose = () => {
        owner._removed = true;
        owner._previewRevision = (owner._previewRevision || 0) + 1;
        observer?.disconnect();
        container.removeEventListener("scroll", scroll, true);
        stopSakuraMotion(owner);
        owner._previewGhost?.remove();
        owner._previewGhost = null;
        selects.clear();
    };
    return owner;
}

