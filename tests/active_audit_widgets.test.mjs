import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createWidgetContext } from './widget_context.mjs';

const source = name => readFileSync(new URL(`../web/${name}.js`, import.meta.url), 'utf8');
const between = (text, start, end) => text.slice(text.indexOf(start), text.indexOf(end, text.indexOf(start)));
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve }; };

function domHarness() {
    const listeners = new Map(), observers = [], frames = [];
    let document;
    class Element {
        constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.style = {}; this.attrs = {}; this.handlers = {}; this.tabIndex = -1; }
        appendChild(child) { this.children.push(child); child.parentNode = this; return child; }
        append(...children) { children.forEach(child => this.appendChild(child)); }
        replaceChildren(...children) { this.children = []; this.append(...children); }
        setAttribute(key, value) { this.attrs[key] = value; }
        addEventListener(key, callback) { this.handlers[key] = callback; }
        remove() { if (!this.parentNode) return; this.parentNode.children = this.parentNode.children.filter(child => child !== this); this.parentNode = null; }
        get isConnected() { return this === document.body || this === document.head || !!this.parentNode?.isConnected; }
        contains(target) { return this === target || this.children.some(child => child.contains(target)); }
        descendants() { return this.children.flatMap(child => [child, ...child.descendants()]); }
        querySelectorAll() { return this.descendants().filter(child => ['BUTTON', 'INPUT', 'TEXTAREA', 'SELECT', 'A'].includes(child.tagName) || child.tabIndex >= 0); }
        querySelector() { return this.querySelectorAll().find(child => ['INPUT', 'TEXTAREA', 'SELECT'].includes(child.tagName) && child.type !== 'hidden' && !child.disabled) || null; }
        closest() { return this.attrs.inert !== undefined ? this : this.parentNode?.closest() || null; }
        getClientRects() { return this.hidden || this.style.display === 'none' ? [] : [{}]; }
        focus() { document.activeElement = this; for (const callback of [...listeners.get('focusin') || []]) callback({ target: this }); }
        select() { this.selected = true; }
        click() { return this.onclick?.(); }
    }
    document = {
        createElement: tag => new Element(tag),
        addEventListener(key, fn) { if (!listeners.has(key)) listeners.set(key, new Set()); listeners.get(key).add(fn); },
        removeEventListener(key, fn) { listeners.get(key)?.delete(fn); },
        getElementById: id => [document.head, document.body].flatMap(root => root.descendants()).find(element => element.id === id),
    };
    document.body = new Element('body'); document.head = new Element('head');
    class Observer {
        constructor(callback) { this.callback = callback; this.active = true; observers.push(this); }
        observe() {}
        disconnect() { this.active = false; }
    }
    const context = createWidgetContext({ document, MutationObserver: Observer,
        requestAnimationFrame: fn => frames.push(fn), setTimeout: () => 0, app: {}, console });
    const common = source('vnccs_common').replace(/^import .*;\n/gm, '').replaceAll('export ', '');
    vm.runInContext(`${common}\nglobalThis.openModal = showModal; globalThis.styles = injectStyles; globalThis.tooltip = getHelpTooltip;`, context);
    return { context, document, listeners, observers, frames, Element,
        flush() { for (const fn of frames.splice(0)) fn(); for (const observer of observers) if (observer.active) observer.callback(); } };
}

function key(modal, name, target, shiftKey = false) {
    const event = { key: name, target, shiftKey, prevented: false, preventDefault() { this.prevented = true; }, stopPropagation() {} };
    modal.handlers.keydown(event);
    return event;
}

test('shared modal labels, traps focus, and restores its opener', () => {
    const h = domHarness();
    const opener = h.document.body.appendChild(new h.Element('button')); opener.focus();
    const input = new h.Element('input');
    const { modal, overlay } = h.context.openModal(h.document.body, 'Edit', () => input, [{ text: 'Cancel' }, { text: 'Save', class: 'primary' }]);
    h.flush();
    assert.equal(modal.attrs.role, 'dialog');
    assert.equal(modal.attrs['aria-modal'], 'true');
    assert.equal(h.document.getElementById(modal.attrs['aria-labelledby']).textContent, 'Edit');
    assert.equal(h.document.activeElement, input);
    const buttons = modal.descendants().filter(item => item.tagName === 'BUTTON');
    buttons.at(-1).focus();
    assert.equal(key(modal, 'Tab', buttons.at(-1)).prevented, true);
    assert.equal(h.document.activeElement, input);
    assert.equal(key(modal, 'Tab', input, true).prevented, true);
    assert.equal(h.document.activeElement, buttons.at(-1));
    opener.focus();
    assert.equal(h.document.activeElement, input);
    key(modal, 'Escape', input);
    assert.equal(overlay.isConnected, false);
    assert.equal(h.document.activeElement, opener);
    assert.equal(h.listeners.get('focusin').size, 0);
    assert.equal(h.observers.filter(item => item.active).length, 0);
});

test('Generator message delegates to the shared dialog and preserves its styling and dismissal', async () => {
    const h = domHarness();
    h.context.showCommonModal = h.context.openModal;
    h.context.root = h.document.body.appendChild(new h.Element('div'));
    const text = source('vnccs_character_generator');
    vm.runInContext(`class Widget {
        ${between(text, '    showModal(title, message) {', '    validateNativeSeedvr(')}
        ${between(text, '    closeModal() {', '    async responseErrorMessage(')}
    }; globalThis.widget = new Widget(); widget.root = root; widget.showModal('Failed', 'Try again');`, h.context);
    h.flush();
    const overlay = h.context.widget.modalEl, modal = overlay.children[0];
    const button = modal.descendants().find(item => item.tagName === 'BUTTON');
    assert.equal(overlay.className, 'vnccs-pipe-modal-backdrop');
    assert.equal(modal.className, 'vnccs-pipe-modal');
    assert.equal(modal.children[0].className, 'vnccs-pipe-modal-title');
    assert.equal(modal.children[1].textContent, 'Try again');
    assert.equal(button.className, 'vnccs-pipe-modal-btn');
    assert.equal(h.document.activeElement, button);
    await button.click();
    assert.equal(h.context.widget.modalEl, null);
    assert.equal(h.listeners.get('focusin').size, 0);
    h.context.widget.showModal('Again', 'Message');
    h.context.widget.modalEl.onclick({ target: h.context.widget.modalEl });
    assert.equal(h.context.widget.modalEl, null);
});

test('Emotion message retains confirm focus, text safety, styling and callbacks through the shared dialog', async () => {
    const h = domHarness();
    h.context.showCommonModal = h.context.openModal;
    h.context.container = h.document.body.appendChild(new h.Element('div'));
    h.context.confirm = () => { h.context.confirmed = true; };
    const text = source('vnccs_emotion_v2');
    vm.runInContext(`${between(text, '                function showModalText(', '                function createCustomEmotionField(')}
        showModalText('Confirm', '<b>literal</b>', confirm);`, h.context);
    h.flush();
    const overlay = h.context.container.children[0], modal = overlay.children[0];
    const buttons = modal.descendants().filter(item => item.tagName === 'BUTTON');
    assert.equal(overlay.className, 'ems-modal-backdrop');
    assert.equal(modal.className, 'ems-modal');
    assert.equal(modal.children[0].hidden, true);
    assert.equal(modal.children[1].children[2].textContent, '<b>literal</b>');
    assert.equal(buttons[0].className, 'ems-modal-btn ems-modal-btn--cancel');
    assert.equal(buttons[1].className, 'ems-modal-btn ems-modal-btn--confirm');
    assert.equal(h.document.activeElement, buttons[1]);
    await buttons[1].click();
    assert.equal(h.context.confirmed, true);
    assert.equal(overlay.isConnected, false);
    assert.equal(h.listeners.get('focusin').size, 0);
});

test('nested modal and container removal clean up focus listeners', () => {
    const h = domHarness();
    const container = h.document.body.appendChild(new h.Element('div'));
    const opener = h.document.body.appendChild(new h.Element('button')); opener.focus();
    const first = h.context.openModal(container, 'First', () => new h.Element('input'), [{ text: 'OK' }]);
    h.flush();
    const parentFocus = h.document.activeElement;
    const second = h.context.openModal(container, 'Second', () => new h.Element('input'), [{ text: 'OK' }]);
    h.flush();
    parentFocus.focus();
    assert.equal(h.document.activeElement, second.content);
    second.overlay.remove();
    assert.equal(h.document.activeElement, parentFocus);
    assert.equal(h.listeners.get('focusin').size, 1);
    container.remove(); h.flush();
    assert.equal(h.document.activeElement, opener);
    assert.equal(h.listeners.get('focusin').size, 0);
    assert.equal(first.overlay.isConnected, false);
});

test('nested error modal restores the original opener after its parent closes first', () => {
    const h = domHarness();
    const opener = h.document.body.appendChild(new h.Element('button')); opener.focus();
    const parent = h.context.openModal(h.document.body, 'Wizard', () => new h.Element('textarea'), [{ text: 'Generate' }]);
    h.flush();
    const child = h.context.openModal(h.document.body, 'Error', () => new h.Element('div'), [{ text: 'OK' }]);
    h.flush();
    parent.overlay.remove(); child.overlay.remove(); h.flush();
    assert.equal(h.document.activeElement, opener);
    assert.equal(h.document.activeElement.isConnected, true);
    assert.equal(h.listeners.get('focusin').size, 0);
});

test('Clothes wizard HTTP error transition restores focus after dismissal', async () => {
    const h = domHarness();
    const opener = h.document.body.appendChild(new h.Element('button')); opener.focus();
    const state = { character: 'Alice', costume: 'Coat', costume_info: {} };
    let parent, child;
    Object.assign(h.context, { state, node: { id: 17 }, els: {},
        hasSelectedEditableCostume: () => true, beginClothesWizardRequest: () => () => true,
        ensureQwenVLReady: async () => true,
        showModal(title, build, buttons) { parent = h.context.openModal(h.document.body, title, build, buttons); },
        showWizardModelError() { child = h.context.openModal(h.document.body, 'Error', () => new h.Element('div'), [{ text: 'OK' }]); },
        showInfo() { assert.fail('Unexpected thrown-error branch'); }, saveState() {}, saveCostumeToBackend: async () => {},
        api: { fetchApi: async () => ({ ok: false, json: async () => ({ message: 'Inference failed' }) }) } });
    const text = source('vnccs_clothes_designer');
    const start = text.indexOf('const openClothesWizard =');
    const end = text.indexOf('\n                };', start) + '\n                };'.length;
    vm.runInContext(`${text.slice(start, end)}\nopenClothesWizard();`, h.context);
    h.flush();
    parent.content.children.find(element => element.tagName === 'TEXTAREA').value = 'coat';
    const generate = parent.modal.descendants().find(element => element.tagName === 'BUTTON' && element.innerText === 'FILL FIELDS');
    await generate.click(); h.flush();
    assert.equal(parent.overlay.isConnected, false);
    assert.equal(child.overlay.isConnected, true);
    child.overlay.remove();
    assert.equal(h.document.activeElement, opener);
    assert.equal(h.listeners.get('focusin').size, 0);
});

test('shared styles are injected once and help tooltip has an accessible ID', () => {
    const h = domHarness();
    h.context.styles('body {}', 'test-style'); h.context.styles('body {}', 'test-style');
    assert.equal(h.document.head.children.length, 1);
    const tooltip = h.context.tooltip();
    assert.equal(tooltip.id, 'vnccs-field-help-tooltip');
    assert.equal(tooltip.attrs.role, 'tooltip');
});

test('Clothes Designer serializes typing before blur and saves to backend on change', async () => {
    const h = domHarness();
    const state = { costume_info: {} }, els = {};
    let saved, backendSaves = 0;
    Object.assign(h.context, { state, els, node: {}, helpFor: () => '', setHelpText() {},
        saveState() { saved = JSON.stringify(state); }, saveCostumeToBackend: async () => { backendSaves++; },
        showInfo() { assert.fail('Unexpected save error'); } });
    const text = source('vnccs_clothes_designer');
    const start = text.indexOf('const createField =');
    const end = text.indexOf('\n                };', start) + '\n                };'.length;
    vm.runInContext(`${text.slice(start, end)}\ncreateField('top', '', true);`, h.context);
    els.top.value = 'silk shirt'; els.top.oninput({ target: els.top });
    assert.equal(JSON.parse(saved).costume_info.top, 'silk shirt');
    assert.equal(backendSaves, 0);
    els.top.onchange({ target: els.top });
    await Promise.resolve();
    assert.equal(backendSaves, 1);
});

test('Cloner serializes text and prompts before blur', () => {
    const h = domHarness();
    const state = { character_info: {} }, els = {};
    let saved;
    Object.assign(h.context, { state, els, helpFor: () => '', setHelpText() {},
        saveState() { saved = JSON.stringify(state); } });
    const text = source('vnccs_character_cloner');
    vm.runInContext(`${between(text, 'const createTraitField =', 'const createSegmentedField =')}\ncreateField('Hair', 'hair');`, h.context);
    els.hair.value = 'blue hair'; els.hair.oninput({ target: els.hair });
    assert.equal(JSON.parse(saved).character_info.hair, 'blue hair');
    vm.runInContext(`${between(text, 'const createTA =', 'botRow.appendChild')}\nglobalThis.promptField = createTA('Prompt', 'negative_prompt');`, h.context);
    const field = h.context.promptField.children.find(element => element.tagName === 'TEXTAREA');
    field.value = 'blurry'; field.oninput({ target: field });
    assert.equal(JSON.parse(saved).character_info.negative_prompt, 'blurry');
});

for (const stale of ['character', 'closed', 'removed', 'superseded', 'current']) {
    for (const ok of [true, false]) {
        test(`Creator wizard ${ok ? 'success' : 'failure'} respects ${stale} ownership`, async () => {
            const h = domHarness(), request = deferred();
            const state = { character: 'Alice', character_info: {} }, node = {};
            let parent, saved = 0, errors = 0;
            Object.assign(h.context, { state, node, ensureQwenVLReady: async () => true,
                showModal(title, build, buttons) { parent = h.context.openModal(h.document.body, title, build, buttons); },
                showCharacterWizardError() { errors++; },
                applyCharacterWizardData(data) { Object.assign(state.character_info, data); saved++; },
                api: { fetchApi: () => request.promise } });
            vm.runInContext(`const beginCharacterWizardRequest = createRequestGuard(node);\nglobalThis.supersede = beginCharacterWizardRequest;\n${between(source('vnccs_character_creator_v2'), 'const openCharacterWizard =', 'const doCreate =')}\nopenCharacterWizard();`, h.context);
            h.flush();
            parent.content.children.find(element => element.tagName === 'TEXTAREA').value = 'Alice idea';
            const button = parent.modal.descendants().find(element => element.tagName === 'BUTTON' && element.innerText === 'FILL FIELDS');
            const work = button.click();
            await new Promise(resolve => setImmediate(resolve));
            if (stale === 'character') state.character = 'Bob';
            if (stale === 'closed') parent.overlay.remove();
            if (stale === 'removed') node.onRemoved();
            if (stale === 'superseded') h.context.supersede();
            request.resolve({ ok, json: async () => ok ? { hair: 'Alice hair' } : { message: 'Failed' } });
            await work;
            assert.equal(saved, stale === 'current' && ok ? 1 : 0);
            assert.equal(errors, stale === 'current' && !ok ? 1 : 0);
            if (stale !== 'current') assert.equal(state.character_info.hair, undefined);
        });
    }
}

for (const stale of ['character', 'reference', 'removed', 'superseded', 'current']) {
    for (const ok of [true, false]) {
        test(`Cloner caption ${ok ? 'success' : 'failure'} respects ${stale} ownership`, async () => {
            const h = domHarness(), request = deferred();
            const state = { character: 'Alice', character_info: {}, source_images: ['Alice.png'], selected_idx: 0 }, node = {};
            const autoGenBtn = h.document.body.appendChild(new h.Element('button'));
            let saves = 0, errors = 0;
            Object.assign(h.context, { state, node, autoGenBtn, container: h.document.body,
                imgList: { querySelectorAll: () => [] }, ensureQwenVLReady: async () => true,
                updateUIFromState() {}, saveState() { saves++; }, showModal() { errors++; },
                console: { log() {} }, api: { fetchApi: () => request.promise } });
            vm.runInContext(`const beginCaptionRequest = createRequestGuard(node);\nglobalThis.supersede = beginCaptionRequest;\n${between(source('vnccs_character_cloner'), 'autoGenBtn.onclick =', '// Helper: Progress Polling')}`, h.context);
            const work = autoGenBtn.click();
            await new Promise(resolve => setImmediate(resolve));
            if (stale === 'character') state.character = 'Bob';
            if (stale === 'reference') state.source_images = ['Other.png'];
            if (stale === 'removed') node.onRemoved();
            if (stale === 'superseded') h.context.supersede();
            request.resolve({ ok, status: ok ? 200 : 500,
                json: async () => ok ? { hair: 'Alice hair' } : { error: 'DEPENDENCY_MISSING', message: 'Failed' } });
            await work;
            assert.equal(saves, stale === 'current' && ok ? 1 : 0);
            assert.equal(errors, stale === 'current' && !ok ? 1 : 0);
            if (stale !== 'current') assert.equal(state.character_info.hair, undefined);
            assert.equal(autoGenBtn.disabled, false);
        });
    }
}
