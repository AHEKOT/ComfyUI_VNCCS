const clone = value => globalThis.structuredClone
    ? globalThis.structuredClone(value)
    : JSON.parse(JSON.stringify(value));

export class FactoryCommandHistory {
    constructor({ limit = 200, maxBytes = 128 * 1024 * 1024, onRestore = () => {}, onPatch = () => {}, onDiscard = () => {}, onChange = () => {} } = {}) {
        this.limit = Math.max(1, Math.floor(Number(limit) || 200));
        this.onRestore = onRestore;
        this.onPatch = onPatch;
        this.onDiscard = onDiscard;
        this.onChange = onChange;
        this.maxBytes = Math.max(1, Number(maxBytes) || 128 * 1024 * 1024);
        this.bytes = 0;
        this.undoStack = [];
        this.redoStack = [];
        this.pending = null;
    }

    begin(label, value) {
        if (this.pending) return;
        this.pending = { label: String(label || "Edit"), before: clone(value) };
    }

    commit(value) {
        if (!this.pending) return false;
        const entry = { ...this.pending, after: clone(value) };
        this.pending = null;
        if (JSON.stringify(entry.before) === JSON.stringify(entry.after)) return false;
        this._append(entry);
        return true;
    }

    pushPatch(label, patches) {
        const changed = patches.filter(patch => JSON.stringify(patch.before) !== JSON.stringify(patch.after));
        if (!changed.length) return false;
        this._append({ label: String(label || "Edit"), patches: clone(changed) });
        return true;
    }

    _append(entry) {
        for (const redo of this.redoStack) this.bytes -= redo.bytes;
        this.redoStack.length = 0;
        entry.bytes = new TextEncoder().encode(JSON.stringify(entry)).byteLength;
        this.undoStack.push(entry);
        this.bytes += entry.bytes;
        let discarded = 0;
        while (this.undoStack.length > this.limit || this.bytes > this.maxBytes) {
            this.bytes -= this.undoStack.shift().bytes;
            discarded += 1;
        }
        if (discarded) this.onDiscard(discarded);
        this.onChange();
    }

    cancel() {
        if (!this.pending) return false;
        const before = this.pending.before;
        this.pending = null;
        this.onRestore(clone(before), { direction: "cancel" });
        return true;
    }

    push(label, before, after) {
        this.pending = { label: String(label || "Edit"), before: clone(before) };
        return this.commit(after);
    }

    undo() {
        const entry = this.undoStack.pop();
        if (!entry) return false;
        this.redoStack.push(entry);
        this.onChange();
        if (entry.patches) this.onPatch(clone(entry.patches), { direction: "undo", label: entry.label });
        else this.onRestore(clone(entry.before), { direction: "undo", label: entry.label });
        return true;
    }

    redo() {
        const entry = this.redoStack.pop();
        if (!entry) return false;
        this.undoStack.push(entry);
        this.onChange();
        if (entry.patches) this.onPatch(clone(entry.patches), { direction: "redo", label: entry.label });
        else this.onRestore(clone(entry.after), { direction: "redo", label: entry.label });
        return true;
    }

    clear() {
        this.pending = null;
        this.undoStack.length = 0;
        this.redoStack.length = 0;
        this.bytes = 0;
        this.onChange();
    }

    get canUndo() { return this.undoStack.length > 0; }
    get canRedo() { return this.redoStack.length > 0; }
}

export function preserveScrollState(root, operation) {
    const states = new Map();
    for (const element of root?.querySelectorAll?.("[data-preserve-scroll]") || []) {
        states.set(element.dataset.preserveScroll, {
            top: element.scrollTop,
            left: element.scrollLeft,
        });
    }
    const result = operation();
    requestAnimationFrame(() => {
        for (const element of root?.querySelectorAll?.("[data-preserve-scroll]") || []) {
            const state = states.get(element.dataset.preserveScroll);
            if (!state) continue;
            element.scrollTop = state.top;
            element.scrollLeft = state.left;
        }
    });
    return result;
}
