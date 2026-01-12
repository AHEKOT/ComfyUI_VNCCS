# ComfyUI Custom Node UI & Widget Development Guide

This document encapsulates findings and best practices for creating advanced custom UIs within ComfyUI nodes, based on the existing VNCCS project widgets.

## Table of Contents
1. [Widget Approaches Overview](#1-widget-approaches-overview)
2. [Hiding Standard Widgets](#2-hiding-standard-widgets)
3. [Approach A: Canvas-Based Widgets](#3-approach-a-canvas-based-widgets)
4. [Approach B: DOM-Based Widgets](#4-approach-b-dom-based-widgets)
5. [WebGL / 3D Content](#5-webgl--3d-content)
6. [Mouse & Keyboard Interaction](#6-mouse--keyboard-interaction)
7. [Backend Communication (API)](#7-backend-communication-api)
8. [State Persistence](#8-state-persistence)
9. [Layout & Resizing](#9-layout--resizing)
10. [Modal Dialogs](#10-modal-dialogs)
11. [Complete Examples](#11-complete-examples)

---

## 1. Widget Approaches Overview

There are **two primary approaches** to building custom widgets in ComfyUI:

| Approach | When to Use | Example Files |
|----------|-------------|---------------|
| **Canvas-Based** | Real-time graphics, 3D viewers, custom sliders, performance-critical | `vnccs_character_studio.js`, `pose_editor.js` |
| **DOM-Based** | Complex forms, image grids, scrollable lists, standard HTML elements | `vnccs_emotion_v2.js` |

### Canvas-Based (Recommended for graphics)
- Uses `onDrawBackground` to render directly on node canvas
- No DOM elements = no Z-index issues, perfect zoom/pan sync
- Requires manual hit-testing for interactions

### DOM-Based (Recommended for complex UI)
- Uses `addDOMWidget` to embed HTML elements
- Standard HTML/CSS capabilities (scrolling, forms, images)
- Must handle zoom/pan sync manually

---

## 2. Hiding Standard Widgets

When you define inputs in Python (`INPUT_TYPES`) but want to replace them with custom controls, you must hide the default widgets **while keeping them functional** for value storage.

### ❌ Anti-Pattern
```javascript
// DON'T: Removing widgets breaks backend sync
this.widgets = this.widgets.filter(w => w.name !== "my_widget");
```

### ✅ Correct Pattern
```javascript
nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);
    
    // Hide widgets but keep them functional
    if (this.widgets) {
        for (const w of this.widgets) {
            // Collapse vertical space (negative height consumes margin)
            w.computeSize = () => [0, -4]; 
            // Prevent rendering
            w.draw = function(ctx) {};     
        }
    }
    
    return r;
};
```

### Alternative: Hide Specific Widgets
```javascript
// For selective hiding - used in vnccs_emotion_v2.js
const dataWidget = this.widgets.find(w => w.name === "hidden_data");
if (dataWidget) dataWidget.hidden = true;
```

---

## 3. Approach A: Canvas-Based Widgets

Use `onDrawBackground` for full custom rendering. This is the "native" approach.

### Basic Structure
```javascript
app.registerExtension({
    name: "VNCCS.MyWidget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "MyNode") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);
                
                // Initialize state
                this.state = {
                    value: 0.5,
                    regions: []  // For hit-testing
                };
                
                // Set default size
                this.setSize([800, 600]);
                
                return r;
            };
            
            // Custom rendering
            nodeType.prototype.onDrawBackground = function(ctx) {
                if (this.flags.collapsed) return;
                
                const w = this.size[0];
                const h = this.size[1];
                
                // Background
                ctx.fillStyle = "#FDF6E3";
                ctx.beginPath();
                ctx.roundRect(0, 0, w, h, 10);
                ctx.fill();
                
                // Reset hit-test regions each frame
                this.state.regions = [];
                
                // Draw custom controls
                this.drawSlider(ctx, 20, 40, 200, 10, "value");
            };
        }
    }
});
```

### Drawing Custom Sliders
```javascript
// Inside onDrawBackground or a helper method
const drawSlider = (ctx, x, y, w, h, label, value, min, max, key) => {
    // Store region for hit-testing
    this.sliderRegions.push({x, y, w, h, min, max, key});
    
    // Label
    ctx.fillStyle = "black";
    ctx.font = "bold 12px Arial";
    ctx.fillText(label, x, y - 4);
    
    // Track
    ctx.fillStyle = "#ddd";
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 4);
    ctx.fill();
    
    // Fill
    const pct = (value - min) / (max - min);
    ctx.fillStyle = "#666";
    ctx.beginPath();
    ctx.roundRect(x, y, w * pct, h, 4);
    ctx.fill();
    
    // Knob
    ctx.fillStyle = "#333";
    ctx.beginPath();
    ctx.arc(x + w * pct, y + h/2, 6, 0, Math.PI * 2);
    ctx.fill();
    
    // Value display
    ctx.fillStyle = "black";
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(2), x + w, y - 4);
    ctx.textAlign = "left";
};
```

---

## 4. Approach B: DOM-Based Widgets

Use `addDOMWidget` for HTML-based UIs with scrolling, forms, images.

### Basic Structure
```javascript
nodeType.prototype.onNodeCreated = function () {
    const r = onNodeCreated?.apply(this, arguments);
    const node = this;
    
    // Create container
    const container = document.createElement("div");
    container.className = "my-widget-container";
    
    // Build UI
    const input = document.createElement("input");
    input.type = "text";
    input.oninput = (e) => {
        // Sync to hidden widget
        const widget = node.widgets.find(w => w.name === "text_value");
        if (widget) widget.value = e.target.value;
    };
    container.appendChild(input);
    
    // Register DOM widget
    node.addDOMWidget("my_ui", "ui", container, {
        serialize: true,
        hideOnZoom: false,
        getValue() { return undefined; },
        setValue(v) { }
    });
    
    return r;
};
```

### Injecting Styles
```javascript
// Inject styles once at module level
const STYLE = `
.my-widget-container {
    display: flex;
    flex-direction: column;
    background: #1e1e1e;
    color: white;
    font-family: monospace;
    padding: 10px;
    box-sizing: border-box;
}
/* ... more styles ... */
`;

const styleEl = document.createElement("style");
styleEl.textContent = STYLE;
document.head.appendChild(styleEl);
```

### Handling Resize
```javascript
node.onResize = function(size) {
    const [w, h] = size;
    // Subtract padding/margins from node chrome
    container.style.width = (w - 20) + "px";
    container.style.height = (h - 60) + "px";
};
```

---

## 5. WebGL / 3D Content

For 3D preview inside a node, use **offscreen rendering** with WebGL.

### Pattern: Offscreen WebGL Viewer
```javascript
class OffscreenModelViewer {
    constructor(width, height) {
        // Create offscreen canvas (NOT attached to DOM)
        this.canvas = document.createElement("canvas");
        this.canvas.width = width;
        this.canvas.height = height;
        
        // Get WebGL context
        this.gl = this.canvas.getContext('webgl') 
            || this.canvas.getContext('experimental-webgl');
        
        this.dirty = true;
        
        if (this.gl) this.initShaders();
    }
    
    render() {
        if (!this.dirty || !this.gl) return;
        
        // ... WebGL rendering code ...
        
        this.dirty = false;
    }
}
```

### Compositing in onDrawBackground
```javascript
nodeType.prototype.onDrawBackground = function(ctx) {
    if (this.flags.collapsed) return;
    
    // Update viewer resolution if needed
    if (this.viewer.canvas.width !== centerW) {
        this.viewer.canvas.width = Math.floor(centerW);
        this.viewer.canvas.height = Math.floor(centerH);
        this.viewer.dirty = true;
    }
    
    // Render 3D scene
    this.viewer.render();
    
    // Composite onto node canvas
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(centerX, centerY, centerW, centerH, 10);
    ctx.clip();
    
    // Draw the WebGL canvas
    ctx.drawImage(this.viewer.canvas, centerX, centerY, centerW, centerH);
    
    ctx.restore();
};
```

### Loading Three.js Dynamically
```javascript
// Used in pose_editor_3d.js
const THREE_VERSION = "0.160.0";
const THREE_SOURCES = {
    core: `https://esm.sh/three@${THREE_VERSION}?dev`,
    orbit: `https://esm.sh/three@${THREE_VERSION}/examples/jsm/controls/OrbitControls?dev`,
    transform: `https://esm.sh/three@${THREE_VERSION}/examples/jsm/controls/TransformControls?dev`
};

const ThreeModuleLoader = {
    promise: null,
    async load() {
        if (!this.promise) {
            this.promise = Promise.all([
                import(THREE_SOURCES.core),
                import(THREE_SOURCES.orbit),
                import(THREE_SOURCES.transform)
            ]).then(([core, orbit, transform]) => ({
                THREE: core,
                OrbitControls: orbit.OrbitControls,
                TransformControls: transform.TransformControls
            }));
        }
        return this.promise;
    }
};
```

---

## 6. Mouse & Keyboard Interaction

### LiteGraph Event Handlers
```javascript
// Mouse events receive local node coordinates
nodeType.prototype.onMouseDown = function(event, pos, graphCanvas) {
    // pos = [x, y] relative to node top-left
    
    // Hit-test sliders
    if (this.sliderRegions) {
        for (const r of this.sliderRegions) {
            if (pos[0] >= r.x && pos[0] <= r.x + r.w &&
                pos[1] >= r.y - 10 && pos[1] <= r.y + r.h + 10) {
                this.activeSlider = r;
                this.onMouseMove(event, pos, graphCanvas);
                return true;  // Consume event
            }
        }
    }
    
    return false;  // Let LiteGraph handle it
};

nodeType.prototype.onMouseMove = function(event, pos, graphCanvas) {
    // CRITICAL: Check button state to prevent "sticky" drags
    if (event.buttons === 0) {
        this.activeSlider = null;
        this.capturingInput = false;
    }
    
    if (this.activeSlider) {
        const r = this.activeSlider;
        let val = (pos[0] - r.x) / r.w;
        val = Math.max(0, Math.min(1, val));
        const actualVal = r.min + val * (r.max - r.min);
        
        // Update state
        this.state[r.key] = actualVal;
        
        // Sync to hidden widget
        const widget = this.widgets.find(w => w.name === r.key);
        if (widget) widget.value = actualVal;
        
        // Request redraw
        this.setDirtyCanvas(true, true);
    }
};

nodeType.prototype.onMouseUp = function(event, pos, graphCanvas) {
    this.activeSlider = null;
    this.capturingInput = false;
};

nodeType.prototype.onWheel = function(event, pos, graphCanvas) {
    if (this.viewer) {
        this.viewer.params.zoomOffset += event.deltaY * 0.05;
        this.viewer.dirty = true;
        this.setDirtyCanvas(true, true);
    }
};
```

### "Sticky Slider" Bug Fix
When a user drags a slider and releases the mouse **outside** the node, `onMouseUp` may not fire. Always check `event.buttons` in `onMouseMove`:

```javascript
nodeType.prototype.onMouseMove = function(event, pos, graphCanvas) {
    // Safety: stop dragging if no buttons are pressed
    if (event.buttons === 0) {
        this.activeSlider = null;
        this.capturingInput = false;
    }
    // ... rest of logic
};
```

---

## 7. Backend Communication (API)

### Registering API Endpoints (Python)
```python
# In __init__.py
def _register_endpoints():
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception:
        return

    @PromptServer.instance.routes.post("/my_node/update")
    async def my_update_handler(request):
        try:
            data = await request.json()
            # Process data...
            return web.json_response({"status": "success", "result": ...})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)})

_register_endpoints()
```

### Calling API from JavaScript
```javascript
// Throttled update pattern
this.pendingUpdate = false;

this.updateMesh = () => {
    if (this.pendingUpdate) return;  // Skip if already pending
    
    this.pendingUpdate = true;
    
    api.fetchApi("/my_node/update", {
        method: "POST",
        body: JSON.stringify({
            param1: this.state.param1,
            param2: this.state.param2
        })
    })
    .then(r => {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
    })
    .then(data => {
        this.pendingUpdate = false;
        if (data.status === "success") {
            this.processResult(data.result);
            this.setDirtyCanvas(true, true);
        }
    })
    .catch(err => {
        this.pendingUpdate = false;
        console.error("API Error:", err);
    });
};
```

---

## 8. State Persistence

### The Problem
`onNodeCreated` runs both when:
1. A node is first added
2. A workflow is loaded from file

When loading, widget values are restored **after** `onNodeCreated`. If you only initialize state there, you'll use defaults instead of saved values.

### The Solution: onConfigure
```javascript
nodeType.prototype.onConfigure = function() {
    if (!this.state) return;
    
    // Map internal state keys to widget names
    const wMap = {
        "myValue": "widget_name",
        "anotherValue": "other_widget"
    };
    
    // Sync UI state <- Widget values
    for (const [key, wName] of Object.entries(wMap)) {
        const w = this.widgets.find(x => x.name === wName);
        if (w) {
            this.state[key] = w.value;
        }
    }
    
    // Refresh visuals
    if (this.updateVisuals) this.updateVisuals();
};
```

### DOM Widget State Restoration
```javascript
function restoreStateFromWidgets() {
    // Restore from hidden text widgets (JSON strings)
    const dataWidget = node.widgets.find(w => w.name === "saved_data");
    if (dataWidget?.value) {
        try {
            const saved = JSON.parse(dataWidget.value);
            state.selections = new Set(saved);
            renderSelections();
        } catch(e) {}
    }
}

// Call after initial data loads
fetch("/api/get_data").then(async (res) => {
    if (res.ok) {
        state.data = await res.json();
        renderData();
        restoreStateFromWidgets();  // Then restore saved selections
    }
});
```

---

## 9. Layout & Resizing

### Enforcing Minimum Size
```javascript
nodeType.prototype.onResize = function(size) {
    const MIN_W = 800;
    const MIN_H = 600;
    
    // Force minimum size
    if (size[0] < MIN_W) size[0] = MIN_W;
    if (size[1] < MIN_H) size[1] = MIN_H;
};
```

### Responsive Layout Pattern
```javascript
nodeType.prototype.onDrawBackground = function(ctx) {
    const w = this.size[0];
    const h = this.size[1];
    
    // Fixed-width columns
    const colW = Math.min(250, w * 0.25);  // Max 250px or 25%
    const margin = 10;
    
    // Fluid center
    const centerX = colW + (margin * 2);
    const centerW = w - (colW * 2) - (margin * 4);
    const centerH = h - 60;
    
    // Draw left column at (margin, 40)
    // Draw right column at (w - colW - margin, 40)
    // Draw center at (centerX, 40, centerW, centerH)
    
    // Store regions for hit-testing
    this.viewerRegion = { x: centerX, y: 40, w: centerW, h: centerH };
};
```

---

## 10. Modal Dialogs

For complex editors that need more space than the node allows (like pose editors):

### Creating a Modal Overlay
```javascript
function ensureStyles() {
    if (document.getElementById("my-dialog-style")) return;
    
    const style = document.createElement("style");
    style.id = "my-dialog-style";
    style.textContent = `
        .my-dialog-overlay {
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.75);
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            pointer-events: none;
            transition: opacity 150ms ease;
            z-index: 9000;
        }
        .my-dialog-overlay.visible {
            opacity: 1;
            pointer-events: auto;
        }
        .my-dialog-panel {
            width: min(1120px, 96vw);
            max-height: 92vh;
            background: #0f131d;
            border-radius: 16px;
            overflow: hidden;
        }
    `;
    document.head.appendChild(style);
}

class MyEditorDialog {
    constructor() {
        ensureStyles();
        
        this.overlay = document.createElement("div");
        this.overlay.className = "my-dialog-overlay";
        
        // Close on backdrop click
        this.overlay.addEventListener("click", (event) => {
            if (event.target === this.overlay) this.close();
        });
        
        // Close on Escape
        this.onEscape = (event) => {
            if (event.key === "Escape") this.close();
        };
        
        // Build panel content...
        this.panel = document.createElement("div");
        this.panel.className = "my-dialog-panel";
        this.overlay.appendChild(this.panel);
        
        document.body.appendChild(this.overlay);
    }
    
    open(data) {
        document.body.classList.add("my-dialog-open");
        document.addEventListener("keydown", this.onEscape);
        this.overlay.classList.add("visible");
        // Load data into editor...
    }
    
    close() {
        this.overlay.classList.remove("visible");
        document.body.classList.remove("my-dialog-open");
        document.removeEventListener("keydown", this.onEscape);
    }
}
```

---

## 11. Complete Examples

### Reference Files in VNCCS Project

| Widget Type | File | Description |
|-------------|------|-------------|
| Canvas + WebGL | [vnccs_character_studio.js](file:///Users/Gleb_Gavrish/Documents/private-development/ComfyUI_VNCCS-1/web/vnccs_character_studio.js) | Sliders + 3D viewer in node |
| DOM-Based | [vnccs_emotion_v2.js](file:///Users/Gleb_Gavrish/Documents/private-development/ComfyUI_VNCCS-1/web/vnccs_emotion_v2.js) | Image grid with checkboxes |
| Canvas + Modal | [pose_editor.js](file:///Users/Gleb_Gavrish/Documents/private-development/ComfyUI_VNCCS-1/web/pose_editor.js) | Preview widget + full editor dialog |
| Three.js | [pose_editor_3d.js](file:///Users/Gleb_Gavrish/Documents/private-development/ComfyUI_VNCCS-1/web/pose_editor_3d.js) | Full Three.js scene in modal |
| Simple Button | [vnccs_emotion_generator.js](file:///Users/Gleb_Gavrish/Documents/private-development/ComfyUI_VNCCS-1/web/vnccs_emotion_generator.js) | Just adds a button to existing node |

### Quick Checklist for New Widgets

- [ ] Register extension with unique name: `app.registerExtension({ name: "VNCCS.MyWidget" })`
- [ ] Check node name in `beforeRegisterNodeDef`: `if (nodeData.name === "MyNode")`
- [ ] Override `onNodeCreated` and call original: `const r = onNodeCreated?.apply(this, arguments)`
- [ ] Hide widgets you're replacing: `w.computeSize = () => [0, -4]; w.draw = () => {}`
- [ ] Initialize state object: `this.state = { ... }`
- [ ] Set default size: `this.setSize([800, 600])`
- [ ] Implement `onDrawBackground` (for canvas) OR `addDOMWidget` (for DOM)
- [ ] Implement mouse handlers with `event.buttons === 0` check
- [ ] Implement `onConfigure` for state restoration
- [ ] Implement `onResize` with minimum size enforcement
- [ ] Sync to hidden widgets on every change: `widget.value = newValue`
- [ ] Use throttling for API calls: check `pendingUpdate` flag
