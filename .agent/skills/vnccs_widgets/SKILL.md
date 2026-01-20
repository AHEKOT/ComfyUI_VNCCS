---
name: vnccs-widgets
description: Best practices and patterns for creating custom UI widgets and extensions in ComfyUI for VNCCS. Use when modifying .js files or creating new frontend features.
---

# VNCCS Widget Development Patterns

This skill documents the established patterns for creating rich, interactive UI components within ComfyUI for the VNCCS project. Based on `vnccs_model_manager.js`, `pose_editor.js`, and `vnccs_autofill.js`.

## Core Concepts

*   **Extensions**: All JS logic registers via `app.registerExtension({...})`.
*   **Shadow DOM**: We often inject raw HTML/CSS rather than using canvas-drawing for complex UIs (like lists, forms).
*   **Styling**: Use dark mode colors (approx `#222` bg, `#ccc` text) to blend with ComfyUI.

## Pattern 1: DOM Embedding (The "Manager" Pattern)
**Best for**: Lists, Tables, Model Managers, Form Inputs that need scrolling.
**Reference**: `vnccs_model_manager.js`

1.  **Usage**: Embed a standard HTML `div` inside a node.
2.  **Implementation**:
    ```javascript
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "MyNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                if(onNodeCreated) onNodeCreated.apply(this, arguments);

                const container = document.createElement("div");
                // Use addDOMWidget to mount it
                this.addDOMWidget("MyWidget", "div", container, {
                     serialize: false,
                     hideOnZoom: false 
                });
                
                // Initialize your logic class
                this.widgetLogic = new MyWidgetLogic(this, container);
            }
        }
    }
    ```
3.  **Styling**: Use `container.style.cssText` or `Object.assign(container.style, {...})`. Avoid external CSS files if possible to keep it self-contained, or inject `<style>` tags dynamically.

## Pattern 2: Modal Editor (The "Pose Editor" Pattern)
**Best for**: Complex graphical editors, Canvas manipulation, 3D Viewers that need full screen.
**Reference**: `pose_editor.js`

1.  **Usage**: A simple "Open Editor" button on the node launches a full-screen overlay.
2.  **Implementation**:
    *   Add a button widget: `this.addWidget("button", "Open", ...)`
    *   Create a singleton `Manager` or `Dialog` class that appends a `div` to `document.body`.
    *   Handle `z-index` to float above ComfyUI graph.
    *   **Data Sync**:
        *   Store data in a **Hidden Widget** on the node (`widget.type = "hidden"`, `computeSize = () => [0, -4]`).
        *   When Modal closes/saves, update the hidden widget's value and call `graph.setDirtyCanvas(true, true)`.

## Pattern 3: Backend-Driven UI (The "Autofill" Pattern)
**Best for**: Dynamic inputs based on server config (e.g., loading character json).
**Reference**: `vnccs_autofill/vnccs_autofill.js`

1.  **Usage**: Fetch data from API and programmatically update existing widgets.
2.  **Key API**: `api.fetchApi('/vnccs/...')` (wrapper around fetch).
3.  **Hot-Patching**:
    *   Loop `node.widgets`.
    *   Update `widget.value`.
    *   **WARNING**: Changing `widget.options.values` dynamically is flaky. Prefer `window.location.reload()` if the *structure* of data changes significantly (like the definition of available characters).

## Best Practices

1.  **Async/Await**: Always use async for API calls.
2.  **Polling**: If downloading large files (Models), use `setInterval` to poll status from backend and update a progress bar in DOM.
3.  **Widget Hiding**: To store data without showing it, use the `computeSize = () => [0, -4]` hack and empty `draw` method.
4.  **Serialization**:
    *   UI State (Scroll position, open tabs) -> `serialize: false`.
    *   Node logic data (Model Name, JSON Data) -> `serialize: true`.

## Code Template: Basic Extension

```javascript
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "VNCCS.MyFeature",
    async setup() {
        // Run once on load
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "VNCCS_MyNode") {
            // Hook node creation
        }
    }
});
```
