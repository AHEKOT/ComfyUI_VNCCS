/**
 * Example: DOM Widget Embedding Pattern
 * 
 * This example shows how to embed a custom DOM element into a ComfyUI node logic.
 * Based on: vnccs_model_manager.js
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "VNCCS.Example.DOMWidget",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VNCCS_ExampleNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                // 1. Create container
                const container = document.createElement("div");
                Object.assign(container.style, {
                    width: "100%",
                    height: "100px",
                    backgroundColor: "#222",
                    display: "flex",
                    flexDirection: "column"
                });

                // 2. Add to node using addDOMWidget
                // 'serialize: false' prevents saving HTML state to workflow JSON
                this.addDOMWidget("ExampleWidget", "div", container, {
                    serialize: false,
                    hideOnZoom: false
                });

                // 3. Delegate logic to a helper class
                this.widgetLogic = new ExampleWidgetLogic(this, container);
            };
        }
    }
});

class ExampleWidgetLogic {
    constructor(node, container) {
        this.node = node;
        this.container = container;
        this.render();
    }

    render() {
        this.container.innerHTML = `<button style="width:100%">Click Me</button>`;
        this.container.querySelector("button").onclick = () => {
            alert("Clicked!");
        };
    }
}
