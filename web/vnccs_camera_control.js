
import { app } from "../../scripts/app.js";

// --- Configuration Constants ---
const CANVAS_SIZE = 320;
const CENTER_X = 160;
const CENTER_Y = 160;
const RADIUS_WIDE = 140;
const RADIUS_MEDIUM = 90;
const RADIUS_CLOSE = 50;

// Colors
const COLOR_BG = "#1a1a1a";
const COLOR_GRID_LINES = "#444";
const COLOR_TEXT = "#888"; 
const COLOR_ACTIVE = "#ffbd45"; // The "yellow"
const COLOR_HIGHLIGHT = "#ffffff";

// Data
const ELEVATION_STEPS = [-30, 0, 30, 60];
const DISTANCE_MAP = {
    "close-up": RADIUS_CLOSE,
    "medium shot": RADIUS_MEDIUM,
    "wide shot": RADIUS_WIDE
};
const DISTANCE_REVERSE_MAP = {
    [RADIUS_CLOSE]: "close-up",
    [RADIUS_MEDIUM]: "medium shot",
    [RADIUS_WIDE]: "wide shot"
};

// --- Custom Widget Class ---
class VNCCS_CameraWidget {
    constructor(node, inputName, inputData, app) {
        this.node = node;
        this.inputName = inputName;
        this.app = app;

        // Internal State
        this.state = {
            azimuth: 0,
            elevation: 0,
            distance: "medium shot",
            include_trigger: true
        };

        // Try load initial state
        try {
            if (this.node.widgets && this.node.widgets[0]) {
                const loaded = JSON.parse(this.node.widgets[0].value);
                this.state = { ...this.state, ...loaded };
            }
        } catch (e) {}

        this.isDragging = false;
        this.dragMode = null; // 'azimuth' or 'elevation'

        // Create Canvas Element
        this.canvas = document.createElement("canvas");
        this.canvas.width = CANVAS_SIZE;
        this.canvas.height = CANVAS_SIZE;
        this.canvas.style.borderRadius = "4px";
        this.ctx = this.canvas.getContext("2d");

        // UI Event Listeners
        this.canvas.addEventListener("mousedown", this.onMouseDown.bind(this));
        document.addEventListener("mousemove", this.onMouseMove.bind(this));
        document.addEventListener("mouseup", this.onMouseUp.bind(this));

        // Initial Draw
        this.draw();
    }

    // --- Drawing Logic ---
    draw() {
        const ctx = this.ctx;
        ctx.fillStyle = COLOR_BG;
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

        this.drawGrid(ctx);
        this.drawSubject(ctx);
        this.drawCameraTriangle(ctx);
        this.drawElevationBar(ctx);
        this.drawInfoText(ctx);
    }

    drawGrid(ctx) {
        // Draw Circles
        ctx.strokeStyle = COLOR_GRID_LINES;
        ctx.lineWidth = 1;

        [RADIUS_CLOSE, RADIUS_MEDIUM, RADIUS_WIDE].forEach(r => {
            ctx.beginPath();
            ctx.arc(CENTER_X, CENTER_Y, r, 0, Math.PI * 2);
            ctx.stroke();
        });

        // Draw Axes (X form for 45 degs)
        ctx.beginPath();
        ctx.moveTo(CENTER_X - RADIUS_WIDE, CENTER_Y);
        ctx.lineTo(CENTER_X + RADIUS_WIDE, CENTER_Y);
        ctx.moveTo(CENTER_X, CENTER_Y - RADIUS_WIDE);
        ctx.lineTo(CENTER_X, CENTER_Y + RADIUS_WIDE);
        
        // Diagonals
        const diag = RADIUS_WIDE * 0.707;
        ctx.moveTo(CENTER_X - diag, CENTER_Y - diag);
        ctx.lineTo(CENTER_X + diag, CENTER_Y + diag);
        ctx.moveTo(CENTER_X + diag, CENTER_Y - diag);
        ctx.lineTo(CENTER_X - diag, CENTER_Y + diag);
        ctx.stroke();
    }

    drawSubject(ctx) {
        // Just a box in the center
        ctx.fillStyle = "#666";
        ctx.fillRect(CENTER_X - 6, CENTER_Y - 6, 12, 12);
    }

    drawCameraTriangle(ctx) {
        const r = DISTANCE_MAP[this.state.distance];
        // Convert azimuth to math angle. 
        // 0 deg = Top ( -PI/2 )
        // 90 deg = Right ( 0 )
        // Wait, standard math: 0 is Right. -PI/2 is Top.
        // We want 0=Front(Top), 90=Right.
        // So MathAngle = (Azimuth - 90) * (PI/180)
        
        const angleRad = (this.state.azimuth - 90) * (Math.PI / 180);
        
        const cx = CENTER_X + r * Math.cos(angleRad);
        const cy = CENTER_Y + r * Math.sin(angleRad);

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angleRad + Math.PI/2); // Point towards center

        // Triangle shape
        ctx.fillStyle = COLOR_ACTIVE;
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, 10); // Pointing IN
        ctx.lineTo(-8, -8);
        ctx.lineTo(8, -8);
        ctx.closePath();
        ctx.fill();
        ctx.stroke(); // Add outline for visibility

        ctx.restore();
    }

    drawElevationBar(ctx) {
        // Simple vertical slider on the right
        const barX = CANVAS_SIZE - 20;
        const barH = 200;
        const barY = (CANVAS_SIZE - barH) / 2;

        // Track line
        ctx.strokeStyle = COLOR_GRID_LINES;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(barX, barY);
        ctx.lineTo(barX, barY + barH);
        ctx.stroke();

        // Ticks for steps
        // -30 (Bottom) to 60 (Top)
        // Map: range = 90. Bar Top = 60, Bar Bottom = -30.
        ELEVATION_STEPS.forEach(step => {
            const norm = (step + 30) / 90; // 0..1
            const y = barY + barH - (norm * barH);
            
            ctx.fillStyle = (step === this.state.elevation) ? COLOR_ACTIVE : "#666";
            ctx.beginPath();
            ctx.arc(barX, y, 4, 0, Math.PI*2);
            ctx.fill();
            
            // Text label
            if (Math.abs(step - this.state.elevation) < 0.1 || step % 30 === 0) {
                 ctx.fillStyle = "#888";
                 ctx.font = "10px sans-serif";
                 ctx.textAlign = "right";
                 ctx.fillText(step + "°", barX - 8, y + 3);
            }
        });

        // Current Indicator Handle
        const currentNorm = (this.state.elevation + 30) / 90;
        const curY = barY + barH - (currentNorm * barH);
        ctx.fillStyle = COLOR_ACTIVE;
        ctx.beginPath();
        ctx.arc(barX, curY, 6, 0, Math.PI*2);
        ctx.fill();
    }

    drawInfoText(ctx) {
        ctx.fillStyle = COLOR_TEXT;
        ctx.font = "12px monospace";
        ctx.textAlign = "left";
        ctx.fillText(`Azimuth:   ${this.state.azimuth}°`, 10, CANVAS_SIZE - 40);
        ctx.fillText(`Elevation: ${this.state.elevation}°`, 10, CANVAS_SIZE - 25);
        ctx.fillText(`Distance:  ${this.state.distance}`, 10, CANVAS_SIZE - 10);
        
        // Trigger status
        ctx.fillStyle = this.state.include_trigger ? "#4a4" : "#a44";
        ctx.fillRect(CANVAS_SIZE - 20, CANVAS_SIZE - 20, 10, 10);
    }
    
    // --- Interaction ---
    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        // Calculate scale factors in case the UI is zoomed
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        // Check Elevation Bar
        const barX = CANVAS_SIZE - 20;
        if (Math.abs(x - barX) < 20) {
            this.isDragging = true;
            this.dragMode = 'elevation';
            this.updateElevation(y);
            return;
        }
        
        // Check Trigger Box (Simple toggle zone)
        if (x > CANVAS_SIZE - 30 && y > CANVAS_SIZE - 30) {
             this.state.include_trigger = !this.state.include_trigger;
             this.updateNode();
             this.draw();
             return;
        }

        // Default: Azimuth/Distance
        this.isDragging = true;
        this.dragMode = 'azimuth';
        this.updatePos(x, y);
    }

    onMouseMove(e) {
        if (!this.isDragging) return;
        const rect = this.canvas.getBoundingClientRect();
        // Calculate scale factors
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        if (this.dragMode === 'elevation') {
            this.updateElevation(y);
        } else {
            this.updatePos(x, y);
        }
    }

    onMouseUp(e) {
        this.isDragging = false;
        this.dragMode = null;
    }

    // Logic updates
    updatePos(x, y) {
        // 1. Calculate Angle
        const dx = x - CENTER_X;
        const dy = y - CENTER_Y;
        let angleRad = Math.atan2(dy, dx);
        
        // Convert to Azimuth (0=Top, 90=Right)
        // atan2 returns angle from X axis (Right).
        // 0 -> 90 deg (Right)
        // -PI/2 -> 0 deg (Top)
        
        let deg = (angleRad * 180 / Math.PI) + 90;
        if (deg < 0) deg += 360;
        
        // Snap to 45 degrees
        this.state.azimuth = Math.round(deg / 45) * 45;
        if (this.state.azimuth >= 360) this.state.azimuth = 0;

        // 2. Calculate Distance (Radius)
        const dist = Math.sqrt(dx*dx + dy*dy);
        
        // Snap to rings
        const dists = [RADIUS_CLOSE, RADIUS_MEDIUM, RADIUS_WIDE];
        const closest = dists.reduce((prev, curr) => 
            Math.abs(curr - dist) < Math.abs(prev - dist) ? curr : prev
        );
        
        this.state.distance = DISTANCE_REVERSE_MAP[closest];
        
        this.updateNode();
        this.draw();
    }

    updateElevation(y) {
        const barH = 200;
        const barY = (CANVAS_SIZE - barH) / 2;
        
        // Inverse map from Y to degree
        // Y = barY + barH - (norm * barH)
        // norm = (barY + barH - Y) / barH
        let norm = (barY + barH - y) / barH;
        if (norm < 0) norm = 0;
        if (norm > 1) norm = 1;
        
        // Deg = norm * 90 - 30
        let deg = norm * 90 - 30;
        
        // Snap to steps [-30, 0, 30, 60]
        const closest = ELEVATION_STEPS.reduce((prev, curr) => 
            Math.abs(curr - deg) < Math.abs(prev - deg) ? curr : prev
        );
        
        this.state.elevation = closest;
        this.updateNode();
        this.draw();
    }

    updateNode() {
        // Serialize state to the hidden widget
        if (this.node.widgets && this.node.widgets[0]) {
            this.node.widgets[0].value = JSON.stringify(this.state);
        }
    }
}


// --- Extension Registration ---
app.registerExtension({
    name: "VNCCS.VisualCameraControl",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "VNCCS_VisualPositionControl") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // Hide the default text input (camera_data)
                // usually mapped to widgets[0] if defined in INPUT_TYPES as required string
                
                // Add Custom Widget
                const widget = new VNCCS_CameraWidget(this, "camera_camera", {}, app);
                
                // Add the canvas to the DOM of the node
                // ComfyUI nodes have `addDOMWidget`
                this.addDOMWidget("CameraControl", "canvas", widget.canvas, {
                    serialize: false, // We don't serialize the canvas itself
                    hideOnZoom: false
                });

                // Force initial update to sync invisible widget
                widget.updateNode();
                
                // Keep dimensions nice
                this.setSize([340, 380]);
            };
        }
    }
});
