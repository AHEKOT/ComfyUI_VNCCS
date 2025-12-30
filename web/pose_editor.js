import { app } from "../../scripts/app.js";
import { Pose3DEditor } from "./pose_editor_3d.js";
import { getBoneColor } from "./bone_colors.js";

const CANVAS_WIDTH = 512;
const CANVAS_HEIGHT = 1536;
const SAFE_MARGIN = 0.075;

const SAFE_ZONE = {
    left: CANVAS_WIDTH * SAFE_MARGIN,
    right: CANVAS_WIDTH * (1 - SAFE_MARGIN),
    top: CANVAS_HEIGHT * SAFE_MARGIN,
    bottom: CANVAS_HEIGHT * (1 - SAFE_MARGIN)
};

const DEFAULT_SKELETON = {
    // OpenPose BODY_25 ordering (without mid_hip)
    nose: [256, 200],          // 0
    neck: [256, 320],          // 1
    r_shoulder: [320, 320],    // 2
    r_elbow: [350, 520],       // 3
    r_wrist: [360, 720],       // 4
    l_shoulder: [192, 320],    // 5
    l_elbow: [162, 520],       // 6
    l_wrist: [152, 720],       // 7
    r_hip: [290, 720],         // 8
    r_knee: [295, 1020],       // 9
    r_ankle: [300, 1320],      // 10
    l_hip: [222, 720],         // 11
    l_knee: [217, 1020],       // 12
    l_ankle: [212, 1320],      // 13
    r_eye: [270, 185],         // 14
    l_eye: [242, 185],         // 15
    r_ear: [285, 195],         // 16
    l_ear: [227, 195]          // 17
};

const BONE_CONNECTIONS = [
    // Upper body
    ["nose", "neck"],
    ["neck", "r_shoulder"],
    ["r_shoulder", "r_elbow"],
    ["r_elbow", "r_wrist"],
    ["neck", "l_shoulder"],
    ["l_shoulder", "l_elbow"],
    ["l_elbow", "l_wrist"],
    ["neck", "r_hip"],
    ["neck", "l_hip"],
    
    // Right leg
    ["r_hip", "r_knee"],
    ["r_knee", "r_ankle"],
    
    // Left leg
    ["l_hip", "l_knee"],
    ["l_knee", "l_ankle"],
    
    // Face
    ["nose", "r_eye"],
    ["r_eye", "r_ear"],
    ["nose", "l_eye"],
    ["l_eye", "l_ear"]
];

const MIRROR_PAIRS = [
    ["l_shoulder", "r_shoulder"],
    ["l_elbow", "r_elbow"],
    ["l_wrist", "r_wrist"],
    ["l_hip", "r_hip"],
    ["l_knee", "r_knee"],
    ["l_ankle", "r_ankle"],
    ["l_eye", "r_eye"],
    ["l_ear", "r_ear"],
    ["l_bigtoe", "r_bigtoe"],
    ["l_smalltoe", "r_smalltoe"],
    ["l_heel", "r_heel"]
];

const JOINT_ALIASES = {};

function range(start, end) {
    const values = [];
    for (let i = start; i < end; i++) {
        values.push(i);
    }
    return values;
}

function loop(start, end) {
    const values = range(start, end);
    if (values.length) {
        values.push(values[0]);
    }
    return values;
}

function vector(from, to) {
    return [to[0] - from[0], to[1] - from[1]];
}

function length(vec) {
    return Math.hypot(vec[0], vec[1]);
}

function normalize(vec) {
    const len = length(vec);
    if (len < 1e-6) {
        return null;
    }
    return [vec[0] / len, vec[1] / len];
}

function perp(vec) {
    return [-vec[1], vec[0]];
}

function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1];
}

// Presets loaded dynamically from server
let PRESETS = [];

const PRESET_BASE_URL = new URL("../presets/poses/", import.meta.url);

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function cloneJoints(joints) {
    return JSON.parse(JSON.stringify(joints));
}

function buildPosePayload(poses) {
    return {
        canvas: { width: CANVAS_WIDTH, height: CANVAS_HEIGHT },
        poses: JSON.parse(JSON.stringify(poses))
    };
}

function computeBounds(joints) {
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    for (const [, [x, y]] of Object.entries(joints)) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
    }

    if (!Number.isFinite(minX)) {
        return { minX: 0, maxX: 0, minY: 0, maxY: 0, width: 0, height: 0 };
    }

    return {
        minX,
        maxX,
        minY,
        maxY,
        width: maxX - minX,
        height: maxY - minY
    };
}

function parsePoseData(raw) {
    // Default: 12 poses
    const defaultPoses = Array(12).fill(null).map(() => cloneJoints(DEFAULT_SKELETON));

    if (!raw) {
        return {
            canvas: { width: CANVAS_WIDTH, height: CANVAS_HEIGHT },
            poses: defaultPoses
        };
    }

    let data = raw;
    if (typeof raw === "string") {
        try {
            data = JSON.parse(raw);
        } catch (error) {
            console.warn("[VNCCS] Failed to parse pose_data, falling back to default", error);
            return {
                canvas: { width: CANVAS_WIDTH, height: CANVAS_HEIGHT },
                poses: defaultPoses
            };
        }
    }

    // Handle DWPose format (array with people objects)
    if (Array.isArray(data)) {
        const poses = [];
        const frameData = data[0]; // DWPose exports have frame data at [0]
        const canvasWidth = frameData?.canvas_width || CANVAS_WIDTH * 3;
        const canvasHeight = frameData?.canvas_height || CANVAS_HEIGHT * 4;
        
        if (frameData?.people) {
            // Sort people by their position (Y first, then X) to get correct grid order
            const peopleWithPos = frameData.people.map((person, idx) => {
                const kp = person.pose_keypoints_2d || [];
                // Use nose position (first keypoint) for sorting
                const x = kp[0] || 0;
                const y = kp[1] || 0;
                return { person, x, y, originalIndex: idx };
            });
            
            // Sort by Y then X to get row-major order
            peopleWithPos.sort((a, b) => {
                const yDiff = a.y - b.y;
                if (Math.abs(yDiff) > 100) return yDiff; // Different rows
                return a.x - b.x; // Same row, sort by X
            });
            
            // Now process in sorted order
            for (let i = 0; i < 12; i++) {
                if (i < peopleWithPos.length) {
                    poses.push(convertDWPoseToJoints(peopleWithPos[i].person, canvasWidth, canvasHeight));
                } else {
                    poses.push(cloneJoints(DEFAULT_SKELETON));
                }
            }
        } else {
            // No people, fill with defaults
            for (let i = 0; i < 12; i++) {
                poses.push(cloneJoints(DEFAULT_SKELETON));
            }
        }
        
        return {
            canvas: { width: CANVAS_WIDTH, height: CANVAS_HEIGHT },
            poses
        };
    }

    // Handle legacy format (single 'joints' object)
    if (data.joints && !data.poses) {
        const singlePose = normalizeJoints(data.joints);
        // Place in first slot, fill rest with default
        const poses = [singlePose, ...defaultPoses.slice(1)];
        return {
            canvas: { width: CANVAS_WIDTH, height: CANVAS_HEIGHT },
            poses
        };
    }

    // Handle new format
    const poses = [];
    const sourcePoses = Array.isArray(data.poses) ? data.poses : [];
    
    for (let i = 0; i < 12; i++) {
        if (i < sourcePoses.length && sourcePoses[i]) {
            // Support both {joints: {...}} and direct {...} formats if needed, 
            // but standard is {joints: ...} inside the array
            const poseData = sourcePoses[i].joints || sourcePoses[i]; 
            poses.push(normalizeJoints(poseData));
        } else {
            poses.push(cloneJoints(DEFAULT_SKELETON));
        }
    }

    return {
        canvas: {
            width: Number(data?.canvas?.width) || CANVAS_WIDTH,
            height: Number(data?.canvas?.height) || CANVAS_HEIGHT
        },
        poses
    };
}

function convertDWPoseToJoints(person, canvasWidth, canvasHeight) {
    // DWPose uses flat array: [x0, y0, conf0, x1, y1, conf1, ...]
    // OpenPose BODY_25 order (18 joints):
    // 0: nose, 1: neck, 2: r_shoulder, 3: r_elbow, 4: r_wrist,
    // 5: l_shoulder, 6: l_elbow, 7: l_wrist, 8: r_hip, 9: r_knee,
    // 10: r_ankle, 11: l_hip, 12: l_knee, 13: l_ankle, 14: r_eye,
    // 15: l_eye, 16: r_ear, 17: l_ear
    
    const keypoints = person.pose_keypoints_2d || [];
    const joints = {};
    
    const jointNames = [
        "nose", "neck", "r_shoulder", "r_elbow", "r_wrist",
        "l_shoulder", "l_elbow", "l_wrist", "r_hip", "r_knee",
        "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye",
        "l_eye", "r_ear", "l_ear"
    ];
    
    // Grid layout: 6 columns × 2 rows
    const gridCols = 6;
    const gridRows = 2;
    const cellWidth = canvasWidth / gridCols;   // 6144/6 = 1024
    const cellHeight = canvasHeight / gridRows; // 6144/2 = 3072
    
    // Determine which cell this person is in based on first keypoint (nose)
    const noseX = keypoints[0] || 0;
    const noseY = keypoints[1] || 0;
    const gridCol = Math.floor(noseX / cellWidth);
    const gridRow = Math.floor(noseY / cellHeight);
    
    // Calculate offset for this cell
    const offsetX = gridCol * cellWidth;
    const offsetY = gridRow * cellHeight;
    
    // Scale factors: cellWidth→CANVAS_WIDTH, cellHeight→CANVAS_HEIGHT
    const scaleX = CANVAS_WIDTH / cellWidth;   // 512/1024 = 0.5
    const scaleY = CANVAS_HEIGHT / cellHeight; // 1536/3072 = 0.5
    
    // Convert keypoints
    for (let i = 0; i < 18 && i * 3 < keypoints.length; i++) {
        const x = keypoints[i * 3];
        const y = keypoints[i * 3 + 1];
        const conf = keypoints[i * 3 + 2];
        
        const jointName = jointNames[i];
        
        // Use confidence to determine if keypoint is valid
        if (conf > 0.1) {
            // Convert to local cell coordinates and scale
            const localX = (x - offsetX) * scaleX;
            const localY = (y - offsetY) * scaleY;
            
            joints[jointName] = [
                clamp(localX, -CANVAS_WIDTH, CANVAS_WIDTH * 2),
                clamp(localY, -CANVAS_HEIGHT, CANVAS_HEIGHT * 2)
            ];
        } else {
            // Use default position for low confidence keypoints
            joints[jointName] = [...DEFAULT_SKELETON[jointName]];
        }
    }
    
    // Fill in any missing joints with defaults
    for (const [name, defaults] of Object.entries(DEFAULT_SKELETON)) {
        if (!joints[name]) {
            joints[name] = [...defaults];
        }
    }
    
    return joints;
}

function normalizeJoints(source) {
    const joints = {};
    const normalizedSource = {};

    for (const [rawName, value] of Object.entries(source)) {
        const mapped = JOINT_ALIASES[rawName] ?? rawName;
        if (!(mapped in normalizedSource)) {
            normalizedSource[mapped] = value;
        }
    }

    for (const [name, defaults] of Object.entries(DEFAULT_SKELETON)) {
        const value = normalizedSource[name];
        if (Array.isArray(value) && value.length >= 2) {
            const x = Number(value[0]);
            const y = Number(value[1]);
            joints[name] = [
                clamp(Number.isFinite(x) ? x : defaults[0], 0, CANVAS_WIDTH),
                clamp(Number.isFinite(y) ? y : defaults[1], 0, CANVAS_HEIGHT)
            ];
        } else {
            joints[name] = [...defaults];
        }
    }
    return joints;
}

function ensurePoseEditorStyles() {
    if (document.getElementById("vnccs-pose-editor-style")) {
        return;
    }

    const style = document.createElement("style");
    style.id = "vnccs-pose-editor-style";
    style.textContent = `
        body.vnccs-pose-editor-open {
            overflow: hidden;
        }
        .vnccs-pose-editor-overlay {
            position: fixed;
            inset: 0;
            background: rgba(4, 8, 16, 0.75);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 32px;
            box-sizing: border-box;
            opacity: 0;
            pointer-events: none;
            transition: opacity 150ms ease;
            z-index: 9000;
        }
        .vnccs-pose-editor-overlay.visible {
            opacity: 1;
            pointer-events: auto;
        }
        .vnccs-pose-editor-panel {
            width: min(1120px, 96vw);
            max-height: 92vh;
            display: flex;
            flex-direction: column;
            background: #0f131d;
            border-radius: 16px;
            box-shadow: 0 32px 90px rgba(0, 0, 0, 0.55);
            border: 1px solid rgba(121, 150, 255, 0.12);
            color: #d7def3;
            overflow: hidden;
        }
        .vnccs-pose-editor-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
            padding: 20px 28px 18px;
            border-bottom: 1px solid rgba(121, 150, 255, 0.15);
        }
        .vnccs-pose-editor-title {
            font-size: 20px;
            font-weight: 600;
            letter-spacing: 0.4px;
            margin-bottom: 4px;
        }
        .vnccs-pose-editor-subtitle {
            font-size: 13px;
            color: #92a3d6;
        }
        .vnccs-close-btn {
            background: rgba(147, 163, 210, 0.14);
            border: 1px solid rgba(147, 163, 210, 0.3);
            color: #d7def3;
            padding: 6px 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: background 120ms ease, color 120ms ease;
        }
        .vnccs-close-btn:hover {
            background: rgba(147, 163, 210, 0.24);
        }
        .vnccs-pose-editor-body {
            display: flex;
            gap: 24px;
            padding: 22px 28px 26px;
            overflow: auto;
            flex: 1;
        }
        .vnccs-pose-editor-canvas-column {
            flex: 1 1 60%;
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-width: 0;
        }
        .vnccs-pose-editor-canvas-wrapper {
            flex: 1;
            background: linear-gradient(180deg, #101626 0%, #0b101b 100%);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(121, 150, 255, 0.12);
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .vnccs-pose-editor-canvas-wrapper canvas {
            display: block;
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
            image-rendering: crisp-edges;
            cursor: crosshair;
            border-radius: 10px;
            box-shadow: 0 18px 40px rgba(10, 17, 30, 0.45);
            transition: transform 0.1s ease-out;
        }
        .vnccs-pose-editor-sidebar {
            flex: 0 0 280px;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }
        .vnccs-panel {
            border: 1px solid rgba(121, 150, 255, 0.12);
            border-radius: 12px;
            background: rgba(14, 18, 30, 0.8);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .vnccs-panel h3 {
            margin: 0;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #a4b6ec;
        }
        .vnccs-panel p {
            margin: 0;
            font-size: 12px;
            color: #8595c4;
            line-height: 1.6;
        }
        .vnccs-button-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .vnccs-btn {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid transparent;
            border-radius: 8px;
            font-size: 12px;
            letter-spacing: 0.3px;
            background: rgba(79, 123, 255, 0.2);
            color: #d7def3;
            cursor: pointer;
            transition: background 120ms ease, border 120ms ease, color 120ms ease;
        }
        .vnccs-btn:hover {
            background: rgba(79, 123, 255, 0.32);
            border-color: rgba(79, 123, 255, 0.45);
        }
        .vnccs-btn.secondary {
            background: rgba(146, 158, 200, 0.14);
            color: #d7def3;
        }
        .vnccs-btn.secondary:hover {
            background: rgba(146, 158, 200, 0.2);
            border-color: rgba(146, 158, 200, 0.4);
        }
        .vnccs-btn.ghost {
            background: transparent;
            border: 1px solid rgba(146, 158, 200, 0.2);
            color: #9eb0dd;
        }
        .vnccs-btn.ghost:hover {
            border-color: rgba(146, 158, 200, 0.45);
            color: #d7def3;
        }
        .vnccs-status {
            font-size: 12px;
            color: #8794ba;
            min-height: 18px;
        }
        .vnccs-toggle {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 12px;
            color: #d7def3;
        }
        .vnccs-toggle input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: #6283ff;
        }
        .vnccs-info-line {
            font-size: 12px;
            color: #96a6d5;
            line-height: 1.5;
        }
        .vnccs-pose-editor-statusbar {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #92a3d6;
            padding: 0 4px;
        }
        .vnccs-select {
            padding: 8px 10px;
            border-radius: 8px;
            border: 1px solid rgba(121, 150, 255, 0.2);
            background: rgba(17, 22, 35, 0.9);
            color: #d7def3;
            font-size: 12px;
        }
        .vnccs-metrics {
            font-size: 12px;
            color: #92a3d6;
            line-height: 1.6;
        }
        .vnccs-metrics strong {
            color: #c5d2ff;
        }
        .vnccs-warning {
            color: #ff9f7d;
        }
        .vnccs-success {
            color: #7dffa9;
        }
        .vnccs-pose-editor-3d-column {
            flex: 1 1 40%;
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-width: 0;
        }
        .vnccs-3d-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }
        .vnccs-3d-title {
            font-size: 16px;
            font-weight: 600;
            letter-spacing: 0.45px;
        }
        .vnccs-3d-subtitle {
            font-size: 12px;
            color: #8795c4;
        }
        .vnccs-3d-toolbar {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .vnccs-3d-btn {
            padding: 6px 12px;
            border-radius: 8px;
            border: 1px solid rgba(121, 150, 255, 0.18);
            background: rgba(49, 72, 110, 0.22);
            color: #d7def3;
            font-size: 12px;
            cursor: pointer;
            transition: background 120ms ease, border 120ms ease, color 120ms ease;
        }
        .vnccs-3d-btn:hover {
            background: rgba(79, 123, 255, 0.28);
            border-color: rgba(79, 123, 255, 0.52);
        }
        .vnccs-3d-btn.active {
            background: rgba(79, 123, 255, 0.48);
            border-color: rgba(79, 123, 255, 0.75);
            color: #f4f6ff;
        }
        .vnccs-3d-canvas-wrapper {
            flex: 1;
            min-height: 420px;
            border: 1px solid rgba(121, 150, 255, 0.12);
            border-radius: 12px;
            background: radial-gradient(circle at 50% 30%, rgba(36, 49, 83, 0.58), rgba(12, 17, 28, 0.92));
            overflow: hidden;
            position: relative;
        }
        .vnccs-3d-canvas-wrapper canvas {
            display: block;
            width: 100%;
            height: 100%;
        }
        .vnccs-3d-footer {
            font-size: 12px;
            color: #92a3d6;
            display: flex;
            justify-content: space-between;
            gap: 12px;
        }
        @media (max-width: 1080px) {
            .vnccs-pose-editor-body {
                flex-direction: column;
            }
            .vnccs-pose-editor-3d-column {
                order: 2;
            }
            .vnccs-pose-editor-sidebar {
                flex-direction: row;
                flex-wrap: wrap;
            }
            .vnccs-panel {
                flex: 1 1 280px;
            }
        }
    `;

    document.head.appendChild(style);
}

class NodePoseIntegration {
    constructor(node, poseWidget) {
        this.node = node;
        this.widget = poseWidget;
        this.suppressWidgetCallback = false;
        this.pose = parsePoseData(poseWidget?.value ?? null);
        this.previewWidget = null;

        if (this.widget) {
            const originalCallback = this.widget.callback;
            this.widget.callback = (value) => {
                if (!this.suppressWidgetCallback) {
                    this.pose = parsePoseData(value);
                    this.requestPreviewRefresh();
                }
                if (originalCallback) {
                    originalCallback(value);
                }
            };
        }
    }

    requestPreviewRefresh() {
        this.node.setDirtyCanvas(true, true);
    }

    get poses() {
        return this.pose.poses;
    }

    updatePoseFromEditor(poses, immediate = false) {
        this.pose = buildPosePayload(poses);
        if (this.widget) {
            const serialized = JSON.stringify(this.pose, null, 2);
            this.widget.value = serialized;
            if (this.widget.callback) {
                this.suppressWidgetCallback = true;
                try {
                    this.widget.callback(serialized);
                } finally {
                    this.suppressWidgetCallback = false;
                }
            }
        }
        this.requestPreviewRefresh();
    }

    installPreview() {
        if (this.previewWidget) {
            return;
        }

        const integration = this;
        this.previewWidget = this.node.addCustomWidget({
            name: "pose_preview",
            type: "vnccs_pose_preview",
            draw(ctx, node, width, y) {
                return PosePreviewRenderer.draw(ctx, integration.pose.poses, width, y);
            },
            computeSize(width) {
                return [width, PosePreviewRenderer.HEIGHT];
            },
            serializeValue() {
                return undefined;
            }
        });
    }
}

class PosePreviewRenderer {
    static HEIGHT = 240;

    static draw(ctx, poses, width, y) {
        const height = PosePreviewRenderer.HEIGHT;
        ctx.save();
        ctx.translate(0, y);

        ctx.fillStyle = "#111822";
        ctx.fillRect(4, 4, width - 8, height - 8);
        ctx.strokeStyle = "rgba(104, 136, 220, 0.35)";
        ctx.lineWidth = 1;
        ctx.strokeRect(4.5, 4.5, width - 9, height - 9);

        // Grid layout: 6 cols, 2 rows
        const cols = 6;
        const rows = 2;
        const padding = 8;
        
        const availableWidth = width - 16;
        const availableHeight = height - 16;
        
        // Calculate cell size to fit
        const cellAspect = CANVAS_WIDTH / CANVAS_HEIGHT; // 1/3
        
        // Try to fit by width
        let cellWidth = availableWidth / cols;
        let cellHeight = cellWidth / cellAspect;
        
        if (cellHeight * rows > availableHeight) {
            // Fit by height
            cellHeight = availableHeight / rows;
            cellWidth = cellHeight * cellAspect;
        }
        
        const startX = 8 + (availableWidth - cellWidth * cols) / 2;
        const startY = 8 + (availableHeight - cellHeight * rows) / 2;

        for (let i = 0; i < 12; i++) {
            const joints = poses[i];
            const col = i % cols;
            const row = Math.floor(i / cols);
            
            const cx = startX + col * cellWidth;
            const cy = startY + row * cellHeight;
            
            ctx.save();
            ctx.translate(cx, cy);
            ctx.scale(cellWidth / CANVAS_WIDTH, cellHeight / CANVAS_HEIGHT);
            
            // Draw connections
            ctx.lineWidth = 8; // Scaled down
            ctx.lineCap = "round";
            for (let i = 0; i < BONE_CONNECTIONS.length; i++) {
                const [start, end] = BONE_CONNECTIONS[i];
                const a = joints[start];
                const b = joints[end];
                if (!a || !b) continue;
                
                ctx.strokeStyle = getBoneColor(start, end, i);
                
                ctx.beginPath();
                ctx.moveTo(a[0], a[1]);
                ctx.lineTo(b[0], b[1]);
                ctx.stroke();
            }
            
            // Draw joints (simplified)
            ctx.fillStyle = "#f9ac5d";
            for (const [name, [jx, jy]] of Object.entries(joints)) {
                ctx.beginPath();
                ctx.arc(jx, jy, 12, 0, Math.PI * 2);
                ctx.fill();
            }
            
            ctx.restore();
        }

        ctx.restore();
        return height;
    }
}

class PoseEditorDialog {
    constructor() {
        ensurePoseEditorStyles();

        this.overlay = document.createElement("div");
        this.overlay.className = "vnccs-pose-editor-overlay";
        this.overlay.addEventListener("click", (event) => {
            if (event.target === this.overlay) {
                this.close();
            }
        });

        this.panel = document.createElement("div");
        this.panel.className = "vnccs-pose-editor-panel";
        this.overlay.appendChild(this.panel);

        this.buildLayout();

        this.canvas = document.createElement("canvas");
        this.canvas.width = CANVAS_WIDTH;
        this.canvas.height = CANVAS_HEIGHT;
        this.canvas.tabIndex = 0;
        this.canvasWrapper.appendChild(this.canvas);
        this.ctx = this.canvas.getContext("2d");

        this.canvas.addEventListener("pointerdown", (event) => this.onPointerDown(event));
        this.canvas.addEventListener("pointermove", (event) => this.onPointerMove(event));
        this.canvas.addEventListener("pointerup", (event) => this.onPointerUp(event));
        this.canvas.addEventListener("pointerleave", (event) => this.onPointerUp(event));
        this.canvas.addEventListener("keydown", (event) => this.onKeyDown(event));
        this.canvas.addEventListener("wheel", (event) => this.onWheel(event), { passive: false });

        this.onEscape = (event) => {
            if (event.key === "Escape") {
                event.preventDefault();
                this.close();
            }
        };

        this.integration = null;
        this.poses = Array(12).fill(null).map(() => cloneJoints(DEFAULT_SKELETON));
        this.currentPoseIndex = 0;
        
        this.hoveredJoint = null;
        this.selectedJoint = null;
        this.dragging = false;
        this.dragPointerId = null;
        this.syncTimer = null;

        this.pose3d = null;
        this.pose3dPromise = null;
        this.pending3DSync = null;
        this.active3DMode = "translate";

        this.showGrid = true;
        this.showLabels = true;
        this.showSafeZone = true;
        this.viewMode = "single"; // 'single' or 'grid'
        this.zoom = 1.0;
        this.minZoom = 0.3;
        this.maxZoom = 3.0;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;
        this.panStartX = 0;
        this.panStartY = 0;
        
        // Load presets list and default poses
        this.loadPresetsList();
        this.loadDefaultPoses();
    }
    
    async loadPresetsList() {
        try {
            const response = await fetch("/vnccs/pose_presets");
            if (response.ok) {
                PRESETS.length = 0;
                const presets = await response.json();
                PRESETS.push(...presets);
                
                // Update preset selector if it exists
                if (this.presetSelect) {
                    this.updatePresetSelect();
                }
            }
        } catch (error) {
            console.warn("[VNCCS] Failed to load presets list", error);
        }
    }
    
    updatePresetSelect() {
        if (!this.presetSelect) return;
        
        // Clear existing options except placeholder
        while (this.presetSelect.options.length > 1) {
            this.presetSelect.remove(1);
        }
        
        // Add presets
        for (const preset of PRESETS) {
            const option = document.createElement("option");
            option.value = preset.id;
            option.textContent = preset.label;
            this.presetSelect.appendChild(option);
        }
    }

    async loadDefaultPoses() {
        try {
            const response = await fetch(new URL("../presets/poses/vnccs_poseset.json", import.meta.url));
            if (response.ok) {
                const payload = await response.json();
                const parsed = parsePoseData(payload);
                this.poses = parsed.poses;
            }
        } catch (error) {
            console.warn("[VNCCS] Failed to load default poses, using skeleton defaults", error);
        }
    }

    get joints() {
        return this.poses[this.currentPoseIndex];
    }

    set joints(value) {
        this.poses[this.currentPoseIndex] = value;
    }

    buildLayout() {
        const header = document.createElement("div");
        header.className = "vnccs-pose-editor-header";

        const titleWrap = document.createElement("div");
        const title = document.createElement("div");
        title.className = "vnccs-pose-editor-title";
        title.textContent = "VNCCS Pose Editor";
        const subtitle = document.createElement("div");
        subtitle.className = "vnccs-pose-editor-subtitle";
        subtitle.textContent = "12-Pose Grid Editor · 512×1536 per pose";
        titleWrap.appendChild(title);
        titleWrap.appendChild(subtitle);

        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.className = "vnccs-close-btn";
        closeBtn.textContent = "Close";
        closeBtn.addEventListener("click", () => this.close());

        header.appendChild(titleWrap);
        header.appendChild(closeBtn);
        this.panel.appendChild(header);

        // Pose Selector Toolbar
        const toolbar = document.createElement("div");
        toolbar.style.padding = "0 28px 12px";
        toolbar.style.display = "flex";
        toolbar.style.gap = "8px";
        toolbar.style.alignItems = "center";
        toolbar.style.borderBottom = "1px solid rgba(121, 150, 255, 0.15)";
        
        const label = document.createElement("span");
        label.textContent = "Active Pose:";
        label.style.fontSize = "12px";
        label.style.color = "#92a3d6";
        toolbar.appendChild(label);

        this.poseButtons = [];
        for (let i = 0; i < 12; i++) {
            const btn = document.createElement("button");
            btn.textContent = `${i + 1}`;
            btn.className = "vnccs-btn secondary";
            btn.style.width = "32px";
            btn.style.padding = "6px 0";
            btn.style.textAlign = "center";
            btn.addEventListener("click", () => this.selectPose(i));
            toolbar.appendChild(btn);
            this.poseButtons.push(btn);
        }
        
        const viewToggle = this.createButton("Grid View", () => this.toggleViewMode(), "ghost");
        viewToggle.style.marginLeft = "auto";
        this.viewToggleBtn = viewToggle;
        toolbar.appendChild(viewToggle);

        this.panel.appendChild(toolbar);

        const body = document.createElement("div");
        body.className = "vnccs-pose-editor-body";
        this.panel.appendChild(body);

        const canvasColumn = document.createElement("div");
        canvasColumn.className = "vnccs-pose-editor-canvas-column";
        this.canvasWrapper = document.createElement("div");
        this.canvasWrapper.className = "vnccs-pose-editor-canvas-wrapper";
        canvasColumn.appendChild(this.canvasWrapper);

        this.statusBar = document.createElement("div");
        this.statusBar.className = "vnccs-pose-editor-statusbar";
        this.statusMessage = document.createElement("span");
        this.jointInfo = document.createElement("span");
        this.statusBar.appendChild(this.statusMessage);
        this.statusBar.appendChild(this.jointInfo);
        canvasColumn.appendChild(this.statusBar);

        body.appendChild(canvasColumn);

    const threeColumn = this.createThreeColumn();
    body.appendChild(threeColumn);

        const sidebar = document.createElement("div");
        sidebar.className = "vnccs-pose-editor-sidebar";
        sidebar.appendChild(this.createPoseToolsPanel());
        sidebar.appendChild(this.createPresetPanel());
        body.appendChild(sidebar);
    }

    selectPose(index) {
        if (index < 0 || index >= 12) return;
        this.currentPoseIndex = index;
        this.updatePoseButtons();
        
        // Update 3D editor
        if (this.pose3d) {
            this.pose3d.setPose(this.joints, true);
        }
        
        this.render();
        this.updateMetrics();
        this.setStatus(`Switched to Pose ${index + 1}`);
    }

    updatePoseButtons() {
        this.poseButtons.forEach((btn, i) => {
            if (i === this.currentPoseIndex) {
                btn.classList.remove("secondary");
                btn.style.background = "rgba(79, 123, 255, 0.4)";
                btn.style.borderColor = "rgba(79, 123, 255, 0.6)";
            } else {
                btn.classList.add("secondary");
                btn.style.background = "";
                btn.style.borderColor = "";
            }
        });
    }

    toggleViewMode() {
        this.viewMode = this.viewMode === "single" ? "grid" : "single";
        this.viewToggleBtn.textContent = this.viewMode === "single" ? "Grid View" : "Single View";
        this.viewToggleBtn.classList.toggle("active", this.viewMode === "grid");
        
        // Reset zoom and pan when switching modes
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.canvas.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
        
        if (this.viewMode === "grid") {
            // Resize canvas for grid
            // We want to show 6x2 grid. 
            // Aspect ratio of single is 1:3.
            // Grid is 6 wide, 2 high.
            // Total aspect: (6*1)/(2*3) = 6/6 = 1:1.
            // So grid is square.
            this.canvas.width = 3072;
            this.canvas.height = 3072; // Wait, 6*512=3072, 2*1536=3072. Yes.
            this.canvas.style.aspectRatio = "1 / 1";
        } else {
            this.canvas.width = CANVAS_WIDTH;
            this.canvas.height = CANVAS_HEIGHT;
            this.canvas.style.aspectRatio = "512 / 1536";
        }
        this.render();
    }

    createPoseToolsPanel() {
        const panel = document.createElement("div");
        panel.className = "vnccs-panel";

        const heading = document.createElement("h3");
        heading.textContent = "Pose Tools";
        const description = document.createElement("p");
        description.textContent = "Refine joint layout, mirror the skeleton, or fit to the safe zone.";

        const row = document.createElement("div");
        row.className = "vnccs-button-row";
        row.appendChild(this.createButton("Reset Pose", () => this.resetPose(), "secondary"));
        row.appendChild(this.createButton("Mirror", () => this.mirrorPose()));
        row.appendChild(this.createButton("Fit Safe Zone", () => this.fitSafeZone()));
        row.appendChild(this.createButton("Center", () => this.centerPose(), "ghost"));

        this.metricsEl = document.createElement("div");
        this.metricsEl.className = "vnccs-metrics";

        panel.appendChild(heading);
        panel.appendChild(description);
        panel.appendChild(row);
        panel.appendChild(this.metricsEl);
        return panel;
    }

    createPresetPanel() {
        const panel = document.createElement("div");
        panel.className = "vnccs-panel";

        const heading = document.createElement("h3");
        heading.textContent = "Presets & IO";
        const description = document.createElement("p");
        description.textContent = "Load curated poses or import/export JSON files.";

        this.presetSelect = document.createElement("select");
        this.presetSelect.className = "vnccs-select";

        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Choose preset";
        this.presetSelect.appendChild(placeholder);

        // Presets will be populated by updatePresetSelect() after loading
        this.updatePresetSelect();

        this.presetSelect.addEventListener("change", (event) => {
            const presetId = event.target.value;
            if (presetId) {
                this.loadPreset(presetId);
            }
        });

        const row = document.createElement("div");
        row.className = "vnccs-button-row";
        row.appendChild(this.createButton("Import", () => this.importPose(), "secondary"));
        row.appendChild(this.createButton("Copy JSON", () => this.copyPoseJson(), "ghost"));
        
        const row2 = document.createElement("div");
        row2.className = "vnccs-button-row";
        row2.appendChild(this.createButton("Save Pose", () => this.downloadPose()));
        row2.appendChild(this.createButton("Save Set", () => this.downloadSet()));

        this.fileInput = document.createElement("input");
        this.fileInput.type = "file";
        this.fileInput.accept = "application/json";
        this.fileInput.style.display = "none";
        this.fileInput.addEventListener("change", (event) => this.onFileSelected(event));

        this.statusEl = document.createElement("div");
        this.statusEl.className = "vnccs-status";

        panel.appendChild(heading);
        panel.appendChild(description);
        panel.appendChild(this.presetSelect);
        panel.appendChild(row);
        panel.appendChild(row2);
        panel.appendChild(this.statusEl);
        panel.appendChild(this.fileInput);
        return panel;
    }

    createOptionsPanel() {
        // Removed - options enabled by default
        return null;
    }

    createThreeColumn() {
        const column = document.createElement("div");
        column.className = "vnccs-pose-editor-3d-column";

        const header = document.createElement("div");
        header.className = "vnccs-3d-header";

        const titleBox = document.createElement("div");
        const title = document.createElement("div");
        title.className = "vnccs-3d-title";
        title.textContent = "3D Pose Explorer";
        const subtitle = document.createElement("div");
        subtitle.className = "vnccs-3d-subtitle";
        subtitle.textContent = "Orbit, move, and rotate joints in space.";
        titleBox.appendChild(title);
        titleBox.appendChild(subtitle);

        this.threeToolbar = document.createElement("div");
        this.threeToolbar.className = "vnccs-3d-toolbar";

        this.threeBtnMove = this.create3DButton("Move", () => this.setTransformMode3D("translate"));
        this.threeBtnRotate = this.create3DButton("Rotate", () => this.setTransformMode3D("rotate"));
        const frameBtn = this.create3DButton("Frame", () => {
            this.ensurePose3D().then((editor) => editor?.frameAll());
        });
        const resetBtn = this.create3DButton("Reset View", () => {
            this.ensurePose3D().then((editor) => editor?.resetView());
        });

        this.threeToolbar.appendChild(this.threeBtnMove);
        this.threeToolbar.appendChild(this.threeBtnRotate);
        this.threeToolbar.appendChild(frameBtn);
        this.threeToolbar.appendChild(resetBtn);

        header.appendChild(titleBox);
        header.appendChild(this.threeToolbar);
        column.appendChild(header);

        this.threeViewport = document.createElement("div");
        this.threeViewport.className = "vnccs-3d-canvas-wrapper";
        column.appendChild(this.threeViewport);

        const footer = document.createElement("div");
        footer.className = "vnccs-3d-footer";
        this.threeStatusLeft = document.createElement("span");
        this.threeStatusLeft.textContent = "Loading 3D editor…";
        this.threeStatusRight = document.createElement("span");
        this.threeStatusRight.textContent = "Orbit with right mouse • Scroll to zoom";
        footer.appendChild(this.threeStatusLeft);
        footer.appendChild(this.threeStatusRight);
        column.appendChild(footer);

        this.update3DToolbarMode("translate");

        return column;
    }

    create3DButton(label, handler) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "vnccs-3d-btn";
        button.textContent = label;
        button.addEventListener("click", handler);
        return button;
    }

    update3DToolbarMode(mode) {
        this.active3DMode = mode;
        if (this.threeBtnMove) {
            this.threeBtnMove.classList.toggle("active", mode === "translate");
        }
        if (this.threeBtnRotate) {
            this.threeBtnRotate.classList.toggle("active", mode === "rotate");
        }
    }

    createButton(label, handler, variant = "primary") {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `vnccs-btn ${variant}`;
        button.textContent = label;
        button.addEventListener("click", handler);
        return button;
    }

    createToggle(label, checked, onChange) {
        const wrapper = document.createElement("label");
        wrapper.className = "vnccs-toggle";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = checked;
        input.addEventListener("change", () => onChange(input.checked));
        const text = document.createElement("span");
        text.textContent = label;
        wrapper.appendChild(input);
        wrapper.appendChild(text);
        return wrapper;
    }

    open(integration) {
        this.integration = integration;
        // Deep copy all poses
        this.poses = JSON.parse(JSON.stringify(integration.pose.poses));
        this.currentPoseIndex = 0;
        this.updatePoseButtons();
        
        this.hoveredJoint = null;
        this.selectedJoint = null;
        this.dragging = false;
        this.dragPointerId = null;
        this.presetSelect.value = "";
        this.setStatus("Pose editor ready", "success");

        if (!document.body.contains(this.overlay)) {
            document.body.appendChild(this.overlay);
        }

        requestAnimationFrame(() => {
            this.overlay.classList.add("visible");
        });
        document.body.classList.add("vnccs-pose-editor-open");
        window.addEventListener("keydown", this.onEscape);
        this.handleJointsMutated(true, { skip3D: true });

        if (this.threeStatusLeft) {
            this.threeStatusLeft.textContent = "Drag joints in 3D space.";
        }
        if (this.threeStatusRight) {
            this.threeStatusRight.textContent = this.active3DMode === "rotate"
                ? "Rotate mode · Use colored arcs to twist limbs"
                : "Move mode · Drag gizmo axes to reposition";
        }
        this.update3DToolbarMode(this.active3DMode);
        this.ensurePose3D().then((editor) => {
            if (!editor) {
                return;
            }
            editor.setActive(true);
            editor.setTransformMode(this.active3DMode);
            editor.setPose(this.joints, true);
            if (this.selectedJoint) {
                editor.setSelection(this.selectedJoint, "2d");
            } else {
                editor.setSelection(null, "2d");
            }
            editor.handleResize();
        });
        this.queuePose3DSync();
    }

    isOpenFor(integration) {
        return this.integration === integration;
    }

    close() {
        if (!this.overlay.parentNode) {
            return;
        }

        this.overlay.classList.remove("visible");
        window.removeEventListener("keydown", this.onEscape);
        document.body.classList.remove("vnccs-pose-editor-open");
        setTimeout(() => {
            if (this.overlay.parentNode) {
                this.overlay.parentNode.removeChild(this.overlay);
            }
        }, 160);

        if (this.syncTimer) {
            clearTimeout(this.syncTimer);
            this.syncTimer = null;
        }

        if (this.pending3DSync) {
            cancelAnimationFrame(this.pending3DSync);
            this.pending3DSync = null;
        }
        if (this.pose3d) {
            this.pose3d.setActive(false);
        }

        this.integration = null;
    }

    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#111828";
        ctx.fillRect(0, 0, width, height);

        if (this.viewMode === "grid") {
            this.renderGrid(ctx);
        } else {
            this.renderSingle(ctx, this.joints, 0, 0);
        }
    }

    renderGrid(ctx) {
        const cols = 6;
        const w = CANVAS_WIDTH;
        const h = CANVAS_HEIGHT;
        
        // Draw grid lines
        ctx.strokeStyle = "rgba(80, 115, 180, 0.32)";
        ctx.lineWidth = 2;
        for (let i = 1; i < cols; i++) {
            ctx.beginPath();
            ctx.moveTo(i * w, 0);
            ctx.lineTo(i * w, this.canvas.height);
            ctx.stroke();
        }
        ctx.beginPath();
        ctx.moveTo(0, h);
        ctx.lineTo(this.canvas.width, h);
        ctx.stroke();

        for (let i = 0; i < 12; i++) {
            const col = i % cols;
            const row = Math.floor(i / cols);
            const x = col * w;
            const y = row * h;
            
            // Highlight active pose background
            if (i === this.currentPoseIndex) {
                ctx.fillStyle = "rgba(79, 123, 255, 0.1)";
                ctx.fillRect(x, y, w, h);
                ctx.strokeStyle = "rgba(79, 123, 255, 0.5)";
                ctx.strokeRect(x, y, w, h);
            }
            
            this.renderSingle(ctx, this.poses[i], x, y, i === this.currentPoseIndex);
            
            // Draw pose number
            ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
            ctx.font = "40px sans-serif";
            ctx.fillText(`${i + 1}`, x + 20, y + 50);
        }
    }

    renderSingle(ctx, joints, offsetX, offsetY, isActive = true) {
        ctx.save();
        ctx.translate(offsetX, offsetY);

        if (this.showGrid && isActive) {
            ctx.strokeStyle = "rgba(80, 115, 180, 0.15)";
            ctx.lineWidth = 1;
            const minor = 64;
            for (let x = 0; x <= CANVAS_WIDTH; x += minor) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, CANVAS_HEIGHT);
                ctx.stroke();
            }
            for (let y = 0; y <= CANVAS_HEIGHT; y += minor) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(CANVAS_WIDTH, y);
                ctx.stroke();
            }
        }

        if (this.showSafeZone && isActive) {
            ctx.strokeStyle = "rgba(130, 210, 255, 0.75)";
            ctx.setLineDash([24, 14]);
            ctx.lineWidth = 2;
            ctx.strokeRect(
                SAFE_ZONE.left,
                SAFE_ZONE.top,
                SAFE_ZONE.right - SAFE_ZONE.left,
                SAFE_ZONE.bottom - SAFE_ZONE.top
            );
            ctx.setLineDash([]);
        }

        ctx.lineWidth = 4;
        ctx.lineCap = "round";
        for (let i = 0; i < BONE_CONNECTIONS.length; i++) {
            const [start, end] = BONE_CONNECTIONS[i];
            const a = joints[start];
            const b = joints[end];
            if (!a || !b) {
                continue;
            }
            
            // Use colored bones matching OpenPose standard
            ctx.strokeStyle = getBoneColor(start, end, i);
            
            ctx.beginPath();
            ctx.moveTo(a[0], a[1]);
            ctx.lineTo(b[0], b[1]);
            ctx.stroke();
        }

        const safeLeft = SAFE_ZONE.left;
        const safeRight = SAFE_ZONE.right;
        const safeTop = SAFE_ZONE.top;
        const safeBottom = SAFE_ZONE.bottom;

        for (const [name, [x, y]] of Object.entries(joints)) {
            const outside = x < safeLeft || x > safeRight || y < safeTop || y > safeBottom;
            const hovered = isActive && name === this.hoveredJoint;
            const selected = isActive && name === this.selectedJoint;

            let fill = "#ff7f5a";
            if (outside) {
                fill = "#ff857a";
            }
            if (hovered) {
                fill = "#ffa86f";
            }
            if (selected) {
                fill = "#ffd35a";
            }

            ctx.fillStyle = fill;
            ctx.beginPath();
            ctx.arc(x, y, selected || hovered ? 14 : 10, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = "rgba(13, 19, 35, 0.85)";
            ctx.lineWidth = 2;
            ctx.stroke();

            if (this.showLabels && (hovered || selected)) {
                ctx.font = "20px 'JetBrains Mono', monospace";
                ctx.fillStyle = "#f5f8ff";
                ctx.textAlign = "center";
                ctx.fillText(name, x, y - 18);
            }
        }
        ctx.restore();
    }

    onPointerDown(event) {
        // Middle mouse button (button 1) for panning
        if (event.button === 1) {
            event.preventDefault();
            this.isPanning = true;
            this.panStartX = event.clientX - this.panX;
            this.panStartY = event.clientY - this.panY;
            this.canvas.style.cursor = "grabbing";
            this.canvas.setPointerCapture(event.pointerId);
            return;
        }

        const { x, y } = this.getCanvasCoords(event);
        const { name: joint, poseIndex } = this.findJointAt(x, y);
        
        if (poseIndex !== -1 && poseIndex !== this.currentPoseIndex) {
            this.selectPose(poseIndex);
        }

        if (!joint) {
            this.setSelectedJoint(null, { source: "2d" });
            this.setHoveredJoint(null, { source: "2d" });
            this.canvas.style.cursor = "crosshair";
            return;
        }
        this.setSelectedJoint(joint, { source: "2d" });
        this.setHoveredJoint(joint, { source: "2d" });
        this.dragging = true;
        this.dragPointerId = event.pointerId;
        this.canvas.setPointerCapture(event.pointerId);
        this.canvas.style.cursor = "grabbing";
    }

    onPointerMove(event) {
        // Handle panning
        if (this.isPanning) {
            this.panX = event.clientX - this.panStartX;
            this.panY = event.clientY - this.panStartY;
            this.canvas.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
            return;
        }

        const { x, y } = this.getCanvasCoords(event);
        
        if (this.dragging && this.selectedJoint) {
            // Map global coords to local pose coords
            let localX = x;
            let localY = y;
            
            if (this.viewMode === "grid") {
                const col = this.currentPoseIndex % 6;
                const row = Math.floor(this.currentPoseIndex / 6);
                localX = x - col * CANVAS_WIDTH;
                localY = y - row * CANVAS_HEIGHT;
            }

            this.joints[this.selectedJoint] = [
                clamp(localX, 0, CANVAS_WIDTH),
                clamp(localY, 0, CANVAS_HEIGHT)
            ];
            this.handleJointsMutated(false);
            return;
        }

        const { name: joint } = this.findJointAt(x, y);
        if (joint !== this.hoveredJoint) {
            this.setHoveredJoint(joint, { source: "2d" });
        }
    }

    onPointerUp(event) {
        // End panning
        if (this.isPanning) {
            this.isPanning = false;
            this.canvas.style.cursor = this.hoveredJoint ? "grab" : "crosshair";
            this.canvas.releasePointerCapture(event.pointerId);
            return;
        }

        if (!this.dragging || event.pointerId !== this.dragPointerId) {
            if (event.type === "pointerleave") {
                this.setHoveredJoint(null, { source: "2d" });
                this.canvas.style.cursor = "crosshair";
            }
            return;
        }
        this.dragging = false;
        this.dragPointerId = null;
        this.canvas.releasePointerCapture(event.pointerId);
        this.canvas.style.cursor = this.hoveredJoint ? "grab" : "crosshair";
        this.handleJointsMutated(true);
    }

    onWheel(event) {
        event.preventDefault();
        
        const delta = event.deltaY > 0 ? 0.9 : 1.1;
        const newZoom = clamp(this.zoom * delta, this.minZoom, this.maxZoom);
        
        if (newZoom !== this.zoom) {
            this.zoom = newZoom;
            this.canvas.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
            this.setStatus(`Zoom: ${Math.round(this.zoom * 100)}%`);
        }
    }

    onKeyDown(event) {
        if (!this.selectedJoint) {
            return;
        }
        let dx = 0;
        let dy = 0;
        const step = event.shiftKey ? 10 : 2;
        switch (event.key) {
            case "ArrowUp":
                dy = -step;
                break;
            case "ArrowDown":
                dy = step;
                break;
            case "ArrowLeft":
                dx = -step;
                break;
            case "ArrowRight":
                dx = step;
                break;
            default:
                return;
        }
        event.preventDefault();
        const [x, y] = this.joints[this.selectedJoint];
        this.joints[this.selectedJoint] = [
            clamp(x + dx, 0, CANVAS_WIDTH),
            clamp(y + dy, 0, CANVAS_HEIGHT)
        ];
        this.handleJointsMutated(false);
    }

    getCanvasCoords(event) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (event.clientX - rect.left) * scaleX,
            y: (event.clientY - rect.top) * scaleY
        };
    }

    findJointAt(x, y) {
        const radius = 30;
        let targetPoseIndex = this.currentPoseIndex;
        let localX = x;
        let localY = y;

        if (this.viewMode === "grid") {
            const col = Math.floor(x / CANVAS_WIDTH);
            const row = Math.floor(y / CANVAS_HEIGHT);
            if (col >= 0 && col < 6 && row >= 0 && row < 2) {
                targetPoseIndex = row * 6 + col;
                localX = x % CANVAS_WIDTH;
                localY = y % CANVAS_HEIGHT;
            } else {
                return { name: null, poseIndex: -1 };
            }
        }

        const joints = this.poses[targetPoseIndex];
        let closest = null;
        let minDist = Number.POSITIVE_INFINITY;
        
        for (const [name, [jx, jy]] of Object.entries(joints)) {
            const dist = Math.hypot(jx - localX, jy - localY);
            if (dist < radius && dist < minDist) {
                closest = name;
                minDist = dist;
            }
        }
        return { name: closest, poseIndex: targetPoseIndex };
    }

    resetPose() {
        this.joints = cloneJoints(DEFAULT_SKELETON);
        this.handleJointsMutated(true);
        this.setStatus("Pose reset to default.", "success");
    }

    mirrorPose() {
        const mirrored = {};
        
        // ПРОСТО зеркалим координаты, имена НЕ меняем!
        // Тогда правая рука (оранжевая) окажется слева визуально
        for (const [name, pos] of Object.entries(this.joints)) {
            if (!Array.isArray(pos) || pos.length < 2) {
                continue;
            }
            
            const [x, y] = pos;
            const mirroredX = CANVAS_WIDTH - x;
            
            // Сохраняем под ТЕМ ЖЕ именем с зеркальной координатой
            mirrored[name] = [mirroredX, y];
        }
        
        this.joints = mirrored;
        this.handleJointsMutated(true);
        this.setStatus("Pose mirrored");
    }

    fitSafeZone() {
        const bounds = computeBounds(this.joints);
        if (bounds.width === 0 || bounds.height === 0) {
            return;
        }
        const safeWidth = SAFE_ZONE.right - SAFE_ZONE.left;
        const safeHeight = SAFE_ZONE.bottom - SAFE_ZONE.top;
        const scale = Math.min(safeWidth / bounds.width, safeHeight / bounds.height);
        const centerX = (bounds.minX + bounds.maxX) / 2;
        const centerY = (bounds.minY + bounds.maxY) / 2;
        const targetX = (SAFE_ZONE.left + SAFE_ZONE.right) / 2;
        const targetY = (SAFE_ZONE.top + SAFE_ZONE.bottom) / 2;
        const adjusted = {};
        for (const [name, [x, y]] of Object.entries(this.joints)) {
            const nx = targetX + (x - centerX) * scale;
            const ny = targetY + (y - centerY) * scale;
            adjusted[name] = [
                clamp(nx, SAFE_ZONE.left, SAFE_ZONE.right),
                clamp(ny, SAFE_ZONE.top, SAFE_ZONE.bottom)
            ];
        }
        this.joints = adjusted;
        this.handleJointsMutated(true);
        this.setStatus("Pose normalized to safe zone.");
    }

    centerPose() {
        const bounds = computeBounds(this.joints);
        if (bounds.width === 0 || bounds.height === 0) {
            return;
        }
        const deltaX = (CANVAS_WIDTH / 2) - (bounds.minX + bounds.width / 2);
        const deltaY = (CANVAS_HEIGHT / 2) - (bounds.minY + bounds.height / 2);
        const centered = {};
        for (const [name, [x, y]] of Object.entries(this.joints)) {
            centered[name] = [
                clamp(x + deltaX, 0, CANVAS_WIDTH),
                clamp(y + deltaY, 0, CANVAS_HEIGHT)
            ];
        }
        this.joints = centered;
        this.handleJointsMutated(true);
        this.setStatus("Pose centered on canvas.");
    }

    async loadPreset(id) {
        const preset = PRESETS.find((entry) => entry.id === id);
        if (!preset) {
            return;
        }
        try {
            this.setStatus(`Loading preset “${preset.label}”…`);
            const response = await fetch(`/vnccs/pose_preset/${preset.file}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const payload = await response.json();
            
            // Check if preset is a set or single pose
            if (payload.poses && Array.isArray(payload.poses)) {
                const parsed = parsePoseData(payload);
                this.poses = parsed.poses;
                this.setStatus(`Preset set “${preset.label}” loaded.`, "success");
            } else {
                // Single pose
                const parsed = parsePoseData(payload); // This returns a set with the pose in [0]
                // We want to load it into current slot
                this.joints = parsed.poses[0];
                this.setStatus(`Preset pose “${preset.label}” loaded.`, "success");
            }
            
            this.handleJointsMutated(true);
        } catch (error) {
            console.error("[VNCCS] Failed to load preset", error);
            this.setStatus("Failed to load preset", "warning");
        }
    }

    importPose() {
        this.fileInput.value = "";
        this.fileInput.click();
    }

    onFileSelected(event) {
        const file = event.target.files?.[0];
        if (!file) {
            return;
        }
        const reader = new FileReader();
        reader.onload = () => {
            try {
                const raw = JSON.parse(reader.result);
                
                // Check if it's a pose set (our format or DWPose format)
                const isPoseSet = (raw.poses && Array.isArray(raw.poses)) || 
                                  (Array.isArray(raw) && raw[0]?.people);
                
                if (isPoseSet) {
                    // It's a set
                    const parsed = parsePoseData(raw);
                    this.poses = parsed.poses;
                    this.setStatus(`Imported pose set “${file.name}”.`, "success");
                } else {
                    // It's a single pose (or legacy)
                    // parsePoseData will put it in [0]
                    const parsed = parsePoseData(raw);
                    this.joints = parsed.poses[0];
                    this.setStatus(`Imported pose “${file.name}” into slot ${this.currentPoseIndex + 1}.`, "success");
                }
                this.handleJointsMutated(true);
            } catch (error) {
                console.error("[VNCCS] Failed to import pose", error);
                this.setStatus("Invalid pose JSON.", "warning");
            }
        };
        reader.readAsText(file);
    }

    copyPoseJson() {
        // Copy current pose only? Or set?
        // Let's copy current pose for compatibility with other tools
        const json = JSON.stringify({
            canvas: { width: CANVAS_WIDTH, height: CANVAS_HEIGHT },
            joints: this.joints
        }, null, 2);
        
        if (navigator.clipboard?.writeText) {
            navigator.clipboard.writeText(json)
                .then(() => this.setStatus("Current pose JSON copied.", "success"))
                .catch(() => this.setStatus("Clipboard unavailable.", "warning"));
        } else {
            this.setStatus("Clipboard API unsupported.", "warning");
        }
    }

    downloadPose() {
        const json = JSON.stringify({
            canvas: { width: CANVAS_WIDTH, height: CANVAS_HEIGHT },
            joints: this.joints
        }, null, 2);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `vnccs_pose_${this.currentPoseIndex + 1}_${Date.now()}.json`;
        anchor.click();
        URL.revokeObjectURL(url);
        this.setStatus("Pose JSON downloaded.", "success");
    }

    downloadSet() {
        const json = JSON.stringify(buildPosePayload(this.poses), null, 2);
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = `vnccs_poseset_${Date.now()}.json`;
        anchor.click();
        URL.revokeObjectURL(url);
        this.setStatus("Pose Set JSON downloaded.", "success");
    }

    scheduleSync(immediate = false) {
        if (!this.integration) {
            return;
        }
        if (immediate) {
            if (this.syncTimer) {
                clearTimeout(this.syncTimer);
                this.syncTimer = null;
            }
            this.integration.updatePoseFromEditor(this.poses);
            return;
        }
        if (this.syncTimer) {
            clearTimeout(this.syncTimer);
        }
        this.syncTimer = setTimeout(() => {
            this.integration?.updatePoseFromEditor(this.poses);
            this.syncTimer = null;
        }, 120);
    }

    updateMetrics() {
        const bounds = computeBounds(this.joints);
        const coverageX = bounds.width ? ((bounds.width / CANVAS_WIDTH) * 100).toFixed(1) : "0.0";
        const coverageY = bounds.height ? ((bounds.height / CANVAS_HEIGHT) * 100).toFixed(1) : "0.0";
        const outside = Object.values(this.joints).filter(([x, y]) => (
            x < SAFE_ZONE.left || x > SAFE_ZONE.right || y < SAFE_ZONE.top || y > SAFE_ZONE.bottom
        )).length;

        this.metricsEl.innerHTML = `
            Width: <strong>${bounds.width.toFixed(0)}px</strong> (${coverageX}% of canvas)<br>
            Height: <strong>${bounds.height.toFixed(0)}px</strong> (${coverageY}% of canvas)<br>
            Outside safe zone: <strong class="${outside ? "vnccs-warning" : "vnccs-success"}">${outside}</strong> joints
        `;
    }

    updateJointInfo() {
        const joint = this.selectedJoint || this.hoveredJoint;
        if (!joint) {
            this.jointInfo.textContent = "Select or hover a joint to inspect coordinates.";
            return;
        }
        const [x, y] = this.joints[joint];
        this.jointInfo.textContent = `${joint}: ${x.toFixed(0)} × ${y.toFixed(0)}`;
    }

    setTransformMode3D(mode) {
        if (!mode) {
            return;
        }
        this.update3DToolbarMode(mode);
        if (this.threeStatusRight) {
            this.threeStatusRight.textContent = mode === "translate"
                ? "Move mode · Drag gizmo axes to reposition"
                : "Rotate mode · Use colored arcs to twist limbs";
        }
        this.ensurePose3D().then((editor) => editor?.setTransformMode(mode));
    }

    ensurePose3D() {
        if (this.pose3dPromise) {
            return this.pose3dPromise;
        }
        if (!this.threeViewport) {
            return Promise.resolve(null);
        }
        if (this.threeStatusLeft) {
            this.threeStatusLeft.textContent = "Loading 3D editor…";
        }
        this.pose3dPromise = Pose3DEditor.create(this.threeViewport, {
            canvas: { width: CANVAS_WIDTH, height: CANVAS_HEIGHT },
            defaultPose: DEFAULT_SKELETON,
            connections: BONE_CONNECTIONS,
            onPoseChanged: (map) => this.onPoseFrom3D(map),
            onPoseCommitted: () => this.setStatus("3D adjustments synced.", "success"),
            onSelectJoint: (name) => this.setSelectedJoint(name, { source: "3d" }),
            onHoverJoint: (name) => this.setHoveredJoint(name, { source: "3d" })
        }).then((editor) => {
            if (!editor) {
                return null;
            }
            this.pose3d = editor;
            editor.setActive(false);
            editor.setTransformMode(this.active3DMode);
            if (this.threeStatusLeft) {
                this.threeStatusLeft.textContent = "Drag joints in 3D space.";
            }
            return editor;
        }).catch((error) => {
            console.error("[VNCCS] Failed to initialize 3D pose editor", error);
            if (this.threeStatusLeft) {
                this.threeStatusLeft.textContent = "3D editor failed to load.";
            }
            this.pose3d = null;
            this.pose3dPromise = null;
            return null;
        });
        return this.pose3dPromise;
    }

    queuePose3DSync() {
        if (this.pending3DSync) {
            return;
        }
        this.pending3DSync = requestAnimationFrame(() => {
            this.pending3DSync = null;
            const payload = cloneJoints(this.joints);
            if (this.pose3d) {
                this.pose3d.setPose(payload, true);
            } else if (this.pose3dPromise) {
                this.pose3dPromise.then((editor) => editor?.setPose(payload, true));
            }
        });
    }

    handleJointsMutated(forceImmediate = false, options = {}) {
        const { skip3D = false } = options;
        this.render();
        this.updateMetrics();
        this.updateJointInfo();
        this.scheduleSync(forceImmediate);
        if (!skip3D) {
            this.queuePose3DSync();
        }
    }

    onPoseFrom3D(map) {
        const updated = cloneJoints(this.joints);
        for (const [name, coords] of Object.entries(map)) {
            if (!Array.isArray(coords) || coords.length < 2) {
                continue;
            }
            const x = clamp(coords[0], 0, CANVAS_WIDTH);
            const y = clamp(coords[1], 0, CANVAS_HEIGHT);
            updated[name] = [x, y];
        }
        this.joints = updated;
        this.handleJointsMutated(true, { skip3D: true });
    }

    setSelectedJoint(name, { source = "internal" } = {}) {
        if (this.selectedJoint === name) {
            return;
        }
        this.selectedJoint = name;
        if (source !== "3d") {
            this.ensurePose3D().then((editor) => editor?.setSelection(name, "2d"));
        }
        if (!name && !this.dragging) {
            this.canvas.style.cursor = "crosshair";
        }
        this.updateJointInfo();
        this.render();
    }

    setHoveredJoint(name, { source = "internal" } = {}) {
        if (this.hoveredJoint === name) {
            return;
        }
        this.hoveredJoint = name;
        if (!this.dragging) {
            this.canvas.style.cursor = name ? "grab" : "crosshair";
        }
        if (source !== "3d") {
            this.pose3d?.setHover?.(name, "2d");
        }
        this.updateJointInfo();
        this.render();
    }

    setStatus(message, type = "info") {
        this.statusEl.textContent = message;
        this.statusEl.classList.remove("vnccs-warning", "vnccs-success");
        if (type === "warning") {
            this.statusEl.classList.add("vnccs-warning");
        } else if (type === "success") {
            this.statusEl.classList.add("vnccs-success");
        }
        this.statusMessage.textContent = message;
    }
}

const PoseEditorManager = {
    dialog: null,

    ensureDialog() {
        if (!this.dialog) {
            this.dialog = new PoseEditorDialog();
        }
        return this.dialog;
    },

    open(integration) {
        this.ensureDialog().open(integration);
    },

    closeIf(integration) {
        if (this.dialog?.isOpenFor(integration)) {
            this.dialog.close();
        }
    }
};

app.registerExtension({
    name: "VNCCS.PoseGenerator",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VNCCS_PoseGenerator") {
            return;
        }

        ensurePoseEditorStyles();

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const poseWidget = this.widgets?.find((widget) => widget.name === "pose_data");
            if (poseWidget) {
                poseWidget.hidden = true;
                poseWidget.computeSize = () => [0, -4];
                poseWidget.draw = () => {};
                poseWidget.serializeValue = () => poseWidget.value;
            }

            const integration = new NodePoseIntegration(this, poseWidget);
            this.__vnccsPose = integration;
            integration.installPreview();

            const button = this.addWidget("button", "Open Pose Editor", null, () => {
                PoseEditorManager.open(integration);
                return false;
            });
            button.serialize = false;

            const originalOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                PoseEditorManager.closeIf(integration);
                originalOnRemoved?.apply(this, arguments);
            };

            return result;
        };
    }
});