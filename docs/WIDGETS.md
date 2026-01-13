# VNCCS Widgets Documentation

## Overview

VNCCS uses custom JavaScript widgets to provide enhanced UI for certain nodes. These widgets are loaded from the `web/` directory via the `WEB_DIRECTORY = "web"` setting in `__init__.py`.

---

## Widget Files

| File | Target Node | Lines | Description |
|------|-------------|-------|-------------|
| `vnccs_emotion_v2.js` | `EmotionGeneratorV2` | 637 | Full custom UI with emotion grid, costume checkboxes, character preview |
| `vnccs_emotion_generator.js` | `EmotionGenerator` | 45 | Adds "Add Emotion" button |
| `pose_editor.js` | `VNCCS_PoseGenerator` | 2144 | 2D pose editor with drag-and-drop joints |
| `pose_editor_3d.js` | `VNCCS_PoseGenerator` | ~400 | 3D pose editor using Three.js |
| `pose_editor_3d_body.js` | - | ~200 | 3D body model definitions |
| `pose_editor_3d_render.js` | - | ~300 | 3D rendering utilities |
| `bone_colors.js` | - | ~150 | OpenPose bone color definitions |

---

## Widget Implementation Patterns

### 1. Extension Registration

All widgets register via `app.registerExtension()`:

```javascript
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "VNCCS.EmotionGeneratorV2",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "EmotionGeneratorV2") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                // Custom initialization here
            };
        }
    }
});
```

### 2. DOM Widget for Custom UI

Used in `EmotionGeneratorV2` to create a full custom HTML interface:

```javascript
const container = document.createElement("div");
container.className = "vnccs-container";
// ... build HTML structure ...

const widget = node.addDOMWidget("emotion_ui_v2", "ui", container, {
    serialize: true,
    hideOnZoom: false,
    getValue() { return undefined; },
    setValue(v) { }
});
```

### 3. Hiding Standard Widgets

Replace standard widgets with custom UI:

```javascript
const charWidget = node.widgets.find(w => w.name === "character");
if (charWidget) charWidget.hidden = true;

const costumesDataWidget = node.widgets.find(w => w.name === "costumes_data");
if (costumesDataWidget) costumesDataWidget.hidden = true;
```

### 4. Syncing Widget Values

Update hidden widgets when custom UI changes:

```javascript
function updateCostumesData() {
    const list = Array.from(state.selectedCostumes);
    if (costumesDataWidget) costumesDataWidget.value = JSON.stringify(list);
}

function updateEmotionsData() {
    const list = Array.from(state.selectedEmotions);
    if (emotionsDataWidget) emotionsDataWidget.value = JSON.stringify(list);
}
```

### 5. Callback Hooks

Hook into widget callbacks:

```javascript
if (charWidget) {
    const originalCb = charWidget.callback;
    charWidget.callback = function(v) {
        state.character = v;
        if (charSelect.value !== v) charSelect.value = v;
        fetchCharacterData(v);
        if (originalCb) originalCb(v);
    };
}
```

### 6. Resize Handling

Handle node resize:

```javascript
node.onResize = function(size) {
    const [w, h] = size;
    container.style.width = (w - 20) + "px";
    container.style.height = (h - 60) + "px";
}
```

### 7. Adding Button Widgets

Simple button addition (EmotionGenerator):

```javascript
this.addWidget("button", "Add Emotion", null, () => {
    const emotionsWidget = this.widgets.find(w => w.name === "emotions");
    // Button click handler
});
```

---

## EmotionGeneratorV2 Widget Details

### State Management

```javascript
let state = {
    character: charWidget ? charWidget.value : "",
    costumes: [],
    selectedCostumes: new Set(),
    emotions: [],
    selectedEmotions: new Set(),
    searchTerm: ""
};
```

### API Calls

```javascript
// Fetch emotions list
fetch("/vnccs/get_emotions").then(async (res) => {
    const data = await res.json();
    // Process emotions...
});

// Fetch character costumes
const res = await fetch(`/vnccs/get_character_costumes?character=${encodeURIComponent(charName)}`);
const validCostumes = await res.json();

// Character preview image
charImg.src = `/vnccs/get_character_sheet_preview?character=${encodeURIComponent(charName)}&t=${Date.now()}`;

// Emotion preview image
img.src = `/vnccs/get_emotion_image?name=${encodeURIComponent(e.safe_name)}`;
```

### CSS Classes

- `.vnccs-container` - Main flexbox container
- `.vnccs-left-col` / `.vnccs-right-col` - Two-column layout
- `.vnccs-section` - Gray themed sections
- `.vnccs-char-header` - Character header with select dropdown
- `.vnccs-char-preview-container` / `.vnccs-char-preview` - Character image preview
- `.vnccs-costumes-list` - Costume checkboxes container
- `.vnccs-emotions-container` - 4-column grid for emotions
- `.vnccs-emotion-item` / `.vnccs-emotion-item.selected` - Emotion cards
- `.vnccs-emotion-img` / `.vnccs-emotion-label` - Emotion image and label
- `.vnccs-search-input` - Search input field
- `.vnccs-btn` - Buttons (Select All)

---

## Pose Editor Widget Details

### Key Classes

- `NodePoseIntegration` - Integrates pose data with ComfyUI node widget
- `PosePreviewRenderer` - Renders OpenPose skeleton preview in node
- `PoseEditorDialog` - Full-screen modal editor for pose manipulation

### Data Format

Pose data is stored as JSON string with 12 poses, each containing joint coordinates:

```javascript
[
    {
        "nose": [256, 150],
        "neck": [256, 250],
        "r_shoulder": [200, 280],
        "l_shoulder": [312, 280],
        // ... more joints
    },
    // ... 11 more poses
]
```

### Joint Names (OpenPose BODY_25 format)

```javascript
const DEFAULT_SKELETON = {
    nose: [256, 150],
    neck: [256, 250],
    r_shoulder: [200, 280],
    r_elbow: [170, 420],
    r_wrist: [155, 560],
    l_shoulder: [312, 280],
    l_elbow: [342, 420],
    l_wrist: [357, 560],
    midhip: [256, 650],
    r_hip: [217, 660],
    r_knee: [217, 900],
    r_ankle: [212, 1140],
    l_hip: [295, 660],
    l_knee: [295, 900],
    l_ankle: [300, 1140],
    r_eye: [270, 135],
    l_eye: [242, 135],
    r_ear: [285, 145],
    l_ear: [227, 145]
};
```

### Canvas Dimensions

- Width: 512px
- Height: 1536px
- Output grid: 6x2 (3072x3072 total)
