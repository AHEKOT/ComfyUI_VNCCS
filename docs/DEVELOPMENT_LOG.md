# CharacterDesigner Node: Development & Architecture Documentation

This document describes the complete architecture, technical decisions, and implementation details of the **CharacterDesigner** custom node for ComfyUI.

## 1. Project Goal
Create a self-contained "Visual Designer" node for creating 3D characters based on the MakeHuman engine directly within ComfyUI.
**Key Features:**
*   **Interactive WebGL Preview:** Realtime 3D visualization inside the node.
*   **Native UI:** Custom sliders and controls painted directly on the node canvas (no HTML overlays).
*   **Parametric Generation:** Morphing mesh based on Age, Gender, Weight, Muscle, etc.
*   **Architecture:** Split between Python backend (geometry calculation) and Javascript frontend (rendering & interaction).

---

## 2. Core Architecture

### A. Backend (Python)
The backend is responsible for heavy geometry processing, loading MakeHuman assets, and calculating morphs.

*   **File:** `nodes/mh_main.py`
    *   **Classes:**
        *   `MH_Generator`: Main logic. Loads `base.obj` and applied delta-morphs (targets).
        *   `MH_VisualDesigner`: The frontend node. Acts as a wrapper, exposing `INPUT_TYPES` but delegating logic to Generator. Returns an `IMAGE` tensor (1536x1536) for the workflow.
*   **Data Handling:**
    *   A global `MH_DATA_CACHE` singleton stores loaded assets to efficient reuse across node executions.
    *   Loads standard MakeHuman targets from `makehuman/data/targets`.
*   **Skeleton System (New):**
    *   **Library:** `utils/mh_skeleton.py` (Ported from MakeHuman Core).
    *   **Logic:** Loads `default.mhskel`. When the mesh changes shape (e.g., gets taller), the skeleton is *retargeted* (`updateJointPositions`) to fit the new geometry.
    *   **Math:** Uses `utils/matrix.py` and `utils/transformations.py` for linear algebra (ported from MH).

### B. Frontend (Javascript)
The frontend handles the visual representation and user interaction within the ComfyUI graph.

*   **File:** `js/character_node.js`
*   **Rendering Engine:**
    *   **`OffscreenModelViewer`**: A custom ES6 class that creates a hidden `<canvas>` and uses raw **WebGL**.
    *   **Shading:** Implements a custom Phong Shader (Vertex + Fragment shaders) for decent lighting.
    *   **Camera:** Auto-centering logic that locks rotation pivot to the world origin (X=0, Z=0) to prevent "wobbly" rotation.
*   **UI Framework (The "No-DOM" Approach):**
    *   **Problem:** Standard DOM widgets float above the canvas and break during zooming/panning.
    *   **Solution:** We hide standard LiteGraph widgets using `computeSize = () => [0, -4]` and `draw = () => {}`.
    *   **Custom Drawing:** We override `onDrawBackground` to paint the entire UI (background, sliders, text, 3D viewport) using `ctx.roundRect`, `ctx.fillText`, and `ctx.drawImage`.
*   **Interaction:**
    *   **Event Handling:** `onMouseDown`, `onMouseMove` are captured to manually raycast clicks against our custom UI regions.
    *   **Sync:** When a custom slider moves, we update the *hidden* standard widget's value to ensure the backend receives the data upon execution.

---

## 3. Key Technical Challenges & Solutions

### 1. UI Layout Stability
*   **Issue:** Shrinking the node caused UI elements to float in void space.
*   **Fix:** Implemented `onResize` constraint to strictly forbid width/height below 800x600.

### 2. "Sticky" Sliders
*   **Issue:** Releasing the mouse outside the node left the slider "stuck" to the cursor.
*   **Fix:** Added a check `if (event.buttons === 0)` in `onMouseMove` to force-release the capture.

### 3. Rendering Quality
*   **Issue:** 3D model looked flat and jagged.
*   **Fix:**
    *   Backend: Calculated smooth Vertex Normals using NumPy vectorization.
    *   Frontend: Implemented per-pixel lighting (Phong) in WebGL.
    *   Output: The final output image is rendered at 1536x1536 using a software rasterizer (Painter's Algorithm) in Python to ensure perfect reproducibility regardless of the browser.

### 4. Skeleton Integration
*   **Issue:** MakeHuman relies on a complex hierarchy of bones and helpers.
*   **Fix:** We ported the `Skeleton`, `Bone`, and `VertexBoneWeights` classes. We stripped dependencies (like logging and file handling) to make them standalone utils. The skeleton now dynamically adjusts its joint positions based on the morphed mesh vertices.

### 5. State Persistence
*   **Issue:** On F5 (Refresh), the custom UI reset to defaults while the node remembered values.
*   **Fix:** Added `onConfigure` hook to read values from the hidden widgets back into the UI state object.

---

## 4. Helper Libraries (`utils/`)
*   `mh_parser.py`: Parses MakeHuman `.target` binary/text files.
*   `mh_skeleton.py`: Handles bone hierarchy and .mhskel loading.
*   `matrix.py`: 4x4 Matrix math helpers.
*   `transformations.py`: Quaternion/Rotation math.
*   `obj_loader.py`: Fast OBJ parser.
*   `mesh_processing.py`: Catmull-Clark subdivision (experimental).

---

## 5. Future Roadmap
*   **Pose System:** Apply rotation matrices to the loaded skeleton to pose the character.
*   **Skinning:** Implement Linear Blend Skinning to deform the mesh based on the posed skeleton.
*   **ControlNet Output:** Export OpenPose-compatible JSON/images derived from the actual 3D skeleton.
