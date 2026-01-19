# MakeHuman Skeleton-Mesh Alignment Guide

## Coordinate System

MakeHuman uses a **right-handed coordinate system**:
- **X axis**: Left/Right (positive X = left side of body)
- **Y axis**: Up/Down (positive Y = up, origin at navel)
- **Z axis**: Front/Back (positive Z = front)

### Mesh Bounds (base.obj)
```
X: -4.973 to +4.973  (total width ~10 units)
Y: -8.449 to +8.497  (total height ~17 units, origin at navel)
Z: -1.103 to +3.256  (depth ~4.4 units)
```

### Key Bone Positions
```
spine_01:    head=(0.00, 1.19, -0.10)   - lower spine
upperarm_l:  head=(1.68, 5.25, 0.15)    - left shoulder (positive X!)
upperarm_r:  head=(-1.68, 5.25, 0.15)   - right shoulder (negative X!)
thigh_l:     head=(1.10, 0.49, 0.12)    - left hip
thigh_r:     head=(-1.10, 0.49, 0.12)   - right hip
head:        head=(0.00, 6.97, 0.16)    - top of neck
```

> **IMPORTANT**: MakeHuman naming convention uses `_l` = LEFT = POSITIVE X.

---

## Critical: Skeleton-Mesh Synchronization

### Problem
The skeleton is loaded with `base_mesh`, but the displayed mesh is **morphed** using character parameters (age, gender, weight, etc.). If the skeleton uses different parameters than the mesh, they won't align.

### Solution
When fetching skeleton bone positions, **solve the mesh with the same parameters** that will be used for display:

```python
# In /vnccs/skeleton/get_bones API:

# CRITICAL: Use SAME normalization as mesh API!
mh_age = (age - 1.0) / (90.0 - 1.0)  # age=25 -> 0.27
factors = solver.calculate_factors(mh_age, gender, weight, muscle, height, breast, genital)
current_verts = solver.solve_mesh(base_mesh, targets, factors)

# Update skeleton with current mesh
class MeshWrapper:
    def __init__(self, verts):
        self.vertices = verts

mesh_wrapper = MeshWrapper(current_verts)
skel.updateJointPositions(mesh_wrapper)  # THIS IS CRITICAL
```

### Age Normalization Formula
```python
mh_age = (age_years - 1.0) / (90.0 - 1.0)
# age=25 years -> 0.27
# age=45 years -> 0.49
# age=1 year -> 0.0
# age=90 years -> 1.0
```

---

## Three.js Integration

### No Coordinate Conversion Needed
Both MakeHuman and Three.js use right-handed coordinate systems. **Do NOT negate any axes.**

```javascript
// CORRECT - use coordinates as-is:
sphere.position.set(bone.headPos[0], bone.headPos[1], bone.headPos[2]);

// WRONG - don't do this:
// sphere.position.set(-bone.headPos[0], bone.headPos[1], bone.headPos[2]);
```

### Camera Setup for Human Model
```javascript
// Model is ~17 units tall, origin at navel
camera.position.set(0, 0, 25);
orbitTarget = new THREE.Vector3(0, 0, 0);  // Look at navel

// Grid at foot level
grid.position.y = -8.5;
```

---

## Bone Rotation (Forward Kinematics)

### Key Principle
Bones are **ROTATED**, not moved. Moving bone positions directly breaks the hierarchy.

### Rotation Order
MakeHuman uses Z-Y-X Euler order:
```python
mat = np.dot(rz, np.dot(ry, rx))
bone.matPose = mat
bone.update()  # Propagate to children
```

### After Rotation
Always call `bone.update()` to propagate transforms down the hierarchy, then `skel.updateJointPositions(mesh_wrapper)` to recalculate positions.

---

## API Endpoints

### GET /vnccs/skeleton/get_bones
Returns skeleton bone data with positions matching the current mesh.

**Response:**
```json
{
  "status": "success",
  "bones": [
    {
      "name": "spine_01",
      "headPos": [0.0, 1.19, -0.1],
      "tailPos": [0.0, 2.5, -0.08],
      "parent": "pelvis",
      "length": 1.31
    }
  ]
}
```

### POST /vnccs/character_studio/update_preview
Returns mesh vertices for given character parameters.

**Request:**
```json
{
  "age": 25.0,
  "gender": 0.5,
  "weight": 0.5,
  "muscle": 0.5,
  "height": 0.5,
  "manual_pose": {"bone_name": [rx, ry, rz]}  // optional, degrees
}
```

**Response:**
```json
{
  "status": "success",
  "vertices": [...],  // flat array [x1,y1,z1, x2,y2,z2, ...]
  "indices": [...],
  "normals": [...]
}
```

---

## Debugging Checklist

1. **Skeleton doesn't match mesh size?**
   - Check that `updateJointPositions()` is called with the same morphed mesh
   - Verify age normalization formula is applied consistently

2. **Left/Right reversed?**
   - Don't negate X coordinates
   - Remember: MakeHuman `_l` = positive X

3. **Bones move opposite direction?**
   - Don't negate rotation angles
   - Check rotation order (Z-Y-X)

4. **Mesh explodes when posed?**
   - Bones should be ROTATED, not moved
   - Check that `bone.update()` is called after setting `matPose`

# Unified Visualization Pipeline (Implementation Detail)

## Overview
To guarantee perfect synchronization between the 3D Mesh and the Skeleton visualization, a "Unified Pipeline" approach was implemented. This ensures that the data for both components comes from the exact same source state and is rendered using the exact same transformation matrices.

## 1. Single API Response (Backend)
Instead of fetching `get_bones` and `update_preview` separately (which could lead to race conditions or state mismatches), the `update_preview` endpoint now returns **both** the mesh vertices and the skeleton bone data in a single JSON response.

### Python Implementation (`__init__.py`)
The critical step is correctly calculating the **Visual Position** of the bones. 
- The `bone.headPos` property in MakeHuman represents the *Rest Position* (even after morphing).
- To show the bone where the mesh actually is (after skinning/posing), we must apply the same **Skinning Matrix** (`matPoseVerts`) that deforms the mesh vertices.

**Formula:**
```python
# Transform Rest Position to Visual Posed Position
VisualHead = bone.matPoseVerts * bone.headPos
VisualTail = bone.matPoseVerts * bone.tailPos
```

**Code Snippet:**
```python
# Inside vnccs_character_studio_update_preview
for bone in skel.getBones():
    # 1. Get Rest Position (Morphed but not Posed)
    headPos = bone.headPos
    
    # 2. Get Skinning Matrix (Global Pose * InvRest)
    # This matrix transforms points from Rest Mesh Space to Posed Mesh Space
    mat = np.array(bone.matPoseVerts)
    
    # 3. Apply Transform
    h_vec = np.array([headPos[0], headPos[1], headPos[2], 1.0])
    finalHead = np.dot(mat, h_vec)[:3].tolist()
    
    # Result: 'finalHead' is exactly where the skin moves to.
```

## 2. Shared Rendering Logic (Frontend)
The frontend (`vnccs_debug2.js`) was rewritten to use a `UnifiedViewer` class.
- **Single Camera**: One camera instance controls the view for both mesh and skeleton.
- **Shared Matrices**: The `ModelView` and `Projection` matrices are calculated once per frame and passed to both the Mesh Shader and the Skeleton/Line Shader.

This mathematically guarantees that they cannot visually diverge, provided the input data (from step 1) is correct.

## 3. Interactive Bone Control
To allow bone manipulation without breaking sync:
1. **Raycasting**: Used to select bones. Ray originates from the camera and checks intersection with spheres at `finalHead` positions.
2. **Rotation Mapping**: Mouse movements are mapped to the bone's **Local Rotation Axes**.
   - The Camera's Up/Right vectors are transformed into the bone's **Local Space** (using `Inverse(RestMatrix)`).
   - This ensures that dragging "Up" on screen always rotates the bone along the visible "Up" direction, regardless of the bone's actual orientation.
3. **Server-Authoritative Update**:
   - The client calculates the new Euler rotation.
   - It sends `manual_pose` to the server.
   - The server applies the pose, solves the skinning, and returns the **new Mesh Vertices** AND the **new Bone Positions**.
   - The client blindly renders the result. This ensures the visual state is always true to the server's simulation.

# 4. Mathematical Model of Interaction

To ensure the user can reproduce the movement logic, here is the exact mathematical model used involved in moving a bone.

## A. Frontend: Mouse to Local Rotation
When the user drags the mouse, we want the bone to rotate relative to the **Camera's View**, not the World Axis. If I pull "Up" on screen, the arm should go "Up" relative to the screen, regardless of how the character is standing.

**Step 1: Get Camera Axes in World Space**
From the shared `ModelView` matrix ($):
$$ Axis_{right} = (M_{00}, M_{10}, M_{20}) $$
$$ Axis_{up} = (M_{01}, M_{11}, M_{21}) $$
*(Note: Rows 0 and 1 of View Matrix)*

**Step 2: Transform to Bone Local Space**
The bone's rotation is defined in its **Local Coordinate System**. We must transform the Camera Axes into this local system using the inverse of the bone's **Rest Matrix** ($).
$$ Axis_{localRight} = R^{-1} \cdot Axis_{right} $$
$$ Axis_{localUp} = R^{-1} \cdot Axis_{up} $$

**Step 3: Calculate Rotation Delta**
We create a quaternion rotation based on mouse delta (, dy$):
$$ Q_{delta} = Q(Axis_{localUp}, dx) \times Q(Axis_{localRight}, dy) $$

**Step 4: Apply to Current Pose**
$$ Q_{new} = Q_{delta} \times Q_{current} $$
*We then convert {new}$ back to Euler Angles (Degrees) to send to the server.*

---

## B. Backend: Pose Application & Skinning
The server receives the Euler angles and calculates the final position of both the mesh and the skeleton visualizer.

**Step 1: Reconstruct Rotation Matrix**
For each bone in `manual_pose`, we create a rotation matrix from Euler angles (, ry, rz$) assuming **Z-Y-X order** (standard for MakeHuman):
$$ M_{pose} = R_z(rz) \cdot R_y(ry) \cdot R_x(rx) $$

**Step 2: Update Hierarchy**
We set `bone.matPose = M_{pose}` and call `bone.update()`. This triggers the recursive update of global matrices:
$$ M_{global} = M_{parentGlobal} \cdot M_{localRest} \cdot M_{pose} $$

**Step 3: Calculate Skinning Matrix**
The **Skinning Matrix** ({skin}$) is what deforms the mesh. It represents the transform from **Rest Pose** to **Current Pose**.
$$ M_{skin} = M_{global} \cdot M_{globalRest}^{-1} $$
*(In code this is `bone.matPoseVerts`)*

**Step 4: Deform Mesh (Linear Blend Skinning)**
For each vertex $ with weights $ for bones $:
$$ v_{final} = \sum (w_i \cdot (M_{skin, i} \cdot v_{rest})) $$

**Step 5: Deform Skeleton Visualizer (Sync Magic)**
To ensure the green skeleton lines match the deformed mesh:
$$ Head_{visual} = M_{skin} \cdot Head_{rest} $$
$$ Tail_{visual} = M_{skin} \cdot Tail_{rest} $$

By using {skin}$ for **both** the vertex calculation (Step 4) and the bone visualization (Step 5), synchronization is mathematically guaranteed.
