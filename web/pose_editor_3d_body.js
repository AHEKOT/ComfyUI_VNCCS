import { getBoneColor, hexToRgb } from "./bone_colors.js";

const OPENPOSE_NAMES = [
    "nose",
    "neck",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "r_eye",
    "l_eye",
    "r_ear",
    "l_ear"
];

const BONE_RADIUS = 4.2;
const JOINT_RADIUS = 12;

const BONE_UP = new Float32Array([0, 1, 0]);

const BONE_COLOR_PALETTE = [
    0xff0000,
    0xff5500,
    0xffaa00,
    0xffff00,
    0xaaff00,
    0x55ff00,
    0x00ff00,
    0x00ff55,
    0x00ffaa,
    0x00ffff,
    0x00aaff,
    0x0055ff,
    0x0000ff,
    0x5500ff,
    0xaa00ff,
    0xff00ff,
    0xff0099,
    0xff0066,
    0xff3300,
    0xff6600,
    0xff9900,
    0xffcc00,
    0xffff66,
    0xccff66,
    0x99ffcc,
    0x66ffff,
    0x6699ff
];

export function hexToRgba(hex, alpha = 1) {
    const r = (hex >> 16) & 255;
    const g = (hex >> 8) & 255;
    const b = hex & 255;
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export const JOINT_TREE = [
    { name: "neck", parent: null },
    { name: "nose", parent: "neck" },
    { name: "r_shoulder", parent: "neck" },
    { name: "r_elbow", parent: "r_shoulder" },
    { name: "r_wrist", parent: "r_elbow" },
    { name: "l_shoulder", parent: "neck" },
    { name: "l_elbow", parent: "l_shoulder" },
    { name: "l_wrist", parent: "l_elbow" },
    { name: "r_hip", parent: "neck" },
    { name: "r_knee", parent: "r_hip" },
    { name: "r_ankle", parent: "r_knee" },
    { name: "l_hip", parent: "neck" },
    { name: "l_knee", parent: "l_hip" },
    { name: "l_ankle", parent: "l_knee" },
    { name: "r_eye", parent: "nose" },
    { name: "r_ear", parent: "r_eye" },
    { name: "l_eye", parent: "nose" },
    { name: "l_ear", parent: "l_eye" }
];

function colorForIndex(index) {
    return BONE_COLOR_PALETTE[index % BONE_COLOR_PALETTE.length];
}

export function getBoneColorHex(index) {
    return colorForIndex(index);
}

export function getBoneColorRgba(index, alpha = 0.85) {
    const hex = colorForIndex(index);
    return hexToRgba(hex, alpha);
}

function createJointHandle(THREE, name) {
    const material = new THREE.MeshStandardMaterial({
        color: 0xe2e7ff,
        emissive: 0x111726,
        roughness: 0.42,
        metalness: 0.18
    });
    const sphere = new THREE.Mesh(new THREE.SphereGeometry(JOINT_RADIUS, 28, 18), material);
    sphere.name = `${name}_handle`;
    sphere.userData.jointName = name;
    return { sphere, material };
}

function createBoneMesh(THREE, color) {
    const geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 18, 1, true);
    geometry.translate(0, 0.5, 0);
    const material = new THREE.MeshStandardMaterial({
        color,
        emissive: new THREE.Color(color).multiplyScalar(0.22),
        transparent: true,
        opacity: 0.92,
        roughness: 0.32,
        metalness: 0.08,
        depthWrite: false
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = "pose_bone";
    return mesh;
}

function updateBoneTransform(THREE, bone, startGroup, endGroup) {
    const startPos = new THREE.Vector3();
    const endPos = new THREE.Vector3();
    startGroup.getWorldPosition(startPos);
    endGroup.getWorldPosition(endPos);

    const direction = endPos.clone().sub(startPos);
    const length = direction.length();

    if (length < 1e-3) {
        bone.visible = false;
        return;
    }

    const parentQuaternion = new THREE.Quaternion();
    startGroup.getWorldQuaternion(parentQuaternion);
    const invParentQuat = parentQuaternion.clone().invert();
    const dirLocal = direction.clone().applyQuaternion(invParentQuat);
    const segmentLength = dirLocal.length();

    if (segmentLength < 1e-3) {
        bone.visible = false;
        return;
    }

    const normalizedLocal = dirLocal.clone().normalize();
    const upVector = new THREE.Vector3(BONE_UP[0], BONE_UP[1], BONE_UP[2]);
    const localQuat = new THREE.Quaternion().setFromUnitVectors(upVector, normalizedLocal);

    const maxMargin = Math.min(JOINT_RADIUS * 0.75, segmentLength * 0.45);
    // shrink the segment so it does not pierce the joint spheres
    const effectiveLength = segmentLength - maxMargin * 2;

    if (effectiveLength <= 1e-2) {
        bone.visible = false;
        return;
    }

    bone.visible = true;

    const offset = normalizedLocal.clone().multiplyScalar(maxMargin);
    bone.position.copy(offset);
    bone.quaternion.copy(localQuat);
    bone.scale.set(BONE_RADIUS, effectiveLength, BONE_RADIUS);
    bone.updateMatrixWorld(true);
}

export function createPoseBody(THREE, { connections, defaultDepth }) {
    const root = new THREE.Group();
    root.name = "vnccs_pose_root";

    const jointData = {};
    const pickTargets = [];

    for (const { name, parent, hidden } of JOINT_TREE) {
        const group = new THREE.Group();
        group.name = name;

        let sphere = null;
        let material = null;
        if (!hidden) {
            const handles = createJointHandle(THREE, name);
            sphere = handles.sphere;
            material = handles.material;
            group.add(sphere);
        }

        const data = {
            name,
            group,
            sphere,
            material,
            baseColor: 0xe2e7ff,
            baseEmissive: 0x111726,
            defaultDepth: defaultDepth(name),
            hidden: Boolean(hidden)
        };

        jointData[name] = data;
        if (sphere) {
            pickTargets.push(sphere);
        }

        if (parent) {
            const parentGroup = jointData[parent]?.group;
            if (parentGroup) {
                parentGroup.add(group);
            } else {
                root.add(group);
            }
        } else {
            root.add(group);
        }
    }

    const bones = [];
    connections.forEach(([start, end], index) => {
        const startData = jointData[start];
        const endData = jointData[end];
        if (!startData || !endData) {
            return;
        }
        // Use custom color from config
        const hexColor = getBoneColor(start, end, index);
        const rgb = hexToRgb(hexColor);
        const color = (rgb.r << 16) | (rgb.g << 8) | rgb.b;
        const mesh = createBoneMesh(THREE, color);
        mesh.name = `${start}__${end}_bone`;
        mesh.userData = { start, end };
        startData.group.add(mesh);
        bones.push({ mesh, start, end });
    });

    const updateBones = () => {
        root.updateMatrixWorld(true);
        for (const bone of bones) {
            const startGroup = jointData[bone.start]?.group;
            const endGroup = jointData[bone.end]?.group;
            if (!startGroup || !endGroup) {
                continue;
            }
            updateBoneTransform(THREE, bone.mesh, startGroup, endGroup);
        }
    };

    const jointNames = JOINT_TREE.map((joint) => joint.name);

    return {
        root,
        jointData,
        jointNames,
        pickTargets,
        bones,
        updateBones
    };
}

export { OPENPOSE_NAMES };
