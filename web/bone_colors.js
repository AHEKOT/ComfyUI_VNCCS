/**
 * Bone color configuration
 * Each bone can have its color customized here
 * Format: "bone_name" or ["joint1", "joint2"] -> hex color code
 * 
 * OpenPose BODY_18 limbSeq (1-indexed in reference, converted to 0-indexed joint names):
 * [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], [1,16], [16,18], [3,17], [6,18]]
 * Maps to:
 * 0: neck->r_shoulder, 1: neck->l_shoulder, 2: r_shoulder->r_elbow, 3: r_elbow->r_wrist, 
 * 4: l_shoulder->l_elbow, 5: l_elbow->l_wrist, 6: neck->r_hip, 7: r_hip->r_knee, 8: r_knee->r_ankle,
 * 9: neck->l_hip, 10: l_hip->l_knee, 11: l_knee->l_ankle, 12: neck->nose, 13: nose->r_eye,
 * 14: r_eye->r_ear, 15: nose->l_eye, 16: l_eye->l_ear, 17: r_shoulder->r_ear, 18: l_shoulder->l_ear
 */

// OpenPose standard color palette (18 colors from reference implementation)
const OPENPOSE_COLORS = [
    "#ff0000",  // 0: Red
    "#ff5500",  // 1: Orange
    "#ffaa00",  // 2: Dark orange
    "#ffff00",  // 3: Yellow
    "#aaff00",  // 4: Yellow-green
    "#55ff00",  // 5: Light green
    "#00ff00",  // 6: Green
    "#00ff55",  // 7: Green-cyan
    "#00ffaa",  // 8: Cyan-green
    "#00ffff",  // 9: Cyan
    "#00aaff",  // 10: Cyan-blue
    "#0055ff",  // 11: Light blue
    "#0000ff",  // 12: Blue
    "#5500ff",  // 13: Purple-blue
    "#aa00ff",  // 14: Purple
    "#ff00ff",  // 15: Magenta
    "#ff00aa",  // 16: Pink
    "#ff0055",  // 17: Hot pink
];

export const BONE_COLORS = {
    // Upper body
    "nose-neck": OPENPOSE_COLORS[12],             // 0: Blue
    "neck-r_shoulder": OPENPOSE_COLORS[1],        // 1: Orange - RIGHT is orange/warm
    "r_shoulder-r_elbow": OPENPOSE_COLORS[2],     // 2: Dark orange - RIGHT
    "r_elbow-r_wrist": OPENPOSE_COLORS[3],        // 3: Yellow - RIGHT
    "neck-l_shoulder": OPENPOSE_COLORS[5],        // 4: Light green - LEFT is cool
    "l_shoulder-l_elbow": OPENPOSE_COLORS[6],     // 5: Green - LEFT
    "l_elbow-l_wrist": OPENPOSE_COLORS[7],        // 6: Green-cyan - LEFT
    "neck-r_hip": OPENPOSE_COLORS[8],             // 7: Cyan-green - RIGHT hip
    "neck-l_hip": OPENPOSE_COLORS[11],            // 8: Light blue - LEFT hip

    // Right leg (GREEN)
    "r_hip-r_knee": OPENPOSE_COLORS[5],           // 9: Light green - RIGHT leg
    "r_knee-r_ankle": OPENPOSE_COLORS[6],         // 10: Green - RIGHT leg

    // Left leg (CYAN-BLUE)
    "l_hip-l_knee": OPENPOSE_COLORS[9],           // 11: Cyan - LEFT leg
    "l_knee-l_ankle": OPENPOSE_COLORS[10],        // 12: Cyan-blue - LEFT leg

    // Face
    "nose-r_eye": OPENPOSE_COLORS[14],            // 13: Purple
    "r_eye-r_ear": OPENPOSE_COLORS[16],           // 14: Pink
    "nose-l_eye": OPENPOSE_COLORS[14],            // 15: Purple
    "l_eye-l_ear": OPENPOSE_COLORS[16],           // 16: Pink
};

// Fallback palette - MUST match Python FALLBACK_PALETTE exactly
export const FALLBACK_PALETTE = [
    OPENPOSE_COLORS[12],  // 0: nose->neck (Blue)
    OPENPOSE_COLORS[1],   // 1: neck->r_shoulder (Orange) - RIGHT is orange/warm
    OPENPOSE_COLORS[2],   // 2: r_shoulder->r_elbow (Dark orange) - RIGHT
    OPENPOSE_COLORS[3],   // 3: r_elbow->r_wrist (Yellow) - RIGHT
    OPENPOSE_COLORS[5],   // 4: neck->l_shoulder (Light green) - LEFT is cool
    OPENPOSE_COLORS[6],   // 5: l_shoulder->l_elbow (Green) - LEFT
    OPENPOSE_COLORS[7],   // 6: l_elbow->l_wrist (Green-cyan) - LEFT
    OPENPOSE_COLORS[8],   // 7: neck->r_hip (Cyan-green) - RIGHT hip
    OPENPOSE_COLORS[11],  // 8: neck->l_hip (Light blue) - LEFT hip
    OPENPOSE_COLORS[5],   // 9: r_hip->r_knee (Light green) - RIGHT leg
    OPENPOSE_COLORS[6],   // 10: r_knee->r_ankle (Green) - RIGHT leg
    OPENPOSE_COLORS[9],   // 11: l_hip->l_knee (Cyan) - LEFT leg
    OPENPOSE_COLORS[10],  // 12: l_knee->l_ankle (Cyan-blue) - LEFT leg
    OPENPOSE_COLORS[14],  // 13: nose->r_eye (Purple)
    OPENPOSE_COLORS[16],  // 14: r_eye->r_ear (Pink)
    OPENPOSE_COLORS[14],  // 15: nose->l_eye (Purple)
    OPENPOSE_COLORS[16],  // 16: l_eye->l_ear (Pink)
];

/**
 * Get color for a specific bone connection
 * @param {string} joint1 - First joint name
 * @param {string} joint2 - Second joint name
 * @param {number} boneIndex - Fallback bone index if direct lookup fails
 * @returns {string} Hex color code
 */
export function getBoneColor(joint1, joint2, boneIndex) {
    // Primary: use boneIndex for consistent palette-based colors
    if (boneIndex >= 0 && boneIndex < FALLBACK_PALETTE.length) {
        return FALLBACK_PALETTE[boneIndex];
    }
    
    // Secondary: try direct lookup
    const key1 = `${joint1}-${joint2}`;
    if (BONE_COLORS[key1]) {
        return BONE_COLORS[key1];
    }
    
    // Tertiary: try reverse lookup
    const key2 = `${joint2}-${joint1}`;
    if (BONE_COLORS[key2]) {
        return BONE_COLORS[key2];
    }
    
    return "#ffffff";
}

/**
 * Convert hex color to RGB
 * @param {string} hex - Hex color code (e.g., "#ff0000")
 * @returns {object} Object with r, g, b properties (0-255)
 */
export function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : { r: 255, g: 255, b: 255 };
}

/**
 * Convert hex color to RGBA string for canvas
 * @param {string} hex - Hex color code
 * @param {number} alpha - Alpha value (0-1)
 * @returns {string} RGBA string
 */
export function hexToRgba(hex, alpha = 1) {
    const rgb = hexToRgb(hex);
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

/**
 * Convert hex color to BGR tuple for OpenCV
 * @param {string} hex - Hex color code
 * @returns {array} [B, G, R] array for OpenCV
 */
export function hexToBgr(hex) {
    const rgb = hexToRgb(hex);
    return [rgb.b, rgb.g, rgb.r];
}
