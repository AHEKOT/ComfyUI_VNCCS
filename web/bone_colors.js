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
    // Match BONE_CONNECTIONS order from pose_editor.js with reference colors
    // Head and face (indices 0-4)
    "nose-neck": OPENPOSE_COLORS[12],         // 0: Blue (nose-neck vertical)
    "nose-r_eye": OPENPOSE_COLORS[13],        // 1: Purple-blue
    "r_eye-r_ear": OPENPOSE_COLORS[14],       // 2: Purple
    "nose-l_eye": OPENPOSE_COLORS[15],        // 3: Magenta
    "l_eye-l_ear": OPENPOSE_COLORS[16],       // 4: Pink
    
    // Shoulder line and torso sides (indices 5-7)
    "r_shoulder-l_shoulder": OPENPOSE_COLORS[0],  // 5: Red (shoulder line)
    "neck-r_hip": OPENPOSE_COLORS[6],         // 6: Green (right torso)
    "neck-l_hip": OPENPOSE_COLORS[9],         // 7: Cyan (left torso)
    
    // Right arm (indices 8-9) - RED/ORANGE side
    "r_shoulder-r_elbow": OPENPOSE_COLORS[1], // 8: Orange
    "r_elbow-r_wrist": OPENPOSE_COLORS[2],    // 9: Dark orange
    
    // Left arm (indices 10-11) - YELLOW/GREEN side
    "l_shoulder-l_elbow": OPENPOSE_COLORS[3], // 10: Yellow
    "l_elbow-l_wrist": OPENPOSE_COLORS[4],    // 11: Yellow-green
    
    // Right leg (indices 12-13) - GREEN/CYAN side
    "r_hip-r_knee": OPENPOSE_COLORS[7],       // 12: Green-cyan
    "r_knee-r_ankle": OPENPOSE_COLORS[8],     // 13: Cyan-green
    
    // Left leg (indices 14-15) - CYAN/BLUE side
    "l_hip-l_knee": OPENPOSE_COLORS[10],      // 14: Cyan-blue
    "l_knee-l_ankle": OPENPOSE_COLORS[11],    // 15: Light blue
};

/**
 * Get color for a specific bone connection
 * @param {string} joint1 - First joint name
 * @param {string} joint2 - Second joint name
 * @param {number} boneIndex - Fallback bone index if direct lookup fails
 * @returns {string} Hex color code
 */
export function getBoneColor(joint1, joint2, boneIndex) {
    // Try direct lookup
    const key1 = `${joint1}-${joint2}`;
    if (BONE_COLORS[key1]) {
        return BONE_COLORS[key1];
    }
    
    // Try reverse lookup
    const key2 = `${joint2}-${joint1}`;
    if (BONE_COLORS[key2]) {
        return BONE_COLORS[key2];
    }
    
    // Fallback to generated colors if not found
    return generateColorFromIndex(boneIndex);
}

/**
 * Generate color from bone index (fallback)
 * @param {number} index - Bone index
 * @returns {string} Hex color code
 */
function generateColorFromIndex(index) {
    return OPENPOSE_COLORS[index % OPENPOSE_COLORS.length];
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
