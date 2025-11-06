// 3D Models loader for hands and feet
// Based on sd-webui-3d-open-pose-editor models.ts

// Use VNCCS API endpoint to serve FBX models
const getModelPath = (filename) => {
    return `/vnccs/models/${filename}`;
};

export const MODEL_CONFIGS = {
    hand: {
        url: getModelPath('hand.fbx'),
        meshName: 'shoupolySurface1',
        bonePrefix: 'shoujoint',
        scale: 1.0
    },
    foot: {
        url: getModelPath('foot.fbx'),
        meshName: 'FootObject',
        bonePrefix: 'FootBone',
        scale: 1.0
    }
};

export const EXTREMITIES_MAPPING = {
    left_hand: 'hand',
    right_hand: 'hand',
    left_foot: 'foot',
    right_foot: 'foot'
};

let handObject = null;
let footObject = null;

export async function loadHandModel(FBXLoader, THREE) {
    if (handObject) return handObject;

    const config = MODEL_CONFIGS.hand;
    const loader = new FBXLoader();

    return new Promise((resolve, reject) => {
        console.log('[VNCCS Models] Loading hand model from:', config.url);
        loader.load(
            config.url,
            (fbx) => {
                console.log('[VNCCS Models] Hand FBX loaded, processing...');
                fbx.scale.multiplyScalar(config.scale);

                // Find the mesh
                let mesh = null;
                let meshCount = 0;
                fbx.traverse((child) => {
                    if (child.isMesh) {
                        meshCount++;
                        console.log(`[VNCCS Models] Found mesh: ${child.name}`);
                        if (child.name === config.meshName) {
                            mesh = child;
                        }
                    }
                });

                console.log(`[VNCCS Models] Total meshes found: ${meshCount}`);

                if (!mesh) {
                    console.error(`[VNCCS Models] Mesh ${config.meshName} not found in FBX`);
                    reject(new Error(`Mesh ${config.meshName} not found`));
                    return;
                }

                console.log('[VNCCS Models] Hand mesh found, setting material...');

                // Set material
                mesh.material = new THREE.MeshPhongMaterial({
                    color: 0xffd0a8,
                    shininess: 30
                });

                // Add skeleton support (no visual bone points)
                if (mesh.skeleton) {
                    console.log(`[VNCCS Models] Hand has ${mesh.skeleton.bones.length} bones`);
                    
                    // Add mask for depth rendering
                    const mask = new THREE.Mesh(
                        new THREE.CylinderGeometry(1, 1, 0.4, 32),
                        new THREE.MeshBasicMaterial({ color: 0x000000 })
                    );
                    mask.name = 'hand_mask';
                    mask.visible = false;
                    mask.rotateZ(Math.PI / 2);
                    mesh.skeleton.bones[0].add(mask);
                }

                handObject = fbx;
                console.log('[VNCCS Models] ✓ Hand model loaded successfully');
                resolve(fbx);
            },
            (progress) => {
                console.log(`Loading hand model: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
            },
            (error) => {
                console.error('[VNCCS Models] Failed to load hand model:', error);
                console.error('[VNCCS Models] URL was:', config.url);
                reject(error);
            }
        );
    });
}

export async function loadFootModel(FBXLoader, THREE) {
    if (footObject) return footObject;

    const config = MODEL_CONFIGS.foot;
    const loader = new FBXLoader();

    return new Promise((resolve, reject) => {
        console.log('[VNCCS Models] Loading foot model from:', config.url);
        loader.load(
            config.url,
            (fbx) => {
                console.log('[VNCCS Models] Foot FBX loaded, processing...');
                fbx.scale.multiplyScalar(config.scale);

                // Find the mesh
                let mesh = null;
                let meshCount = 0;
                fbx.traverse((child) => {
                    if (child.isMesh) {
                        meshCount++;
                        console.log(`[VNCCS Models] Found mesh: ${child.name}`);
                        if (child.name === config.meshName) {
                            mesh = child;
                        }
                    }
                });

                console.log(`[VNCCS Models] Total meshes found: ${meshCount}`);

                if (!mesh) {
                    console.error(`[VNCCS Models] Mesh ${config.meshName} not found in FBX`);
                    reject(new Error(`Mesh ${config.meshName} not found`));
                    return;
                }

                console.log('[VNCCS Models] Foot mesh found, setting material...');

                // Set material
                mesh.material = new THREE.MeshPhongMaterial({
                    color: 0xffd0a8,
                    shininess: 30
                });

                // Add skeleton support (no visual bone points)
                if (mesh.skeleton) {
                    console.log(`[VNCCS Models] Foot has ${mesh.skeleton.bones.length} bones`);
                    
                    // Add mask for depth rendering
                    const mask = new THREE.Mesh(
                        new THREE.CylinderGeometry(0.8, 0.8, 0.4, 32),
                        new THREE.MeshBasicMaterial({ color: 0x000000 })
                    );
                    mask.name = 'foot_mask';
                    mask.visible = false;
                    mask.rotateX(Math.PI / 2);
                    mesh.skeleton.bones[0].add(mask);
                }

                footObject = fbx;
                console.log('[VNCCS Models] ✓ Foot model loaded successfully');
                resolve(fbx);
            },
            (progress) => {
                console.log(`Loading foot model: ${(progress.loaded / progress.total * 100).toFixed(1)}%`);
            },
            (error) => {
                console.error('[VNCCS Models] Failed to load foot model:', error);
                console.error('[VNCCS Models] URL was:', config.url);
                reject(error);
            }
        );
    });
}

export function cloneHandModel(THREE, SkeletonUtils) {
    console.log('[VNCCS Models] cloneHandModel called, handObject:', handObject ? 'exists' : 'NULL');
    if (!handObject) {
        console.error('[VNCCS Models] Cannot clone hand - handObject is null!');
        return null;
    }
    console.log('[VNCCS Models] SkeletonUtils:', SkeletonUtils ? 'available' : 'missing');
    const cloned = SkeletonUtils ? SkeletonUtils.clone(handObject) : handObject.clone();
    console.log('[VNCCS Models] Hand cloned:', cloned ? 'success' : 'FAILED');
    return cloned;
}

export function cloneFootModel(THREE, SkeletonUtils) {
    console.log('[VNCCS Models] cloneFootModel called, footObject:', footObject ? 'exists' : 'NULL');
    if (!footObject) {
        console.error('[VNCCS Models] Cannot clone foot - footObject is null!');
        return null;
    }
    console.log('[VNCCS Models] SkeletonUtils:', SkeletonUtils ? 'available' : 'missing');
    const cloned = SkeletonUtils ? SkeletonUtils.clone(footObject) : footObject.clone();
    console.log('[VNCCS Models] Foot cloned:', cloned ? 'success' : 'FAILED');
    return cloned;
}

export function isExtremityBone(boneName) {
    return boneName.startsWith(MODEL_CONFIGS.hand.bonePrefix) || 
           boneName.startsWith(MODEL_CONFIGS.foot.bonePrefix);
}

export { handObject, footObject };
