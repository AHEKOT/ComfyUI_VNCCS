// Advanced rendering utilities for depth, normal, and canny edge detection
// Based on sd-webui-3d-open-pose-editor capture methods

export class AdvancedRenderer {
    constructor(THREE, scene, camera, renderer) {
        this.THREE = THREE;
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        
        // Create offscreen canvas for captures
        this.captureCanvas = document.createElement('canvas');
        this.captureContext = this.captureCanvas.getContext('2d', { willReadFrequently: true });
        
        this.initDepthMaterial();
        this.initNormalMaterial();
    }

    initDepthMaterial() {
        // Depth material - renders Z-depth as grayscale
        this.depthMaterial = new this.THREE.ShaderMaterial({
            vertexShader: `
                varying vec4 vPosition;
                void main() {
                    vPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_Position = projectionMatrix * vPosition;
                }
            `,
            fragmentShader: `
                varying vec4 vPosition;
                uniform float cameraNear;
                uniform float cameraFar;
                
                void main() {
                    float depth = -vPosition.z;
                    float normalized = (depth - cameraNear) / (cameraFar - cameraNear);
                    normalized = clamp(normalized, 0.0, 1.0);
                    
                    // Invert so closer = whiter
                    float inverted = 1.0 - normalized;
                    gl_FragColor = vec4(vec3(inverted), 1.0);
                }
            `,
            uniforms: {
                cameraNear: { value: this.camera.near },
                cameraFar: { value: this.camera.far }
            }
        });
    }

    initNormalMaterial() {
        // Normal material - renders surface normals as RGB
        this.normalMaterial = new this.THREE.MeshNormalMaterial();
    }

    captureDepth(width, height) {
        // Save original materials
        const originalMaterials = new Map();
        this.scene.traverse((obj) => {
            if (obj.isMesh && obj.material) {
                originalMaterials.set(obj, obj.material);
                obj.material = this.depthMaterial;
            }
        });

        // Update uniforms
        this.depthMaterial.uniforms.cameraNear.value = this.camera.near;
        this.depthMaterial.uniforms.cameraFar.value = this.camera.far;

        // Render
        const originalSize = new this.THREE.Vector2();
        this.renderer.getSize(originalSize);
        this.renderer.setSize(width, height, false);
        
        this.renderer.setClearColor(0x000000, 1);
        this.renderer.render(this.scene, this.camera);
        
        const dataURL = this.renderer.domElement.toDataURL('image/png');

        // Restore
        this.renderer.setSize(originalSize.x, originalSize.y, false);
        this.scene.traverse((obj) => {
            if (originalMaterials.has(obj)) {
                obj.material = originalMaterials.get(obj);
            }
        });

        return dataURL;
    }

    captureNormal(width, height) {
        // Save original materials
        const originalMaterials = new Map();
        this.scene.traverse((obj) => {
            if (obj.isMesh && obj.material) {
                originalMaterials.set(obj, obj.material);
                obj.material = this.normalMaterial;
            }
        });

        // Render
        const originalSize = new this.THREE.Vector2();
        this.renderer.getSize(originalSize);
        this.renderer.setSize(width, height, false);
        
        this.renderer.setClearColor(0x000000, 1);
        this.renderer.render(this.scene, this.camera);
        
        const dataURL = this.renderer.domElement.toDataURL('image/png');

        // Restore
        this.renderer.setSize(originalSize.x, originalSize.y, false);
        this.scene.traverse((obj) => {
            if (originalMaterials.has(obj)) {
                obj.material = originalMaterials.get(obj);
            }
        });

        return dataURL;
    }

    captureCanny(width, height, lowThreshold = 50, highThreshold = 100) {
        // First capture normal render
        const originalSize = new this.THREE.Vector2();
        this.renderer.getSize(originalSize);
        this.renderer.setSize(width, height, false);
        
        this.renderer.setClearColor(0x000000, 1);
        this.renderer.render(this.scene, this.camera);
        
        // Get image data
        this.captureCanvas.width = width;
        this.captureCanvas.height = height;
        this.captureContext.drawImage(this.renderer.domElement, 0, 0);
        const imageData = this.captureContext.getImageData(0, 0, width, height);
        
        // Apply Sobel edge detection
        const edges = this.sobelEdgeDetection(imageData, lowThreshold, highThreshold);
        
        // Put back to canvas
        this.captureContext.putImageData(edges, 0, 0);
        const dataURL = this.captureCanvas.toDataURL('image/png');

        // Restore
        this.renderer.setSize(originalSize.x, originalSize.y, false);

        return dataURL;
    }

    sobelEdgeDetection(imageData, lowThreshold, highThreshold) {
        const width = imageData.width;
        const height = imageData.height;
        const data = imageData.data;
        
        // Convert to grayscale
        const gray = new Uint8Array(width * height);
        for (let i = 0; i < data.length; i += 4) {
            const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
            gray[i / 4] = avg;
        }

        // Sobel kernels
        const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];

        const gradient = new Uint8Array(width * height);
        
        // Apply Sobel operator
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                let gx = 0;
                let gy = 0;
                
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++) {
                        const idx = (y + ky) * width + (x + kx);
                        const kernelIdx = (ky + 1) * 3 + (kx + 1);
                        gx += gray[idx] * sobelX[kernelIdx];
                        gy += gray[idx] * sobelY[kernelIdx];
                    }
                }
                
                const magnitude = Math.sqrt(gx * gx + gy * gy);
                const idx = y * width + x;
                gradient[idx] = magnitude;
            }
        }

        // Apply thresholding with hysteresis
        const result = new ImageData(width, height);
        for (let i = 0; i < gradient.length; i++) {
            const value = gradient[i];
            let edge = 0;
            
            if (value >= highThreshold) {
                edge = 255;
            } else if (value >= lowThreshold) {
                // Check if connected to strong edge
                const x = i % width;
                const y = Math.floor(i / width);
                
                for (let dy = -1; dy <= 1; dy++) {
                    for (let dx = -1; dx <= 1; dx++) {
                        const nx = x + dx;
                        const ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            const nIdx = ny * width + nx;
                            if (gradient[nIdx] >= highThreshold) {
                                edge = 255;
                                break;
                            }
                        }
                    }
                    if (edge) break;
                }
            }
            
            const pixelIdx = i * 4;
            result.data[pixelIdx] = edge;
            result.data[pixelIdx + 1] = edge;
            result.data[pixelIdx + 2] = edge;
            result.data[pixelIdx + 3] = 255;
        }

        return result;
    }

    captureOpenPose(width, height) {
        // Standard OpenPose render
        const originalSize = new this.THREE.Vector2();
        this.renderer.getSize(originalSize);
        this.renderer.setSize(width, height, false);
        
        this.renderer.setClearColor(0x000000, 1);
        this.renderer.render(this.scene, this.camera);
        
        const dataURL = this.renderer.domElement.toDataURL('image/png');

        this.renderer.setSize(originalSize.x, originalSize.y, false);

        return dataURL;
    }

    makeAllImages(width, height) {
        return {
            pose: this.captureOpenPose(width, height),
            depth: this.captureDepth(width, height),
            normal: this.captureNormal(width, height),
            canny: this.captureCanny(width, height)
        };
    }

    dispose() {
        if (this.depthMaterial) {
            this.depthMaterial.dispose();
        }
        if (this.normalMaterial) {
            this.normalMaterial.dispose();
        }
    }
}

export function createAdvancedRenderer(THREE, scene, camera, renderer) {
    return new AdvancedRenderer(THREE, scene, camera, renderer);
}
