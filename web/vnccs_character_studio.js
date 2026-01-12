import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// WebGL Shader Sources (Phong Shading)
const VS_SOURCE = `
    attribute vec3 position;
    attribute vec3 normal;
    uniform mat4 uModelViewMatrix;
    uniform mat4 uProjectionMatrix;
    uniform mat4 uNormalMatrix;
    varying highp vec3 vNormal;
    varying highp vec3 vViewPosition;
    
    void main(void) {
        vec4 pos = uModelViewMatrix * vec4(position, 1.0);
        gl_Position = uProjectionMatrix * pos;
        vViewPosition = pos.xyz;
        vNormal = normalize((uNormalMatrix * vec4(normal, 0.0)).xyz);
    }
`;

const FS_SOURCE = `
    precision highp float;
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    
    void main(void) {
        // Light Setup
        vec3 lightDir = normalize(vec3(0.5, 0.8, 1.0));
        vec3 normal = normalize(vNormal);
        
        // Ambient
        vec3 ambient = vec3(0.3, 0.25, 0.25);
        
        // Diffuse
        float diff = max(dot(normal, lightDir), 0.0);
        vec3 diffuse = vec3(0.8, 0.7, 0.6) * diff;
        
        // Specular
        vec3 viewDir = normalize(-vViewPosition);
        vec3 reflectDir = reflect(-lightDir, normal);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
        vec3 specular = vec3(0.2) * spec;
        
        gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
    }
`;

function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) return null;
    return shader;
}

/**
 * Offscreen 3D Model Viewer using WebGL
 */
class OffscreenModelViewer {
    constructor(width, height) {
        this.canvas = document.createElement("canvas");
        this.canvas.width = width;
        this.canvas.height = height;
        this.gl = this.canvas.getContext('webgl') || this.canvas.getContext('experimental-webgl');
        this.program = null;
        this.buffers = {};
        this.params = { rotX: 0, rotY: 0, zoomOffset: 0.0 };
        this.dirty = true;
        this.indicesCount = 0;
        this.bounds = { center: [0, 9, 0], size: [0, 18, 0] };

        if (this.gl) this.initShaders();
    }

    initShaders() {
        const gl = this.gl;
        const vs = createShader(gl, gl.VERTEX_SHADER, VS_SOURCE);
        const fs = createShader(gl, gl.FRAGMENT_SHADER, FS_SOURCE);
        this.program = gl.createProgram();
        gl.attachShader(this.program, vs);
        gl.attachShader(this.program, fs);
        gl.linkProgram(this.program);

        this.attribLoc = {
            vertex: gl.getAttribLocation(this.program, 'position'),
            normal: gl.getAttribLocation(this.program, 'normal'),
        };
        this.unifLoc = {
            uModelView: gl.getUniformLocation(this.program, 'uModelViewMatrix'),
            uProj: gl.getUniformLocation(this.program, 'uProjectionMatrix'),
            uNormal: gl.getUniformLocation(this.program, 'uNormalMatrix'),
        };
    }

    updateGeometry(vertices, indices, normals) {
        if (!this.gl || !vertices || !indices) return;
        const gl = this.gl;

        // Vertices buffer
        if (!this.buffers.position) this.buffers.position = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

        // Indices buffer
        if (!this.buffers.indices) this.buffers.indices = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.indices);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Int32Array(indices), gl.STATIC_DRAW);

        const ext = gl.getExtension('OES_element_index_uint');
        this.indexType = ext ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT;

        // Normals buffer
        let normalData;
        if (normals && normals.length === vertices.length) {
            normalData = new Float32Array(normals);
        } else {
            normalData = new Float32Array(vertices.length);
        }

        if (!this.buffers.normal) this.buffers.normal = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.normal);
        gl.bufferData(gl.ARRAY_BUFFER, normalData, gl.STATIC_DRAW);

        this.indicesCount = indices.length;

        // Calculate bounds
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (let i = 0; i < vertices.length; i += 3) {
            const x = vertices[i];
            const y = vertices[i + 1];
            const z = vertices[i + 2];
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
            if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
        }

        this.bounds = {
            center: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2],
            size: [maxX - minX, maxY - minY, maxZ - minZ],
            min: [minX, minY, minZ],
            max: [maxX, maxY, maxZ]
        };

        this.dirty = true;
    }

    render() {
        if (!this.dirty || !this.gl) return;
        const gl = this.gl;
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0.85, 0.85, 0.85, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);

        gl.useProgram(this.program);

        const aspect = this.canvas.width / this.canvas.height;
        const fov = 45 * Math.PI / 180;
        const zNear = 0.1, zFar = 1000.0;
        const f = 1.0 / Math.tan(fov / 2);

        const pMat = new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (zFar + zNear) / (zNear - zFar), -1,
            0, 0, (2 * zFar * zNear) / (zNear - zFar), 0
        ]);

        // Auto camera logic
        const modelH = Math.max(this.bounds.size[1], 5.0);
        const margin = 1.15;
        const autoDist = (modelH * margin) * f / 2;
        const finalDist = autoDist + this.params.zoomOffset;

        // Rotation
        const cx = Math.cos(this.params.rotX), sx = Math.sin(this.params.rotX);
        const cy = Math.cos(this.params.rotY), sy = Math.sin(this.params.rotY);

        const tx = 0;
        const ty = -this.bounds.center[1];
        const tz = 0;

        // Rotation columns
        const r00 = cy, r01 = 0, r02 = sy;
        const r10 = sx * sy, r11 = cx, r12 = -sx * cy;
        const r20 = -cx * sy, r21 = sx, r22 = cx * cy;

        // Translation vector
        const tX = r00 * tx + r10 * ty + r20 * tz;
        const tY = r01 * tx + r11 * ty + r21 * tz;
        const tZ = r02 * tx + r12 * ty + r22 * tz - finalDist;

        const mvMat = new Float32Array([
            r00, r10, r20, 0,
            r01, r11, r21, 0,
            r02, r12, r22, 0,
            tX, tY, tZ, 1
        ]);

        gl.uniformMatrix4fv(this.unifLoc.uProj, false, pMat);
        gl.uniformMatrix4fv(this.unifLoc.uModelView, false, mvMat);

        gl.uniformMatrix4fv(this.unifLoc.uNormal, false, new Float32Array([
            r00, r10, r20, 0,
            r01, r11, r21, 0,
            r02, r12, r22, 0,
            0, 0, 0, 1
        ]));

        if (this.buffers.position && this.indicesCount > 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.position);
            gl.vertexAttribPointer(this.attribLoc.vertex, 3, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(this.attribLoc.vertex);

            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.normal);
            gl.vertexAttribPointer(this.attribLoc.normal, 3, gl.FLOAT, false, 0, 0);
            gl.enableVertexAttribArray(this.attribLoc.normal);

            gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buffers.indices);
            gl.drawElements(gl.TRIANGLES, this.indicesCount, this.indexType, 0);
        }
        this.dirty = false;
    }

    getCanvas() {
        return this.canvas;
    }
}

/**
 * VNCCS Character Studio Widget Extension
 */
app.registerExtension({
    name: "VNCCS.CharacterStudio",
    async beforeRegisterNodeDef(nodeType, nodeData, app_local) {
        if (nodeData.name === "VNCCS_CharacterStudio") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Hide default widgets
                if (this.widgets) {
                    for (const w of this.widgets) {
                        w.computeSize = () => [0, -4];
                        w.draw = function (ctx) { };
                    }
                }

                // Initialize state from widgets
                this.state = {
                    age: this.widgets.find(w => w.name == "age")?.value || 25,
                    gender: this.widgets.find(w => w.name == "gender")?.value || 0.5,
                    weight: this.widgets.find(w => w.name == "weight")?.value || 0.5,
                    muscle: this.widgets.find(w => w.name == "muscle")?.value || 0.5,
                    height: this.widgets.find(w => w.name == "height")?.value || 0.5,
                    breast: this.widgets.find(w => w.name == "breast_size")?.value || 0.5,
                    genital: this.widgets.find(w => w.name == "genital_size")?.value || 0.5,
                };

                this.setSize([800, 600]);
                this.viewer = new OffscreenModelViewer(400, 500);
                this.activeSlider = null;
                this.lastError = null;
                this.pendingUpdate = false;

                // Throttled mesh update
                this.updateMesh = () => {
                    if (this.pendingUpdate) return;

                    this.pendingUpdate = true;
                    this.lastError = null;

                    api.fetchApi("/vnccs/character_studio/update_preview", {
                        method: "POST",
                        body: JSON.stringify({
                            age: this.state.age,
                            gender: this.state.gender,
                            weight: this.state.weight,
                            muscle: this.state.muscle,
                            height: this.state.height,
                            breast_size: this.state.breast,
                            genital_size: this.state.genital
                        })
                    }).then(r => {
                        if (!r.ok) throw new Error("HTTP " + r.status);
                        return r.json();
                    })
                        .then(data => {
                            this.pendingUpdate = false;
                            if (data.status === "success" && this.viewer) {
                                this.viewer.updateGeometry(data.vertices, data.indices, data.normals);
                                this.setDirtyCanvas(true, true);
                            } else if (data.status === "error") {
                                throw new Error(data.message || "Unknown API Error");
                            }
                        })
                        .catch(err => {
                            this.pendingUpdate = false;
                            console.error("VNCCS CharacterStudio Error:", err);
                        });
                };

                this.updateMesh();
                return r;
            };

            // Sync UI with widgets after graph load
            nodeType.prototype.onConfigure = function () {
                if (!this.state) return;

                const wMap = {
                    "age": "age", "gender": "gender", "weight": "weight",
                    "muscle": "muscle", "height": "height",
                    "breast": "breast_size", "genital": "genital_size"
                };

                for (const [key, wName] of Object.entries(wMap)) {
                    const w = this.widgets.find(x => x.name === wName);
                    if (w) {
                        this.state[key] = w.value;
                    }
                }
                if (this.updateMesh) this.updateMesh();
            };

            // Enforce minimum size
            nodeType.prototype.onResize = function (size) {
                const MIN_W = 800;
                const MIN_H = 600;

                if (size[0] < MIN_W) size[0] = MIN_W;
                if (size[1] < MIN_H) size[1] = MIN_H;
            };

            // Draw custom UI
            nodeType.prototype.onDrawBackground = function (ctx) {
                if (this.flags.collapsed) return;

                const w = this.size[0];
                const h = this.size[1];

                const colW = Math.min(250, w * 0.25);
                const margin = 10;

                // Background
                ctx.fillStyle = "#FDF6E3";
                ctx.beginPath();
                ctx.roundRect(0, 0, w, h, 10);
                ctx.fill();

                const drawSlider = (x, y, sw, sh, label, value, min, max, key) => {
                    if (!this.sliderRegions) this.sliderRegions = [];
                    this.sliderRegions.push({ x, y, w: sw, h: sh, min, max, key });

                    ctx.fillStyle = "black";
                    ctx.font = "bold 12px Arial";
                    ctx.fillText(label, x, y - 4);

                    ctx.fillStyle = "#ddd";
                    ctx.beginPath();
                    ctx.roundRect(x, y, sw, sh, 4);
                    ctx.fill();

                    const pct = (value - min) / (max - min);
                    ctx.fillStyle = "#666";
                    ctx.beginPath();
                    ctx.roundRect(x, y, sw * pct, sh, 4);
                    ctx.fill();

                    ctx.fillStyle = "#333";
                    ctx.beginPath();
                    ctx.arc(x + sw * pct, y + sh / 2, 6, 0, Math.PI * 2);
                    ctx.fill();

                    ctx.fillStyle = "black";
                    ctx.textAlign = "right";
                    ctx.fillText(key === "age" ? Math.round(value) : value.toFixed(2), x + sw, y - 4);
                    ctx.textAlign = "left";
                };

                this.sliderRegions = [];
                this.viewerRegion = null;

                let y = 40;
                const panelSpacing = 15;

                // Left Column
                const leftX = margin;

                // Profile panel (Age/Gender)
                const h1 = 100;
                ctx.fillStyle = "rgba(0,0,0,0.05)";
                ctx.strokeStyle = "#ccc";
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.roundRect(leftX, y, colW, h1, 8);
                ctx.fill();
                ctx.stroke();
                ctx.fillStyle = "black";
                ctx.font = "bold 14px Arial";
                ctx.fillText("Profile:", leftX, y - 5);
                drawSlider(leftX + 10, y + 25, colW - 20, 8, "Age", this.state.age, 1, 90, "age");
                drawSlider(leftX + 10, y + 60, colW - 20, 8, "Gender (F-M)", this.state.gender, 0, 1, "gender");
                y += h1 + panelSpacing + 20;

                // Body panel
                const h2 = 140;
                ctx.fillStyle = "rgba(0,0,0,0.05)";
                ctx.beginPath();
                ctx.roundRect(leftX, y, colW, h2, 8);
                ctx.fill();
                ctx.stroke();
                ctx.fillStyle = "black";
                ctx.fillText("Body:", leftX, y - 5);
                drawSlider(leftX + 10, y + 25, colW - 20, 8, "Weight", this.state.weight, 0, 1, "weight");
                drawSlider(leftX + 10, y + 60, colW - 20, 8, "Muscle", this.state.muscle, 0, 1, "muscle");
                drawSlider(leftX + 10, y + 95, colW - 20, 8, "Height", this.state.height, 0, 1, "height");
                y += h2 + panelSpacing + 20;

                // Details panel
                const h3 = 100;
                ctx.fillStyle = "rgba(0,0,0,0.05)";
                ctx.beginPath();
                ctx.roundRect(leftX, y, colW, h3, 8);
                ctx.fill();
                ctx.stroke();
                ctx.fillStyle = "black";
                ctx.fillText("Details:", leftX, y - 5);
                drawSlider(leftX + 10, y + 25, colW - 20, 8, "Breast Size", this.state.breast, 0, 1, "breast");
                drawSlider(leftX + 10, y + 60, colW - 20, 8, "Genital Size", this.state.genital, 0, 1, "genital");

                // Right Column (placeholders for future features)
                const rightX = w - colW - margin;
                y = 40;

                ctx.strokeRect(rightX, y, colW, 100);
                ctx.fillText("Hair (TODO)", rightX, y - 5);
                y += 120;

                ctx.strokeRect(rightX, y, colW, h - y - 20);
                ctx.fillText("Face (TODO)", rightX, y - 5);

                // Center Viewer
                const centerX = colW + (margin * 2);
                const centerY = 40;
                const centerW = w - (colW * 2) - (margin * 4);
                const centerH = h - 60;

                this.viewerRegion = { x: centerX, y: centerY, w: centerW, h: centerH };

                if (this.viewer) {
                    if (this.viewer.canvas.width !== Math.floor(centerW) ||
                        this.viewer.canvas.height !== Math.floor(centerH)) {
                        this.viewer.canvas.width = Math.floor(centerW);
                        this.viewer.canvas.height = Math.floor(centerH);
                        this.viewer.dirty = true;
                    }

                    this.viewer.render();
                    ctx.save();
                    ctx.beginPath();
                    ctx.roundRect(centerX, centerY, centerW, centerH, 10);
                    ctx.clip();

                    if (this.viewer.canvas) {
                        ctx.drawImage(this.viewer.canvas, centerX, centerY, centerW, centerH);
                    }

                    if (this.lastError) {
                        ctx.fillStyle = "red";
                        ctx.textAlign = "center";
                        ctx.fillText("Error: " + this.lastError, centerX + centerW / 2, centerY + centerH / 2);
                    } else if (this.viewer.indicesCount === 0) {
                        ctx.fillStyle = "#999";
                        ctx.textAlign = "center";
                        ctx.fillText("Model Loading...", centerX + centerW / 2, centerY + centerH / 2);
                    }
                    ctx.restore();
                    ctx.lineWidth = 3;
                    ctx.strokeStyle = "#1a1a1a";
                    ctx.strokeRect(centerX, centerY, centerW, centerH);
                }
            };

            // Mouse handlers
            nodeType.prototype.onMouseDown = function (event, pos, graphCanvas) {
                // Check sliders
                if (this.sliderRegions) {
                    for (const r of this.sliderRegions) {
                        if (pos[0] >= r.x && pos[0] <= r.x + r.w &&
                            pos[1] >= r.y - 10 && pos[1] <= r.y + r.h + 10) {
                            this.activeSlider = r;
                            this.onMouseMove(event, pos, graphCanvas);
                            return true;
                        }
                    }
                }

                // Check viewer
                if (this.viewerRegion &&
                    pos[0] > this.viewerRegion.x && pos[0] < this.viewerRegion.x + this.viewerRegion.w &&
                    pos[1] > this.viewerRegion.y && pos[1] < this.viewerRegion.y + this.viewerRegion.h) {
                    this.capturingInput = true;
                    this.lastMousePos = [pos[0], pos[1]];
                    return true;
                }
            };

            nodeType.prototype.onMouseMove = function (event, pos, graphCanvas) {
                // Release if no buttons pressed
                if (event.buttons === 0) {
                    this.activeSlider = null;
                    this.capturingInput = false;
                }

                if (this.activeSlider) {
                    const r = this.activeSlider;
                    let val = (pos[0] - r.x) / r.w;
                    val = Math.max(0, Math.min(1, val));
                    const actualVal = r.min + val * (r.max - r.min);
                    this.state[r.key] = actualVal;

                    const wMap = {
                        "age": "age", "gender": "gender", "weight": "weight",
                        "muscle": "muscle", "height": "height",
                        "breast": "breast_size", "genital": "genital_size"
                    };
                    if (wMap[r.key]) {
                        const widget = this.widgets.find(w => w.name === wMap[r.key]);
                        if (widget) widget.value = actualVal;
                    }
                    this.setDirtyCanvas(true, true);
                    this.updateMesh();
                } else if (this.capturingInput && this.viewer) {
                    const dx = pos[0] - this.lastMousePos[0];
                    this.viewer.params.rotY += dx * 0.01;
                    this.viewer.dirty = true;
                    this.lastMousePos = [pos[0], pos[1]];
                    this.setDirtyCanvas(true, true);
                }
            };

            nodeType.prototype.onMouseUp = function (event, pos, graphCanvas) {
                this.activeSlider = null;
                this.capturingInput = false;
            };

            nodeType.prototype.onWheel = function (event, pos, graphCanvas) {
                if (this.viewer) {
                    this.viewer.params.zoom += event.deltaY * 0.05;
                    this.viewer.dirty = true;
                    this.setDirtyCanvas(true, true);
                }
            };
        }
    }
});
