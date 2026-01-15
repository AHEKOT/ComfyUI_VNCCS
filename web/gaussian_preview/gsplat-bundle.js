var GSPLAT = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // gsplat-entry.js
  var gsplat_entry_exports = {};
  __export(gsplat_entry_exports, {
    Camera: () => wt,
    CameraData: () => Rt,
    Color32: () => dt,
    FPSControls: () => Gt,
    FadeInPass: () => Nt,
    IntersectionTester: () => Kt,
    Loader: () => Tt,
    Matrix3: () => _,
    Matrix4: () => K,
    Object3D: () => st,
    OrbitControls: () => Lt,
    PLYLoader: () => bt,
    Plane: () => vt,
    Quaternion: () => y,
    RenderData: () => Ut,
    RenderProgram: () => Ct,
    Scene: () => kt,
    ShaderPass: () => _t,
    ShaderProgram: () => lt,
    Splat: () => X,
    SplatData: () => Y,
    Splatv: () => tt,
    SplatvData: () => ot,
    SplatvLoader: () => xt,
    Vector3: () => R,
    Vector4: () => z,
    VideoRenderProgram: () => Yt,
    WebGLRenderer: () => Mt
  });

  // node_modules/gsplat/dist/index.es.js
  var import_meta = {};
  var R = class _R {
    constructor(t = 0, n = 0, i = 0) {
      this.x = t, this.y = n, this.z = i;
    }
    equals(t) {
      return !(this.x !== t.x || this.y !== t.y || this.z !== t.z);
    }
    add(t) {
      return typeof t == "number" ? new _R(this.x + t, this.y + t, this.z + t) : new _R(this.x + t.x, this.y + t.y, this.z + t.z);
    }
    subtract(t) {
      return typeof t == "number" ? new _R(this.x - t, this.y - t, this.z - t) : new _R(this.x - t.x, this.y - t.y, this.z - t.z);
    }
    multiply(t) {
      return typeof t == "number" ? new _R(this.x * t, this.y * t, this.z * t) : t instanceof _R ? new _R(this.x * t.x, this.y * t.y, this.z * t.z) : new _R(
        this.x * t.buffer[0] + this.y * t.buffer[4] + this.z * t.buffer[8] + t.buffer[12],
        this.x * t.buffer[1] + this.y * t.buffer[5] + this.z * t.buffer[9] + t.buffer[13],
        this.x * t.buffer[2] + this.y * t.buffer[6] + this.z * t.buffer[10] + t.buffer[14]
      );
    }
    divide(t) {
      return typeof t == "number" ? new _R(this.x / t, this.y / t, this.z / t) : new _R(this.x / t.x, this.y / t.y, this.z / t.z);
    }
    cross(t) {
      const n = this.y * t.z - this.z * t.y, i = this.z * t.x - this.x * t.z, e = this.x * t.y - this.y * t.x;
      return new _R(n, i, e);
    }
    dot(t) {
      return this.x * t.x + this.y * t.y + this.z * t.z;
    }
    lerp(t, n) {
      return new _R(this.x + (t.x - this.x) * n, this.y + (t.y - this.y) * n, this.z + (t.z - this.z) * n);
    }
    min(t) {
      return new _R(Math.min(this.x, t.x), Math.min(this.y, t.y), Math.min(this.z, t.z));
    }
    max(t) {
      return new _R(Math.max(this.x, t.x), Math.max(this.y, t.y), Math.max(this.z, t.z));
    }
    getComponent(t) {
      switch (t) {
        case 0:
          return this.x;
        case 1:
          return this.y;
        case 2:
          return this.z;
        default:
          throw new Error(`Invalid component index: ${t}`);
      }
    }
    minComponent() {
      return this.x < this.y && this.x < this.z ? 0 : this.y < this.z ? 1 : 2;
    }
    maxComponent() {
      return this.x > this.y && this.x > this.z ? 0 : this.y > this.z ? 1 : 2;
    }
    magnitude() {
      return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
    }
    distanceTo(t) {
      return Math.sqrt((this.x - t.x) ** 2 + (this.y - t.y) ** 2 + (this.z - t.z) ** 2);
    }
    normalize() {
      const t = this.magnitude();
      return new _R(this.x / t, this.y / t, this.z / t);
    }
    flat() {
      return [this.x, this.y, this.z];
    }
    clone() {
      return new _R(this.x, this.y, this.z);
    }
    toString() {
      return `[${this.flat().join(", ")}]`;
    }
    static One(t = 1) {
      return new _R(t, t, t);
    }
  };
  var y = class _y {
    constructor(t = 0, n = 0, i = 0, e = 1) {
      this.x = t, this.y = n, this.z = i, this.w = e;
    }
    equals(t) {
      return !(this.x !== t.x || this.y !== t.y || this.z !== t.z || this.w !== t.w);
    }
    normalize() {
      const t = Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
      return new _y(this.x / t, this.y / t, this.z / t, this.w / t);
    }
    multiply(t) {
      const n = this.w, i = this.x, e = this.y, A = this.z, o = t.w, s = t.x, r = t.y, Q = t.z;
      return new _y(
        n * s + i * o + e * Q - A * r,
        n * r - i * Q + e * o + A * s,
        n * Q + i * r - e * s + A * o,
        n * o - i * s - e * r - A * Q
      );
    }
    inverse() {
      const t = this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
      return new _y(-this.x / t, -this.y / t, -this.z / t, this.w / t);
    }
    apply(t) {
      const n = new _y(t.x, t.y, t.z, 0), i = new _y(-this.x, -this.y, -this.z, this.w), e = this.multiply(n).multiply(i);
      return new R(e.x, e.y, e.z);
    }
    flat() {
      return [this.x, this.y, this.z, this.w];
    }
    clone() {
      return new _y(this.x, this.y, this.z, this.w);
    }
    static FromEuler(t) {
      const n = t.x / 2, i = t.y / 2, e = t.z / 2, A = Math.cos(i), o = Math.sin(i), s = Math.cos(n), r = Math.sin(n), Q = Math.cos(e), I = Math.sin(e);
      return new _y(
        A * r * Q + o * s * I,
        o * s * Q - A * r * I,
        A * s * I - o * r * Q,
        A * s * Q + o * r * I
      );
    }
    toEuler() {
      const t = 2 * (this.w * this.x + this.y * this.z), n = 1 - 2 * (this.x * this.x + this.y * this.y), i = Math.atan2(t, n);
      let e;
      const A = 2 * (this.w * this.y - this.z * this.x);
      Math.abs(A) >= 1 ? e = Math.sign(A) * Math.PI / 2 : e = Math.asin(A);
      const o = 2 * (this.w * this.z + this.x * this.y), s = 1 - 2 * (this.y * this.y + this.z * this.z), r = Math.atan2(o, s);
      return new R(i, e, r);
    }
    static FromMatrix3(t) {
      const n = t.buffer, i = n[0] + n[4] + n[8];
      let e, A, o, s;
      if (i > 0) {
        const r = 0.5 / Math.sqrt(i + 1);
        s = 0.25 / r, e = (n[7] - n[5]) * r, A = (n[2] - n[6]) * r, o = (n[3] - n[1]) * r;
      } else if (n[0] > n[4] && n[0] > n[8]) {
        const r = 2 * Math.sqrt(1 + n[0] - n[4] - n[8]);
        s = (n[7] - n[5]) / r, e = 0.25 * r, A = (n[1] + n[3]) / r, o = (n[2] + n[6]) / r;
      } else if (n[4] > n[8]) {
        const r = 2 * Math.sqrt(1 + n[4] - n[0] - n[8]);
        s = (n[2] - n[6]) / r, e = (n[1] + n[3]) / r, A = 0.25 * r, o = (n[5] + n[7]) / r;
      } else {
        const r = 2 * Math.sqrt(1 + n[8] - n[0] - n[4]);
        s = (n[3] - n[1]) / r, e = (n[2] + n[6]) / r, A = (n[5] + n[7]) / r, o = 0.25 * r;
      }
      return new _y(e, A, o, s);
    }
    static FromAxisAngle(t, n) {
      const i = n / 2, e = Math.sin(i), A = Math.cos(i);
      return new _y(t.x * e, t.y * e, t.z * e, A);
    }
    static LookRotation(t) {
      const n = new R(0, 0, 1), i = n.dot(t);
      if (Math.abs(i - -1) < 1e-6)
        return new _y(0, 1, 0, Math.PI);
      if (Math.abs(i - 1) < 1e-6)
        return new _y();
      const e = Math.acos(i), A = n.cross(t).normalize();
      return _y.FromAxisAngle(A, e);
    }
    toString() {
      return `[${this.flat().join(", ")}]`;
    }
  };
  var Ft = class {
    constructor() {
      const t = /* @__PURE__ */ new Map();
      this.addEventListener = (n, i) => {
        t.has(n) || t.set(n, /* @__PURE__ */ new Set()), t.get(n).add(i);
      }, this.removeEventListener = (n, i) => {
        t.has(n) && t.get(n).delete(i);
      }, this.hasEventListener = (n, i) => t.has(n) ? t.get(n).has(i) : false, this.dispatchEvent = (n) => {
        if (t.has(n.type))
          for (const i of t.get(n.type))
            i(n);
      };
    }
  };
  var K = class _K {
    // prettier-ignore
    constructor(t = 1, n = 0, i = 0, e = 0, A = 0, o = 1, s = 0, r = 0, Q = 0, I = 0, d = 1, a = 0, U = 0, F = 0, g = 0, B = 1) {
      this.buffer = [
        t,
        n,
        i,
        e,
        A,
        o,
        s,
        r,
        Q,
        I,
        d,
        a,
        U,
        F,
        g,
        B
      ];
    }
    equals(t) {
      if (this.buffer.length !== t.buffer.length)
        return false;
      if (this.buffer === t.buffer)
        return true;
      for (let n = 0; n < this.buffer.length; n++)
        if (this.buffer[n] !== t.buffer[n])
          return false;
      return true;
    }
    multiply(t) {
      const n = this.buffer, i = t.buffer;
      return new _K(
        i[0] * n[0] + i[1] * n[4] + i[2] * n[8] + i[3] * n[12],
        i[0] * n[1] + i[1] * n[5] + i[2] * n[9] + i[3] * n[13],
        i[0] * n[2] + i[1] * n[6] + i[2] * n[10] + i[3] * n[14],
        i[0] * n[3] + i[1] * n[7] + i[2] * n[11] + i[3] * n[15],
        i[4] * n[0] + i[5] * n[4] + i[6] * n[8] + i[7] * n[12],
        i[4] * n[1] + i[5] * n[5] + i[6] * n[9] + i[7] * n[13],
        i[4] * n[2] + i[5] * n[6] + i[6] * n[10] + i[7] * n[14],
        i[4] * n[3] + i[5] * n[7] + i[6] * n[11] + i[7] * n[15],
        i[8] * n[0] + i[9] * n[4] + i[10] * n[8] + i[11] * n[12],
        i[8] * n[1] + i[9] * n[5] + i[10] * n[9] + i[11] * n[13],
        i[8] * n[2] + i[9] * n[6] + i[10] * n[10] + i[11] * n[14],
        i[8] * n[3] + i[9] * n[7] + i[10] * n[11] + i[11] * n[15],
        i[12] * n[0] + i[13] * n[4] + i[14] * n[8] + i[15] * n[12],
        i[12] * n[1] + i[13] * n[5] + i[14] * n[9] + i[15] * n[13],
        i[12] * n[2] + i[13] * n[6] + i[14] * n[10] + i[15] * n[14],
        i[12] * n[3] + i[13] * n[7] + i[14] * n[11] + i[15] * n[15]
      );
    }
    clone() {
      const t = this.buffer;
      return new _K(
        t[0],
        t[1],
        t[2],
        t[3],
        t[4],
        t[5],
        t[6],
        t[7],
        t[8],
        t[9],
        t[10],
        t[11],
        t[12],
        t[13],
        t[14],
        t[15]
      );
    }
    determinant() {
      const t = this.buffer;
      return t[12] * t[9] * t[6] * t[3] - t[8] * t[13] * t[6] * t[3] - t[12] * t[5] * t[10] * t[3] + t[4] * t[13] * t[10] * t[3] + t[8] * t[5] * t[14] * t[3] - t[4] * t[9] * t[14] * t[3] - t[12] * t[9] * t[2] * t[7] + t[8] * t[13] * t[2] * t[7] + t[12] * t[1] * t[10] * t[7] - t[0] * t[13] * t[10] * t[7] - t[8] * t[1] * t[14] * t[7] + t[0] * t[9] * t[14] * t[7] + t[12] * t[5] * t[2] * t[11] - t[4] * t[13] * t[2] * t[11] - t[12] * t[1] * t[6] * t[11] + t[0] * t[13] * t[6] * t[11] + t[4] * t[1] * t[14] * t[11] - t[0] * t[5] * t[14] * t[11] - t[8] * t[5] * t[2] * t[15] + t[4] * t[9] * t[2] * t[15] + t[8] * t[1] * t[6] * t[15] - t[0] * t[9] * t[6] * t[15] - t[4] * t[1] * t[10] * t[15] + t[0] * t[5] * t[10] * t[15];
    }
    invert() {
      const t = this.buffer, n = this.determinant();
      if (n === 0)
        throw new Error("Matrix is not invertible.");
      const i = 1 / n;
      return new _K(
        i * (t[5] * t[10] * t[15] - t[5] * t[11] * t[14] - t[9] * t[6] * t[15] + t[9] * t[7] * t[14] + t[13] * t[6] * t[11] - t[13] * t[7] * t[10]),
        i * (-t[1] * t[10] * t[15] + t[1] * t[11] * t[14] + t[9] * t[2] * t[15] - t[9] * t[3] * t[14] - t[13] * t[2] * t[11] + t[13] * t[3] * t[10]),
        i * (t[1] * t[6] * t[15] - t[1] * t[7] * t[14] - t[5] * t[2] * t[15] + t[5] * t[3] * t[14] + t[13] * t[2] * t[7] - t[13] * t[3] * t[6]),
        i * (-t[1] * t[6] * t[11] + t[1] * t[7] * t[10] + t[5] * t[2] * t[11] - t[5] * t[3] * t[10] - t[9] * t[2] * t[7] + t[9] * t[3] * t[6]),
        i * (-t[4] * t[10] * t[15] + t[4] * t[11] * t[14] + t[8] * t[6] * t[15] - t[8] * t[7] * t[14] - t[12] * t[6] * t[11] + t[12] * t[7] * t[10]),
        i * (t[0] * t[10] * t[15] - t[0] * t[11] * t[14] - t[8] * t[2] * t[15] + t[8] * t[3] * t[14] + t[12] * t[2] * t[11] - t[12] * t[3] * t[10]),
        i * (-t[0] * t[6] * t[15] + t[0] * t[7] * t[14] + t[4] * t[2] * t[15] - t[4] * t[3] * t[14] - t[12] * t[2] * t[7] + t[12] * t[3] * t[6]),
        i * (t[0] * t[6] * t[11] - t[0] * t[7] * t[10] - t[4] * t[2] * t[11] + t[4] * t[3] * t[10] + t[8] * t[2] * t[7] - t[8] * t[3] * t[6]),
        i * (t[4] * t[9] * t[15] - t[4] * t[11] * t[13] - t[8] * t[5] * t[15] + t[8] * t[7] * t[13] + t[12] * t[5] * t[11] - t[12] * t[7] * t[9]),
        i * (-t[0] * t[9] * t[15] + t[0] * t[11] * t[13] + t[8] * t[1] * t[15] - t[8] * t[3] * t[13] - t[12] * t[1] * t[11] + t[12] * t[3] * t[9]),
        i * (t[0] * t[5] * t[15] - t[0] * t[7] * t[13] - t[4] * t[1] * t[15] + t[4] * t[3] * t[13] + t[12] * t[1] * t[7] - t[12] * t[3] * t[5]),
        i * (-t[0] * t[5] * t[11] + t[0] * t[7] * t[9] + t[4] * t[1] * t[11] - t[4] * t[3] * t[9] - t[8] * t[1] * t[7] + t[8] * t[3] * t[5]),
        i * (-t[4] * t[9] * t[14] + t[4] * t[10] * t[13] + t[8] * t[5] * t[14] - t[8] * t[6] * t[13] - t[12] * t[5] * t[10] + t[12] * t[6] * t[9]),
        i * (t[0] * t[9] * t[14] - t[0] * t[10] * t[13] - t[8] * t[1] * t[14] + t[8] * t[2] * t[13] + t[12] * t[1] * t[10] - t[12] * t[2] * t[9]),
        i * (-t[0] * t[5] * t[14] + t[0] * t[6] * t[13] + t[4] * t[1] * t[14] - t[4] * t[2] * t[13] - t[12] * t[1] * t[6] + t[12] * t[2] * t[5]),
        i * (t[0] * t[5] * t[10] - t[0] * t[6] * t[9] - t[4] * t[1] * t[10] + t[4] * t[2] * t[9] + t[8] * t[1] * t[6] - t[8] * t[2] * t[5])
      );
    }
    static Compose(t, n, i) {
      const e = n.x, A = n.y, o = n.z, s = n.w, r = e + e, Q = A + A, I = o + o, d = e * r, a = e * Q, U = e * I, F = A * Q, g = A * I, B = o * I, C = s * r, c = s * Q, p = s * I, u = i.x, S = i.y, W = i.z;
      return new _K(
        (1 - (F + B)) * u,
        (a + p) * u,
        (U - c) * u,
        0,
        (a - p) * S,
        (1 - (d + B)) * S,
        (g + C) * S,
        0,
        (U + c) * W,
        (g - C) * W,
        (1 - (d + F)) * W,
        0,
        t.x,
        t.y,
        t.z,
        1
      );
    }
    toString() {
      return `[${this.buffer.join(", ")}]`;
    }
  };
  var It = class extends Event {
    constructor(t) {
      super("objectAdded"), this.object = t;
    }
  };
  var ht = class extends Event {
    constructor(t) {
      super("objectRemoved"), this.object = t;
    }
  };
  var ct = class extends Event {
    constructor(t) {
      super("objectChanged"), this.object = t;
    }
  };
  var st = class extends Ft {
    constructor() {
      super(), this.positionChanged = false, this.rotationChanged = false, this.scaleChanged = false, this._position = new R(), this._rotation = new y(), this._scale = new R(1, 1, 1), this._transform = new K(), this._changeEvent = new ct(this), this.update = () => {
      }, this.applyPosition = () => {
        this.position = new R();
      }, this.applyRotation = () => {
        this.rotation = new y();
      }, this.applyScale = () => {
        this.scale = new R(1, 1, 1);
      }, this.raiseChangeEvent = () => {
        this.dispatchEvent(this._changeEvent);
      };
    }
    _updateMatrix() {
      this._transform = K.Compose(this._position, this._rotation, this._scale);
    }
    get position() {
      return this._position;
    }
    set position(t) {
      this._position.equals(t) || (this._position = t, this.positionChanged = true, this._updateMatrix(), this.dispatchEvent(this._changeEvent));
    }
    get rotation() {
      return this._rotation;
    }
    set rotation(t) {
      this._rotation.equals(t) || (this._rotation = t, this.rotationChanged = true, this._updateMatrix(), this.dispatchEvent(this._changeEvent));
    }
    get scale() {
      return this._scale;
    }
    set scale(t) {
      this._scale.equals(t) || (this._scale = t, this.scaleChanged = true, this._updateMatrix(), this.dispatchEvent(this._changeEvent));
    }
    get forward() {
      let t = new R(0, 0, 1);
      return t = this.rotation.apply(t), t;
    }
    get transform() {
      return this._transform;
    }
  };
  var _ = class __ {
    // prettier-ignore
    constructor(t = 1, n = 0, i = 0, e = 0, A = 1, o = 0, s = 0, r = 0, Q = 1) {
      this.buffer = [
        t,
        n,
        i,
        e,
        A,
        o,
        s,
        r,
        Q
      ];
    }
    equals(t) {
      if (this.buffer.length !== t.buffer.length)
        return false;
      if (this.buffer === t.buffer)
        return true;
      for (let n = 0; n < this.buffer.length; n++)
        if (this.buffer[n] !== t.buffer[n])
          return false;
      return true;
    }
    multiply(t) {
      const n = this.buffer, i = t.buffer;
      return new __(
        i[0] * n[0] + i[3] * n[1] + i[6] * n[2],
        i[1] * n[0] + i[4] * n[1] + i[7] * n[2],
        i[2] * n[0] + i[5] * n[1] + i[8] * n[2],
        i[0] * n[3] + i[3] * n[4] + i[6] * n[5],
        i[1] * n[3] + i[4] * n[4] + i[7] * n[5],
        i[2] * n[3] + i[5] * n[4] + i[8] * n[5],
        i[0] * n[6] + i[3] * n[7] + i[6] * n[8],
        i[1] * n[6] + i[4] * n[7] + i[7] * n[8],
        i[2] * n[6] + i[5] * n[7] + i[8] * n[8]
      );
    }
    clone() {
      const t = this.buffer;
      return new __(
        t[0],
        t[1],
        t[2],
        t[3],
        t[4],
        t[5],
        t[6],
        t[7],
        t[8]
      );
    }
    static Eye(t = 1) {
      return new __(t, 0, 0, 0, t, 0, 0, 0, t);
    }
    static Diagonal(t) {
      return new __(t.x, 0, 0, 0, t.y, 0, 0, 0, t.z);
    }
    static RotationFromQuaternion(t) {
      return new __(
        1 - 2 * t.y * t.y - 2 * t.z * t.z,
        2 * t.x * t.y - 2 * t.z * t.w,
        2 * t.x * t.z + 2 * t.y * t.w,
        2 * t.x * t.y + 2 * t.z * t.w,
        1 - 2 * t.x * t.x - 2 * t.z * t.z,
        2 * t.y * t.z - 2 * t.x * t.w,
        2 * t.x * t.z - 2 * t.y * t.w,
        2 * t.y * t.z + 2 * t.x * t.w,
        1 - 2 * t.x * t.x - 2 * t.y * t.y
      );
    }
    static RotationFromEuler(t) {
      const n = Math.cos(t.x), i = Math.sin(t.x), e = Math.cos(t.y), A = Math.sin(t.y), o = Math.cos(t.z), s = Math.sin(t.z), r = [
        e * o + A * i * s,
        -e * s + A * i * o,
        A * n,
        n * s,
        n * o,
        -i,
        -A * o + e * i * s,
        A * s + e * i * o,
        e * n
      ];
      return new __(...r);
    }
    toString() {
      return `[${this.buffer.join(", ")}]`;
    }
  };
  var _Y = class _Y {
    constructor(t = 0, n = null, i = null, e = null, A = null) {
      this.changed = false, this.detached = false, this._vertexCount = t, this._positions = n || new Float32Array(0), this._rotations = i || new Float32Array(0), this._scales = e || new Float32Array(0), this._colors = A || new Uint8Array(0), this._selection = new Uint8Array(this.vertexCount), this.translate = (o) => {
        for (let s = 0; s < this.vertexCount; s++)
          this.positions[3 * s + 0] += o.x, this.positions[3 * s + 1] += o.y, this.positions[3 * s + 2] += o.z;
        this.changed = true;
      }, this.rotate = (o) => {
        const s = _.RotationFromQuaternion(o).buffer;
        for (let r = 0; r < this.vertexCount; r++) {
          const Q = this.positions[3 * r + 0], I = this.positions[3 * r + 1], d = this.positions[3 * r + 2];
          this.positions[3 * r + 0] = s[0] * Q + s[1] * I + s[2] * d, this.positions[3 * r + 1] = s[3] * Q + s[4] * I + s[5] * d, this.positions[3 * r + 2] = s[6] * Q + s[7] * I + s[8] * d;
          const a = new y(
            this.rotations[4 * r + 1],
            this.rotations[4 * r + 2],
            this.rotations[4 * r + 3],
            this.rotations[4 * r + 0]
          ), U = o.multiply(a);
          this.rotations[4 * r + 1] = U.x, this.rotations[4 * r + 2] = U.y, this.rotations[4 * r + 3] = U.z, this.rotations[4 * r + 0] = U.w;
        }
        this.changed = true;
      }, this.scale = (o) => {
        for (let s = 0; s < this.vertexCount; s++)
          this.positions[3 * s + 0] *= o.x, this.positions[3 * s + 1] *= o.y, this.positions[3 * s + 2] *= o.z, this.scales[3 * s + 0] *= o.x, this.scales[3 * s + 1] *= o.y, this.scales[3 * s + 2] *= o.z;
        this.changed = true;
      }, this.serialize = () => {
        const o = new Uint8Array(this.vertexCount * _Y.RowLength), s = new Float32Array(o.buffer), r = new Uint8Array(o.buffer);
        for (let Q = 0; Q < this.vertexCount; Q++)
          s[8 * Q + 0] = this.positions[3 * Q + 0], s[8 * Q + 1] = this.positions[3 * Q + 1], s[8 * Q + 2] = this.positions[3 * Q + 2], r[32 * Q + 24 + 0] = this.colors[4 * Q + 0], r[32 * Q + 24 + 1] = this.colors[4 * Q + 1], r[32 * Q + 24 + 2] = this.colors[4 * Q + 2], r[32 * Q + 24 + 3] = this.colors[4 * Q + 3], s[8 * Q + 3 + 0] = this.scales[3 * Q + 0], s[8 * Q + 3 + 1] = this.scales[3 * Q + 1], s[8 * Q + 3 + 2] = this.scales[3 * Q + 2], r[32 * Q + 28 + 0] = this.rotations[4 * Q + 0] * 128 + 128 & 255, r[32 * Q + 28 + 1] = this.rotations[4 * Q + 1] * 128 + 128 & 255, r[32 * Q + 28 + 2] = this.rotations[4 * Q + 2] * 128 + 128 & 255, r[32 * Q + 28 + 3] = this.rotations[4 * Q + 3] * 128 + 128 & 255;
        return o;
      }, this.reattach = (o, s, r, Q, I) => {
        console.assert(
          o.byteLength === this.vertexCount * 3 * 4,
          `Expected ${this.vertexCount * 3 * 4} bytes, got ${o.byteLength} bytes`
        ), this._positions = new Float32Array(o), this._rotations = new Float32Array(s), this._scales = new Float32Array(r), this._colors = new Uint8Array(Q), this._selection = new Uint8Array(I), this.detached = false;
      };
    }
    static Deserialize(t) {
      const n = t.length / _Y.RowLength, i = new Float32Array(3 * n), e = new Float32Array(4 * n), A = new Float32Array(3 * n), o = new Uint8Array(4 * n), s = new Float32Array(t.buffer), r = new Uint8Array(t.buffer);
      for (let Q = 0; Q < n; Q++)
        i[3 * Q + 0] = s[8 * Q + 0], i[3 * Q + 1] = s[8 * Q + 1], i[3 * Q + 2] = s[8 * Q + 2], e[4 * Q + 0] = (r[32 * Q + 28 + 0] - 128) / 128, e[4 * Q + 1] = (r[32 * Q + 28 + 1] - 128) / 128, e[4 * Q + 2] = (r[32 * Q + 28 + 2] - 128) / 128, e[4 * Q + 3] = (r[32 * Q + 28 + 3] - 128) / 128, A[3 * Q + 0] = s[8 * Q + 3 + 0], A[3 * Q + 1] = s[8 * Q + 3 + 1], A[3 * Q + 2] = s[8 * Q + 3 + 2], o[4 * Q + 0] = r[32 * Q + 24 + 0], o[4 * Q + 1] = r[32 * Q + 24 + 1], o[4 * Q + 2] = r[32 * Q + 24 + 2], o[4 * Q + 3] = r[32 * Q + 24 + 3];
      return new _Y(n, i, e, A, o);
    }
    get vertexCount() {
      return this._vertexCount;
    }
    get positions() {
      return this._positions;
    }
    get rotations() {
      return this._rotations;
    }
    get scales() {
      return this._scales;
    }
    get colors() {
      return this._colors;
    }
    get selection() {
      return this._selection;
    }
    clone() {
      return new _Y(
        this.vertexCount,
        new Float32Array(this.positions),
        new Float32Array(this.rotations),
        new Float32Array(this.scales),
        new Uint8Array(this.colors)
      );
    }
  };
  _Y.RowLength = 3 * 4 + 3 * 4 + 4 + 4;
  var Y = _Y;
  var _ot = class _ot {
    constructor(t, n, i, e, A) {
      this._vertexCount = t, this._positions = n, this._data = i, this._width = e, this._height = A, this.serialize = () => new Uint8Array(this._data.buffer);
    }
    static Deserialize(t, n, i) {
      const e = new Uint32Array(t.buffer), A = new Float32Array(t.buffer), o = Math.floor(A.byteLength / this.RowLength), s = new Float32Array(o * 3);
      for (let r = 0; r < o; r++)
        s[3 * r + 0] = A[16 * r + 0], s[3 * r + 1] = A[16 * r + 1], s[3 * r + 2] = A[16 * r + 2], s[3 * r + 0] = A[16 * r + 3];
      return new _ot(o, s, e, n, i);
    }
    get vertexCount() {
      return this._vertexCount;
    }
    get positions() {
      return this._positions;
    }
    get data() {
      return this._data;
    }
    get width() {
      return this._width;
    }
    get height() {
      return this._height;
    }
  };
  _ot.RowLength = 64;
  var ot = _ot;
  var _et = class _et {
    static SplatToPLY(t, n) {
      let i = `ply
format binary_little_endian 1.0
`;
      i += `element vertex ${n}
`;
      const e = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"];
      for (let B = 0; B < 45; B++)
        e.push(`f_rest_${B}`);
      e.push("opacity"), e.push("scale_0"), e.push("scale_1"), e.push("scale_2"), e.push("rot_0"), e.push("rot_1"), e.push("rot_2"), e.push("rot_3");
      for (const B of e)
        i += `property float ${B}
`;
      i += `end_header
`;
      const A = new TextEncoder().encode(i), o = 4 * 3 + 4 * 3 + 4 * 3 + 4 * 45 + 4 + 4 * 3 + 4 * 4, s = n * o, r = new DataView(new ArrayBuffer(A.length + s));
      new Uint8Array(r.buffer).set(A, 0);
      const Q = new Float32Array(t), I = new Uint8Array(t), d = A.length, a = 4 * 3 + 4 * 3, U = a + 4 * 3 + 4 * 45, F = U + 4, g = F + 4 * 3;
      for (let B = 0; B < n; B++) {
        const C = Q[8 * B + 0], c = Q[8 * B + 1], p = Q[8 * B + 2], u = (I[32 * B + 24 + 0] / 255 - 0.5) / this.SH_C0, S = (I[32 * B + 24 + 1] / 255 - 0.5) / this.SH_C0, W = (I[32 * B + 24 + 2] / 255 - 0.5) / this.SH_C0, Z = I[32 * B + 24 + 3] / 255, k = Math.log(Z / (1 - Z)), f = Math.log(Q[8 * B + 3 + 0]), H = Math.log(Q[8 * B + 3 + 1]), L = Math.log(Q[8 * B + 3 + 2]);
        let M = new y(
          (I[32 * B + 28 + 1] - 128) / 128,
          (I[32 * B + 28 + 2] - 128) / 128,
          (I[32 * B + 28 + 3] - 128) / 128,
          (I[32 * B + 28 + 0] - 128) / 128
        );
        M = M.normalize();
        const V = M.w, w = M.x, P = M.y, T = M.z;
        r.setFloat32(d + o * B + 0, C, true), r.setFloat32(d + o * B + 4, c, true), r.setFloat32(d + o * B + 8, p, true), r.setFloat32(d + o * B + a + 0, u, true), r.setFloat32(d + o * B + a + 4, S, true), r.setFloat32(d + o * B + a + 8, W, true), r.setFloat32(d + o * B + U, k, true), r.setFloat32(d + o * B + F + 0, f, true), r.setFloat32(d + o * B + F + 4, H, true), r.setFloat32(d + o * B + F + 8, L, true), r.setFloat32(d + o * B + g + 0, V, true), r.setFloat32(d + o * B + g + 4, w, true), r.setFloat32(d + o * B + g + 8, P, true), r.setFloat32(d + o * B + g + 12, T, true);
      }
      return r.buffer;
    }
  };
  _et.SH_C0 = 0.28209479177387814;
  var et = _et;
  var nt = class {
    constructor(t, n) {
      this.min = t, this.max = n;
    }
    contains(t) {
      return t.x >= this.min.x && t.x <= this.max.x && t.y >= this.min.y && t.y <= this.max.y && t.z >= this.min.z && t.z <= this.max.z;
    }
    intersects(t) {
      return this.max.x >= t.min.x && this.min.x <= t.max.x && this.max.y >= t.min.y && this.min.y <= t.max.y && this.max.z >= t.min.z && this.min.z <= t.max.z;
    }
    size() {
      return this.max.subtract(this.min);
    }
    center() {
      return this.min.add(this.max).divide(2);
    }
    expand(t) {
      this.min = this.min.min(t), this.max = this.max.max(t);
    }
    permute() {
      const t = this.min, n = this.max;
      this.min = new R(Math.min(t.x, n.x), Math.min(t.y, n.y), Math.min(t.z, n.z)), this.max = new R(Math.max(t.x, n.x), Math.max(t.y, n.y), Math.max(t.z, n.z));
    }
  };
  var X = class _X extends st {
    constructor(t = void 0) {
      super(), this.selectedChanged = false, this.colorTransformChanged = false, this._selected = false, this._colorTransforms = [], this._colorTransformsMap = /* @__PURE__ */ new Map(), this._data = t || new Y(), this._bounds = new nt(
        new R(1 / 0, 1 / 0, 1 / 0),
        new R(-1 / 0, -1 / 0, -1 / 0)
      ), this.recalculateBounds = () => {
        this._bounds = new nt(
          new R(1 / 0, 1 / 0, 1 / 0),
          new R(-1 / 0, -1 / 0, -1 / 0)
        );
        for (let n = 0; n < this._data.vertexCount; n++)
          this._bounds.expand(
            new R(
              this._data.positions[3 * n],
              this._data.positions[3 * n + 1],
              this._data.positions[3 * n + 2]
            )
          );
      }, this.applyPosition = () => {
        this.data.translate(this.position), this.position = new R();
      }, this.applyRotation = () => {
        this.data.rotate(this.rotation), this.rotation = new y();
      }, this.applyScale = () => {
        this.data.scale(this.scale), this.scale = new R(1, 1, 1);
      }, this.recalculateBounds();
    }
    saveToFile(t = null, n = "splat") {
      if (!document) return;
      if (!t) {
        const s = /* @__PURE__ */ new Date();
        t = `splat-${s.getFullYear()}-${s.getMonth() + 1}-${s.getDate()}.${n}`;
      }
      const i = this.clone();
      i.applyRotation(), i.applyScale(), i.applyPosition();
      const e = i.data.serialize();
      let A;
      if (n === "ply") {
        const s = et.SplatToPLY(e.buffer, i.data.vertexCount);
        A = new Blob([s], { type: "application/octet-stream" });
      } else
        A = new Blob([e.buffer], { type: "application/octet-stream" });
      const o = document.createElement("a");
      o.download = t, o.href = URL.createObjectURL(A), o.click();
    }
    get data() {
      return this._data;
    }
    get selected() {
      return this._selected;
    }
    set selected(t) {
      this._selected !== t && (this._selected = t, this.selectedChanged = true, this.dispatchEvent(this._changeEvent));
    }
    get colorTransforms() {
      return this._colorTransforms;
    }
    get colorTransformsMap() {
      return this._colorTransformsMap;
    }
    get bounds() {
      let t = this._bounds.center();
      t = t.add(this.position);
      let n = this._bounds.size();
      return n = n.multiply(this.scale), new nt(t.subtract(n.divide(2)), t.add(n.divide(2)));
    }
    clone() {
      const t = new _X(this.data.clone());
      return t.position = this.position.clone(), t.rotation = this.rotation.clone(), t.scale = this.scale.clone(), t;
    }
  };
  var tt = class extends st {
    constructor(t) {
      super(), this._data = t;
    }
    get data() {
      return this._data;
    }
  };
  var Rt = class {
    constructor() {
      this._fx = 1132, this._fy = 1132, this._near = 0.1, this._far = 100, this._width = 512, this._height = 512, this._projectionMatrix = new K(), this._viewMatrix = new K(), this._viewProj = new K(), this._updateProjectionMatrix = () => {
        this._projectionMatrix = new K(
          2 * this.fx / this.width,
          0,
          0,
          0,
          0,
          -2 * this.fy / this.height,
          0,
          0,
          0,
          0,
          this.far / (this.far - this.near),
          1,
          0,
          0,
          -(this.far * this.near) / (this.far - this.near),
          0
        ), this._viewProj = this.projectionMatrix.multiply(this.viewMatrix);
      }, this.update = (t, n) => {
        const i = _.RotationFromQuaternion(n).buffer, e = t.flat();
        this._viewMatrix = new K(
          i[0],
          i[1],
          i[2],
          0,
          i[3],
          i[4],
          i[5],
          0,
          i[6],
          i[7],
          i[8],
          0,
          -e[0] * i[0] - e[1] * i[3] - e[2] * i[6],
          -e[0] * i[1] - e[1] * i[4] - e[2] * i[7],
          -e[0] * i[2] - e[1] * i[5] - e[2] * i[8],
          1
        ), this._viewProj = this.projectionMatrix.multiply(this.viewMatrix);
      }, this.setSize = (t, n) => {
        this._width = t, this._height = n, this._updateProjectionMatrix();
      };
    }
    get fx() {
      return this._fx;
    }
    set fx(t) {
      this._fx !== t && (this._fx = t, this._updateProjectionMatrix());
    }
    get fy() {
      return this._fy;
    }
    set fy(t) {
      this._fy !== t && (this._fy = t, this._updateProjectionMatrix());
    }
    get near() {
      return this._near;
    }
    set near(t) {
      this._near !== t && (this._near = t, this._updateProjectionMatrix());
    }
    get far() {
      return this._far;
    }
    set far(t) {
      this._far !== t && (this._far = t, this._updateProjectionMatrix());
    }
    get width() {
      return this._width;
    }
    get height() {
      return this._height;
    }
    get projectionMatrix() {
      return this._projectionMatrix;
    }
    get viewMatrix() {
      return this._viewMatrix;
    }
    get viewProj() {
      return this._viewProj;
    }
  };
  var z = class _z {
    constructor(t = 0, n = 0, i = 0, e = 0) {
      this.x = t, this.y = n, this.z = i, this.w = e;
    }
    equals(t) {
      return !(this.x !== t.x || this.y !== t.y || this.z !== t.z || this.w !== t.w);
    }
    add(t) {
      return typeof t == "number" ? new _z(this.x + t, this.y + t, this.z + t, this.w + t) : new _z(this.x + t.x, this.y + t.y, this.z + t.z, this.w + t.w);
    }
    subtract(t) {
      return typeof t == "number" ? new _z(this.x - t, this.y - t, this.z - t, this.w - t) : new _z(this.x - t.x, this.y - t.y, this.z - t.z, this.w - t.w);
    }
    multiply(t) {
      return typeof t == "number" ? new _z(this.x * t, this.y * t, this.z * t, this.w * t) : t instanceof _z ? new _z(this.x * t.x, this.y * t.y, this.z * t.z, this.w * t.w) : new _z(
        this.x * t.buffer[0] + this.y * t.buffer[4] + this.z * t.buffer[8] + this.w * t.buffer[12],
        this.x * t.buffer[1] + this.y * t.buffer[5] + this.z * t.buffer[9] + this.w * t.buffer[13],
        this.x * t.buffer[2] + this.y * t.buffer[6] + this.z * t.buffer[10] + this.w * t.buffer[14],
        this.x * t.buffer[3] + this.y * t.buffer[7] + this.z * t.buffer[11] + this.w * t.buffer[15]
      );
    }
    dot(t) {
      return this.x * t.x + this.y * t.y + this.z * t.z + this.w * t.w;
    }
    lerp(t, n) {
      return new _z(
        this.x + (t.x - this.x) * n,
        this.y + (t.y - this.y) * n,
        this.z + (t.z - this.z) * n,
        this.w + (t.w - this.w) * n
      );
    }
    magnitude() {
      return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w);
    }
    distanceTo(t) {
      return Math.sqrt((this.x - t.x) ** 2 + (this.y - t.y) ** 2 + (this.z - t.z) ** 2 + (this.w - t.w) ** 2);
    }
    normalize() {
      const t = this.magnitude();
      return new _z(this.x / t, this.y / t, this.z / t, this.w / t);
    }
    flat() {
      return [this.x, this.y, this.z, this.w];
    }
    clone() {
      return new _z(this.x, this.y, this.z, this.w);
    }
    toString() {
      return `[${this.flat().join(", ")}]`;
    }
  };
  var wt = class extends st {
    constructor(t = void 0) {
      super(), this._data = t || new Rt(), this._position = new R(0, 0, -5), this.update = () => {
        this.data.update(this.position, this.rotation);
      }, this.screenPointToRay = (n, i) => {
        const e = new z(n, i, -1, 1), A = this._data.projectionMatrix.invert(), o = e.multiply(A), s = this._data.viewMatrix.invert(), r = o.multiply(s);
        return new R(
          r.x / r.w,
          r.y / r.w,
          r.z / r.w
        ).subtract(this.position).normalize();
      };
    }
    get data() {
      return this._data;
    }
  };
  var kt = class extends Ft {
    constructor() {
      super(), this._objects = [], this.addObject = (t) => {
        this.objects.push(t), this.dispatchEvent(new It(t));
      }, this.removeObject = (t) => {
        const n = this.objects.indexOf(t);
        if (n < 0)
          throw new Error("Object not found in scene");
        this.objects.splice(n, 1), this.dispatchEvent(new ht(t));
      }, this.findObject = (t) => {
        for (const n of this.objects)
          if (t(n))
            return n;
      }, this.findObjectOfType = (t) => {
        for (const n of this.objects)
          if (n instanceof t)
            return n;
      }, this.reset = () => {
        const t = this.objects.slice();
        for (const n of t)
          this.removeObject(n);
      }, this.reset();
    }
    getMergedSceneDataBuffer(t = "splat") {
      const n = [];
      let i = 0;
      for (const o of this.objects)
        if (o instanceof X) {
          const s = o.clone();
          s.applyRotation(), s.applyScale(), s.applyPosition();
          const r = s.data.serialize();
          n.push(r), i += s.data.vertexCount;
        }
      const e = new Uint8Array(i * Y.RowLength);
      let A = 0;
      for (const o of n)
        e.set(o, A), A += o.length;
      return t === "ply" ? et.SplatToPLY(e.buffer, i) : e.buffer;
    }
    saveToFile(t = null, n = "splat") {
      if (!document) return;
      if (!t) {
        const o = /* @__PURE__ */ new Date();
        t = `scene-${o.getFullYear()}-${o.getMonth() + 1}-${o.getDate()}.${n}`;
      }
      const i = this.getMergedSceneDataBuffer(n), e = new Blob([i], { type: "application/octet-stream" }), A = document.createElement("a");
      A.download = t, A.href = URL.createObjectURL(e), A.click();
    }
    get objects() {
      return this._objects;
    }
  };
  async function rt(E, t) {
    const n = await fetch(E, {
      mode: "cors",
      credentials: "omit",
      cache: t ? "force-cache" : "default"
    });
    if (n.status != 200)
      throw new Error(n.status + " Unable to load " + n.url);
    return n;
  }
  async function At(E, t) {
    const n = E.body.getReader(), i = E.headers.get("content-length"), e = i && !isNaN(parseInt(i)) ? parseInt(i) : void 0, A = [];
    let o = 0;
    for (; ; ) {
      const { done: Q, value: I } = await n.read();
      if (Q) break;
      if (A.push(I), o += I.length, t && e) {
        const d = o / e, a = Math.min(d * 0.95, 0.95);
        t(a);
      }
    }
    const s = new Uint8Array(o);
    let r = 0;
    for (const Q of A)
      s.set(Q, r), r += Q.length;
    return t && t(1), s;
  }
  var Tt = class {
    static async LoadAsync(t, n, i, e = false) {
      const A = await rt(t, e), o = await At(A, i);
      return this.LoadFromArrayBuffer(o.buffer, n);
    }
    static async LoadFromFileAsync(t, n, i) {
      const e = new FileReader();
      let A = new X();
      return e.onload = (o) => {
        A = this.LoadFromArrayBuffer(o.target.result, n);
      }, e.onprogress = (o) => {
        i?.(o.loaded / o.total);
      }, e.readAsArrayBuffer(t), await new Promise((o) => {
        e.onloadend = () => {
          o();
        };
      }), A;
    }
    static LoadFromArrayBuffer(t, n) {
      const i = new Uint8Array(t), e = Y.Deserialize(i), A = new X(e);
      return n.addObject(A), A;
    }
  };
  var bt = class {
    static async LoadAsync(t, n, i, e = "", A = false) {
      const o = await rt(t, A), s = await At(o, i);
      if (s[0] !== 112 || s[1] !== 108 || s[2] !== 121 || s[3] !== 10)
        throw new Error("Invalid PLY file");
      return this.LoadFromArrayBuffer(s.buffer, n, e);
    }
    static async LoadFromFileAsync(t, n, i, e = "") {
      const A = new FileReader();
      let o = new X();
      return A.onload = (s) => {
        o = this.LoadFromArrayBuffer(s.target.result, n, e);
      }, A.onprogress = (s) => {
        i?.(s.loaded / s.total);
      }, A.readAsArrayBuffer(t), await new Promise((s) => {
        A.onloadend = () => {
          s();
        };
      }), o;
    }
    static LoadFromArrayBuffer(t, n, i = "") {
      const e = new Uint8Array(this._ParsePLYBuffer(t, i)), A = Y.Deserialize(e), o = new X(A);
      return n.addObject(o), o;
    }
    static _ParsePLYBuffer(t, n) {
      const i = new Uint8Array(t), e = new TextDecoder().decode(i.slice(0, 1024 * 10)), A = `end_header
`, o = e.indexOf(A);
      if (o < 0) throw new Error("Unable to read .ply file header");
      const s = parseInt(/element vertex (\d+)\n/.exec(e)[1]);
      let r = 0;
      const Q = {
        double: 8,
        int: 4,
        uint: 4,
        float: 4,
        short: 2,
        ushort: 2,
        uchar: 1
      }, I = [];
      for (const F of e.slice(0, o).split(`
`).filter((g) => g.startsWith("property "))) {
        const [g, B, C] = F.split(" ");
        if (I.push({ name: C, type: B, offset: r }), !Q[B]) throw new Error(`Unsupported property type: ${B}`);
        r += Q[B];
      }
      const d = new DataView(t, o + A.length), a = new ArrayBuffer(Y.RowLength * s), U = y.FromEuler(new R(Math.PI / 2, 0, 0));
      for (let F = 0; F < s; F++) {
        const g = new Float32Array(a, F * Y.RowLength, 3), B = new Float32Array(a, F * Y.RowLength + 12, 3), C = new Uint8ClampedArray(a, F * Y.RowLength + 24, 4), c = new Uint8ClampedArray(a, F * Y.RowLength + 28, 4);
        let p = 255, u = 0, S = 0, W = 0;
        I.forEach((k) => {
          let f;
          switch (k.type) {
            case "float":
              f = d.getFloat32(k.offset + F * r, true);
              break;
            case "int":
              f = d.getInt32(k.offset + F * r, true);
              break;
            default:
              throw new Error(`Unsupported property type: ${k.type}`);
          }
          switch (k.name) {
            case "x":
              g[0] = f;
              break;
            case "y":
              g[1] = f;
              break;
            case "z":
              g[2] = f;
              break;
            case "scale_0":
            case "scaling_0":
              B[0] = Math.exp(f);
              break;
            case "scale_1":
            case "scaling_1":
              B[1] = Math.exp(f);
              break;
            case "scale_2":
            case "scaling_2":
              B[2] = Math.exp(f);
              break;
            case "red":
              C[0] = f;
              break;
            case "green":
              C[1] = f;
              break;
            case "blue":
              C[2] = f;
              break;
            case "f_dc_0":
            case "features_0":
              C[0] = (0.5 + et.SH_C0 * f) * 255;
              break;
            case "f_dc_1":
            case "features_1":
              C[1] = (0.5 + et.SH_C0 * f) * 255;
              break;
            case "f_dc_2":
            case "features_2":
              C[2] = (0.5 + et.SH_C0 * f) * 255;
              break;
            case "f_dc_3":
              C[3] = (0.5 + et.SH_C0 * f) * 255;
              break;
            case "opacity":
            case "opacity_0":
              C[3] = 1 / (1 + Math.exp(-f)) * 255;
              break;
            case "rot_0":
            case "rotation_0":
              p = f;
              break;
            case "rot_1":
            case "rotation_1":
              u = f;
              break;
            case "rot_2":
            case "rotation_2":
              S = f;
              break;
            case "rot_3":
            case "rotation_3":
              W = f;
              break;
          }
        });
        let Z = new y(u, S, W, p);
        switch (n) {
          case "polycam": {
            const k = g[1];
            g[1] = -g[2], g[2] = k, Z = U.multiply(Z);
            break;
          }
          case "":
            break;
          default:
            throw new Error(`Unsupported format: ${n}`);
        }
        Z = Z.normalize(), c[0] = Z.w * 128 + 128, c[1] = Z.x * 128 + 128, c[2] = Z.y * 128 + 128, c[3] = Z.z * 128 + 128;
      }
      return a;
    }
  };
  var xt = class {
    static async LoadAsync(t, n, i, e, A = false) {
      const o = await rt(t, A), s = await At(o, e);
      return this._ParseSplatvBuffer(s.buffer, n, i);
    }
    static async LoadFromFileAsync(t, n, i, e) {
      const A = new FileReader();
      let o = null;
      if (A.onload = (s) => {
        o = this._ParseSplatvBuffer(s.target.result, n, i);
      }, A.onprogress = (s) => {
        e?.(s.loaded / s.total);
      }, A.readAsArrayBuffer(t), await new Promise((s) => {
        A.onloadend = () => {
          s();
        };
      }), !o)
        throw new Error("Failed to load splatv file");
      return o;
    }
    static _ParseSplatvBuffer(t, n, i) {
      let e = null;
      const A = (a, U, F) => {
        if (a.type === "magic") {
          const g = new Int32Array(U.buffer);
          if (g[0] !== 26443)
            throw new Error("Invalid splatv file");
          F.push({ size: g[1], type: "chunks" });
        } else if (a.type === "chunks") {
          const g = JSON.parse(new TextDecoder("utf-8").decode(U));
          if (g.length == 0)
            throw new Error("Invalid splatv file");
          g.length > 1 && console.warn("Splatv file contains more than one chunk, only the first one will be loaded");
          const B = g[0], C = B.cameras;
          if (i && C && C.length) {
            const c = C[0], p = new R(
              c.position[0],
              c.position[1],
              c.position[2]
            ), u = y.FromMatrix3(
              new _(
                c.rotation[0][0],
                c.rotation[0][1],
                c.rotation[0][2],
                c.rotation[1][0],
                c.rotation[1][1],
                c.rotation[1][2],
                c.rotation[2][0],
                c.rotation[2][1],
                c.rotation[2][2]
              )
            );
            i.position = p, i.rotation = u;
          }
          F.push(B);
        } else if (a.type === "splat") {
          const g = ot.Deserialize(U, a.texwidth, a.texheight), B = new tt(g);
          n.addObject(B), e = B;
        }
      }, o = new Uint8Array(t), s = [
        { size: 8, type: "magic", texwidth: 0, texheight: 0 }
      ];
      let r = s.shift(), Q = new Uint8Array(r.size), I = 0, d = 0;
      for (; r; ) {
        for (; I < r.size; ) {
          const a = Math.min(r.size - I, o.length - d);
          Q.set(o.subarray(d, d + a), I), I += a, d += a;
        }
        if (A(r, Q, s), e)
          return e;
        r = s.shift(), r && (Q = new Uint8Array(r.size), I = 0);
      }
      throw new Error("Invalid splatv file");
    }
  };
  var Bt = "dmFyIGVBID0gZnVuY3Rpb24oQyA9IHt9KSB7CiAgdmFyIHIsIEkgPSBDLCBsID0gaW1wb3J0Lm1ldGEudXJsLCBqID0gIiIsIFU7CiAgewogICAgdHJ5IHsKICAgICAgaiA9IG5ldyBVUkwoIi4iLCBsKS5ocmVmOwogICAgfSBjYXRjaCB7CiAgICB9CiAgICBVID0gKEEpID0+IHsKICAgICAgdmFyIEIgPSBuZXcgWE1MSHR0cFJlcXVlc3QoKTsKICAgICAgcmV0dXJuIEIub3BlbigiR0VUIiwgQSwgITEpLCBCLnJlc3BvbnNlVHlwZSA9ICJhcnJheWJ1ZmZlciIsIEIuc2VuZChudWxsKSwgbmV3IFVpbnQ4QXJyYXkoQi5yZXNwb25zZSk7CiAgICB9OwogIH0KICBjb25zb2xlLmxvZy5iaW5kKGNvbnNvbGUpLCBjb25zb2xlLmVycm9yLmJpbmQoY29uc29sZSk7CiAgdmFyIE0sIHUsIEw7CiAgZnVuY3Rpb24gcSgpIHsKICAgIHZhciBBID0gdS5idWZmZXI7CiAgICBJLkhFQVBVOCA9IEwgPSBuZXcgVWludDhBcnJheShBKSwgSS5IRUFQVTMyID0gbmV3IFVpbnQzMkFycmF5KEEpLCBJLkhFQVBGMzIgPSBuZXcgRmxvYXQzMkFycmF5KEEpLCBuZXcgQmlnSW50NjRBcnJheShBKSwgbmV3IEJpZ1VpbnQ2NEFycmF5KEEpOwogIH0KICBmdW5jdGlvbiBQKCkgewogICAgaWYgKEkucHJlUnVuKQogICAgICBmb3IgKHR5cGVvZiBJLnByZVJ1biA9PSAiZnVuY3Rpb24iICYmIChJLnByZVJ1biA9IFtJLnByZVJ1bl0pOyBJLnByZVJ1bi5sZW5ndGg7ICkKICAgICAgICBDQShJLnByZVJ1bi5zaGlmdCgpKTsKICAgIHgoVCk7CiAgfQogIGZ1bmN0aW9uIGIoKSB7CiAgICB5LmMoKTsKICB9CiAgZnVuY3Rpb24gVygpIHsKICAgIGlmIChJLnBvc3RSdW4pCiAgICAgIGZvciAodHlwZW9mIEkucG9zdFJ1biA9PSAiZnVuY3Rpb24iICYmIChJLnBvc3RSdW4gPSBbSS5wb3N0UnVuXSk7IEkucG9zdFJ1bi5sZW5ndGg7ICkKICAgICAgICBnQShJLnBvc3RSdW4uc2hpZnQoKSk7CiAgICB4KHYpOwogIH0KICB2YXIgcyA9IDAsIGYgPSBudWxsOwogIGZ1bmN0aW9uIF8oQSkgewogICAgcysrLCBJLm1vbml0b3JSdW5EZXBlbmRlbmNpZXM/LihzKTsKICB9CiAgZnVuY3Rpb24geihBKSB7CiAgICBpZiAocy0tLCBJLm1vbml0b3JSdW5EZXBlbmRlbmNpZXM/LihzKSwgcyA9PSAwICYmIGYpIHsKICAgICAgdmFyIEIgPSBmOwogICAgICBmID0gbnVsbCwgQigpOwogICAgfQogIH0KICB2YXIgUzsKICBmdW5jdGlvbiBWKCkgewogICAgcmV0dXJuIEJBKCJBR0Z6YlFFQUFBQUJHUVJnQVg4QmYyQUJmd0JnQ1g5L2YzOS9mMzkvZndCZ0FBQUNCd0VCWVFGaEFBQURCZ1VBQVFBQ0F3VUhBUUdDQW9DQUFnWUlBWDhCUVlDTUJBc0hGUVVCWWdJQUFXTUFCUUZrQUFRQlpRQURBV1lBQWd3QkFRck5PQVZQQVFKL1FZQUlLQUlBSWdFZ0FFRUhha0Y0Y1NJQ2FpRUFBa0FnQWtFQUlBQWdBVTBiUlFSQUlBQS9BRUVRZEUwTkFTQUFFQUFOQVF0QmhBaEJNRFlDQUVGL0R3dEJnQWdnQURZQ0FDQUJDOXdMQVFoL0FrQWdBRVVOQUNBQVFRaHJJZ01nQUVFRWF5Z0NBQ0lDUVhoeElnQnFJUVVDUUNBQ1FRRnhEUUFnQWtFQ2NVVU5BU0FESUFNb0FnQWlCR3NpQTBHWUNDZ0NBRWtOQVNBQUlBUnFJUUFDUUFKQUFrQkJuQWdvQWdBZ0EwY0VRQ0FES0FJTUlRRWdCRUgvQVUwRVFDQUJJQU1vQWdnaUFrY05Ba0dJQ0VHSUNDZ0NBRUYrSUFSQkEzWjNjVFlDQUF3RkN5QURLQUlZSVFjZ0FTQURSd1JBSUFNb0FnZ2lBaUFCTmdJTUlBRWdBallDQ0F3RUN5QURLQUlVSWdJRWZ5QURRUlJxQlNBREtBSVFJZ0pGRFFNZ0EwRVFhZ3NoQkFOQUlBUWhCaUFDSWdGQkZHb2hCQ0FCS0FJVUlnSU5BQ0FCUVJCcUlRUWdBU2dDRUNJQ0RRQUxJQVpCQURZQ0FBd0RDeUFGS0FJRUlnSkJBM0ZCQTBjTkEwR1FDQ0FBTmdJQUlBVWdBa0YrY1RZQ0JDQURJQUJCQVhJMkFnUWdCU0FBTmdJQUR3c2dBaUFCTmdJTUlBRWdBallDQ0F3Q0MwRUFJUUVMSUFkRkRRQUNRQ0FES0FJY0lnUkJBblJCdUFwcUlnSW9BZ0FnQTBZRVFDQUNJQUUyQWdBZ0FRMEJRWXdJUVl3SUtBSUFRWDRnQkhkeE5nSUFEQUlMQWtBZ0F5QUhLQUlRUmdSQUlBY2dBVFlDRUF3QkN5QUhJQUUyQWhRTElBRkZEUUVMSUFFZ0J6WUNHQ0FES0FJUUlnSUVRQ0FCSUFJMkFoQWdBaUFCTmdJWUN5QURLQUlVSWdKRkRRQWdBU0FDTmdJVUlBSWdBVFlDR0FzZ0F5QUZUdzBBSUFVb0FnUWlCRUVCY1VVTkFBSkFBa0FDUUFKQUlBUkJBbkZGQkVCQm9BZ29BZ0FnQlVZRVFFR2dDQ0FETmdJQVFaUUlRWlFJS0FJQUlBQnFJZ0EyQWdBZ0F5QUFRUUZ5TmdJRUlBTkJuQWdvQWdCSERRWkJrQWhCQURZQ0FFR2NDRUVBTmdJQUR3dEJuQWdvQWdBaUJ5QUZSZ1JBUVp3SUlBTTJBZ0JCa0FoQmtBZ29BZ0FnQUdvaUFEWUNBQ0FESUFCQkFYSTJBZ1FnQUNBRGFpQUFOZ0lBRHdzZ0JFRjRjU0FBYWlFQUlBVW9BZ3doQVNBRVFmOEJUUVJBSUFVb0FnZ2lBaUFCUmdSQVFZZ0lRWWdJS0FJQVFYNGdCRUVEZG5keE5nSUFEQVVMSUFJZ0FUWUNEQ0FCSUFJMkFnZ01CQXNnQlNnQ0dDRUlJQUVnQlVjRVFDQUZLQUlJSWdJZ0FUWUNEQ0FCSUFJMkFnZ01Bd3NnQlNnQ0ZDSUNCSDhnQlVFVWFnVWdCU2dDRUNJQ1JRMENJQVZCRUdvTElRUURRQ0FFSVFZZ0FpSUJRUlJxSVFRZ0FTZ0NGQ0lDRFFBZ0FVRVFhaUVFSUFFb0FoQWlBZzBBQ3lBR1FRQTJBZ0FNQWdzZ0JTQUVRWDV4TmdJRUlBTWdBRUVCY2pZQ0JDQUFJQU5xSUFBMkFnQU1Bd3RCQUNFQkN5QUlSUTBBQWtBZ0JTZ0NIQ0lFUVFKMFFiZ0thaUlDS0FJQUlBVkdCRUFnQWlBQk5nSUFJQUVOQVVHTUNFR01DQ2dDQUVGK0lBUjNjVFlDQUF3Q0N3SkFJQVVnQ0NnQ0VFWUVRQ0FJSUFFMkFoQU1BUXNnQ0NBQk5nSVVDeUFCUlEwQkN5QUJJQWcyQWhnZ0JTZ0NFQ0lDQkVBZ0FTQUNOZ0lRSUFJZ0FUWUNHQXNnQlNnQ0ZDSUNSUTBBSUFFZ0FqWUNGQ0FDSUFFMkFoZ0xJQU1nQUVFQmNqWUNCQ0FBSUFOcUlBQTJBZ0FnQXlBSFJ3MEFRWkFJSUFBMkFnQVBDeUFBUWY4QlRRUkFJQUJCZUhGQnNBaHFJUUlDZjBHSUNDZ0NBQ0lFUVFFZ0FFRURkblFpQUhGRkJFQkJpQWdnQUNBRWNqWUNBQ0FDREFFTElBSW9BZ2dMSVFBZ0FpQUROZ0lJSUFBZ0F6WUNEQ0FESUFJMkFnd2dBeUFBTmdJSUR3dEJIeUVCSUFCQi8vLy9CMDBFUUNBQVFTWWdBRUVJZG1jaUFtdDJRUUZ4SUFKQkFYUnJRVDVxSVFFTElBTWdBVFlDSENBRFFnQTNBaEFnQVVFQ2RFRzRDbW9oQkFKL0FrQUNmMEdNQ0NnQ0FDSUdRUUVnQVhRaUFuRkZCRUJCakFnZ0FpQUdjallDQUNBRUlBTTJBZ0JCR0NFQlFRZ01BUXNnQUVFWklBRkJBWFpyUVFBZ0FVRWZSeHQwSVFFZ0JDZ0NBQ0VFQTBBZ0JDSUNLQUlFUVhoeElBQkdEUUlnQVVFZGRpRUVJQUZCQVhRaEFTQUNJQVJCQkhGcUlnWW9BaEFpQkEwQUN5QUdJQU0yQWhCQkdDRUJJQUloQkVFSUN5RUFJQU1pQWd3QkN5QUNLQUlJSWdRZ0F6WUNEQ0FDSUFNMkFnaEJHQ0VBUVFnaEFVRUFDeUVHSUFFZ0Eyb2dCRFlDQUNBRElBSTJBZ3dnQUNBRGFpQUdOZ0lBUWFnSVFhZ0lLQUlBUVFGcklnQkJmeUFBR3pZQ0FBc0wwU2NCQzM4akFFRVFheUlLSkFBQ1FBSkFBa0FDUUFKQUFrQUNRQUpBQWtBQ1FDQUFRZlFCVFFSQVFZZ0lLQUlBSWdSQkVDQUFRUXRxUWZnRGNTQUFRUXRKR3lJR1FRTjJJZ0IySWdGQkEzRUVRQUpBSUFGQmYzTkJBWEVnQUdvaUFrRURkQ0lCUWJBSWFpSUFJQUZCdUFocUtBSUFJZ0VvQWdnaUJVWUVRRUdJQ0NBRVFYNGdBbmR4TmdJQURBRUxJQVVnQURZQ0RDQUFJQVUyQWdnTElBRkJDR29oQUNBQklBSkJBM1FpQWtFRGNqWUNCQ0FCSUFKcUlnRWdBU2dDQkVFQmNqWUNCQXdMQ3lBR1FaQUlLQUlBSWdoTkRRRWdBUVJBQWtCQkFpQUFkQ0lDUVFBZ0FtdHlJQUVnQUhSeGFDSUJRUU4wSWdCQnNBaHFJZ0lnQUVHNENHb29BZ0FpQUNnQ0NDSUZSZ1JBUVlnSUlBUkJmaUFCZDNFaUJEWUNBQXdCQ3lBRklBSTJBZ3dnQWlBRk5nSUlDeUFBSUFaQkEzSTJBZ1FnQUNBR2FpSUhJQUZCQTNRaUFTQUdheUlGUVFGeU5nSUVJQUFnQVdvZ0JUWUNBQ0FJQkVBZ0NFRjRjVUd3Q0dvaEFVR2NDQ2dDQUNFQ0FuOGdCRUVCSUFoQkEzWjBJZ054UlFSQVFZZ0lJQU1nQkhJMkFnQWdBUXdCQ3lBQktBSUlDeUVESUFFZ0FqWUNDQ0FESUFJMkFnd2dBaUFCTmdJTUlBSWdBellDQ0FzZ0FFRUlhaUVBUVp3SUlBYzJBZ0JCa0FnZ0JUWUNBQXdMQzBHTUNDZ0NBQ0lMUlEwQklBdG9RUUowUWJnS2FpZ0NBQ0lDS0FJRVFYaHhJQVpySVFNZ0FpRUJBMEFDUUNBQktBSVFJZ0JGQkVBZ0FTZ0NGQ0lBUlEwQkN5QUFLQUlFUVhoeElBWnJJZ0VnQXlBQklBTkpJZ0ViSVFNZ0FDQUNJQUViSVFJZ0FDRUJEQUVMQ3lBQ0tBSVlJUWtnQWlBQ0tBSU1JZ0JIQkVBZ0FpZ0NDQ0lCSUFBMkFnd2dBQ0FCTmdJSURBb0xJQUlvQWhRaUFRUi9JQUpCRkdvRklBSW9BaEFpQVVVTkF5QUNRUkJxQ3lFRkEwQWdCU0VISUFFaUFFRVVhaUVGSUFBb0FoUWlBUTBBSUFCQkVHb2hCU0FBS0FJUUlnRU5BQXNnQjBFQU5nSUFEQWtMUVg4aEJpQUFRYjkvU3cwQUlBQkJDMm9pQVVGNGNTRUdRWXdJS0FJQUlnZEZEUUJCSHlFSVFRQWdCbXNoQXlBQVFmVC8vd2ROQkVBZ0JrRW1JQUZCQ0habklnQnJka0VCY1NBQVFRRjBhMEUrYWlFSUN3SkFBa0FDUUNBSVFRSjBRYmdLYWlnQ0FDSUJSUVJBUVFBaEFBd0JDMEVBSVFBZ0JrRVpJQWhCQVhaclFRQWdDRUVmUnh0MElRSURRQUpBSUFFb0FnUkJlSEVnQm1zaUJDQURUdzBBSUFFaEJTQUVJZ01OQUVFQUlRTWdBU0VBREFNTElBQWdBU2dDRkNJRUlBUWdBU0FDUVIxMlFRUnhhaWdDRUNJQlJoc2dBQ0FFR3lFQUlBSkJBWFFoQWlBQkRRQUxDeUFBSUFWeVJRUkFRUUFoQlVFQ0lBaDBJZ0JCQUNBQWEzSWdCM0VpQUVVTkF5QUFhRUVDZEVHNENtb29BZ0FoQUFzZ0FFVU5BUXNEUUNBQUtBSUVRWGh4SUFacklnSWdBMGtoQVNBQ0lBTWdBUnNoQXlBQUlBVWdBUnNoQlNBQUtBSVFJZ0VFZnlBQkJTQUFLQUlVQ3lJQURRQUxDeUFGUlEwQUlBTkJrQWdvQWdBZ0JtdFBEUUFnQlNnQ0dDRUlJQVVnQlNnQ0RDSUFSd1JBSUFVb0FnZ2lBU0FBTmdJTUlBQWdBVFlDQ0F3SUN5QUZLQUlVSWdFRWZ5QUZRUlJxQlNBRktBSVFJZ0ZGRFFNZ0JVRVFhZ3NoQWdOQUlBSWhCQ0FCSWdCQkZHb2hBaUFBS0FJVUlnRU5BQ0FBUVJCcUlRSWdBQ2dDRUNJQkRRQUxJQVJCQURZQ0FBd0hDeUFHUVpBSUtBSUFJZ1ZOQkVCQm5BZ29BZ0FoQUFKQUlBVWdCbXNpQVVFUVR3UkFJQUFnQm1vaUFpQUJRUUZ5TmdJRUlBQWdCV29nQVRZQ0FDQUFJQVpCQTNJMkFnUU1BUXNnQUNBRlFRTnlOZ0lFSUFBZ0JXb2lBU0FCS0FJRVFRRnlOZ0lFUVFBaEFrRUFJUUVMUVpBSUlBRTJBZ0JCbkFnZ0FqWUNBQ0FBUVFocUlRQU1DUXNnQmtHVUNDZ0NBQ0lDU1FSQVFaUUlJQUlnQm1zaUFUWUNBRUdnQ0VHZ0NDZ0NBQ0lBSUFacUlnSTJBZ0FnQWlBQlFRRnlOZ0lFSUFBZ0JrRURjallDQkNBQVFRaHFJUUFNQ1F0QkFDRUFJQVpCTDJvaUF3Si9RZUFMS0FJQUJFQkI2QXNvQWdBTUFRdEI3QXRDZnpjQ0FFSGtDMEtBb0lDQWdJQUVOd0lBUWVBTElBcEJER3BCY0hGQjJLclZxZ1Z6TmdJQVFmUUxRUUEyQWdCQnhBdEJBRFlDQUVHQUlBc2lBV29pQkVFQUlBRnJJZ2R4SWdFZ0JrME5DRUhBQ3lnQ0FDSUZCRUJCdUFzb0FnQWlDQ0FCYWlJSklBaE5JQVVnQ1VseURRa0xBa0JCeEFzdEFBQkJCSEZGQkVBQ1FBSkFBa0FDUUVHZ0NDZ0NBQ0lGQkVCQnlBc2hBQU5BSUFBb0FnQWlDQ0FGVFFSQUlBVWdDQ0FBS0FJRWFra05Bd3NnQUNnQ0NDSUFEUUFMQzBFQUVBRWlBa0YvUmcwRElBRWhCRUhrQ3lnQ0FDSUFRUUZySWdVZ0FuRUVRQ0FCSUFKcklBSWdCV3BCQUNBQWEzRnFJUVFMSUFRZ0JrME5BMEhBQ3lnQ0FDSUFCRUJCdUFzb0FnQWlCU0FFYWlJSElBVk5JQUFnQjBseURRUUxJQVFRQVNJQUlBSkhEUUVNQlFzZ0JDQUNheUFIY1NJRUVBRWlBaUFBS0FJQUlBQW9BZ1JxUmcwQklBSWhBQXNnQUVGL1JnMEJJQVpCTUdvZ0JFMEVRQ0FBSVFJTUJBdEI2QXNvQWdBaUFpQURJQVJyYWtFQUlBSnJjU0lDRUFGQmYwWU5BU0FDSUFScUlRUWdBQ0VDREFNTElBSkJmMGNOQWd0QnhBdEJ4QXNvQWdCQkJISTJBZ0FMSUFFUUFTSUNRWDlHUVFBUUFTSUFRWDlHY2lBQUlBSk5jZzBGSUFBZ0Ftc2lCQ0FHUVNocVRRMEZDMEc0QzBHNEN5Z0NBQ0FFYWlJQU5nSUFRYndMS0FJQUlBQkpCRUJCdkFzZ0FEWUNBQXNDUUVHZ0NDZ0NBQ0lEQkVCQnlBc2hBQU5BSUFJZ0FDZ0NBQ0lCSUFBb0FnUWlCV3BHRFFJZ0FDZ0NDQ0lBRFFBTERBUUxRWmdJS0FJQUlnQkJBQ0FBSUFKTkcwVUVRRUdZQ0NBQ05nSUFDMEVBSVFCQnpBc2dCRFlDQUVISUN5QUNOZ0lBUWFnSVFYODJBZ0JCckFoQjRBc29BZ0EyQWdCQjFBdEJBRFlDQUFOQUlBQkJBM1FpQVVHNENHb2dBVUd3Q0dvaUJUWUNBQ0FCUWJ3SWFpQUZOZ0lBSUFCQkFXb2lBRUVnUncwQUMwR1VDQ0FFUVNocklnQkJlQ0FDYTBFSGNTSUJheUlGTmdJQVFhQUlJQUVnQW1vaUFUWUNBQ0FCSUFWQkFYSTJBZ1FnQUNBQ2FrRW9OZ0lFUWFRSVFmQUxLQUlBTmdJQURBUUxJQUlnQTAwZ0FTQURTM0lOQWlBQUtBSU1RUWh4RFFJZ0FDQUVJQVZxTmdJRVFhQUlJQU5CZUNBRGEwRUhjU0lBYWlJQk5nSUFRWlFJUVpRSUtBSUFJQVJxSWdJZ0FHc2lBRFlDQUNBQklBQkJBWEkyQWdRZ0FpQURha0VvTmdJRVFhUUlRZkFMS0FJQU5nSUFEQU1MUVFBaEFBd0dDMEVBSVFBTUJBdEJtQWdvQWdBZ0Frc0VRRUdZQ0NBQ05nSUFDeUFDSUFScUlRVkJ5QXNoQUFKQUEwQWdCU0FBS0FJQUlnRkhCRUFnQUNnQ0NDSUFEUUVNQWdzTElBQXRBQXhCQ0hGRkRRTUxRY2dMSVFBRFFBSkFJQUFvQWdBaUFTQURUUVJBSUFNZ0FTQUFLQUlFYWlJRlNRMEJDeUFBS0FJSUlRQU1BUXNMUVpRSUlBUkJLR3NpQUVGNElBSnJRUWR4SWdGcklnYzJBZ0JCb0FnZ0FTQUNhaUlCTmdJQUlBRWdCMEVCY2pZQ0JDQUFJQUpxUVNnMkFnUkJwQWhCOEFzb0FnQTJBZ0FnQXlBRlFTY2dCV3RCQjNGcVFTOXJJZ0FnQUNBRFFSQnFTUnNpQVVFYk5nSUVJQUZCMEFzcEFnQTNBaEFnQVVISUN5a0NBRGNDQ0VIUUN5QUJRUWhxTmdJQVFjd0xJQVEyQWdCQnlBc2dBallDQUVIVUMwRUFOZ0lBSUFGQkdHb2hBQU5BSUFCQkJ6WUNCQ0FBUVFocUlBQkJCR29oQUNBRlNRMEFDeUFCSUFOR0RRQWdBU0FCS0FJRVFYNXhOZ0lFSUFNZ0FTQURheUlDUVFGeU5nSUVJQUVnQWpZQ0FBSi9JQUpCL3dGTkJFQWdBa0Y0Y1VHd0NHb2hBQUovUVlnSUtBSUFJZ0ZCQVNBQ1FRTjJkQ0lDY1VVRVFFR0lDQ0FCSUFKeU5nSUFJQUFNQVFzZ0FDZ0NDQXNoQVNBQUlBTTJBZ2dnQVNBRE5nSU1RUXdoQWtFSURBRUxRUjhoQUNBQ1FmLy8vd2ROQkVBZ0FrRW1JQUpCQ0habklnQnJka0VCY1NBQVFRRjBhMEUrYWlFQUN5QURJQUEyQWh3Z0EwSUFOd0lRSUFCQkFuUkJ1QXBxSVFFQ1FBSkFRWXdJS0FJQUlnVkJBU0FBZENJRWNVVUVRRUdNQ0NBRUlBVnlOZ0lBSUFFZ0F6WUNBQXdCQ3lBQ1FSa2dBRUVCZG10QkFDQUFRUjlIRzNRaEFDQUJLQUlBSVFVRFFDQUZJZ0VvQWdSQmVIRWdBa1lOQWlBQVFSMTJJUVVnQUVFQmRDRUFJQUVnQlVFRWNXb2lCQ2dDRUNJRkRRQUxJQVFnQXpZQ0VBc2dBeUFCTmdJWVFRZ2hBaUFESWdFaEFFRU1EQUVMSUFFb0FnZ2lBQ0FETmdJTUlBRWdBellDQ0NBRElBQTJBZ2hCQUNFQVFSZ2hBa0VNQ3lBRGFpQUJOZ0lBSUFJZ0Eyb2dBRFlDQUF0QmxBZ29BZ0FpQUNBR1RRMEFRWlFJSUFBZ0Jtc2lBVFlDQUVHZ0NFR2dDQ2dDQUNJQUlBWnFJZ0kyQWdBZ0FpQUJRUUZ5TmdJRUlBQWdCa0VEY2pZQ0JDQUFRUWhxSVFBTUJBdEJoQWhCTURZQ0FFRUFJUUFNQXdzZ0FDQUNOZ0lBSUFBZ0FDZ0NCQ0FFYWpZQ0JDQUNRWGdnQW10QkIzRnFJZ2dnQmtFRGNqWUNCQ0FCUVhnZ0FXdEJCM0ZxSWdRZ0JpQUlhaUlEYXlFSEFrQkJvQWdvQWdBZ0JFWUVRRUdnQ0NBRE5nSUFRWlFJUVpRSUtBSUFJQWRxSWdBMkFnQWdBeUFBUVFGeU5nSUVEQUVMUVp3SUtBSUFJQVJHQkVCQm5BZ2dBellDQUVHUUNFR1FDQ2dDQUNBSGFpSUFOZ0lBSUFNZ0FFRUJjallDQkNBQUlBTnFJQUEyQWdBTUFRc2dCQ2dDQkNJQVFRTnhRUUZHQkVBZ0FFRjRjU0VKSUFRb0Fnd2hBZ0pBSUFCQi93Rk5CRUFnQkNnQ0NDSUJJQUpHQkVCQmlBaEJpQWdvQWdCQmZpQUFRUU4yZDNFMkFnQU1BZ3NnQVNBQ05nSU1JQUlnQVRZQ0NBd0JDeUFFS0FJWUlRWUNRQ0FDSUFSSEJFQWdCQ2dDQ0NJQUlBSTJBZ3dnQWlBQU5nSUlEQUVMQWtBZ0JDZ0NGQ0lBQkg4Z0JFRVVhZ1VnQkNnQ0VDSUFSUTBCSUFSQkVHb0xJUUVEUUNBQklRVWdBQ0lDUVJScUlRRWdBQ2dDRkNJQURRQWdBa0VRYWlFQklBSW9BaEFpQUEwQUN5QUZRUUEyQWdBTUFRdEJBQ0VDQ3lBR1JRMEFBa0FnQkNnQ0hDSUFRUUowUWJnS2FpSUJLQUlBSUFSR0JFQWdBU0FDTmdJQUlBSU5BVUdNQ0VHTUNDZ0NBRUYrSUFCM2NUWUNBQXdDQ3dKQUlBUWdCaWdDRUVZRVFDQUdJQUkyQWhBTUFRc2dCaUFDTmdJVUN5QUNSUTBCQ3lBQ0lBWTJBaGdnQkNnQ0VDSUFCRUFnQWlBQU5nSVFJQUFnQWpZQ0dBc2dCQ2dDRkNJQVJRMEFJQUlnQURZQ0ZDQUFJQUkyQWhnTElBY2dDV29oQnlBRUlBbHFJZ1FvQWdRaEFBc2dCQ0FBUVg1eE5nSUVJQU1nQjBFQmNqWUNCQ0FESUFkcUlBYzJBZ0FnQjBIL0FVMEVRQ0FIUVhoeFFiQUlhaUVBQW45QmlBZ29BZ0FpQVVFQklBZEJBM1owSWdKeFJRUkFRWWdJSUFFZ0FuSTJBZ0FnQUF3QkN5QUFLQUlJQ3lFQklBQWdBellDQ0NBQklBTTJBZ3dnQXlBQU5nSU1JQU1nQVRZQ0NBd0JDMEVmSVFJZ0IwSC8vLzhIVFFSQUlBZEJKaUFIUVFoMlp5SUFhM1pCQVhFZ0FFRUJkR3RCUG1vaEFnc2dBeUFDTmdJY0lBTkNBRGNDRUNBQ1FRSjBRYmdLYWlFQUFrQUNRRUdNQ0NnQ0FDSUJRUUVnQW5RaUJYRkZCRUJCakFnZ0FTQUZjallDQUNBQUlBTTJBZ0FNQVFzZ0IwRVpJQUpCQVhaclFRQWdBa0VmUnh0MElRSWdBQ2dDQUNFQkEwQWdBU0lBS0FJRVFYaHhJQWRHRFFJZ0FrRWRkaUVCSUFKQkFYUWhBaUFBSUFGQkJIRnFJZ1VvQWhBaUFRMEFDeUFGSUFNMkFoQUxJQU1nQURZQ0dDQURJQU0yQWd3Z0F5QUROZ0lJREFFTElBQW9BZ2dpQVNBRE5nSU1JQUFnQXpZQ0NDQURRUUEyQWhnZ0F5QUFOZ0lNSUFNZ0FUWUNDQXNnQ0VFSWFpRUFEQUlMQWtBZ0NFVU5BQUpBSUFVb0Fod2lBVUVDZEVHNENtb2lBaWdDQUNBRlJnUkFJQUlnQURZQ0FDQUFEUUZCakFnZ0IwRitJQUYzY1NJSE5nSUFEQUlMQWtBZ0JTQUlLQUlRUmdSQUlBZ2dBRFlDRUF3QkN5QUlJQUEyQWhRTElBQkZEUUVMSUFBZ0NEWUNHQ0FGS0FJUUlnRUVRQ0FBSUFFMkFoQWdBU0FBTmdJWUN5QUZLQUlVSWdGRkRRQWdBQ0FCTmdJVUlBRWdBRFlDR0FzQ1FDQURRUTlOQkVBZ0JTQURJQVpxSWdCQkEzSTJBZ1FnQUNBRmFpSUFJQUFvQWdSQkFYSTJBZ1FNQVFzZ0JTQUdRUU55TmdJRUlBVWdCbW9pQkNBRFFRRnlOZ0lFSUFNZ0JHb2dBellDQUNBRFFmOEJUUVJBSUFOQmVIRkJzQWhxSVFBQ2YwR0lDQ2dDQUNJQlFRRWdBMEVEZG5RaUFuRkZCRUJCaUFnZ0FTQUNjallDQUNBQURBRUxJQUFvQWdnTElRRWdBQ0FFTmdJSUlBRWdCRFlDRENBRUlBQTJBZ3dnQkNBQk5nSUlEQUVMUVI4aEFDQURRZi8vL3dkTkJFQWdBMEVtSUFOQkNIWm5JZ0JyZGtFQmNTQUFRUUYwYTBFK2FpRUFDeUFFSUFBMkFod2dCRUlBTndJUUlBQkJBblJCdUFwcUlRRUNRQUpBSUFkQkFTQUFkQ0lDY1VVRVFFR01DQ0FDSUFkeU5nSUFJQUVnQkRZQ0FDQUVJQUUyQWhnTUFRc2dBMEVaSUFCQkFYWnJRUUFnQUVFZlJ4dDBJUUFnQVNnQ0FDRUJBMEFnQVNJQ0tBSUVRWGh4SUFOR0RRSWdBRUVkZGlFQklBQkJBWFFoQUNBQ0lBRkJCSEZxSWdjb0FoQWlBUTBBQ3lBSElBUTJBaEFnQkNBQ05nSVlDeUFFSUFRMkFnd2dCQ0FFTmdJSURBRUxJQUlvQWdnaUFDQUVOZ0lNSUFJZ0JEWUNDQ0FFUVFBMkFoZ2dCQ0FDTmdJTUlBUWdBRFlDQ0FzZ0JVRUlhaUVBREFFTEFrQWdDVVVOQUFKQUlBSW9BaHdpQVVFQ2RFRzRDbW9pQlNnQ0FDQUNSZ1JBSUFVZ0FEWUNBQ0FBRFFGQmpBZ2dDMEYrSUFGM2NUWUNBQXdDQ3dKQUlBSWdDU2dDRUVZRVFDQUpJQUEyQWhBTUFRc2dDU0FBTmdJVUN5QUFSUTBCQ3lBQUlBazJBaGdnQWlnQ0VDSUJCRUFnQUNBQk5nSVFJQUVnQURZQ0dBc2dBaWdDRkNJQlJRMEFJQUFnQVRZQ0ZDQUJJQUEyQWhnTEFrQWdBMEVQVFFSQUlBSWdBeUFHYWlJQVFRTnlOZ0lFSUFBZ0Ftb2lBQ0FBS0FJRVFRRnlOZ0lFREFFTElBSWdCa0VEY2pZQ0JDQUNJQVpxSWdVZ0EwRUJjallDQkNBRElBVnFJQU0yQWdBZ0NBUkFJQWhCZUhGQnNBaHFJUUJCbkFnb0FnQWhBUUovUVFFZ0NFRURkblFpQnlBRWNVVUVRRUdJQ0NBRUlBZHlOZ0lBSUFBTUFRc2dBQ2dDQ0FzaEJDQUFJQUUyQWdnZ0JDQUJOZ0lNSUFFZ0FEWUNEQ0FCSUFRMkFnZ0xRWndJSUFVMkFnQkJrQWdnQXpZQ0FBc2dBa0VJYWlFQUN5QUtRUkJxSkFBZ0FBdkdCQUlHZndwOVFmLy8vLzhISVF4QmdJQ0FnSGdoRFVGL0lRa0RRQ0FESUFwR0JFQkJBQ0VBSUFoQkFFR0FnQkQ4Q3dCREFQOS9SeUFOSUF4cnNwVWhEd1VnQkNBS1FReHNhaUlMS2dJQUlSTWdDeW9DQ0NFVUlBc3FBZ1FoRlNBSklBSWdDa0VDZENJT2FpZ0NBQ0lMUndSQUlBRWdDMEhRQUd4cUlna3FBandnQUNvQ09DSVBsQ0FKS2dJNElBQXFBaWdpRUpRZ0NTb0NNQ0FBS2dJSUloR1VJQUFxQWhnaUVpQUpLZ0kwbEpLU2tpRVdJQWtxQWl3Z0Q1UWdDU29DS0NBUWxDQUpLZ0lnSUJHVUlCSWdDU29DSkpTU2twSWhGeUFKS2dJY0lBK1VJQWtxQWhnZ0VKUWdDU29DRUNBUmxDQVNJQWtxQWhTVWtwS1NJUmdnQ1NvQ0RDQVBsQ0FKS2dJSUlCQ1VJQWtxQWdBZ0VaUWdDU29DQkNBU2xKS1NraUVQSUFzaENRc2dCU0FPYWlBV0lCY2dGSlFnRHlBVGxDQVZJQmlVa3BLU1F3QUFnRVdVL0FBaUN6WUNBQ0FNSUFzZ0N5QU1TaHNoRENBTklBc2dDeUFOU0JzaERTQUtRUUZxSVFvTUFRc0xBMEFnQUNBRFJrVUVRQ0FGSUFCQkFuUnFJZ0VnRHlBQktBSUFJQXhyczVUOEFTSUJOZ0lBSUFnZ0FVRUNkR29pQVNBQktBSUFRUUZxTmdJQUlBQkJBV29oQUF3QkN3dEJBQ0VBSUFkQkFEWUNBQ0FJUVFScklRRkJBQ0VNUVFFaENnTkFJQXBCZ0lBRVJnUkFBMEFDUUNBQUlBTkdEUUFnQnlBRklBQkJBblJxS0FJQVFRSjBhaUlCSUFFb0FnQWlBVUVCYWpZQ0FDQUdJQUZCQW5ScUlBQTJBZ0FnQUVFQmFpRUFEQUVMQ3dVZ0J5QUtRUUowSWdKcUlBRWdBbW9vQWdBZ0RHb2lERFlDQUNBS1FRRnFJUW9NQVFzTEN3SUFDd3NKQVFCQmdRZ0xBZ1lCIik7CiAgfQogIGZ1bmN0aW9uIE8oQSkgewogICAgaWYgKEFycmF5QnVmZmVyLmlzVmlldyhBKSkKICAgICAgcmV0dXJuIEE7CiAgICBpZiAoQSA9PSBTICYmIE0pCiAgICAgIHJldHVybiBuZXcgVWludDhBcnJheShNKTsKICAgIGlmIChVKQogICAgICByZXR1cm4gVShBKTsKICAgIHRocm93ICdzeW5jIGZldGNoaW5nIG9mIHRoZSB3YXNtIGZhaWxlZDogeW91IGNhbiBwcmVsb2FkIGl0IHRvIE1vZHVsZVsid2FzbUJpbmFyeSJdIG1hbnVhbGx5LCBvciBlbWNjLnB5IHdpbGwgZG8gdGhhdCBmb3IgeW91IHdoZW4gZ2VuZXJhdGluZyBIVE1MIChidXQgbm90IEpTKSc7CiAgfQogIGZ1bmN0aW9uICQoQSwgQikgewogICAgdmFyIHQsIEUgPSBPKEEpOwogICAgdCA9IG5ldyBXZWJBc3NlbWJseS5Nb2R1bGUoRSk7CiAgICB2YXIgaSA9IG5ldyBXZWJBc3NlbWJseS5JbnN0YW5jZSh0LCBCKTsKICAgIHJldHVybiBbaSwgdF07CiAgfQogIGZ1bmN0aW9uIEFBKCkgewogICAgcmV0dXJuIHsgYTogbkEgfTsKICB9CiAgZnVuY3Rpb24gSUEoKSB7CiAgICBmdW5jdGlvbiBBKEUsIGkpIHsKICAgICAgcmV0dXJuIHkgPSBFLmV4cG9ydHMsIHUgPSB5LmIsIHEoKSwgckEoeSksIHooKSwgeTsKICAgIH0KICAgIF8oKTsKICAgIHZhciBCID0gQUEoKTsKICAgIGlmIChJLmluc3RhbnRpYXRlV2FzbSkKICAgICAgcmV0dXJuIG5ldyBQcm9taXNlKChFLCBpKSA9PiB7CiAgICAgICAgSS5pbnN0YW50aWF0ZVdhc20oQiwgKG8sIGgpID0+IHsKICAgICAgICAgIEUoQShvKSk7CiAgICAgICAgfSk7CiAgICAgIH0pOwogICAgUyA/Pz0gVigpOwogICAgdmFyIHQgPSAkKFMsIEIpOwogICAgcmV0dXJuIEEodFswXSk7CiAgfQogIGZvciAodmFyIHggPSAoQSkgPT4gewogICAgZm9yICg7IEEubGVuZ3RoID4gMDsgKQogICAgICBBLnNoaWZ0KCkoSSk7CiAgfSwgdiA9IFtdLCBnQSA9IChBKSA9PiB2LnB1c2goQSksIFQgPSBbXSwgQ0EgPSAoQSkgPT4gVC5wdXNoKEEpLCBCQSA9IChBKSA9PiB7CiAgICBmb3IgKHZhciBCLCB0LCBFID0gMCwgaSA9IDAsIG8gPSBBLmxlbmd0aCwgaCA9IG5ldyBVaW50OEFycmF5KChvICogMyA+PiAyKSAtIChBW28gLSAyXSA9PSAiPSIpIC0gKEFbbyAtIDFdID09ICI9IikpOyBFIDwgbzsgRSArPSA0LCBpICs9IDMpCiAgICAgIEIgPSBuW0EuY2hhckNvZGVBdChFICsgMSldLCB0ID0gbltBLmNoYXJDb2RlQXQoRSArIDIpXSwgaFtpXSA9IG5bQS5jaGFyQ29kZUF0KEUpXSA8PCAyIHwgQiA+PiA0LCBoW2kgKyAxXSA9IEIgPDwgNCB8IHQgPj4gMiwgaFtpICsgMl0gPSB0IDw8IDYgfCBuW0EuY2hhckNvZGVBdChFICsgMyldOwogICAgcmV0dXJuIGg7CiAgfSwgUUEgPSAoKSA9PiAyMTQ3NDgzNjQ4LCBFQSA9IChBLCBCKSA9PiBNYXRoLmNlaWwoQSAvIEIpICogQiwgdEEgPSAoQSkgPT4gewogICAgdmFyIEIgPSB1LmJ1ZmZlciwgdCA9IChBIC0gQi5ieXRlTGVuZ3RoICsgNjU1MzUpIC8gNjU1MzYgfCAwOwogICAgdHJ5IHsKICAgICAgcmV0dXJuIHUuZ3Jvdyh0KSwgcSgpLCAxOwogICAgfSBjYXRjaCB7CiAgICB9CiAgfSwgaUEgPSAoQSkgPT4gewogICAgdmFyIEIgPSBMLmxlbmd0aDsKICAgIEEgPj4+PSAwOwogICAgdmFyIHQgPSBRQSgpOwogICAgaWYgKEEgPiB0KQogICAgICByZXR1cm4gITE7CiAgICBmb3IgKHZhciBFID0gMTsgRSA8PSA0OyBFICo9IDIpIHsKICAgICAgdmFyIGkgPSBCICogKDEgKyAwLjIgLyBFKTsKICAgICAgaSA9IE1hdGgubWluKGksIEEgKyAxMDA2NjMyOTYpOwogICAgICB2YXIgbyA9IE1hdGgubWluKHQsIEVBKE1hdGgubWF4KEEsIGkpLCA2NTUzNikpLCBoID0gdEEobyk7CiAgICAgIGlmIChoKQogICAgICAgIHJldHVybiAhMDsKICAgIH0KICAgIHJldHVybiAhMTsKICB9LCBuID0gbmV3IFVpbnQ4QXJyYXkoMTIzKSwgZSA9IDI1OyBlID49IDA7IC0tZSkKICAgIG5bNDggKyBlXSA9IDUyICsgZSwgbls2NSArIGVdID0gZSwgbls5NyArIGVdID0gMjYgKyBlOwogIG5bNDNdID0gNjIsIG5bNDddID0gNjMsIEkubm9FeGl0UnVudGltZSAmJiBJLm5vRXhpdFJ1bnRpbWUsIEkucHJpbnQgJiYgSS5wcmludCwgSS5wcmludEVyciAmJiBJLnByaW50RXJyLCBJLndhc21CaW5hcnkgJiYgKE0gPSBJLndhc21CaW5hcnkpLCBJLmFyZ3VtZW50cyAmJiBJLmFyZ3VtZW50cywgSS50aGlzUHJvZ3JhbSAmJiBJLnRoaXNQcm9ncmFtOwogIGZ1bmN0aW9uIHJBKEEpIHsKICAgIEkuX3NvcnQgPSBBLmQsIEkuX21hbGxvYyA9IEEuZSwgSS5fZnJlZSA9IEEuZjsKICB9CiAgdmFyIG5BID0geyBhOiBpQSB9LCB5ID0gSUEoKTsKICBmdW5jdGlvbiBtKCkgewogICAgaWYgKHMgPiAwKSB7CiAgICAgIGYgPSBtOwogICAgICByZXR1cm47CiAgICB9CiAgICBpZiAoUCgpLCBzID4gMCkgewogICAgICBmID0gbTsKICAgICAgcmV0dXJuOwogICAgfQogICAgZnVuY3Rpb24gQSgpIHsKICAgICAgSS5jYWxsZWRSdW4gPSAhMCwgYigpLCBJLm9uUnVudGltZUluaXRpYWxpemVkPy4oKSwgVygpOwogICAgfQogICAgSS5zZXRTdGF0dXMgPyAoSS5zZXRTdGF0dXMoIlJ1bm5pbmcuLi4iKSwgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgIHNldFRpbWVvdXQoKCkgPT4gSS5zZXRTdGF0dXMoIiIpLCAxKSwgQSgpOwogICAgfSwgMSkpIDogQSgpOwogIH0KICBmdW5jdGlvbiBhQSgpIHsKICAgIGlmIChJLnByZUluaXQpCiAgICAgIGZvciAodHlwZW9mIEkucHJlSW5pdCA9PSAiZnVuY3Rpb24iICYmIChJLnByZUluaXQgPSBbSS5wcmVJbml0XSk7IEkucHJlSW5pdC5sZW5ndGggPiAwOyApCiAgICAgICAgSS5wcmVJbml0LnNoaWZ0KCkoKTsKICB9CiAgcmV0dXJuIGFBKCksIG0oKSwgciA9IEksIHI7Cn07CmxldCBnLCBRLCBkLCBjLCBOLCB3LCBwLCBSLCBrLCBKLCBhID0gMCwgWSA9IDAsIEggPSBbXSwgRyA9ICEwLCBEID0gITEsIEYgPSAhMSwgSyA9ICExOwphc3luYyBmdW5jdGlvbiBvQSgpIHsKICBpZiAoIWcgJiYgKGcgPSBhd2FpdCBlQSgpLCAhZyB8fCAhZy5IRUFQRjMyIHx8ICFnLl9zb3J0KSkKICAgIHRocm93IG5ldyBFcnJvcigiV0FTTSBtb2R1bGUgZmFpbGVkIHRvIGluaXRpYWxpemUgcHJvcGVybHkiKTsKfQpjb25zdCBaID0gYXN5bmMgKCkgPT4gewogIGlmIChEKSB7CiAgICBGID0gITA7CiAgICByZXR1cm47CiAgfQogIEQgPSAhMCwgRiA9ICExLCBnIHx8IGF3YWl0IG9BKCk7CiAgY29uc3QgQyA9IE1hdGgucG93KDIsIE1hdGguY2VpbChNYXRoLmxvZzIoUS52ZXJ0ZXhDb3VudCkpKTsKICBhIDwgQyAmJiAoYSA+IDAgJiYgKGcuX2ZyZWUoZCksIGcuX2ZyZWUoTiksIGcuX2ZyZWUodyksIGcuX2ZyZWUocCksIGcuX2ZyZWUoUiksIGcuX2ZyZWUoayksIGcuX2ZyZWUoSikpLCBhID0gQywgZCA9IGcuX21hbGxvYygxNiAqIDQpLCBOID0gZy5fbWFsbG9jKGEgKiA0KSwgdyA9IGcuX21hbGxvYygzICogYSAqIDQpLCBwID0gZy5fbWFsbG9jKGEgKiA0KSwgUiA9IGcuX21hbGxvYyhhICogNCksIGsgPSBnLl9tYWxsb2MoYSAqIDQpLCBKID0gZy5fbWFsbG9jKGEgKiA0KSksIFkgPCBRLnRyYW5zZm9ybXMubGVuZ3RoICYmIChZID4gMCAmJiBnLl9mcmVlKGMpLCBZID0gUS50cmFuc2Zvcm1zLmxlbmd0aCwgYyA9IGcuX21hbGxvYyhZICogNCkpLCBEID0gITEsIEYgJiYgKEYgPSAhMSwgYXdhaXQgWigpKTsKfSwgc0EgPSAoKSA9PiB7CiAgaWYgKCEoRCB8fCBGIHx8ICFnIHx8ICFRKSkgewogICAgRCA9ICEwOwogICAgdHJ5IHsKICAgICAgY29uc3QgQyA9IGcuSEVBUEYzMiwgciA9IGcuSEVBUFUzMjsKICAgICAgaWYgKHcgLyA0ICsgUS5wb3NpdGlvbnMubGVuZ3RoID4gQy5sZW5ndGgpCiAgICAgICAgdGhyb3cgbmV3IEVycm9yKCJQb3NpdGlvbnMgYnVmZmVyIG92ZXJmbG93Iik7CiAgICAgIGlmIChjIC8gNCArIFEudHJhbnNmb3Jtcy5sZW5ndGggPiBDLmxlbmd0aCkKICAgICAgICB0aHJvdyBuZXcgRXJyb3IoIlRyYW5zZm9ybXMgYnVmZmVyIG92ZXJmbG93Iik7CiAgICAgIGlmIChOIC8gNCArIFEudHJhbnNmb3JtSW5kaWNlcy5sZW5ndGggPiByLmxlbmd0aCkKICAgICAgICB0aHJvdyBuZXcgRXJyb3IoIlRyYW5zZm9ybSBpbmRpY2VzIGJ1ZmZlciBvdmVyZmxvdyIpOwogICAgICBpZiAoQy5zZXQoUS5wb3NpdGlvbnMsIHcgLyA0KSwgQy5zZXQoUS50cmFuc2Zvcm1zLCBjIC8gNCksIHIuc2V0KFEudHJhbnNmb3JtSW5kaWNlcywgTiAvIDQpLCBDLnNldChuZXcgRmxvYXQzMkFycmF5KEgpLCBkIC8gNCksIGcuX3NvcnQoCiAgICAgICAgZCwKICAgICAgICBjLAogICAgICAgIE4sCiAgICAgICAgUS52ZXJ0ZXhDb3VudCwKICAgICAgICB3LAogICAgICAgIHAsCiAgICAgICAgUiwKICAgICAgICBrLAogICAgICAgIEoKICAgICAgKSwgUiArIFEudmVydGV4Q291bnQgKiA0ID4gci5idWZmZXIuYnl0ZUxlbmd0aCkKICAgICAgICB0aHJvdyBuZXcgRXJyb3IoIkRlcHRoIGluZGV4IGJ1ZmZlciBvdmVyZmxvdyIpOwogICAgICBjb25zdCBJID0gbmV3IFVpbnQzMkFycmF5KHIuYnVmZmVyLCBSLCBRLnZlcnRleENvdW50KSwgbCA9IG5ldyBVaW50MzJBcnJheShJLnNsaWNlKCkuYnVmZmVyKTsKICAgICAgc2VsZi5wb3N0TWVzc2FnZSh7IGRlcHRoSW5kZXg6IGwgfSwgW2wuYnVmZmVyXSk7CiAgICB9IGNhdGNoIHsKICAgICAgc2VsZi5wb3N0TWVzc2FnZSh7IGRlcHRoSW5kZXg6IG5ldyBVaW50MzJBcnJheSgwKSB9LCBbXSk7CiAgICB9CiAgICBEID0gITEsIEcgPSAhMTsKICB9Cn0sIFggPSAoKSA9PiB7CiAgSyB8fCAoSyA9ICEwLCBHICYmIHNBKCksIHNldFRpbWVvdXQoKCkgPT4gewogICAgSyA9ICExLCBYKCk7CiAgfSkpOwp9OwpzZWxmLm9ubWVzc2FnZSA9IChDKSA9PiB7CiAgQy5kYXRhLnNvcnREYXRhICYmIChRID8gKFEucG9zaXRpb25zLnNldChDLmRhdGEuc29ydERhdGEucG9zaXRpb25zKSwgUS50cmFuc2Zvcm1zLnNldChDLmRhdGEuc29ydERhdGEudHJhbnNmb3JtcyksIFEudHJhbnNmb3JtSW5kaWNlcy5zZXQoQy5kYXRhLnNvcnREYXRhLnRyYW5zZm9ybUluZGljZXMpLCBRLnZlcnRleENvdW50ID0gQy5kYXRhLnNvcnREYXRhLnZlcnRleENvdW50KSA6IFEgPSB7CiAgICBwb3NpdGlvbnM6IG5ldyBGbG9hdDMyQXJyYXkoQy5kYXRhLnNvcnREYXRhLnBvc2l0aW9ucyksCiAgICB0cmFuc2Zvcm1zOiBuZXcgRmxvYXQzMkFycmF5KEMuZGF0YS5zb3J0RGF0YS50cmFuc2Zvcm1zKSwKICAgIHRyYW5zZm9ybUluZGljZXM6IG5ldyBVaW50MzJBcnJheShDLmRhdGEuc29ydERhdGEudHJhbnNmb3JtSW5kaWNlcyksCiAgICB2ZXJ0ZXhDb3VudDogQy5kYXRhLnNvcnREYXRhLnZlcnRleENvdW50CiAgfSwgRyA9ICEwLCBaKCkpLCBDLmRhdGEudmlld1Byb2ogJiYgKEMuZGF0YS52aWV3UHJvai5ldmVyeSgocikgPT4gSC5pbmNsdWRlcyhyKSkgPT09ICExICYmIChIID0gQy5kYXRhLnZpZXdQcm9qLCBHID0gITApLCBYKCkpOwp9OwovLyMgc291cmNlTWFwcGluZ1VSTD1Tb3J0V29ya2VyLURRTDdVVEVoLmpzLm1hcAo=";
  var Et = (E) => Uint8Array.from(atob(E), (t) => t.charCodeAt(0));
  var Qt = typeof self < "u" && self.Blob && new Blob(["URL.revokeObjectURL(import.meta.url);", Et(Bt)], { type: "text/javascript;charset=utf-8" });
  function St(E) {
    let t;
    try {
      if (t = Qt && (self.URL || self.webkitURL).createObjectURL(Qt), !t) throw "";
      const n = new Worker(t, {
        type: "module",
        name: E?.name
      });
      return n.addEventListener("error", () => {
        (self.URL || self.webkitURL).revokeObjectURL(t);
      }), n;
    } catch {
      return new Worker(
        "data:text/javascript;base64," + Bt,
        {
          type: "module",
          name: E?.name
        }
      );
    }
  }
  var lt = class {
    constructor(t, n) {
      this._scene = null, this._camera = null, this._started = false, this._initialized = false, this._renderer = t;
      const i = t.gl;
      this._program = i.createProgram(), this._passes = n || [];
      const e = i.createShader(i.VERTEX_SHADER);
      i.shaderSource(e, this._getVertexSource()), i.compileShader(e), i.getShaderParameter(e, i.COMPILE_STATUS) || console.error(i.getShaderInfoLog(e));
      const A = i.createShader(i.FRAGMENT_SHADER);
      i.shaderSource(A, this._getFragmentSource()), i.compileShader(A), i.getShaderParameter(A, i.COMPILE_STATUS) || console.error(i.getShaderInfoLog(A)), i.attachShader(this.program, e), i.attachShader(this.program, A), i.linkProgram(this.program), i.getProgramParameter(this.program, i.LINK_STATUS) || console.error(i.getProgramInfoLog(this.program)), this.resize = () => {
        i.useProgram(this._program), this._resize();
      }, this.initialize = () => {
        console.assert(!this._initialized, "ShaderProgram already initialized"), i.useProgram(this._program), this._initialize();
        for (const o of this.passes)
          o.initialize(this);
        this._initialized = true, this._started = true;
      }, this.render = (o, s) => {
        i.useProgram(this._program), (this._scene !== o || this._camera !== s) && (this.dispose(), this._scene = o, this._camera = s, this.initialize());
        for (const r of this.passes)
          r.render();
        this._render();
      }, this.dispose = () => {
        if (this._initialized) {
          i.useProgram(this._program);
          for (const o of this.passes)
            o.dispose();
          this._dispose(), this._scene = null, this._camera = null, this._initialized = false;
        }
      };
    }
    get renderer() {
      return this._renderer;
    }
    get scene() {
      return this._scene;
    }
    get camera() {
      return this._camera;
    }
    get program() {
      return this._program;
    }
    get passes() {
      return this._passes;
    }
    get started() {
      return this._started;
    }
  };
  var gt = "dmFyIG9BID0gZnVuY3Rpb24oQyA9IHt9KSB7CiAgdmFyIGUsIEEgPSBDLCBoID0gaW1wb3J0Lm1ldGEudXJsLCBrID0gIiIsIHk7CiAgewogICAgdHJ5IHsKICAgICAgayA9IG5ldyBVUkwoIi4iLCBoKS5ocmVmOwogICAgfSBjYXRjaCB7CiAgICB9CiAgICB5ID0gKGcpID0+IHsKICAgICAgdmFyIEIgPSBuZXcgWE1MSHR0cFJlcXVlc3QoKTsKICAgICAgcmV0dXJuIEIub3BlbigiR0VUIiwgZywgITEpLCBCLnJlc3BvbnNlVHlwZSA9ICJhcnJheWJ1ZmZlciIsIEIuc2VuZChudWxsKSwgbmV3IFVpbnQ4QXJyYXkoQi5yZXNwb25zZSk7CiAgICB9OwogIH0KICBjb25zb2xlLmxvZy5iaW5kKGNvbnNvbGUpLCBjb25zb2xlLmVycm9yLmJpbmQoY29uc29sZSk7CiAgdmFyIEYsIEQsIE47CiAgZnVuY3Rpb24gVSgpIHsKICAgIHZhciBnID0gRC5idWZmZXI7CiAgICBBLkhFQVBVOCA9IE4gPSBuZXcgVWludDhBcnJheShnKSwgQS5IRUFQVTMyID0gbmV3IFVpbnQzMkFycmF5KGcpLCBBLkhFQVBGMzIgPSBuZXcgRmxvYXQzMkFycmF5KGcpLCBuZXcgQmlnSW50NjRBcnJheShnKSwgbmV3IEJpZ1VpbnQ2NEFycmF5KGcpOwogIH0KICBmdW5jdGlvbiBpKCkgewogICAgaWYgKEEucHJlUnVuKQogICAgICBmb3IgKHR5cGVvZiBBLnByZVJ1biA9PSAiZnVuY3Rpb24iICYmIChBLnByZVJ1biA9IFtBLnByZVJ1bl0pOyBBLnByZVJ1bi5sZW5ndGg7ICkKICAgICAgICAkKEEucHJlUnVuLnNoaWZ0KCkpOwogICAgeChUKTsKICB9CiAgZnVuY3Rpb24gSCgpIHsKICAgIGYuYygpOwogIH0KICBmdW5jdGlvbiBiKCkgewogICAgaWYgKEEucG9zdFJ1bikKICAgICAgZm9yICh0eXBlb2YgQS5wb3N0UnVuID09ICJmdW5jdGlvbiIgJiYgKEEucG9zdFJ1biA9IFtBLnBvc3RSdW5dKTsgQS5wb3N0UnVuLmxlbmd0aDsgKQogICAgICAgIE8oQS5wb3N0UnVuLnNoaWZ0KCkpOwogICAgeChqKTsKICB9CiAgdmFyIHMgPSAwLCB3ID0gbnVsbDsKICBmdW5jdGlvbiB2KGcpIHsKICAgIHMrKywgQS5tb25pdG9yUnVuRGVwZW5kZW5jaWVzPy4ocyk7CiAgfQogIGZ1bmN0aW9uIFAoZykgewogICAgaWYgKHMtLSwgQS5tb25pdG9yUnVuRGVwZW5kZW5jaWVzPy4ocyksIHMgPT0gMCAmJiB3KSB7CiAgICAgIHZhciBCID0gdzsKICAgICAgdyA9IG51bGwsIEIoKTsKICAgIH0KICB9CiAgdmFyIG07CiAgZnVuY3Rpb24gWCgpIHsKICAgIHJldHVybiBBQSgiQUdGemJRRUFBQUFCSmdaZ0FYOEJmMkFDZlgwQmYyQUJmUUYvWUFGL0FHQUxmMzkvZjM5L2YzOS9mMzhBWUFBQUFnY0JBV0VCWVFBQUF3Z0hBQUVDQXdBRUJRVUhBUUdDQW9DQUFnWUlBWDhCUVlDTUJBc0hGUVVCWWdJQUFXTUFCd0ZrQUFZQlpRQUZBV1lBQkF3QkFRcXFRQWRQQVFKL1FZQUlLQUlBSWdFZ0FFRUhha0Y0Y1NJQ2FpRUFBa0FnQWtFQUlBQWdBVTBiUlFSQUlBQS9BRUVRZEUwTkFTQUFFQUFOQVF0QmhBaEJNRFlDQUVGL0R3dEJnQWdnQURZQ0FDQUJDdzRBSUFBUUF5QUJFQU5CRUhSeUMzSUJCSDhnQUx3aUJFSC8vLzhEY1NFQkFrQWdCRUVYZGtIL0FYRWlBa1VOQUNBQ1FmQUFUUVJBSUFGQmdJQ0FCSEpCOFFBZ0FtdDJJUUVNQVFzZ0FrR05BVXNFUUVHQStBRWhBMEVBSVFFTUFRc2dBa0VLZEVHQWdBZHJJUU1MSUFNZ0JFRVFka0dBZ0FKeGNpQUJRUTEyY2d2Y0N3RUlmd0pBSUFCRkRRQWdBRUVJYXlJRElBQkJCR3NvQWdBaUFrRjRjU0lBYWlFRkFrQWdBa0VCY1EwQUlBSkJBbkZGRFFFZ0F5QURLQUlBSWdScklnTkJtQWdvQWdCSkRRRWdBQ0FFYWlFQUFrQUNRQUpBUVp3SUtBSUFJQU5IQkVBZ0F5Z0NEQ0VCSUFSQi93Rk5CRUFnQVNBREtBSUlJZ0pIRFFKQmlBaEJpQWdvQWdCQmZpQUVRUU4yZDNFMkFnQU1CUXNnQXlnQ0dDRUhJQUVnQTBjRVFDQURLQUlJSWdJZ0FUWUNEQ0FCSUFJMkFnZ01CQXNnQXlnQ0ZDSUNCSDhnQTBFVWFnVWdBeWdDRUNJQ1JRMERJQU5CRUdvTElRUURRQ0FFSVFZZ0FpSUJRUlJxSVFRZ0FTZ0NGQ0lDRFFBZ0FVRVFhaUVFSUFFb0FoQWlBZzBBQ3lBR1FRQTJBZ0FNQXdzZ0JTZ0NCQ0lDUVFOeFFRTkhEUU5Ca0FnZ0FEWUNBQ0FGSUFKQmZuRTJBZ1FnQXlBQVFRRnlOZ0lFSUFVZ0FEWUNBQThMSUFJZ0FUWUNEQ0FCSUFJMkFnZ01BZ3RCQUNFQkN5QUhSUTBBQWtBZ0F5Z0NIQ0lFUVFKMFFiZ0thaUlDS0FJQUlBTkdCRUFnQWlBQk5nSUFJQUVOQVVHTUNFR01DQ2dDQUVGK0lBUjNjVFlDQUF3Q0N3SkFJQU1nQnlnQ0VFWUVRQ0FISUFFMkFoQU1BUXNnQnlBQk5nSVVDeUFCUlEwQkN5QUJJQWMyQWhnZ0F5Z0NFQ0lDQkVBZ0FTQUNOZ0lRSUFJZ0FUWUNHQXNnQXlnQ0ZDSUNSUTBBSUFFZ0FqWUNGQ0FDSUFFMkFoZ0xJQU1nQlU4TkFDQUZLQUlFSWdSQkFYRkZEUUFDUUFKQUFrQUNRQ0FFUVFKeFJRUkFRYUFJS0FJQUlBVkdCRUJCb0FnZ0F6WUNBRUdVQ0VHVUNDZ0NBQ0FBYWlJQU5nSUFJQU1nQUVFQmNqWUNCQ0FEUVp3SUtBSUFSdzBHUVpBSVFRQTJBZ0JCbkFoQkFEWUNBQThMUVp3SUtBSUFJZ2NnQlVZRVFFR2NDQ0FETmdJQVFaQUlRWkFJS0FJQUlBQnFJZ0EyQWdBZ0F5QUFRUUZ5TmdJRUlBQWdBMm9nQURZQ0FBOExJQVJCZUhFZ0FHb2hBQ0FGS0FJTUlRRWdCRUgvQVUwRVFDQUZLQUlJSWdJZ0FVWUVRRUdJQ0VHSUNDZ0NBRUYrSUFSQkEzWjNjVFlDQUF3RkN5QUNJQUUyQWd3Z0FTQUNOZ0lJREFRTElBVW9BaGdoQ0NBQklBVkhCRUFnQlNnQ0NDSUNJQUUyQWd3Z0FTQUNOZ0lJREFNTElBVW9BaFFpQWdSL0lBVkJGR29GSUFVb0FoQWlBa1VOQWlBRlFSQnFDeUVFQTBBZ0JDRUdJQUlpQVVFVWFpRUVJQUVvQWhRaUFnMEFJQUZCRUdvaEJDQUJLQUlRSWdJTkFBc2dCa0VBTmdJQURBSUxJQVVnQkVGK2NUWUNCQ0FESUFCQkFYSTJBZ1FnQUNBRGFpQUFOZ0lBREFNTFFRQWhBUXNnQ0VVTkFBSkFJQVVvQWh3aUJFRUNkRUc0Q21vaUFpZ0NBQ0FGUmdSQUlBSWdBVFlDQUNBQkRRRkJqQWhCakFnb0FnQkJmaUFFZDNFMkFnQU1BZ3NDUUNBRklBZ29BaEJHQkVBZ0NDQUJOZ0lRREFFTElBZ2dBVFlDRkFzZ0FVVU5BUXNnQVNBSU5nSVlJQVVvQWhBaUFnUkFJQUVnQWpZQ0VDQUNJQUUyQWhnTElBVW9BaFFpQWtVTkFDQUJJQUkyQWhRZ0FpQUJOZ0lZQ3lBRElBQkJBWEkyQWdRZ0FDQURhaUFBTmdJQUlBTWdCMGNOQUVHUUNDQUFOZ0lBRHdzZ0FFSC9BVTBFUUNBQVFYaHhRYkFJYWlFQ0FuOUJpQWdvQWdBaUJFRUJJQUJCQTNaMElnQnhSUVJBUVlnSUlBQWdCSEkyQWdBZ0Fnd0JDeUFDS0FJSUN5RUFJQUlnQXpZQ0NDQUFJQU0yQWd3Z0F5QUNOZ0lNSUFNZ0FEWUNDQThMUVI4aEFTQUFRZi8vL3dkTkJFQWdBRUVtSUFCQkNIWm5JZ0pyZGtFQmNTQUNRUUYwYTBFK2FpRUJDeUFESUFFMkFod2dBMElBTndJUUlBRkJBblJCdUFwcUlRUUNmd0pBQW45QmpBZ29BZ0FpQmtFQklBRjBJZ0p4UlFSQVFZd0lJQUlnQm5JMkFnQWdCQ0FETmdJQVFSZ2hBVUVJREFFTElBQkJHU0FCUVFGMmEwRUFJQUZCSDBjYmRDRUJJQVFvQWdBaEJBTkFJQVFpQWlnQ0JFRjRjU0FBUmcwQ0lBRkJIWFloQkNBQlFRRjBJUUVnQWlBRVFRUnhhaUlHS0FJUUlnUU5BQXNnQmlBRE5nSVFRUmdoQVNBQ0lRUkJDQXNoQUNBRElnSU1BUXNnQWlnQ0NDSUVJQU0yQWd3Z0FpQUROZ0lJUVJnaEFFRUlJUUZCQUFzaEJpQUJJQU5xSUFRMkFnQWdBeUFDTmdJTUlBQWdBMm9nQmpZQ0FFR29DRUdvQ0NnQ0FFRUJheUlBUVg4Z0FCczJBZ0FMQzlFbkFRdC9Jd0JCRUdzaUNpUUFBa0FDUUFKQUFrQUNRQUpBQWtBQ1FBSkFBa0FnQUVIMEFVMEVRRUdJQ0NnQ0FDSUVRUkFnQUVFTGFrSDRBM0VnQUVFTFNSc2lCa0VEZGlJQWRpSUJRUU54QkVBQ1FDQUJRWDl6UVFGeElBQnFJZ0pCQTNRaUFVR3dDR29pQUNBQlFiZ0lhaWdDQUNJQktBSUlJZ1ZHQkVCQmlBZ2dCRUYrSUFKM2NUWUNBQXdCQ3lBRklBQTJBZ3dnQUNBRk5nSUlDeUFCUVFocUlRQWdBU0FDUVFOMElnSkJBM0kyQWdRZ0FTQUNhaUlCSUFFb0FnUkJBWEkyQWdRTUN3c2dCa0dRQ0NnQ0FDSUlUUTBCSUFFRVFBSkFRUUlnQUhRaUFrRUFJQUpyY2lBQklBQjBjV2dpQVVFRGRDSUFRYkFJYWlJQ0lBQkJ1QWhxS0FJQUlnQW9BZ2dpQlVZRVFFR0lDQ0FFUVg0Z0FYZHhJZ1EyQWdBTUFRc2dCU0FDTmdJTUlBSWdCVFlDQ0FzZ0FDQUdRUU55TmdJRUlBQWdCbW9pQnlBQlFRTjBJZ0VnQm1zaUJVRUJjallDQkNBQUlBRnFJQVUyQWdBZ0NBUkFJQWhCZUhGQnNBaHFJUUZCbkFnb0FnQWhBZ0ovSUFSQkFTQUlRUU4yZENJRGNVVUVRRUdJQ0NBRElBUnlOZ0lBSUFFTUFRc2dBU2dDQ0FzaEF5QUJJQUkyQWdnZ0F5QUNOZ0lNSUFJZ0FUWUNEQ0FDSUFNMkFnZ0xJQUJCQ0dvaEFFR2NDQ0FITmdJQVFaQUlJQVUyQWdBTUN3dEJqQWdvQWdBaUMwVU5BU0FMYUVFQ2RFRzRDbW9vQWdBaUFpZ0NCRUY0Y1NBR2F5RURJQUloQVFOQUFrQWdBU2dDRUNJQVJRUkFJQUVvQWhRaUFFVU5BUXNnQUNnQ0JFRjRjU0FHYXlJQklBTWdBU0FEU1NJQkd5RURJQUFnQWlBQkd5RUNJQUFoQVF3QkN3c2dBaWdDR0NFSklBSWdBaWdDRENJQVJ3UkFJQUlvQWdnaUFTQUFOZ0lNSUFBZ0FUWUNDQXdLQ3lBQ0tBSVVJZ0VFZnlBQ1FSUnFCU0FDS0FJUUlnRkZEUU1nQWtFUWFnc2hCUU5BSUFVaEJ5QUJJZ0JCRkdvaEJTQUFLQUlVSWdFTkFDQUFRUkJxSVFVZ0FDZ0NFQ0lCRFFBTElBZEJBRFlDQUF3SkMwRi9JUVlnQUVHL2Ywc05BQ0FBUVF0cUlnRkJlSEVoQmtHTUNDZ0NBQ0lIUlEwQVFSOGhDRUVBSUFacklRTWdBRUgwLy84SFRRUkFJQVpCSmlBQlFRaDJaeUlBYTNaQkFYRWdBRUVCZEd0QlBtb2hDQXNDUUFKQUFrQWdDRUVDZEVHNENtb29BZ0FpQVVVRVFFRUFJUUFNQVF0QkFDRUFJQVpCR1NBSVFRRjJhMEVBSUFoQkgwY2JkQ0VDQTBBQ1FDQUJLQUlFUVhoeElBWnJJZ1FnQTA4TkFDQUJJUVVnQkNJRERRQkJBQ0VESUFFaEFBd0RDeUFBSUFFb0FoUWlCQ0FFSUFFZ0FrRWRka0VFY1dvb0FoQWlBVVliSUFBZ0JCc2hBQ0FDUVFGMElRSWdBUTBBQ3dzZ0FDQUZja1VFUUVFQUlRVkJBaUFJZENJQVFRQWdBR3R5SUFkeElnQkZEUU1nQUdoQkFuUkJ1QXBxS0FJQUlRQUxJQUJGRFFFTEEwQWdBQ2dDQkVGNGNTQUdheUlDSUFOSklRRWdBaUFESUFFYklRTWdBQ0FGSUFFYklRVWdBQ2dDRUNJQkJIOGdBUVVnQUNnQ0ZBc2lBQTBBQ3dzZ0JVVU5BQ0FEUVpBSUtBSUFJQVpyVHcwQUlBVW9BaGdoQ0NBRklBVW9BZ3dpQUVjRVFDQUZLQUlJSWdFZ0FEWUNEQ0FBSUFFMkFnZ01DQXNnQlNnQ0ZDSUJCSDhnQlVFVWFnVWdCU2dDRUNJQlJRMERJQVZCRUdvTElRSURRQ0FDSVFRZ0FTSUFRUlJxSVFJZ0FDZ0NGQ0lCRFFBZ0FFRVFhaUVDSUFBb0FoQWlBUTBBQ3lBRVFRQTJBZ0FNQndzZ0JrR1FDQ2dDQUNJRlRRUkFRWndJS0FJQUlRQUNRQ0FGSUFacklnRkJFRThFUUNBQUlBWnFJZ0lnQVVFQmNqWUNCQ0FBSUFWcUlBRTJBZ0FnQUNBR1FRTnlOZ0lFREFFTElBQWdCVUVEY2pZQ0JDQUFJQVZxSWdFZ0FTZ0NCRUVCY2pZQ0JFRUFJUUpCQUNFQkMwR1FDQ0FCTmdJQVFad0lJQUkyQWdBZ0FFRUlhaUVBREFrTElBWkJsQWdvQWdBaUFra0VRRUdVQ0NBQ0lBWnJJZ0UyQWdCQm9BaEJvQWdvQWdBaUFDQUdhaUlDTmdJQUlBSWdBVUVCY2pZQ0JDQUFJQVpCQTNJMkFnUWdBRUVJYWlFQURBa0xRUUFoQUNBR1FTOXFJZ01DZjBIZ0N5Z0NBQVJBUWVnTEtBSUFEQUVMUWV3TFFuODNBZ0JCNUF0Q2dLQ0FnSUNBQkRjQ0FFSGdDeUFLUVF4cVFYQnhRZGlxMWFvRmN6WUNBRUgwQzBFQU5nSUFRY1FMUVFBMkFnQkJnQ0FMSWdGcUlnUkJBQ0FCYXlJSGNTSUJJQVpORFFoQndBc29BZ0FpQlFSQVFiZ0xLQUlBSWdnZ0FXb2lDU0FJVFNBRklBbEpjZzBKQ3dKQVFjUUxMUUFBUVFSeFJRUkFBa0FDUUFKQUFrQkJvQWdvQWdBaUJRUkFRY2dMSVFBRFFDQUFLQUlBSWdnZ0JVMEVRQ0FGSUFnZ0FDZ0NCR3BKRFFNTElBQW9BZ2dpQUEwQUN3dEJBQkFCSWdKQmYwWU5BeUFCSVFSQjVBc29BZ0FpQUVFQmF5SUZJQUp4QkVBZ0FTQUNheUFDSUFWcVFRQWdBR3R4YWlFRUN5QUVJQVpORFFOQndBc29BZ0FpQUFSQVFiZ0xLQUlBSWdVZ0JHb2lCeUFGVFNBQUlBZEpjZzBFQ3lBRUVBRWlBQ0FDUncwQkRBVUxJQVFnQW1zZ0IzRWlCQkFCSWdJZ0FDZ0NBQ0FBS0FJRWFrWU5BU0FDSVFBTElBQkJmMFlOQVNBR1FUQnFJQVJOQkVBZ0FDRUNEQVFMUWVnTEtBSUFJZ0lnQXlBRWEycEJBQ0FDYTNFaUFoQUJRWDlHRFFFZ0FpQUVhaUVFSUFBaEFnd0RDeUFDUVg5SERRSUxRY1FMUWNRTEtBSUFRUVJ5TmdJQUN5QUJFQUVpQWtGL1JrRUFFQUVpQUVGL1JuSWdBQ0FDVFhJTkJTQUFJQUpySWdRZ0JrRW9hazBOQlF0QnVBdEJ1QXNvQWdBZ0JHb2lBRFlDQUVHOEN5Z0NBQ0FBU1FSQVFid0xJQUEyQWdBTEFrQkJvQWdvQWdBaUF3UkFRY2dMSVFBRFFDQUNJQUFvQWdBaUFTQUFLQUlFSWdWcVJnMENJQUFvQWdnaUFBMEFDd3dFQzBHWUNDZ0NBQ0lBUVFBZ0FDQUNUUnRGQkVCQm1BZ2dBallDQUF0QkFDRUFRY3dMSUFRMkFnQkJ5QXNnQWpZQ0FFR29DRUYvTmdJQVFhd0lRZUFMS0FJQU5nSUFRZFFMUVFBMkFnQURRQ0FBUVFOMElnRkJ1QWhxSUFGQnNBaHFJZ1UyQWdBZ0FVRzhDR29nQlRZQ0FDQUFRUUZxSWdCQklFY05BQXRCbEFnZ0JFRW9heUlBUVhnZ0FtdEJCM0VpQVdzaUJUWUNBRUdnQ0NBQklBSnFJZ0UyQWdBZ0FTQUZRUUZ5TmdJRUlBQWdBbXBCS0RZQ0JFR2tDRUh3Q3lnQ0FEWUNBQXdFQ3lBQ0lBTk5JQUVnQTB0eURRSWdBQ2dDREVFSWNRMENJQUFnQkNBRmFqWUNCRUdnQ0NBRFFYZ2dBMnRCQjNFaUFHb2lBVFlDQUVHVUNFR1VDQ2dDQUNBRWFpSUNJQUJySWdBMkFnQWdBU0FBUVFGeU5nSUVJQUlnQTJwQktEWUNCRUdrQ0VId0N5Z0NBRFlDQUF3REMwRUFJUUFNQmd0QkFDRUFEQVFMUVpnSUtBSUFJQUpMQkVCQm1BZ2dBallDQUFzZ0FpQUVhaUVGUWNnTElRQUNRQU5BSUFVZ0FDZ0NBQ0lCUndSQUlBQW9BZ2dpQUEwQkRBSUxDeUFBTFFBTVFRaHhSUTBEQzBISUN5RUFBMEFDUUNBQUtBSUFJZ0VnQTAwRVFDQURJQUVnQUNnQ0JHb2lCVWtOQVFzZ0FDZ0NDQ0VBREFFTEMwR1VDQ0FFUVNocklnQkJlQ0FDYTBFSGNTSUJheUlITmdJQVFhQUlJQUVnQW1vaUFUWUNBQ0FCSUFkQkFYSTJBZ1FnQUNBQ2FrRW9OZ0lFUWFRSVFmQUxLQUlBTmdJQUlBTWdCVUVuSUFWclFRZHhha0V2YXlJQUlBQWdBMEVRYWtrYklnRkJHellDQkNBQlFkQUxLUUlBTndJUUlBRkJ5QXNwQWdBM0FnaEIwQXNnQVVFSWFqWUNBRUhNQ3lBRU5nSUFRY2dMSUFJMkFnQkIxQXRCQURZQ0FDQUJRUmhxSVFBRFFDQUFRUWMyQWdRZ0FFRUlhaUFBUVFScUlRQWdCVWtOQUFzZ0FTQURSZzBBSUFFZ0FTZ0NCRUYrY1RZQ0JDQURJQUVnQTJzaUFrRUJjallDQkNBQklBSTJBZ0FDZnlBQ1FmOEJUUVJBSUFKQmVIRkJzQWhxSVFBQ2YwR0lDQ2dDQUNJQlFRRWdBa0VEZG5RaUFuRkZCRUJCaUFnZ0FTQUNjallDQUNBQURBRUxJQUFvQWdnTElRRWdBQ0FETmdJSUlBRWdBellDREVFTUlRSkJDQXdCQzBFZklRQWdBa0gvLy84SFRRUkFJQUpCSmlBQ1FRaDJaeUlBYTNaQkFYRWdBRUVCZEd0QlBtb2hBQXNnQXlBQU5nSWNJQU5DQURjQ0VDQUFRUUowUWJnS2FpRUJBa0FDUUVHTUNDZ0NBQ0lGUVFFZ0FIUWlCSEZGQkVCQmpBZ2dCQ0FGY2pZQ0FDQUJJQU0yQWdBTUFRc2dBa0VaSUFCQkFYWnJRUUFnQUVFZlJ4dDBJUUFnQVNnQ0FDRUZBMEFnQlNJQktBSUVRWGh4SUFKR0RRSWdBRUVkZGlFRklBQkJBWFFoQUNBQklBVkJCSEZxSWdRb0FoQWlCUTBBQ3lBRUlBTTJBaEFMSUFNZ0FUWUNHRUVJSVFJZ0F5SUJJUUJCREF3QkN5QUJLQUlJSWdBZ0F6WUNEQ0FCSUFNMkFnZ2dBeUFBTmdJSVFRQWhBRUVZSVFKQkRBc2dBMm9nQVRZQ0FDQUNJQU5xSUFBMkFnQUxRWlFJS0FJQUlnQWdCazBOQUVHVUNDQUFJQVpySWdFMkFnQkJvQWhCb0Fnb0FnQWlBQ0FHYWlJQ05nSUFJQUlnQVVFQmNqWUNCQ0FBSUFaQkEzSTJBZ1FnQUVFSWFpRUFEQVFMUVlRSVFUQTJBZ0JCQUNFQURBTUxJQUFnQWpZQ0FDQUFJQUFvQWdRZ0JHbzJBZ1FnQWtGNElBSnJRUWR4YWlJSUlBWkJBM0kyQWdRZ0FVRjRJQUZyUVFkeGFpSUVJQVlnQ0dvaUEyc2hCd0pBUWFBSUtBSUFJQVJHQkVCQm9BZ2dBellDQUVHVUNFR1VDQ2dDQUNBSGFpSUFOZ0lBSUFNZ0FFRUJjallDQkF3QkMwR2NDQ2dDQUNBRVJnUkFRWndJSUFNMkFnQkJrQWhCa0Fnb0FnQWdCMm9pQURZQ0FDQURJQUJCQVhJMkFnUWdBQ0FEYWlBQU5nSUFEQUVMSUFRb0FnUWlBRUVEY1VFQlJnUkFJQUJCZUhFaENTQUVLQUlNSVFJQ1FDQUFRZjhCVFFSQUlBUW9BZ2dpQVNBQ1JnUkFRWWdJUVlnSUtBSUFRWDRnQUVFRGRuZHhOZ0lBREFJTElBRWdBallDRENBQ0lBRTJBZ2dNQVFzZ0JDZ0NHQ0VHQWtBZ0FpQUVSd1JBSUFRb0FnZ2lBQ0FDTmdJTUlBSWdBRFlDQ0F3QkN3SkFJQVFvQWhRaUFBUi9JQVJCRkdvRklBUW9BaEFpQUVVTkFTQUVRUkJxQ3lFQkEwQWdBU0VGSUFBaUFrRVVhaUVCSUFBb0FoUWlBQTBBSUFKQkVHb2hBU0FDS0FJUUlnQU5BQXNnQlVFQU5nSUFEQUVMUVFBaEFnc2dCa1VOQUFKQUlBUW9BaHdpQUVFQ2RFRzRDbW9pQVNnQ0FDQUVSZ1JBSUFFZ0FqWUNBQ0FDRFFGQmpBaEJqQWdvQWdCQmZpQUFkM0UyQWdBTUFnc0NRQ0FFSUFZb0FoQkdCRUFnQmlBQ05nSVFEQUVMSUFZZ0FqWUNGQXNnQWtVTkFRc2dBaUFHTmdJWUlBUW9BaEFpQUFSQUlBSWdBRFlDRUNBQUlBSTJBaGdMSUFRb0FoUWlBRVVOQUNBQ0lBQTJBaFFnQUNBQ05nSVlDeUFISUFscUlRY2dCQ0FKYWlJRUtBSUVJUUFMSUFRZ0FFRitjVFlDQkNBRElBZEJBWEkyQWdRZ0F5QUhhaUFITmdJQUlBZEIvd0ZOQkVBZ0IwRjRjVUd3Q0dvaEFBSi9RWWdJS0FJQUlnRkJBU0FIUVFOMmRDSUNjVVVFUUVHSUNDQUJJQUp5TmdJQUlBQU1BUXNnQUNnQ0NBc2hBU0FBSUFNMkFnZ2dBU0FETmdJTUlBTWdBRFlDRENBRElBRTJBZ2dNQVF0Qkh5RUNJQWRCLy8vL0IwMEVRQ0FIUVNZZ0IwRUlkbWNpQUd0MlFRRnhJQUJCQVhSclFUNXFJUUlMSUFNZ0FqWUNIQ0FEUWdBM0FoQWdBa0VDZEVHNENtb2hBQUpBQWtCQmpBZ29BZ0FpQVVFQklBSjBJZ1Z4UlFSQVFZd0lJQUVnQlhJMkFnQWdBQ0FETmdJQURBRUxJQWRCR1NBQ1FRRjJhMEVBSUFKQkgwY2JkQ0VDSUFBb0FnQWhBUU5BSUFFaUFDZ0NCRUY0Y1NBSFJnMENJQUpCSFhZaEFTQUNRUUYwSVFJZ0FDQUJRUVJ4YWlJRktBSVFJZ0VOQUFzZ0JTQUROZ0lRQ3lBRElBQTJBaGdnQXlBRE5nSU1JQU1nQXpZQ0NBd0JDeUFBS0FJSUlnRWdBellDRENBQUlBTTJBZ2dnQTBFQU5nSVlJQU1nQURZQ0RDQURJQUUyQWdnTElBaEJDR29oQUF3Q0N3SkFJQWhGRFFBQ1FDQUZLQUljSWdGQkFuUkJ1QXBxSWdJb0FnQWdCVVlFUUNBQ0lBQTJBZ0FnQUEwQlFZd0lJQWRCZmlBQmQzRWlCellDQUF3Q0N3SkFJQVVnQ0NnQ0VFWUVRQ0FJSUFBMkFoQU1BUXNnQ0NBQU5nSVVDeUFBUlEwQkN5QUFJQWcyQWhnZ0JTZ0NFQ0lCQkVBZ0FDQUJOZ0lRSUFFZ0FEWUNHQXNnQlNnQ0ZDSUJSUTBBSUFBZ0FUWUNGQ0FCSUFBMkFoZ0xBa0FnQTBFUFRRUkFJQVVnQXlBR2FpSUFRUU55TmdJRUlBQWdCV29pQUNBQUtBSUVRUUZ5TmdJRURBRUxJQVVnQmtFRGNqWUNCQ0FGSUFacUlnUWdBMEVCY2pZQ0JDQURJQVJxSUFNMkFnQWdBMEgvQVUwRVFDQURRWGh4UWJBSWFpRUFBbjlCaUFnb0FnQWlBVUVCSUFOQkEzWjBJZ0p4UlFSQVFZZ0lJQUVnQW5JMkFnQWdBQXdCQ3lBQUtBSUlDeUVCSUFBZ0JEWUNDQ0FCSUFRMkFnd2dCQ0FBTmdJTUlBUWdBVFlDQ0F3QkMwRWZJUUFnQTBILy8vOEhUUVJBSUFOQkppQURRUWgyWnlJQWEzWkJBWEVnQUVFQmRHdEJQbW9oQUFzZ0JDQUFOZ0ljSUFSQ0FEY0NFQ0FBUVFKMFFiZ0thaUVCQWtBQ1FDQUhRUUVnQUhRaUFuRkZCRUJCakFnZ0FpQUhjallDQUNBQklBUTJBZ0FnQkNBQk5nSVlEQUVMSUFOQkdTQUFRUUYyYTBFQUlBQkJIMGNiZENFQUlBRW9BZ0FoQVFOQUlBRWlBaWdDQkVGNGNTQURSZzBDSUFCQkhYWWhBU0FBUVFGMElRQWdBaUFCUVFSeGFpSUhLQUlRSWdFTkFBc2dCeUFFTmdJUUlBUWdBallDR0FzZ0JDQUVOZ0lNSUFRZ0JEWUNDQXdCQ3lBQ0tBSUlJZ0FnQkRZQ0RDQUNJQVEyQWdnZ0JFRUFOZ0lZSUFRZ0FqWUNEQ0FFSUFBMkFnZ0xJQVZCQ0dvaEFBd0JDd0pBSUFsRkRRQUNRQ0FDS0FJY0lnRkJBblJCdUFwcUlnVW9BZ0FnQWtZRVFDQUZJQUEyQWdBZ0FBMEJRWXdJSUF0QmZpQUJkM0UyQWdBTUFnc0NRQ0FDSUFrb0FoQkdCRUFnQ1NBQU5nSVFEQUVMSUFrZ0FEWUNGQXNnQUVVTkFRc2dBQ0FKTmdJWUlBSW9BaEFpQVFSQUlBQWdBVFlDRUNBQklBQTJBaGdMSUFJb0FoUWlBVVVOQUNBQUlBRTJBaFFnQVNBQU5nSVlDd0pBSUFOQkQwMEVRQ0FDSUFNZ0Jtb2lBRUVEY2pZQ0JDQUFJQUpxSWdBZ0FDZ0NCRUVCY2pZQ0JBd0JDeUFDSUFaQkEzSTJBZ1FnQWlBR2FpSUZJQU5CQVhJMkFnUWdBeUFGYWlBRE5nSUFJQWdFUUNBSVFYaHhRYkFJYWlFQVFad0lLQUlBSVFFQ2YwRUJJQWhCQTNaMElnY2dCSEZGQkVCQmlBZ2dCQ0FIY2pZQ0FDQUFEQUVMSUFBb0FnZ0xJUVFnQUNBQk5nSUlJQVFnQVRZQ0RDQUJJQUEyQWd3Z0FTQUVOZ0lJQzBHY0NDQUZOZ0lBUVpBSUlBTTJBZ0FMSUFKQkNHb2hBQXNnQ2tFUWFpUUFJQUFMb1FzQ0MzOEpmU01BUWFBQmF5SUxKQUFnQzBFd2FrRUFRU1Q4Q3dBRFFDQUJJQTVIQkVBZ0FpQU9RUU5zSWd4QkFtcEJBblFpRDJvcUFnQWhGeUFDSUF4QkFXcEJBblFpRUdvcUFnQWhHQ0FJSUF4QkFuUWlFV29nQWlBUmFpb0NBQ0laT0FJQUlBZ2dFR29nR0RnQ0FDQUlJQTlxSUJjNEFnQWdCeUFPUVFWMGFpSU5RUUEyQWd3Z0RTQVhPQUlJSUEwZ0dEZ0NCQ0FOSUJrNEFnQUNRQ0FBUlFSQUlBWWdEbW90QUFCRkRRRUxJQTFCZ0lDQUNEWUNEQXNnRFNBRklBNUJBblFpREVFQmNpSVNhaTBBQUVFSWRDQUZJQXhxTFFBQWNpQUZJQXhCQW5JaUUyb3RBQUJCRUhSeUlBVWdERUVEY2lJTWFpMEFBRUVZZEhJMkFod2dDeUFESUJKQkFuUWlFbW9xQWdBaUZ6Z0NrQUVnQ3lBRElCTkJBblFpRTJvcUFnQWlHRGdDbEFFZ0N5QURJQXhCQW5RaUZHb3FBZ0FpR1RnQ21BRWdDeUFESUE1QkJIUWlGV29xQWdDTUlobzRBcHdCSUF0QjRBQnFJZ3dnQ3lvQ21BRWlGa01BQUFEQWxDQVdsQ0FMS2dLVUFTSVdRd0FBQU1DVUlCYVVRd0FBZ0QrU2tqZ0NBQ0FNSUFzcUFwQUJJaFlnRnBJZ0N5b0NsQUdVSUFzcUFwZ0JJaFlnRnBJZ0N5b0NuQUdVa3pnQ0JDQU1JQXNxQXBBQkloWWdGcElnQ3lvQ21BR1VJQXNxQXBRQkloWWdGcElnQ3lvQ25BR1VramdDQ0NBTUlBc3FBcEFCSWhZZ0ZwSWdDeW9DbEFHVUlBc3FBcGdCSWhZZ0ZwSWdDeW9DbkFHVWtqZ0NEQ0FNSUFzcUFwZ0JJaFpEQUFBQXdKUWdGcFFnQ3lvQ2tBRWlGa01BQUFEQWxDQVdsRU1BQUlBL2twSTRBaEFnRENBTEtnS1VBU0lXSUJhU0lBc3FBcGdCbENBTEtnS1FBU0lXSUJhU0lBc3FBcHdCbEpNNEFoUWdEQ0FMS2dLUUFTSVdJQmFTSUFzcUFwZ0JsQ0FMS2dLVUFTSVdJQmFTSUFzcUFwd0JsSk00QWhnZ0RDQUxLZ0tVQVNJV0lCYVNJQXNxQXBnQmxDQUxLZ0tRQVNJV0lCYVNJQXNxQXB3QmxKSTRBaHdnRENBTEtnS1VBU0lXUXdBQUFNQ1VJQmFVSUFzcUFwQUJJaFpEQUFBQXdKUWdGcFJEQUFDQVA1S1NPQUlnSUFrZ0ZXb2dGemdDQUNBSklCSnFJQmc0QWdBZ0NTQVRhaUFaT0FJQUlBa2dGR29nR2pnQ0FDQUxJQVFnRVdvcUFnQWlGemdDTUNBTElBUWdFR29xQWdBaUdEZ0NRQ0FMSUFRZ0Qyb3FBZ0FpR1RnQ1VDQUtJQkZxSUJjNEFnQWdDaUFRYWlBWU9BSUFJQW9nRDJvZ0dUZ0NBQ0FMSUF3cUFoZ2dDeW9DT0pRZ0RDb0NBQ0FMS2dJd2xDQU1LZ0lNSUFzcUFqU1VrcEk0QWdBZ0N5QU1LZ0ljSUFzcUFqaVVJQXdxQWdRZ0N5b0NNSlFnRENvQ0VDQUxLZ0kwbEpLU09BSUVJQXNnRENvQ0lDQUxLZ0k0bENBTUtnSUlJQXNxQWpDVUlBd3FBaFFnQ3lvQ05KU1NramdDQ0NBTElBd3FBaGdnQ3lvQ1JKUWdEQ29DQUNBTEtnSThsQ0FNS2dJTUlBc3FBa0NVa3BJNEFnd2dDeUFNS2dJY0lBc3FBa1NVSUF3cUFnUWdDeW9DUEpRZ0RDb0NFQ0FMS2dKQWxKS1NPQUlRSUFzZ0RDb0NJQ0FMS2dKRWxDQU1LZ0lJSUFzcUFqeVVJQXdxQWhRZ0N5b0NRSlNTa2pnQ0ZDQUxJQXdxQWhnZ0N5b0NVSlFnRENvQ0FDQUxLZ0pJbENBTUtnSU1JQXNxQWt5VWtwSTRBaGdnQ3lBTUtnSWNJQXNxQWxDVUlBd3FBZ1FnQ3lvQ1NKUWdEQ29DRUNBTEtnSk1sSktTT0FJY0lBc2dEQ29DSUNBTEtnSlFsQ0FNS2dJSUlBc3FBa2lVSUF3cUFoUWdDeW9DVEpTU2tqZ0NJQ0FMS2dJZ0lSY2dDeW9DQ0NFWUlBc3FBaFFoR1NBTklBc3FBaGdpR2lBYWxDQUxLZ0lBSWhZZ0ZwUWdDeW9DRENJYklCdVVrcEpEQUFDQVFKUWdHaUFMS2dJY0loeVVJQllnQ3lvQ0JDSWRsQ0FiSUFzcUFoQWlIcFNTa2tNQUFJQkFsQkFDTmdJUUlBMGdHaUFYbENBV0lCaVVJQnNnR1pTU2trTUFBSUJBbENBY0lCeVVJQjBnSFpRZ0hpQWVsSktTUXdBQWdFQ1VFQUkyQWhRZ0RTQWNJQmVVSUIwZ0dKUWdIaUFabEpLU1F3QUFnRUNVSUJjZ0Y1UWdHQ0FZbENBWklCbVVrcEpEQUFDQVFKUVFBallDR0NBT1FRRnFJUTRNQVFzTElBdEJvQUZxSkFBTEFnQUxDd2tCQUVHQkNBc0NCZ0U9Iik7CiAgfQogIGZ1bmN0aW9uIFcoZykgewogICAgaWYgKEFycmF5QnVmZmVyLmlzVmlldyhnKSkKICAgICAgcmV0dXJuIGc7CiAgICBpZiAoZyA9PSBtICYmIEYpCiAgICAgIHJldHVybiBuZXcgVWludDhBcnJheShGKTsKICAgIGlmICh5KQogICAgICByZXR1cm4geShnKTsKICAgIHRocm93ICdzeW5jIGZldGNoaW5nIG9mIHRoZSB3YXNtIGZhaWxlZDogeW91IGNhbiBwcmVsb2FkIGl0IHRvIE1vZHVsZVsid2FzbUJpbmFyeSJdIG1hbnVhbGx5LCBvciBlbWNjLnB5IHdpbGwgZG8gdGhhdCBmb3IgeW91IHdoZW4gZ2VuZXJhdGluZyBIVE1MIChidXQgbm90IEpTKSc7CiAgfQogIGZ1bmN0aW9uIF8oZywgQikgewogICAgdmFyIEUsIFEgPSBXKGcpOwogICAgRSA9IG5ldyBXZWJBc3NlbWJseS5Nb2R1bGUoUSk7CiAgICB2YXIgbyA9IG5ldyBXZWJBc3NlbWJseS5JbnN0YW5jZShFLCBCKTsKICAgIHJldHVybiBbbywgRV07CiAgfQogIGZ1bmN0aW9uIFYoKSB7CiAgICByZXR1cm4geyBhOiBFQSB9OwogIH0KICBmdW5jdGlvbiB6KCkgewogICAgZnVuY3Rpb24gZyhRLCBvKSB7CiAgICAgIHJldHVybiBmID0gUS5leHBvcnRzLCBEID0gZi5iLCBVKCksIFFBKGYpLCBQKCksIGY7CiAgICB9CiAgICB2KCk7CiAgICB2YXIgQiA9IFYoKTsKICAgIGlmIChBLmluc3RhbnRpYXRlV2FzbSkKICAgICAgcmV0dXJuIG5ldyBQcm9taXNlKChRLCBvKSA9PiB7CiAgICAgICAgQS5pbnN0YW50aWF0ZVdhc20oQiwgKHIsIGMpID0+IHsKICAgICAgICAgIFEoZyhyKSk7CiAgICAgICAgfSk7CiAgICAgIH0pOwogICAgbSA/Pz0gWCgpOwogICAgdmFyIEUgPSBfKG0sIEIpOwogICAgcmV0dXJuIGcoRVswXSk7CiAgfQogIGZvciAodmFyIHggPSAoZykgPT4gewogICAgZm9yICg7IGcubGVuZ3RoID4gMDsgKQogICAgICBnLnNoaWZ0KCkoQSk7CiAgfSwgaiA9IFtdLCBPID0gKGcpID0+IGoucHVzaChnKSwgVCA9IFtdLCAkID0gKGcpID0+IFQucHVzaChnKSwgQUEgPSAoZykgPT4gewogICAgZm9yICh2YXIgQiwgRSwgUSA9IDAsIG8gPSAwLCByID0gZy5sZW5ndGgsIGMgPSBuZXcgVWludDhBcnJheSgociAqIDMgPj4gMikgLSAoZ1tyIC0gMl0gPT0gIj0iKSAtIChnW3IgLSAxXSA9PSAiPSIpKTsgUSA8IHI7IFEgKz0gNCwgbyArPSAzKQogICAgICBCID0gbltnLmNoYXJDb2RlQXQoUSArIDEpXSwgRSA9IG5bZy5jaGFyQ29kZUF0KFEgKyAyKV0sIGNbb10gPSBuW2cuY2hhckNvZGVBdChRKV0gPDwgMiB8IEIgPj4gNCwgY1tvICsgMV0gPSBCIDw8IDQgfCBFID4+IDIsIGNbbyArIDJdID0gRSA8PCA2IHwgbltnLmNoYXJDb2RlQXQoUSArIDMpXTsKICAgIHJldHVybiBjOwogIH0sIGdBID0gKCkgPT4gMjE0NzQ4MzY0OCwgSUEgPSAoZywgQikgPT4gTWF0aC5jZWlsKGcgLyBCKSAqIEIsIENBID0gKGcpID0+IHsKICAgIHZhciBCID0gRC5idWZmZXIsIEUgPSAoZyAtIEIuYnl0ZUxlbmd0aCArIDY1NTM1KSAvIDY1NTM2IHwgMDsKICAgIHRyeSB7CiAgICAgIHJldHVybiBELmdyb3coRSksIFUoKSwgMTsKICAgIH0gY2F0Y2ggewogICAgfQogIH0sIEJBID0gKGcpID0+IHsKICAgIHZhciBCID0gTi5sZW5ndGg7CiAgICBnID4+Pj0gMDsKICAgIHZhciBFID0gZ0EoKTsKICAgIGlmIChnID4gRSkKICAgICAgcmV0dXJuICExOwogICAgZm9yICh2YXIgUSA9IDE7IFEgPD0gNDsgUSAqPSAyKSB7CiAgICAgIHZhciBvID0gQiAqICgxICsgMC4yIC8gUSk7CiAgICAgIG8gPSBNYXRoLm1pbihvLCBnICsgMTAwNjYzMjk2KTsKICAgICAgdmFyIHIgPSBNYXRoLm1pbihFLCBJQShNYXRoLm1heChnLCBvKSwgNjU1MzYpKSwgYyA9IENBKHIpOwogICAgICBpZiAoYykKICAgICAgICByZXR1cm4gITA7CiAgICB9CiAgICByZXR1cm4gITE7CiAgfSwgbiA9IG5ldyBVaW50OEFycmF5KDEyMyksIGEgPSAyNTsgYSA+PSAwOyAtLWEpCiAgICBuWzQ4ICsgYV0gPSA1MiArIGEsIG5bNjUgKyBhXSA9IGEsIG5bOTcgKyBhXSA9IDI2ICsgYTsKICBuWzQzXSA9IDYyLCBuWzQ3XSA9IDYzLCBBLm5vRXhpdFJ1bnRpbWUgJiYgQS5ub0V4aXRSdW50aW1lLCBBLnByaW50ICYmIEEucHJpbnQsIEEucHJpbnRFcnIgJiYgQS5wcmludEVyciwgQS53YXNtQmluYXJ5ICYmIChGID0gQS53YXNtQmluYXJ5KSwgQS5hcmd1bWVudHMgJiYgQS5hcmd1bWVudHMsIEEudGhpc1Byb2dyYW0gJiYgQS50aGlzUHJvZ3JhbTsKICBmdW5jdGlvbiBRQShnKSB7CiAgICBBLl9wYWNrID0gZy5kLCBBLl9tYWxsb2MgPSBnLmUsIEEuX2ZyZWUgPSBnLmY7CiAgfQogIHZhciBFQSA9IHsgYTogQkEgfSwgZiA9IHooKTsKICBmdW5jdGlvbiBxKCkgewogICAgaWYgKHMgPiAwKSB7CiAgICAgIHcgPSBxOwogICAgICByZXR1cm47CiAgICB9CiAgICBpZiAoaSgpLCBzID4gMCkgewogICAgICB3ID0gcTsKICAgICAgcmV0dXJuOwogICAgfQogICAgZnVuY3Rpb24gZygpIHsKICAgICAgQS5jYWxsZWRSdW4gPSAhMCwgSCgpLCBBLm9uUnVudGltZUluaXRpYWxpemVkPy4oKSwgYigpOwogICAgfQogICAgQS5zZXRTdGF0dXMgPyAoQS5zZXRTdGF0dXMoIlJ1bm5pbmcuLi4iKSwgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgIHNldFRpbWVvdXQoKCkgPT4gQS5zZXRTdGF0dXMoIiIpLCAxKSwgZygpOwogICAgfSwgMSkpIDogZygpOwogIH0KICBmdW5jdGlvbiBpQSgpIHsKICAgIGlmIChBLnByZUluaXQpCiAgICAgIGZvciAodHlwZW9mIEEucHJlSW5pdCA9PSAiZnVuY3Rpb24iICYmIChBLnByZUluaXQgPSBbQS5wcmVJbml0XSk7IEEucHJlSW5pdC5sZW5ndGggPiAwOyApCiAgICAgICAgQS5wcmVJbml0LnNoaWZ0KCkoKTsKICB9CiAgcmV0dXJuIGlBKCksIHEoKSwgZSA9IEEsIGU7Cn07CmxldCBJOwphc3luYyBmdW5jdGlvbiB0QSgpIHsKICBpZiAoSSA9IGF3YWl0IG9BKCksICFJIHx8ICFJLkhFQVBGMzIgfHwgIUkuX3BhY2spCiAgICB0aHJvdyBuZXcgRXJyb3IoIldBU00gbW9kdWxlIGZhaWxlZCB0byBpbml0aWFsaXplIHByb3Blcmx5Iik7Cn0KbGV0IHQgPSAwOwpjb25zdCBsID0gbmV3IEFycmF5KCk7CmxldCBLID0gITEsIHAgPSAhMSwgWSwgUiwgRywgdSwgZCwgTSwgUywgTCwgSjsKY29uc3QgZUEgPSBhc3luYyAoQykgPT4gewogIGZvciAoOyBwOyApCiAgICBhd2FpdCBuZXcgUHJvbWlzZSgoSCkgPT4gc2V0VGltZW91dChILCAwKSk7CiAgSSB8fCAocCA9ICEwLCBhd2FpdCB0QSgpLCBwID0gITEpOwogIGNvbnN0IGUgPSBNYXRoLnBvdygyLCBNYXRoLmNlaWwoTWF0aC5sb2cyKEMudmVydGV4Q291bnQpKSk7CiAgZSA+IHQgJiYgKHQgPiAwICYmIChJLl9mcmVlKFkpLCBJLl9mcmVlKFIpLCBJLl9mcmVlKEcpLCBJLl9mcmVlKHUpLCBJLl9mcmVlKGQpLCBJLl9mcmVlKE0pLCBJLl9mcmVlKFMpLCBJLl9mcmVlKEwpLCBJLl9mcmVlKEopKSwgdCA9IGUsIFkgPSBJLl9tYWxsb2MoMyAqIHQgKiA0KSwgUiA9IEkuX21hbGxvYyg0ICogdCAqIDQpLCBHID0gSS5fbWFsbG9jKDMgKiB0ICogNCksIHUgPSBJLl9tYWxsb2MoNCAqIHQpLCBkID0gSS5fbWFsbG9jKHQpLCBNID0gSS5fbWFsbG9jKDggKiB0ICogNCksIFMgPSBJLl9tYWxsb2MoMyAqIHQgKiA0KSwgTCA9IEkuX21hbGxvYyg0ICogdCAqIDQpLCBKID0gSS5fbWFsbG9jKDMgKiB0ICogNCkpLCBJLkhFQVBGMzIuc2V0KEMucG9zaXRpb25zLCBZIC8gNCksIEkuSEVBUEYzMi5zZXQoQy5yb3RhdGlvbnMsIFIgLyA0KSwgSS5IRUFQRjMyLnNldChDLnNjYWxlcywgRyAvIDQpLCBJLkhFQVBVOC5zZXQoQy5jb2xvcnMsIHUpLCBJLkhFQVBVOC5zZXQoQy5zZWxlY3Rpb24sIGQpLCBJLl9wYWNrKAogICAgQy5zZWxlY3RlZCwKICAgIEMudmVydGV4Q291bnQsCiAgICBZLAogICAgUiwKICAgIEcsCiAgICB1LAogICAgZCwKICAgIE0sCiAgICBTLAogICAgTCwKICAgIEoKICApOwogIGNvbnN0IEEgPSBuZXcgVWludDMyQXJyYXkoSS5IRUFQVTMyLmJ1ZmZlciwgTSwgQy52ZXJ0ZXhDb3VudCAqIDgpLCBoID0gbmV3IFVpbnQzMkFycmF5KEEuc2xpY2UoKS5idWZmZXIpLCBrID0gbmV3IEZsb2F0MzJBcnJheShJLkhFQVBGMzIuYnVmZmVyLCBTLCBDLnZlcnRleENvdW50ICogMyksIHkgPSBuZXcgRmxvYXQzMkFycmF5KGsuc2xpY2UoKS5idWZmZXIpLCBGID0gbmV3IEZsb2F0MzJBcnJheShJLkhFQVBGMzIuYnVmZmVyLCBMLCBDLnZlcnRleENvdW50ICogNCksIEQgPSBuZXcgRmxvYXQzMkFycmF5KEYuc2xpY2UoKS5idWZmZXIpLCBOID0gbmV3IEZsb2F0MzJBcnJheShJLkhFQVBGMzIuYnVmZmVyLCBKLCBDLnZlcnRleENvdW50ICogMyksIFUgPSBuZXcgRmxvYXQzMkFycmF5KE4uc2xpY2UoKS5idWZmZXIpLCBpID0gewogICAgZGF0YTogaCwKICAgIHdvcmxkUG9zaXRpb25zOiB5LAogICAgd29ybGRSb3RhdGlvbnM6IEQsCiAgICB3b3JsZFNjYWxlczogVSwKICAgIG9mZnNldDogQy5vZmZzZXQsCiAgICB2ZXJ0ZXhDb3VudDogQy52ZXJ0ZXhDb3VudCwKICAgIHBvc2l0aW9uczogQy5wb3NpdGlvbnMuYnVmZmVyLAogICAgcm90YXRpb25zOiBDLnJvdGF0aW9ucy5idWZmZXIsCiAgICBzY2FsZXM6IEMuc2NhbGVzLmJ1ZmZlciwKICAgIGNvbG9yczogQy5jb2xvcnMuYnVmZmVyLAogICAgc2VsZWN0aW9uOiBDLnNlbGVjdGlvbi5idWZmZXIKICB9OwogIHNlbGYucG9zdE1lc3NhZ2UoeyByZXNwb25zZTogaSB9LCBbCiAgICBpLmRhdGEuYnVmZmVyLAogICAgaS53b3JsZFBvc2l0aW9ucy5idWZmZXIsCiAgICBpLndvcmxkUm90YXRpb25zLmJ1ZmZlciwKICAgIGkud29ybGRTY2FsZXMuYnVmZmVyLAogICAgaS5wb3NpdGlvbnMsCiAgICBpLnJvdGF0aW9ucywKICAgIGkuc2NhbGVzLAogICAgaS5jb2xvcnMsCiAgICBpLnNlbGVjdGlvbgogIF0pLCBLID0gITE7Cn0sIFogPSAoKSA9PiB7CiAgaWYgKGwubGVuZ3RoICE9PSAwICYmICFLKSB7CiAgICBLID0gITA7CiAgICBjb25zdCBDID0gbC5zaGlmdCgpOwogICAgZUEoQyksIHNldFRpbWVvdXQoKCkgPT4gewogICAgICBLID0gITEsIFooKTsKICAgIH0sIDApOwogIH0KfTsKc2VsZi5vbm1lc3NhZ2UgPSAoQykgPT4gewogIGlmIChDLmRhdGEuc3BsYXQpIHsKICAgIGNvbnN0IGUgPSBDLmRhdGEuc3BsYXQ7CiAgICBmb3IgKGNvbnN0IFtBLCBoXSBvZiBsLmVudHJpZXMoKSkKICAgICAgaWYgKGgub2Zmc2V0ID09PSBlLm9mZnNldCkgewogICAgICAgIGxbQV0gPSBlOwogICAgICAgIHJldHVybjsKICAgICAgfQogICAgbC5wdXNoKGUpLCBaKCk7CiAgfQp9OwovLyMgc291cmNlTWFwcGluZ1VSTD1EYXRhV29ya2VyLUJ6RWRfWmo2LmpzLm1hcAo=";
  var ut = (E) => Uint8Array.from(atob(E), (t) => t.charCodeAt(0));
  var at = typeof self < "u" && self.Blob && new Blob(["URL.revokeObjectURL(import.meta.url);", ut(gt)], { type: "text/javascript;charset=utf-8" });
  function Jt(E) {
    let t;
    try {
      if (t = at && (self.URL || self.webkitURL).createObjectURL(at), !t) throw "";
      const n = new Worker(t, {
        type: "module",
        name: E?.name
      });
      return n.addEventListener("error", () => {
        (self.URL || self.webkitURL).revokeObjectURL(t);
      }), n;
    } catch {
      return new Worker(
        "data:text/javascript;base64," + gt,
        {
          type: "module",
          name: E?.name
        }
      );
    }
  }
  var mt = function(E = {}) {
    var t, n = E, i = import_meta.url, e = "", A;
    {
      try {
        e = new URL(".", i).href;
      } catch {
      }
      A = (h) => {
        var m = new XMLHttpRequest();
        return m.open("GET", h, false), m.responseType = "arraybuffer", m.send(null), new Uint8Array(m.response);
      };
    }
    console.log.bind(console), console.error.bind(console);
    var o, s, r;
    function Q() {
      var h = s.buffer;
      n.HEAPU8 = r = new Uint8Array(h), n.HEAPU32 = new Uint32Array(h), n.HEAPF32 = new Float32Array(h), new BigInt64Array(h), new BigUint64Array(h);
    }
    function I() {
      if (n.preRun)
        for (typeof n.preRun == "function" && (n.preRun = [n.preRun]); n.preRun.length; )
          L(n.preRun.shift());
      Z(H);
    }
    function d() {
      j.c();
    }
    function a() {
      if (n.postRun)
        for (typeof n.postRun == "function" && (n.postRun = [n.postRun]); n.postRun.length; )
          f(n.postRun.shift());
      Z(k);
    }
    var U = 0, F = null;
    function g(h) {
      U++, n.monitorRunDependencies?.(U);
    }
    function B(h) {
      if (U--, n.monitorRunDependencies?.(U), U == 0 && F) {
        var m = F;
        F = null, m();
      }
    }
    var C;
    function c() {
      return M("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");
    }
    function p(h) {
      if (ArrayBuffer.isView(h))
        return h;
      if (h == C && o)
        return new Uint8Array(o);
      if (A)
        return A(h);
      throw 'sync fetching of the wasm failed: you can preload it to Module["wasmBinary"] manually, or emcc.py will do that for you when generating HTML (but not JS)';
    }
    function u(h, m) {
      var x, N = p(h);
      x = new WebAssembly.Module(N);
      var v = new WebAssembly.Instance(x, m);
      return [v, x];
    }
    function S() {
      return { a: b };
    }
    function W() {
      function h(N, v) {
        return j = N.exports, s = j.b, Q(), D(j), B(), j;
      }
      g();
      var m = S();
      if (n.instantiateWasm)
        return new Promise((N, v) => {
          n.instantiateWasm(m, (O, $) => {
            N(h(O));
          });
        });
      C ?? (C = c());
      var x = u(C, m);
      return h(x[0]);
    }
    for (var Z = (h) => {
      for (; h.length > 0; )
        h.shift()(n);
    }, k = [], f = (h) => k.push(h), H = [], L = (h) => H.push(h), M = (h) => {
      for (var m, x, N = 0, v = 0, O = h.length, $ = new Uint8Array((O * 3 >> 2) - (h[O - 2] == "=") - (h[O - 1] == "=")); N < O; N += 4, v += 3)
        m = l[h.charCodeAt(N + 1)], x = l[h.charCodeAt(N + 2)], $[v] = l[h.charCodeAt(N)] << 2 | m >> 4, $[v + 1] = m << 4 | x >> 2, $[v + 2] = x << 6 | l[h.charCodeAt(N + 3)];
      return $;
    }, V = () => 2147483648, w = (h, m) => Math.ceil(h / m) * m, P = (h) => {
      var m = s.buffer, x = (h - m.byteLength + 65535) / 65536 | 0;
      try {
        return s.grow(x), Q(), 1;
      } catch {
      }
    }, T = (h) => {
      var m = r.length;
      h >>>= 0;
      var x = V();
      if (h > x)
        return false;
      for (var N = 1; N <= 4; N *= 2) {
        var v = m * (1 + 0.2 / N);
        v = Math.min(v, h + 100663296);
        var O = Math.min(x, w(Math.max(h, v), 65536)), $ = P(O);
        if ($)
          return true;
      }
      return false;
    }, l = new Uint8Array(123), J = 25; J >= 0; --J)
      l[48 + J] = 52 + J, l[65 + J] = J, l[97 + J] = 26 + J;
    l[43] = 62, l[47] = 63, n.noExitRuntime && n.noExitRuntime, n.print && n.print, n.printErr && n.printErr, n.wasmBinary && (o = n.wasmBinary), n.arguments && n.arguments, n.thisProgram && n.thisProgram;
    function D(h) {
      n._pack = h.d, n._malloc = h.e, n._free = h.f;
    }
    var b = { a: T }, j = W();
    function q() {
      if (U > 0) {
        F = q;
        return;
      }
      if (I(), U > 0) {
        F = q;
        return;
      }
      function h() {
        n.calledRun = true, d(), n.onRuntimeInitialized?.(), a();
      }
      n.setStatus ? (n.setStatus("Running..."), setTimeout(() => {
        setTimeout(() => n.setStatus(""), 1), h();
      }, 1)) : h();
    }
    function G() {
      if (n.preInit)
        for (typeof n.preInit == "function" && (n.preInit = [n.preInit]); n.preInit.length > 0; )
          n.preInit.shift()();
    }
    return G(), q(), t = n, t;
  };
  var ft = () => new Jt();
  var Ut = class {
    constructor(t) {
      this.dataChanged = false, this.transformsChanged = false, this.colorTransformsChanged = false, this._updating = /* @__PURE__ */ new Set(), this._dirty = /* @__PURE__ */ new Set();
      let n = 0, i = 0;
      this._splatIndices = /* @__PURE__ */ new Map(), this._offsets = /* @__PURE__ */ new Map();
      const e = /* @__PURE__ */ new Map();
      for (const a of t.objects)
        a instanceof X && (this._splatIndices.set(a, i), this._offsets.set(a, n), e.set(n, a), n += a.data.vertexCount, i++);
      this._vertexCount = n, this._width = 2048, this._height = Math.ceil(2 * this.vertexCount / this.width), this._data = new Uint32Array(this.width * this.height * 4), this._transformsWidth = 5, this._transformsHeight = e.size, this._transforms = new Float32Array(this._transformsWidth * this._transformsHeight * 4), this._transformIndicesWidth = 1024, this._transformIndicesHeight = Math.ceil(this.vertexCount / this._transformIndicesWidth), this._transformIndices = new Uint32Array(this._transformIndicesWidth * this._transformIndicesHeight), this._colorTransformsWidth = 4, this._colorTransformsHeight = 64, this._colorTransforms = new Float32Array(this._colorTransformsWidth * this._colorTransformsHeight * 4), this._colorTransforms.fill(0), this._colorTransforms[0] = 1, this._colorTransforms[5] = 1, this._colorTransforms[10] = 1, this._colorTransforms[15] = 1, this._colorTransformIndicesWidth = 1024, this._colorTransformIndicesHeight = Math.ceil(this.vertexCount / this._colorTransformIndicesWidth), this._colorTransformIndices = new Uint32Array(
        this._colorTransformIndicesWidth * this._colorTransformIndicesHeight
      ), this.colorTransformIndices.fill(0), this._positions = new Float32Array(this.vertexCount * 3), this._rotations = new Float32Array(this.vertexCount * 4), this._scales = new Float32Array(this.vertexCount * 3), this._worker = ft();
      const A = (a) => {
        const U = this._splatIndices.get(a);
        this._transforms.set(a.transform.buffer, U * 20), this._transforms[U * 20 + 16] = a.selected ? 1 : 0, a.positionChanged = false, a.rotationChanged = false, a.scaleChanged = false, a.selectedChanged = false, this.transformsChanged = true;
      }, o = () => {
        let a = false;
        for (const g of this._splatIndices.keys())
          if (g.colorTransformChanged) {
            a = true;
            break;
          }
        if (!a)
          return;
        const U = [new K()];
        this._colorTransformIndices.fill(0);
        let F = 1;
        for (const g of this._splatIndices.keys()) {
          const B = this._offsets.get(g);
          for (const C of g.colorTransforms)
            U.includes(C) || (U.push(C), F++);
          for (const C of g.colorTransformsMap.keys()) {
            const c = g.colorTransformsMap.get(C);
            this._colorTransformIndices[C + B] = c + F - 1;
          }
          g.colorTransformChanged = false;
        }
        for (let g = 0; g < U.length; g++) {
          const B = U[g];
          this._colorTransforms.set(B.buffer, g * 16);
        }
        this.colorTransformsChanged = true;
      };
      this._worker.onmessage = (a) => {
        if (a.data.response) {
          const U = a.data.response, F = e.get(U.offset);
          A(F), o();
          const g = this._splatIndices.get(F);
          for (let B = 0; B < F.data.vertexCount; B++)
            this._transformIndices[U.offset + B] = g;
          this._data.set(U.data, U.offset * 8), F.data.reattach(
            U.positions,
            U.rotations,
            U.scales,
            U.colors,
            U.selection
          ), this._positions.set(U.worldPositions, U.offset * 3), this._rotations.set(U.worldRotations, U.offset * 4), this._scales.set(U.worldScales, U.offset * 3), this._updating.delete(F), F.selectedChanged = false, this.dataChanged = true;
        }
      };
      let s;
      async function r() {
        s = await mt();
      }
      r();
      async function Q() {
        for (; !s; )
          await new Promise((a) => setTimeout(a, 0));
      }
      const I = (a) => {
        if (!s) {
          Q().then(() => {
            I(a);
          });
          return;
        }
        A(a);
        const U = s._malloc(3 * a.data.vertexCount * 4), F = s._malloc(4 * a.data.vertexCount * 4), g = s._malloc(3 * a.data.vertexCount * 4), B = s._malloc(4 * a.data.vertexCount), C = s._malloc(a.data.vertexCount), c = s._malloc(8 * a.data.vertexCount * 4), p = s._malloc(3 * a.data.vertexCount * 4), u = s._malloc(4 * a.data.vertexCount * 4), S = s._malloc(3 * a.data.vertexCount * 4);
        s.HEAPF32.set(a.data.positions, U / 4), s.HEAPF32.set(a.data.rotations, F / 4), s.HEAPF32.set(a.data.scales, g / 4), s.HEAPU8.set(a.data.colors, B), s.HEAPU8.set(a.data.selection, C), s._pack(
          a.selected,
          a.data.vertexCount,
          U,
          F,
          g,
          B,
          C,
          c,
          p,
          u,
          S
        );
        const W = new Uint32Array(s.HEAPU32.buffer, c, a.data.vertexCount * 8), Z = new Float32Array(
          s.HEAPF32.buffer,
          p,
          a.data.vertexCount * 3
        ), k = new Float32Array(
          s.HEAPF32.buffer,
          u,
          a.data.vertexCount * 4
        ), f = new Float32Array(s.HEAPF32.buffer, S, a.data.vertexCount * 3), H = this._splatIndices.get(a), L = this._offsets.get(a);
        for (let M = 0; M < a.data.vertexCount; M++)
          this._transformIndices[L + M] = H;
        this._data.set(W, L * 8), this._positions.set(Z, L * 3), this._rotations.set(k, L * 4), this._scales.set(f, L * 3), s._free(U), s._free(F), s._free(g), s._free(B), s._free(C), s._free(c), s._free(p), s._free(u), s._free(S), this.dataChanged = true, this.colorTransformsChanged = true;
      }, d = (a) => {
        if ((a.positionChanged || a.rotationChanged || a.scaleChanged || a.selectedChanged) && A(a), a.colorTransformChanged && o(), !a.data.changed || a.data.detached) return;
        const U = {
          position: new Float32Array(a.position.flat()),
          rotation: new Float32Array(a.rotation.flat()),
          scale: new Float32Array(a.scale.flat()),
          selected: a.selected,
          vertexCount: a.data.vertexCount,
          positions: a.data.positions,
          rotations: a.data.rotations,
          scales: a.data.scales,
          colors: a.data.colors,
          selection: a.data.selection,
          offset: this._offsets.get(a)
        };
        this._worker.postMessage(
          {
            splat: U
          },
          [
            U.position.buffer,
            U.rotation.buffer,
            U.scale.buffer,
            U.positions.buffer,
            U.rotations.buffer,
            U.scales.buffer,
            U.colors.buffer,
            U.selection.buffer
          ]
        ), this._updating.add(a), a.data.detached = true;
      };
      this.getSplat = (a) => {
        let U = null;
        for (const [F, g] of this._offsets)
          if (a >= g)
            U = F;
          else
            break;
        return U;
      }, this.getLocalIndex = (a, U) => {
        const F = this._offsets.get(a);
        return U - F;
      }, this.markDirty = (a) => {
        this._dirty.add(a);
      }, this.rebuild = () => {
        for (const a of this._dirty)
          d(a);
        this._dirty.clear();
      }, this.dispose = () => {
        this._worker.terminate();
      };
      for (const a of this._splatIndices.keys())
        I(a);
      o();
    }
    get offsets() {
      return this._offsets;
    }
    get data() {
      return this._data;
    }
    get width() {
      return this._width;
    }
    get height() {
      return this._height;
    }
    get transforms() {
      return this._transforms;
    }
    get transformsWidth() {
      return this._transformsWidth;
    }
    get transformsHeight() {
      return this._transformsHeight;
    }
    get transformIndices() {
      return this._transformIndices;
    }
    get transformIndicesWidth() {
      return this._transformIndicesWidth;
    }
    get transformIndicesHeight() {
      return this._transformIndicesHeight;
    }
    get colorTransforms() {
      return this._colorTransforms;
    }
    get colorTransformsWidth() {
      return this._colorTransformsWidth;
    }
    get colorTransformsHeight() {
      return this._colorTransformsHeight;
    }
    get colorTransformIndices() {
      return this._colorTransformIndices;
    }
    get colorTransformIndicesWidth() {
      return this._colorTransformIndicesWidth;
    }
    get colorTransformIndicesHeight() {
      return this._colorTransformIndicesHeight;
    }
    get positions() {
      return this._positions;
    }
    get rotations() {
      return this._rotations;
    }
    get scales() {
      return this._scales;
    }
    get vertexCount() {
      return this._vertexCount;
    }
    get needsRebuild() {
      return this._dirty.size > 0;
    }
    get updating() {
      return this._updating.size > 0;
    }
  };
  var dt = class {
    constructor(t = 0, n = 0, i = 0, e = 255) {
      this.r = t, this.g = n, this.b = i, this.a = e;
    }
    flat() {
      return [this.r, this.g, this.b, this.a];
    }
    flatNorm() {
      return [this.r / 255, this.g / 255, this.b / 255, this.a / 255];
    }
    toHexString() {
      return "#" + this.flat().map((t) => t.toString(16).padStart(2, "0")).join("");
    }
    toString() {
      return `[${this.flat().join(", ")}]`;
    }
  };
  var Vt = () => new St();
  var Zt = (
    /* glsl */
    `#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform highp sampler2D u_transforms;
uniform highp usampler2D u_transformIndices;
uniform highp sampler2D u_colorTransforms;
uniform highp usampler2D u_colorTransformIndices;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;

uniform bool useDepthFade;
uniform float depthFade;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;
out float vSize;
out float vSelected;

void main () {
    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    float selected = float((cen.w >> 24) & 0xffu);

    uint transformIndex = texelFetch(u_transformIndices, ivec2(uint(index) & 0x3ffu, uint(index) >> 10), 0).x;
    mat4 transform = mat4(
        texelFetch(u_transforms, ivec2(0, transformIndex), 0),
        texelFetch(u_transforms, ivec2(1, transformIndex), 0),
        texelFetch(u_transforms, ivec2(2, transformIndex), 0),
        texelFetch(u_transforms, ivec2(3, transformIndex), 0)
    );

    if (selected < 0.5) {
        selected = texelFetch(u_transforms, ivec2(4, transformIndex), 0).x;
    }

    mat4 viewTransform = view * transform;

    vec4 cam = viewTransform * vec4(uintBitsToFloat(cen.xyz), 1);
    vec4 pos2d = projection * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -pos2d.w || pos2d.z > pos2d.w || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z), 
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z), 
        0., 0., 0.
    );

    mat3 T = transpose(mat3(viewTransform)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    //ref: https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L110-L111
    cov2d[0][0] += 0.3;
    cov2d[1][1] += 0.3;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if (lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    uint colorTransformIndex = texelFetch(u_colorTransformIndices, ivec2(uint(index) & 0x3ffu, uint(index) >> 10), 0).x;
    mat4 colorTransform = mat4(
        texelFetch(u_colorTransforms, ivec2(0, colorTransformIndex), 0),
        texelFetch(u_colorTransforms, ivec2(1, colorTransformIndex), 0),
        texelFetch(u_colorTransforms, ivec2(2, colorTransformIndex), 0),
        texelFetch(u_colorTransforms, ivec2(3, colorTransformIndex), 0)
    );

    vec4 color = vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
    vColor = colorTransform * color;

    vPosition = position;
    vSize = length(majorAxis);
    vSelected = selected;

    float scalingFactor = 1.0;

    if (useDepthFade) {
        float depthNorm = (pos2d.z / pos2d.w + 1.0) / 2.0;
        float near = 0.1; float far = 100.0;
        float normalizedDepth = (2.0 * near) / (far + near - depthNorm * (far - near));
        float start = max(normalizedDepth - 0.1, 0.0);
        float end = min(normalizedDepth + 0.1, 1.0);
        scalingFactor = clamp((depthFade - start) / (end - start), 0.0, 1.0);
    }

    vec2 vCenter = vec2(pos2d) / pos2d.w;
    gl_Position = vec4(
        vCenter 
        + position.x * majorAxis * scalingFactor / viewport
        + position.y * minorAxis * scalingFactor / viewport, 0.0, 1.0);
}
`
  );
  var pt = (
    /* glsl */
    `#version 300 es
precision highp float;

uniform float outlineThickness;
uniform vec4 outlineColor;

in vec4 vColor;
in vec2 vPosition;
in float vSize;
in float vSelected;

out vec4 fragColor;

void main () {
    float A = -dot(vPosition, vPosition);

    if (A < -4.0) discard;

    if (vSelected < 0.5) {
        float B = exp(A) * vColor.a;
        fragColor = vec4(B * vColor.rgb, B);
        return;
    }

    float outlineThreshold = -4.0 + (outlineThickness / vSize);

    if (A < outlineThreshold) {
        fragColor = outlineColor;
    } 
    else {
        float B = exp(A) * vColor.a;
        fragColor = vec4(B * vColor.rgb, B);
    }
}
`
  );
  var Ct = class extends lt {
    constructor(t, n) {
      super(t, n), this._outlineThickness = 10, this._outlineColor = new dt(255, 165, 0, 255), this._renderData = null, this._depthIndex = new Uint32Array(), this._splatTexture = null, this._worker = null;
      const i = t.canvas, e = t.gl;
      let A, o, s, r, Q, I, d, a, U, F, g, B, C, c, p, u, S, W, Z;
      this._resize = () => {
        this._camera && (this._camera.data.setSize(i.width, i.height), this._camera.update(), A = e.getUniformLocation(this.program, "projection"), e.uniformMatrix4fv(A, false, this._camera.data.projectionMatrix.buffer), o = e.getUniformLocation(this.program, "viewport"), e.uniform2fv(o, new Float32Array([i.width, i.height])));
      };
      const k = () => {
        this._worker = Vt(), this._worker.onmessage = (V) => {
          if (V.data.depthIndex) {
            const { depthIndex: w } = V.data;
            this._depthIndex = w, e.bindBuffer(e.ARRAY_BUFFER, Z), e.bufferData(e.ARRAY_BUFFER, w, e.STATIC_DRAW);
          }
        };
      };
      this._initialize = () => {
        if (!this._scene || !this._camera) {
          console.error("Cannot render without scene and camera");
          return;
        }
        this._resize(), this._scene.addEventListener("objectAdded", f), this._scene.addEventListener("objectRemoved", H);
        for (const V of this._scene.objects)
          V instanceof X && V.addEventListener("objectChanged", L);
        this._renderData = new Ut(this._scene), s = e.getUniformLocation(this.program, "focal"), e.uniform2fv(s, new Float32Array([this._camera.data.fx, this._camera.data.fy])), r = e.getUniformLocation(this.program, "view"), e.uniformMatrix4fv(r, false, this._camera.data.viewMatrix.buffer), F = e.getUniformLocation(this.program, "outlineThickness"), e.uniform1f(F, this.outlineThickness), g = e.getUniformLocation(this.program, "outlineColor"), e.uniform4fv(g, new Float32Array(this.outlineColor.flatNorm())), this._splatTexture = e.createTexture(), Q = e.getUniformLocation(this.program, "u_texture"), e.uniform1i(Q, 0), c = e.createTexture(), I = e.getUniformLocation(this.program, "u_transforms"), e.uniform1i(I, 1), p = e.createTexture(), d = e.getUniformLocation(this.program, "u_transformIndices"), e.uniform1i(d, 2), u = e.createTexture(), a = e.getUniformLocation(this.program, "u_colorTransforms"), e.uniform1i(a, 3), S = e.createTexture(), U = e.getUniformLocation(
          this.program,
          "u_colorTransformIndices"
        ), e.uniform1i(U, 4), W = e.createBuffer(), e.bindBuffer(e.ARRAY_BUFFER, W), e.bufferData(e.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), e.STATIC_DRAW), B = e.getAttribLocation(this.program, "position"), e.enableVertexAttribArray(B), e.vertexAttribPointer(B, 2, e.FLOAT, false, 0, 0), Z = e.createBuffer(), C = e.getAttribLocation(this.program, "index"), e.enableVertexAttribArray(C), e.bindBuffer(e.ARRAY_BUFFER, Z), k();
      };
      const f = (V) => {
        const w = V;
        w.object instanceof X && w.object.addEventListener("objectChanged", L), M();
      }, H = (V) => {
        const w = V;
        w.object instanceof X && w.object.removeEventListener("objectChanged", L), M();
      }, L = (V) => {
        const w = V;
        w.object instanceof X && this._renderData && this._renderData.markDirty(w.object);
      }, M = () => {
        this._renderData?.dispose(), this._renderData = new Ut(this._scene), this._worker?.terminate(), k();
      };
      this._render = () => {
        if (!this._scene || !this._camera || !this.renderData) {
          console.error("Cannot render without scene and camera");
          return;
        }
        if (this.renderData.needsRebuild && this.renderData.rebuild(), this.renderData.dataChanged || this.renderData.transformsChanged || this.renderData.colorTransformsChanged) {
          this.renderData.dataChanged && (e.activeTexture(e.TEXTURE0), e.bindTexture(e.TEXTURE_2D, this.splatTexture), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
            e.TEXTURE_2D,
            0,
            e.RGBA32UI,
            this.renderData.width,
            this.renderData.height,
            0,
            e.RGBA_INTEGER,
            e.UNSIGNED_INT,
            this.renderData.data
          )), this.renderData.transformsChanged && (e.activeTexture(e.TEXTURE1), e.bindTexture(e.TEXTURE_2D, c), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
            e.TEXTURE_2D,
            0,
            e.RGBA32F,
            this.renderData.transformsWidth,
            this.renderData.transformsHeight,
            0,
            e.RGBA,
            e.FLOAT,
            this.renderData.transforms
          ), e.activeTexture(e.TEXTURE2), e.bindTexture(e.TEXTURE_2D, p), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
            e.TEXTURE_2D,
            0,
            e.R32UI,
            this.renderData.transformIndicesWidth,
            this.renderData.transformIndicesHeight,
            0,
            e.RED_INTEGER,
            e.UNSIGNED_INT,
            this.renderData.transformIndices
          )), this.renderData.colorTransformsChanged && (e.activeTexture(e.TEXTURE3), e.bindTexture(e.TEXTURE_2D, u), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
            e.TEXTURE_2D,
            0,
            e.RGBA32F,
            this.renderData.colorTransformsWidth,
            this.renderData.colorTransformsHeight,
            0,
            e.RGBA,
            e.FLOAT,
            this.renderData.colorTransforms
          ), e.activeTexture(e.TEXTURE4), e.bindTexture(e.TEXTURE_2D, S), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
            e.TEXTURE_2D,
            0,
            e.R32UI,
            this.renderData.colorTransformIndicesWidth,
            this.renderData.colorTransformIndicesHeight,
            0,
            e.RED_INTEGER,
            e.UNSIGNED_INT,
            this.renderData.colorTransformIndices
          ));
          const V = new Float32Array(this.renderData.positions.slice().buffer), w = new Float32Array(this.renderData.transforms.slice().buffer), P = new Uint32Array(this.renderData.transformIndices.slice().buffer);
          this._worker?.postMessage(
            {
              sortData: {
                positions: V,
                transforms: w,
                transformIndices: P,
                vertexCount: this.renderData.vertexCount
              }
            },
            [V.buffer, w.buffer, P.buffer]
          ), this.renderData.dataChanged = false, this.renderData.transformsChanged = false, this.renderData.colorTransformsChanged = false;
        }
        this._camera.update(), this._worker?.postMessage({ viewProj: this._camera.data.viewProj.buffer }), e.viewport(0, 0, i.width, i.height), e.clearColor(0, 0, 0, 0), e.clear(e.COLOR_BUFFER_BIT), e.disable(e.DEPTH_TEST), e.enable(e.BLEND), e.blendFuncSeparate(e.ONE_MINUS_DST_ALPHA, e.ONE, e.ONE_MINUS_DST_ALPHA, e.ONE), e.blendEquationSeparate(e.FUNC_ADD, e.FUNC_ADD), e.uniformMatrix4fv(A, false, this._camera.data.projectionMatrix.buffer), e.uniformMatrix4fv(r, false, this._camera.data.viewMatrix.buffer), e.bindBuffer(e.ARRAY_BUFFER, W), e.vertexAttribPointer(B, 2, e.FLOAT, false, 0, 0), e.bindBuffer(e.ARRAY_BUFFER, Z), e.bufferData(e.ARRAY_BUFFER, this.depthIndex, e.STATIC_DRAW), e.vertexAttribIPointer(C, 1, e.INT, 0, 0), e.vertexAttribDivisor(C, 1), e.drawArraysInstanced(e.TRIANGLE_FAN, 0, 4, this.depthIndex.length);
      }, this._dispose = () => {
        if (!this._scene || !this._camera || !this.renderData) {
          console.error("Cannot dispose without scene and camera");
          return;
        }
        this._scene.removeEventListener("objectAdded", f), this._scene.removeEventListener("objectRemoved", H);
        for (const V of this._scene.objects)
          V instanceof X && V.removeEventListener("objectChanged", L);
        this._worker?.terminate(), this.renderData.dispose(), e.deleteTexture(this.splatTexture), e.deleteTexture(c), e.deleteTexture(p), e.deleteBuffer(Z), e.deleteBuffer(W);
      }, this._setOutlineThickness = (V) => {
        this._outlineThickness = V, this._initialized && e.uniform1f(F, V);
      }, this._setOutlineColor = (V) => {
        this._outlineColor = V, this._initialized && e.uniform4fv(g, new Float32Array(V.flatNorm()));
      };
    }
    get renderData() {
      return this._renderData;
    }
    get depthIndex() {
      return this._depthIndex;
    }
    get splatTexture() {
      return this._splatTexture;
    }
    get outlineThickness() {
      return this._outlineThickness;
    }
    set outlineThickness(t) {
      this._setOutlineThickness(t);
    }
    get outlineColor() {
      return this._outlineColor;
    }
    set outlineColor(t) {
      this._setOutlineColor(t);
    }
    get worker() {
      return this._worker;
    }
    _getVertexSource() {
      return Zt;
    }
    _getFragmentSource() {
      return pt;
    }
  };
  var Nt = class {
    constructor(t = 1) {
      let n = 0, i = false, e, A, o, s;
      this.initialize = (r) => {
        if (!(r instanceof Ct))
          throw new Error("FadeInPass requires a RenderProgram");
        n = r.started ? 1 : 0, i = true, e = r, A = r.renderer.gl, o = A.getUniformLocation(e.program, "useDepthFade"), A.uniform1i(o, 1), s = A.getUniformLocation(e.program, "depthFade"), A.uniform1f(s, n);
      }, this.render = () => {
        !i || e.renderData?.updating || (A.useProgram(e.program), n = Math.min(n + t * 0.01, 1), n >= 1 && (i = false, A.uniform1i(o, 0)), A.uniform1f(s, n));
      };
    }
    dispose() {
    }
  };
  var Mt = class {
    constructor(t = null, n = null) {
      this._backgroundColor = new dt();
      const i = t || document.createElement("canvas");
      t || (i.style.display = "block", i.style.boxSizing = "border-box", i.style.width = "100%", i.style.height = "100%", i.style.margin = "0", i.style.padding = "0", document.body.appendChild(i)), i.style.background = this._backgroundColor.toHexString(), this._canvas = i, this._gl = i.getContext("webgl2", { antialias: false });
      const e = n || [];
      n || e.push(new Nt()), this._renderProgram = new Ct(this, e);
      const A = [this._renderProgram];
      this.resize = () => {
        const o = i.clientWidth, s = i.clientHeight;
        (i.width !== o || i.height !== s) && this.setSize(o, s);
      }, this.setSize = (o, s) => {
        i.width = o, i.height = s, this._gl.viewport(0, 0, i.width, i.height);
        for (const r of A)
          r.resize();
      }, this.render = (o, s) => {
        for (const r of A)
          r.render(o, s);
      }, this.dispose = () => {
        for (const o of A)
          o.dispose();
      }, this.addProgram = (o) => {
        A.push(o);
      }, this.removeProgram = (o) => {
        const s = A.indexOf(o);
        if (s < 0)
          throw new Error("Program not found");
        A.splice(s, 1);
      }, this.resize();
    }
    get canvas() {
      return this._canvas;
    }
    get gl() {
      return this._gl;
    }
    get renderProgram() {
      return this._renderProgram;
    }
    get backgroundColor() {
      return this._backgroundColor;
    }
    set backgroundColor(t) {
      this._backgroundColor = t, this._canvas.style.background = t.toHexString();
    }
  };
  var Lt = class {
    constructor(t, n, i = 0.5, e = 0.5, A = 5, o = true, s = new R()) {
      this.minAngle = -90, this.maxAngle = 90, this.minZoom = 0.1, this.maxZoom = 30, this.orbitSpeed = 1, this.panSpeed = 1, this.zoomSpeed = 1, this.dampening = 0.12, this.setCameraTarget = () => {
      };
      let r = s.clone(), Q = r.clone(), I = i, d = e, a = A, U = false, F = false, g = 0, B = 0, C = 0;
      const c = {};
      let p = false;
      const u = () => {
        if (p) return;
        const l = t.rotation.toEuler();
        I = -l.y, d = -l.x;
        const J = t.position.x - a * Math.sin(I) * Math.cos(d), D = t.position.y + a * Math.sin(d), b = t.position.z + a * Math.cos(I) * Math.cos(d);
        Q = new R(J, D, b);
      };
      t.addEventListener("objectChanged", u), this.setCameraTarget = (l) => {
        const J = l.x - t.position.x, D = l.y - t.position.y, b = l.z - t.position.z;
        a = Math.sqrt(J * J + D * D + b * b), d = Math.atan2(D, Math.sqrt(J * J + b * b)), I = -Math.atan2(J, b), Q = new R(l.x, l.y, l.z);
      };
      const S = () => 0.1 + 0.9 * (a - this.minZoom) / (this.maxZoom - this.minZoom), W = (l) => {
        c[l.code] = true, l.code === "ArrowUp" && (c.KeyW = true), l.code === "ArrowDown" && (c.KeyS = true), l.code === "ArrowLeft" && (c.KeyA = true), l.code === "ArrowRight" && (c.KeyD = true);
      }, Z = (l) => {
        c[l.code] = false, l.code === "ArrowUp" && (c.KeyW = false), l.code === "ArrowDown" && (c.KeyS = false), l.code === "ArrowLeft" && (c.KeyA = false), l.code === "ArrowRight" && (c.KeyD = false);
      }, k = (l) => {
        T(l), U = true, F = l.button === 2, B = l.clientX, C = l.clientY, window.addEventListener("mouseup", f);
      }, f = (l) => {
        T(l), U = false, F = false, window.removeEventListener("mouseup", f);
      }, H = (l) => {
        if (T(l), !U || !t) return;
        const J = l.clientX - B, D = l.clientY - C;
        if (F) {
          const b = S(), j = -J * this.panSpeed * 0.01 * b, q = -D * this.panSpeed * 0.01 * b, G = _.RotationFromQuaternion(t.rotation).buffer, h = new R(G[0], G[3], G[6]), m = new R(G[1], G[4], G[7]);
          Q = Q.add(h.multiply(j)), Q = Q.add(m.multiply(q));
        } else
          I -= J * this.orbitSpeed * 3e-3, d += D * this.orbitSpeed * 3e-3, d = Math.min(
            Math.max(d, this.minAngle * Math.PI / 180),
            this.maxAngle * Math.PI / 180
          );
        B = l.clientX, C = l.clientY;
      }, L = (l) => {
        T(l);
        const J = S();
        a += l.deltaY * this.zoomSpeed * 0.025 * J, a = Math.min(Math.max(a, this.minZoom), this.maxZoom);
      }, M = (l) => {
        if (T(l), l.touches.length === 1)
          U = true, F = false, B = l.touches[0].clientX, C = l.touches[0].clientY, g = 0;
        else if (l.touches.length === 2) {
          U = true, F = true, B = (l.touches[0].clientX + l.touches[1].clientX) / 2, C = (l.touches[0].clientY + l.touches[1].clientY) / 2;
          const J = l.touches[0].clientX - l.touches[1].clientX, D = l.touches[0].clientY - l.touches[1].clientY;
          g = Math.sqrt(J * J + D * D);
        }
      }, V = (l) => {
        T(l), U = false, F = false;
      }, w = (l) => {
        if (T(l), !(!U || !t))
          if (F) {
            const J = S(), D = l.touches[0].clientX - l.touches[1].clientX, b = l.touches[0].clientY - l.touches[1].clientY, j = Math.sqrt(D * D + b * b), q = g - j;
            a += q * this.zoomSpeed * 0.1 * J, a = Math.min(Math.max(a, this.minZoom), this.maxZoom), g = j;
            const G = (l.touches[0].clientX + l.touches[1].clientX) / 2, h = (l.touches[0].clientY + l.touches[1].clientY) / 2, m = G - B, x = h - C, N = _.RotationFromQuaternion(t.rotation).buffer, v = new R(N[0], N[3], N[6]), O = new R(N[1], N[4], N[7]);
            Q = Q.add(v.multiply(-m * this.panSpeed * 0.025 * J)), Q = Q.add(O.multiply(-x * this.panSpeed * 0.025 * J)), B = G, C = h;
          } else {
            const J = l.touches[0].clientX - B, D = l.touches[0].clientY - C;
            I -= J * this.orbitSpeed * 3e-3, d += D * this.orbitSpeed * 3e-3, d = Math.min(
              Math.max(d, this.minAngle * Math.PI / 180),
              this.maxAngle * Math.PI / 180
            ), B = l.touches[0].clientX, C = l.touches[0].clientY;
          }
      }, P = (l, J, D) => (1 - D) * l + D * J;
      this.update = () => {
        p = true, i = P(i, I, this.dampening), e = P(e, d, this.dampening), A = P(A, a, this.dampening), r = r.lerp(Q, this.dampening);
        const l = r.x + A * Math.sin(i) * Math.cos(e), J = r.y - A * Math.sin(e), D = r.z - A * Math.cos(i) * Math.cos(e);
        t.position = new R(l, J, D);
        const b = r.subtract(t.position).normalize(), j = Math.asin(-b.y), q = Math.atan2(b.x, b.z);
        t.rotation = y.FromEuler(new R(j, q, 0));
        const G = 0.025, h = 0.01, m = _.RotationFromQuaternion(t.rotation).buffer, x = new R(-m[2], -m[5], -m[8]), N = new R(m[0], m[3], m[6]);
        c.KeyS && (Q = Q.add(x.multiply(G))), c.KeyW && (Q = Q.subtract(x.multiply(G))), c.KeyA && (Q = Q.subtract(N.multiply(G))), c.KeyD && (Q = Q.add(N.multiply(G))), c.KeyE && (I += h), c.KeyQ && (I -= h), c.KeyR && (d += h), c.KeyF && (d -= h), p = false;
      };
      const T = (l) => {
        l.preventDefault(), l.stopPropagation();
      };
      this.dispose = () => {
        n.removeEventListener("dragenter", T), n.removeEventListener("dragover", T), n.removeEventListener("dragleave", T), n.removeEventListener("contextmenu", T), n.removeEventListener("mousedown", k), n.removeEventListener("mousemove", H), n.removeEventListener("wheel", L), n.removeEventListener("touchstart", M), n.removeEventListener("touchend", V), n.removeEventListener("touchmove", w), o && (window.removeEventListener("keydown", W), window.removeEventListener("keyup", Z));
      }, o && (window.addEventListener("keydown", W), window.addEventListener("keyup", Z)), n.addEventListener("dragenter", T), n.addEventListener("dragover", T), n.addEventListener("dragleave", T), n.addEventListener("contextmenu", T), n.addEventListener("mousedown", k), n.addEventListener("mousemove", H), n.addEventListener("wheel", L), n.addEventListener("touchstart", M), n.addEventListener("touchend", V), n.addEventListener("touchmove", w), this.update();
    }
  };
  var Gt = class {
    constructor(t, n) {
      this.moveSpeed = 1.5, this.lookSpeed = 0.7, this.dampening = 0.5;
      const i = {};
      let e = t.rotation.toEuler().x, A = t.rotation.toEuler().y, o = t.position, s = false;
      const r = () => {
        n.requestPointerLock();
      }, Q = () => {
        s = document.pointerLockElement === n, s ? n.addEventListener("mousemove", I) : n.removeEventListener("mousemove", I);
      }, I = (F) => {
        const g = F.movementX, B = F.movementY;
        A += g * this.lookSpeed * 1e-3, e -= B * this.lookSpeed * 1e-3, e = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, e));
      }, d = (F) => {
        i[F.code] = true, F.code === "ArrowUp" && (i.KeyW = true), F.code === "ArrowDown" && (i.KeyS = true), F.code === "ArrowLeft" && (i.KeyA = true), F.code === "ArrowRight" && (i.KeyD = true);
      }, a = (F) => {
        i[F.code] = false, F.code === "ArrowUp" && (i.KeyW = false), F.code === "ArrowDown" && (i.KeyS = false), F.code === "ArrowLeft" && (i.KeyA = false), F.code === "ArrowRight" && (i.KeyD = false), F.code === "Escape" && document.exitPointerLock();
      };
      this.update = () => {
        const F = _.RotationFromQuaternion(t.rotation).buffer, g = new R(-F[2], -F[5], -F[8]), B = new R(F[0], F[3], F[6]);
        let C = new R(0, 0, 0);
        i.KeyS && (C = C.add(g)), i.KeyW && (C = C.subtract(g)), i.KeyA && (C = C.subtract(B)), i.KeyD && (C = C.add(B)), C = new R(C.x, 0, C.z), C.magnitude() > 0 && (C = C.normalize()), o = o.add(C.multiply(this.moveSpeed * 0.01)), t.position = t.position.add(o.subtract(t.position).multiply(this.dampening)), t.rotation = y.FromEuler(new R(e, A, 0));
      };
      const U = (F) => {
        F.preventDefault(), F.stopPropagation();
      };
      this.dispose = () => {
        n.removeEventListener("dragenter", U), n.removeEventListener("dragover", U), n.removeEventListener("dragleave", U), n.removeEventListener("contextmenu", U), n.removeEventListener("mousedown", r), document.removeEventListener("pointerlockchange", Q), window.removeEventListener("keydown", d), window.removeEventListener("keyup", a);
      }, window.addEventListener("keydown", d), window.addEventListener("keyup", a), n.addEventListener("dragenter", U), n.addEventListener("dragover", U), n.addEventListener("dragleave", U), n.addEventListener("contextmenu", U), n.addEventListener("mousedown", r), document.addEventListener("pointerlockchange", Q), this.update();
    }
  };
  var vt = class {
    constructor(t, n) {
      this.normal = t, this.point = n;
    }
    intersect(t, n) {
      const i = this.normal.dot(n);
      if (Math.abs(i) < 1e-4)
        return null;
      const e = this.normal.dot(this.point.subtract(t)) / i;
      return e < 0 ? null : t.add(n.multiply(e));
    }
  };
  var _t = class {
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    initialize(t) {
    }
    render() {
    }
    dispose() {
    }
  };
  var Wt = (
    /* glsl */
    `#version 300 es
precision highp float;
precision highp int;
  
uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;
uniform float time;
  
in vec2 position;
in int index;
  
out vec4 vColor;
out vec2 vPosition;
  
void main () {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);

    uvec4 motion1 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 3u, uint(index) >> 10), 0);
    vec2 trbf = unpackHalf2x16(motion1.w);
    float dt = time - trbf.x;

    float topacity = exp(-1.0 * pow(dt / trbf.y, 2.0));
    if(topacity < 0.02) return;

    uvec4 motion0 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 2u, uint(index) >> 10), 0);
    uvec4 static0 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2), uint(index) >> 10), 0);

    vec2 m0 = unpackHalf2x16(motion0.x), m1 = unpackHalf2x16(motion0.y), m2 = unpackHalf2x16(motion0.z), 
         m3 = unpackHalf2x16(motion0.w), m4 = unpackHalf2x16(motion1.x); 
      
    vec4 trot = vec4(unpackHalf2x16(motion1.y).xy, unpackHalf2x16(motion1.z).xy) * dt;
    vec3 tpos = (vec3(m0.xy, m1.x) * dt + vec3(m1.y, m2.xy) * dt*dt + vec3(m3.xy, m4.x) * dt*dt*dt);
      
    vec4 cam = view * vec4(uintBitsToFloat(static0.xyz) + tpos, 1);
    vec4 pos = projection * cam;
  
    float clip = 1.2 * pos.w;
    if (pos.z < -clip || pos.x < -clip || pos.x > clip || pos.y < -clip || pos.y > clip) return;
    uvec4 static1 = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 2) | 1u, uint(index) >> 10), 0);

    vec4 rot = vec4(unpackHalf2x16(static0.w).xy, unpackHalf2x16(static1.x).xy) + trot;
    vec3 scale = vec3(unpackHalf2x16(static1.y).xy, unpackHalf2x16(static1.z).x);
    rot /= sqrt(dot(rot, rot));
  
    mat3 S = mat3(scale.x, 0.0, 0.0, 0.0, scale.y, 0.0, 0.0, 0.0, scale.z);
    mat3 R = mat3(
        1.0 - 2.0 * (rot.z * rot.z + rot.w * rot.w), 2.0 * (rot.y * rot.z - rot.x * rot.w), 2.0 * (rot.y * rot.w + rot.x * rot.z),
        2.0 * (rot.y * rot.z + rot.x * rot.w), 1.0 - 2.0 * (rot.y * rot.y + rot.w * rot.w), 2.0 * (rot.z * rot.w - rot.x * rot.y),
        2.0 * (rot.y * rot.w - rot.x * rot.z), 2.0 * (rot.z * rot.w + rot.x * rot.y), 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z));
    mat3 M = S * R;
    mat3 Vrk = 4.0 * transpose(M) * M;
    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z), 
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z), 
        0., 0., 0.
    );
  
    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;
  
    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;
  
    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);
      
    uint rgba = static1.w;
    vColor = 
        clamp(pos.z/pos.w+1.0, 0.0, 1.0) * 
        vec4(1.0, 1.0, 1.0, topacity) *
        vec4(
            (rgba) & 0xffu, 
            (rgba >> 8) & 0xffu, 
            (rgba >> 16) & 0xffu, 
            (rgba >> 24) & 0xffu) / 255.0;

    vec2 vCenter = vec2(pos) / pos.w;
    gl_Position = vec4(
        vCenter 
        + position.x * majorAxis / viewport 
        + position.y * minorAxis / viewport, 0.0, 1.0);

    vPosition = position;
}
`
  );
  var Dt = (
    /* glsl */
    `#version 300 es
precision highp float;
  
in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
}
`
  );
  var Yt = class extends lt {
    constructor(t, n = []) {
      super(t, n), this._renderData = null, this._depthIndex = new Uint32Array(), this._splatTexture = null;
      const i = t.canvas, e = t.gl;
      let A, o, s, r, Q, I, d, a, U, F, g;
      this._resize = () => {
        this._camera && (this._camera.data.setSize(i.width, i.height), this._camera.update(), o = e.getUniformLocation(this.program, "projection"), e.uniformMatrix4fv(o, false, this._camera.data.projectionMatrix.buffer), s = e.getUniformLocation(this.program, "viewport"), e.uniform2fv(s, new Float32Array([i.width, i.height])));
      };
      const B = () => {
        if (t.renderProgram.worker === null) {
          console.error("Render program is not initialized. Cannot render without worker");
          return;
        }
        A = t.renderProgram.worker, A.onmessage = (u) => {
          if (u.data.depthIndex) {
            const { depthIndex: S } = u.data;
            this._depthIndex = S, e.bindBuffer(e.ARRAY_BUFFER, g), e.bufferData(e.ARRAY_BUFFER, S, e.STATIC_DRAW);
          }
        };
      };
      this._initialize = () => {
        if (!this._scene || !this._camera) {
          console.error("Cannot render without scene and camera");
          return;
        }
        this._resize(), this._scene.addEventListener("objectAdded", C), this._scene.addEventListener("objectRemoved", c);
        for (const Z of this._scene.objects)
          Z instanceof tt && (this._renderData === null ? (this._renderData = Z.data, Z.addEventListener("objectChanged", p)) : console.warn("Multiple Splatv objects are not currently supported"));
        if (this._renderData === null) {
          console.error("Cannot render without Splatv object");
          return;
        }
        r = e.getUniformLocation(this.program, "focal"), e.uniform2fv(r, new Float32Array([this._camera.data.fx, this._camera.data.fy])), Q = e.getUniformLocation(this.program, "view"), e.uniformMatrix4fv(Q, false, this._camera.data.viewMatrix.buffer), this._splatTexture = e.createTexture(), I = e.getUniformLocation(this.program, "u_texture"), e.uniform1i(I, 0), d = e.getUniformLocation(this.program, "time"), e.uniform1f(d, Math.sin(Date.now() / 1e3) / 2 + 1 / 2), F = e.createBuffer(), e.bindBuffer(e.ARRAY_BUFFER, F), e.bufferData(e.ARRAY_BUFFER, new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]), e.STATIC_DRAW), a = e.getAttribLocation(this.program, "position"), e.enableVertexAttribArray(a), e.vertexAttribPointer(a, 2, e.FLOAT, false, 0, 0), g = e.createBuffer(), U = e.getAttribLocation(this.program, "index"), e.enableVertexAttribArray(U), e.bindBuffer(e.ARRAY_BUFFER, g), B(), e.activeTexture(e.TEXTURE0), e.bindTexture(e.TEXTURE_2D, this._splatTexture), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_S, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_WRAP_T, e.CLAMP_TO_EDGE), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MIN_FILTER, e.NEAREST), e.texParameteri(e.TEXTURE_2D, e.TEXTURE_MAG_FILTER, e.NEAREST), e.texImage2D(
          e.TEXTURE_2D,
          0,
          e.RGBA32UI,
          this._renderData.width,
          this._renderData.height,
          0,
          e.RGBA_INTEGER,
          e.UNSIGNED_INT,
          this._renderData.data
        );
        const u = this._renderData.positions, S = new Float32Array(new K().buffer), W = new Uint32Array(this._renderData.vertexCount);
        W.fill(0), A.postMessage(
          {
            sortData: {
              positions: u,
              transforms: S,
              transformIndices: W,
              vertexCount: this._renderData.vertexCount
            }
          },
          [u.buffer, S.buffer, W.buffer]
        );
      };
      const C = (u) => {
        const S = u;
        S.object instanceof tt && (this._renderData === null ? (this._renderData = S.object.data, S.object.addEventListener("objectChanged", p)) : console.warn("Splatv not supported by default RenderProgram. Use VideoRenderProgram instead.")), this.dispose();
      }, c = (u) => {
        const S = u;
        S.object instanceof tt && this._renderData === S.object.data && (this._renderData = null, S.object.removeEventListener("objectChanged", p)), this.dispose();
      }, p = (u) => {
        const S = u;
        S.object instanceof tt && this._renderData === S.object.data && this.dispose();
      };
      this._render = () => {
        if (!this._scene || !this._camera) {
          console.error("Cannot render without scene and camera");
          return;
        }
        if (!this._renderData) {
          console.warn("Cannot render without Splatv object");
          return;
        }
        this._camera.update(), A.postMessage({ viewProj: this._camera.data.viewProj.buffer }), e.viewport(0, 0, i.width, i.height), e.clearColor(0, 0, 0, 0), e.clear(e.COLOR_BUFFER_BIT), e.disable(e.DEPTH_TEST), e.enable(e.BLEND), e.blendFuncSeparate(e.ONE_MINUS_DST_ALPHA, e.ONE, e.ONE_MINUS_DST_ALPHA, e.ONE), e.blendEquationSeparate(e.FUNC_ADD, e.FUNC_ADD), e.uniformMatrix4fv(o, false, this._camera.data.projectionMatrix.buffer), e.uniformMatrix4fv(Q, false, this._camera.data.viewMatrix.buffer), e.uniform1f(d, Math.sin(Date.now() / 1e3) / 2 + 1 / 2), e.bindBuffer(e.ARRAY_BUFFER, F), e.vertexAttribPointer(a, 2, e.FLOAT, false, 0, 0), e.bindBuffer(e.ARRAY_BUFFER, g), e.bufferData(e.ARRAY_BUFFER, this._depthIndex, e.STATIC_DRAW), e.vertexAttribIPointer(U, 1, e.INT, 0, 0), e.vertexAttribDivisor(U, 1), e.drawArraysInstanced(e.TRIANGLE_FAN, 0, 4, this._renderData.vertexCount);
      }, this._dispose = () => {
        if (!this._scene || !this._camera) {
          console.error("Cannot dispose without scene and camera");
          return;
        }
        this._scene.removeEventListener("objectAdded", C), this._scene.removeEventListener("objectRemoved", c);
        for (const u of this._scene.objects)
          u instanceof tt && this._renderData === u.data && (this._renderData = null, u.removeEventListener("objectChanged", p));
        A?.terminate(), e.deleteTexture(this._splatTexture), e.deleteBuffer(g), e.deleteBuffer(F);
      };
    }
    get renderData() {
      return this._renderData;
    }
    _getVertexSource() {
      return Wt;
    }
    _getFragmentSource() {
      return Dt;
    }
  };
  var it = class _it {
    constructor(t, n, i) {
      this.bounds = t, this.boxes = n, this.left = null, this.right = null, this.pointIndices = [], i.length > 1 ? this.split(t, n, i) : i.length > 0 && (this.pointIndices = i);
    }
    split(t, n, i) {
      const e = t.size().maxComponent();
      i.sort((r, Q) => n[r].center().getComponent(e) - n[Q].center().getComponent(e));
      const A = Math.floor(i.length / 2), o = i.slice(0, A), s = i.slice(A);
      this.left = new _it(t, n, o), this.right = new _it(t, n, s);
    }
    queryRange(t) {
      return this.bounds.intersects(t) ? this.left !== null && this.right !== null ? this.left.queryRange(t).concat(this.right.queryRange(t)) : this.pointIndices.filter((n) => t.intersects(this.boxes[n])) : [];
    }
  };
  var yt = class {
    constructor(t, n) {
      const i = n.map((e, A) => A);
      this.root = new it(t, n, i);
    }
    queryRange(t) {
      return this.root.queryRange(t);
    }
  };
  var Kt = class {
    constructor(t, n = 100, i = 1) {
      let e = 0, A = null, o = [];
      const s = () => {
        if (t.renderData === null) {
          console.error("IntersectionTester cannot be called before renderProgram has been initialized");
          return;
        }
        o = [];
        const r = t.renderData, Q = new Array(r.offsets.size);
        let I = 0;
        const d = new nt(
          new R(1 / 0, 1 / 0, 1 / 0),
          new R(-1 / 0, -1 / 0, -1 / 0)
        );
        for (const a of r.offsets.keys()) {
          const U = a.bounds;
          Q[I++] = U, d.expand(U.min), d.expand(U.max), o.push(a);
        }
        d.permute(), A = new yt(d, Q), e = r.vertexCount;
      };
      this.testPoint = (r, Q) => {
        if (t.renderData === null || t.camera === null)
          return console.error("IntersectionTester cannot be called before renderProgram has been initialized"), null;
        if (s(), A === null)
          return console.error("Failed to build octree for IntersectionTester"), null;
        const I = t.renderData, d = t.camera;
        e !== I.vertexCount && console.warn("IntersectionTester has not been rebuilt since the last render");
        const a = d.screenPointToRay(r, Q);
        for (let U = 0; U < n; U += i) {
          const F = d.position.add(a.multiply(U)), g = new R(
            F.x - i / 2,
            F.y - i / 2,
            F.z - i / 2
          ), B = new R(
            F.x + i / 2,
            F.y + i / 2,
            F.z + i / 2
          ), C = new nt(g, B), c = A.queryRange(C);
          if (c.length > 0)
            return o[c[0]];
        }
        return null;
      };
    }
  };
  return __toCommonJS(gsplat_entry_exports);
})();
