# Local Gaussian renderer provenance

- SparkJS `v2.1.0` (`@sparkjsdev/spark`), commit
  `fec6d05d0caa1ab1b3ac8a3d480ce13383ff1c96`, MIT license.
- Three.js `r180`, MIT license.

These renderer assets are carried from VNCCS-Utils with their original licenses.
Spark imports the adjacent Three.js and Pass modules. Its worker and WASM payloads
are embedded; no CDN or Factory code is required. Background Creator imports this
runtime only when opening a model. Unused Factory controls and model loaders are omitted.
