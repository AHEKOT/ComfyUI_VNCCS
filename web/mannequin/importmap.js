
const LOCAL = true; // use local/github or global/npm path

const PATH = LOCAL ? '../src' : 'https://cdn.jsdelivr.net/npm/mannequin-js@5.0.0/src';

const im = document.createElement( 'script' );
im.type = 'importmap';
im.textContent = JSON.stringify( {
	imports: {
		"three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
		"three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/",
		"mannequin": PATH+"/mannequin.js",
		"mannequin/": PATH+"/"
	}
} );
document.currentScript.after( im );
