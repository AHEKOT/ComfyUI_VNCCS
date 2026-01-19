import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
/*import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";*/
import { GROUND_LEVEL } from "./globals.js";



var renderer, scene, camera, light, controls;


// add favicon
var link = document.createElement( 'link' );
link.rel = 'icon';
document.head.appendChild( link );
if ( window.location.pathname.indexOf( 'src/editor' )<0 )
	link.href = '../assets/logo/logo.png';
else
	link.href = '../../assets/logo/logo.png';



// add meta tag for mobile devices
var meta = document.createElement( 'meta' );
meta.name = "viewport";
meta.content = "width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0";
document.head.appendChild( meta );



var clock = new THREE.Clock();



function systemAnimate() {

	if ( controls ) controls.update();

	if ( stage.animationLoop ) stage.animationLoop( clock.getElapsedTime() );

	renderer.render( scene, camera );

}



// initialize Three.js elements that are created by default
function initStage( ) {

	renderer = new THREE.WebGLRenderer( { antialias: true } );
	renderer.setSize( window.innerWidth, window.innerHeight );
	renderer.outputColorSpace = THREE.SRGBColorSpace;
	renderer.domElement.style = 'width:100%; height:100%; position:fixed; top:0; left:0;';
	renderer.setAnimationLoop( systemAnimate );
	document.body.appendChild( renderer.domElement );

	scene = new THREE.Scene();
	scene.background = new THREE.Color( 'gainsboro' );

	camera = new THREE.PerspectiveCamera( 30, window.innerWidth / window.innerHeight, 0.1, 2000 );
	camera.position.set( 0, 0, 5 );

	light = new THREE.DirectionalLight( 'white', 2.75 );
	light.decay = 0;
	light.penumbra = 0.5;
	light.angle = 0.8;
	light.position.set( 0, 2, 1 ).setLength( 15 );
	scene.add( light, new THREE.AmbientLight( 'white', 0.5 ) );

	function onWindowResize( /*event*/ ) {

		camera.aspect = window.innerWidth / window.innerHeight;
		camera.updateProjectionMatrix();

		renderer.setSize( window.innerWidth, window.innerHeight, true );

	}

	window.addEventListener( 'resize', onWindowResize, false );
	onWindowResize();

	return {
		renderer: renderer,
		scene: scene,
		camera: camera,
		light: light,
		animationLoop: null,
	};

}


// the stage is already created, just add ground, shadows, etc
function createStage( animationLoop ) {

	// turn on shadows
	renderer.shadowMap.enabled = true;
	renderer.shadowMap.type = THREE.PCFSoftShadowMap;

	// add shadows to light
	light.shadow.mapSize.width = Math.min( 4 * 1024, renderer.capabilities.maxTextureSize / 2 );
	light.shadow.mapSize.height = light.shadow.mapSize.width;
	light.shadow.camera.near = 13;
	light.shadow.camera.far = 18.5;
	light.shadow.camera.left = -5;
	light.shadow.camera.right = 5;
	light.shadow.camera.top = 5;
	light.shadow.camera.bottom = -5;
	light.shadow.normalBias = 0.005;
	light.autoUpdate = false;
	light.castShadow = true;

	//scene.add( new THREE.CameraHelper(light.shadow.camera));


	// add ground
	var canvas = document.createElement( 'CANVAS' );
	canvas.width = 512;
	canvas.height = 512;

	var context = canvas.getContext( '2d' );
	context.fillStyle = 'white';
	context.filter = 'blur(40px)';
	context.beginPath();
	context.arc( 256, 256, 150, 0, 2*Math.PI );
	context.fill();

	var ground = new THREE.Mesh(
		new THREE.CircleGeometry( 50 ),
		new THREE.MeshLambertMaterial(
			{
				color: 'antiquewhite',
				transparent: true,
				map: new THREE.CanvasTexture( canvas )
			} )
	);
	ground.receiveShadow = true;
	ground.position.y = GROUND_LEVEL;
	ground.rotation.x = -Math.PI / 2;
	ground.renderOrder = -1;
	scene.add( ground );

	// add conntrols
	controls = new OrbitControls( camera, renderer.domElement );
	controls.enableDamping = true;

	stage.animationLoop = animationLoop;

	// add new properties to the stage
	stage.ground = ground;
	stage.controls = controls;

	/*
	const loader = new GLTFLoader();
	loader.load( 'https://threejs.org/examples/models/gltf/Soldier.glb', function ( gltf ) {

		const model = gltf.scene;

var		rightArm = model.getObjectByName( 'mixamorigRightArm' );
console.log(model)
		scene.add( model );

	} );
*/

} // createScene



var stage = initStage( );



function getStage( ) {

	return stage;

}



export { renderer, scene, camera, light, controls, createStage, getStage, systemAnimate, clock };
