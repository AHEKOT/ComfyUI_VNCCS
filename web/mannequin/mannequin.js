// mannequin.js
//
// getVersion( )
// getPostureVersion( )
// getGroundLevel( )
// blend( posture0, posture1, k )
// addLabel( joint, text, y, z=0.5, x=0, rotY=1, rotZ=0, rotX=0 )
// createStage( animateFunction )
//export * from './bodies/Mannequin.js';
//export * from './bodies/Female.js';
//export * from './bodies/Male.js';
//export * from './bodies/Child.js';



import { GROUND_LEVEL, VERSION, VERSION_POSTURE } from './globals.js';



// return the software version
function getVersion( ) {

	return VERSION;

}



// return the posture data version
function getPostureVersion( ) {

	return VERSION_POSTURE;

}



// return the vertical position of the ground
function getGroundLevel( ) {

	return GROUND_LEVEL;

}



// blends two postures and returns the resulting posture
function blend( posture0, posture1, k ) {

	if ( posture0.version != posture1.version )
		throw 'Incompatibe posture blending.';

	function lerp( data0, data1, k ) {

		if ( data0 instanceof Array ) {

			var result = [];
			for ( var i in data0 )
				result.push( lerp( data0[ i ], data1[ i ], k ) );
			return result;

		} else {

			return data0 * ( 1 - k ) + k * data1;

		}

	}

	return {
		version: posture1.version,
		data: lerp( posture0.data, posture1.data, k )
	};

} // blend



export * from './bodies/Mannequin.js';
export * from './bodies/Female.js';
export * from './bodies/Male.js';
export * from './bodies/Child.js';
export { createStage, getStage } from './scene.js';
export { getVersion, getPostureVersion, getGroundLevel, blend };
