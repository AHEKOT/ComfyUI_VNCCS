import * as THREE from "three";
import { Phalange } from './Phalange.js';
import { grad, rad } from "../globals.js";

// size-x, size-y, size-z, alpha, dAlpha, offset, scale, rad
class Finger extends Phalange {

	constructor( parentJoint, leftOrRight, number ) {

		var thumb = ( number==0 );

		var sca = [ 1.1, 0.95, 1, 0.95, 0.8 ][ number ];
		var fat = [ 1.0, 0.95, 1, 0.95, 0.8 ][ number ];
		var fat2 = [ 1.5, 1, 1, 1, 1 ][ number ];

		// var minX = [ 0, -25, -10, -15, -35][number] * leftOrRight;
		// var maxX = [60,  25,  10,   5,  10][number] * leftOrRight;

		var minX = [ 0, -20, -15, -25, -35 ][ number ] * leftOrRight;
		var maxX = [ 50, 35, 15, 15, 20 ][ number ] * leftOrRight;

		super( parentJoint, [ 0.8*fat, 0.8*sca*( thumb?1.4:1 ), 0.8*fat2, 0, 45, 0.3, 0.4, 0.25 ], 0 );

		this.position.x = [ -0.3, 0.0, 0.15, 0.15, 0.03 ][ number ];
		this.position.y = [ 0.5, 2.2, 2.3, 2.2, 2.1 ][ number ];
		this.position.z = [ 0.8, 0.7, 0.225, -0.25, -0.7 ][ number ] * leftOrRight;

		this.mid = new Phalange( this, [ 0.6*fat, 0.7*sca*( thumb?1.1:1 ), 0.6*fat2, 0, 60, 0.3, 0.4, 0.15 ], 0 );
		this.tip = new Phalange( this.mid, [ 0.5*fat, 0.6*sca*( thumb?1.1:1 ), 0.5*fat2, 0, 60, 0.3, 0.4, 0.1 ], fat2 );

		this.leftOrRight = leftOrRight;

		this.y = thumb ? -this.leftOrRight * 90 : 0;

		this.minRot = new THREE.Vector3( Math.min( minX, maxX ), Math.min( this.y, 2*this.y ), thumb?-90:-10 );
		this.maxRot = new THREE.Vector3( Math.max( minX, maxX ), Math.max( this.y, 2*this.y ), thumb?45:120 );

		this.mid.minRot = new THREE.Vector3( 0, 0, 0 );
		this.mid.maxRot = new THREE.Vector3( 0, 0, thumb?90:120 );

		this.tip.minRot = new THREE.Vector3( 0, 0, 0 );
		this.tip.maxRot = new THREE.Vector3( 0, 0, thumb?90:120 );

		this.name = 'Finger';

	} // Finger.constructor

	get bend() {

		return this.z;

	}

	set bend( angle ) {

		this.z = angle;

	}

	get straddle() {

		return -this.leftOrRight * this.x;

	}

	set straddle( angle ) {

		this.x = -this.leftOrRight * angle;

	}


	get turn() {

		return -this.leftOrRight * this.y;

	}

	set turn( angle ) {

		this.y = -this.leftOrRight * angle;

	}


	get posture() {

		this.rotation.reorder( 'XYZ' );
		this.mid.rotation.reorder( 'XYZ' );
		this.tip.rotation.reorder( 'XYZ' );
		return [
			grad( this.rotation.x ),
			grad( this.rotation.y ),
			grad( this.rotation.z ),

			grad( this.mid.rotation.x ),
			grad( this.mid.rotation.z ),

			grad( this.tip.rotation.x ),
			grad( this.tip.rotation.z ),
		];

	}

	set posture( pos ) {

		this.rotation.reorder( 'XYZ' );
		this.rotation.x = rad( pos[ 0 ]);
		this.rotation.y = rad( pos[ 1 ]);
		this.rotation.z = rad( pos[ 2 ]);
		this.mid.rotation.set( rad( pos[ 3 ]), 0, rad( pos[ 4 ]), 'XYZ' );
		this.tip.rotation.set( rad( pos[ 5 ]), 0, rad( pos[ 6 ]), 'XYZ' );

	}

} // Finger




export { Finger };
