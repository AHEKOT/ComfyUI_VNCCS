import * as THREE from "three";
import { LimbShape } from '../shapes/LimbShape.js';
import { Joint } from "./Joint.js";


class Wrist extends Joint {

	constructor( parentJoint ) {

		super( parentJoint, null, [ 1, 2.2, 2.5, -90, 120, 0.5, 0.3, 1 / 2 ], LimbShape );
		this.leftOrRight = parentJoint.parentJoint.leftOrRight;

		this.imageWrapper.rotation.set( 0, -this.leftOrRight * Math.PI / 2, 0 );

		if ( this.leftOrRight == -1 ) {

			this.minRot = new THREE.Vector3( -20, -90, -90 );
			this.maxRot = new THREE.Vector3( 35, 90, 90 );

		} else {

			this.minRot = new THREE.Vector3( -35, -90, -90 );
			this.maxRot = new THREE.Vector3( 20, 90, 90 );

		}

		this.name = 'Wrist';

	} // Wrist.constructor

	biologicallyImpossibleLevel() {

		// return 0 if the rotation is possible
		// return >0 if it is not possible, the higher the result, the more impossible it is

		var result = 0;

		var wristX = new THREE.Vector3(),
			wristY = new THREE.Vector3(),
			wristZ = new THREE.Vector3();
		this.matrixWorld.extractBasis( wristX, wristY, wristZ );

		var elbowX = new THREE.Vector3(),
			elbowY = new THREE.Vector3(),
			elbowZ = new THREE.Vector3();
		this.parentJoint.matrixWorld.extractBasis( elbowX, elbowY, elbowZ );

		var dot1 = wristY.dot( elbowY );
		if ( dot1 < 0 ) result += -dot1;

		var dot2 = wristZ.dot( elbowZ );
		if ( dot2 < 0 ) result += -dot2;

		return result;

	} // Wrist.biologicallyImpossibleLevel

	get bend() {

		return -this.leftOrRight * this.x;

	}

	set bend( angle ) {

		this.x = -this.leftOrRight * angle;

	}

	get tilt() {

		return this.leftOrRight * this.z;

	}

	set tilt( angle ) {

		this.z = this.leftOrRight * angle;

	}

	get turn() {

		return this.leftOrRight * this.y;

	}

	set turn( angle ) {

		this.y = this.leftOrRight * angle;

	}

} // Wrist




export { Wrist };
