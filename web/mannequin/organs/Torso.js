import * as THREE from "three";
import { TorsoShape } from '../shapes/TorsoShape.js';
import { Joint } from "./Joint.js";


class Torso extends Joint {

	constructor( parentJoint ) {

		super( parentJoint, [ -2, 4, 0 ], [ 5, 17, 10, parentJoint.feminine ? 10 : 80, parentJoint.feminine ? 520 : 380, parentJoint.feminine ? 0.8 : 0.9, parentJoint.feminine ? 0.25 : 0.2 ], TorsoShape );

		this.minRot = new THREE.Vector3( -25, -50, -60 );
		this.maxRot = new THREE.Vector3( 25, 50, 25 );

		this.name = 'Torso';

	} // Torso.constructor

	get bend() {

		return -this.z;

	}

	set bend( angle ) {

		this.z = -angle;

	}

	get tilt() {

		return -this.x;

	}

	set tilt( angle ) {

		this.x = -angle;

	}

	get turn() {

		return this.y;

	}

	set turn( angle ) {

		this.y = angle;

	}

} // Torso




export { Torso };
