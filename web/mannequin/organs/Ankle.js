import * as THREE from "three";
import { ShoeShape } from '../shapes/ShoeShape.js';
import { Joint } from "./Joint.js";


class Ankle extends Joint {

	static SIZE = { sx: 1, sy: 4, sz: 2 };

	constructor( parentJoint ) {

		super( parentJoint, null, Ankle.SIZE, ShoeShape );
		this.leftOrRight = parentJoint.parentJoint.leftOrRight; // i.e. leg

		this.minRot = new THREE.Vector3( -25, -30, -70 );
		this.maxRot = new THREE.Vector3( 25, 30, 80 );

		this.name = 'Ankle';

	} // Ankle.constructor

	get bend() {

		return -this.z;

	}

	set bend( angle ) {

		this.z = -angle;

	}

	get tilt() {

		return this.leftOrRight * this.x;

	}

	set tilt( angle ) {

		this.x = this.leftOrRight * angle;

	}

	get turn() {

		return this.leftOrRight * this.y;

	}

	set turn( angle ) {

		this.y = this.leftOrRight * angle;

	}

} // Ankle




export { Ankle };
