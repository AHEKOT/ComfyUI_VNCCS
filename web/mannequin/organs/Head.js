import * as THREE from "three";
import { HeadShape } from '../shapes/HeadShape.js';
import { Joint } from "./Joint.js";
import { grad, rad } from "../globals.js";

class Head extends Joint {

	static SIZE = { sx: 3*1.5, sy: 4*1.5, sz: 2.5*1.5 };

	constructor( parentJoint ) {

		super( parentJoint, [ 1, 3+1.5, 0 ], Head.SIZE, HeadShape );

		this.minRot = new THREE.Vector3( -45 / 2, -90 / 2, -60 / 2 );
		this.maxRot = new THREE.Vector3( 45 / 2, 90 / 2, 50 / 2 );

		this.name = 'Head';

	} // Head.constructor

	get nod() {

		return -2 * this.z;

	}

	set nod( angle ) {

		this.z = -angle / 2;
		this.parentJoint.z = -angle / 2;

	}

	get tilt() {

		return -2 * this.x;

	}

	set tilt( angle ) {

		this.x = -angle / 2;
		this.parentJoint.x = -angle / 2;

	}

	get turn() {

		return 2 * this.y;

	}

	set turn( angle ) {

		this.y = angle / 2;
		this.parentJoint.y = angle / 2;

	}

	get posture() {

		this.rotation.reorder( 'XYZ' );
		return [ grad( this.rotation.x ), grad( this.rotation.y ), grad( this.rotation.z ) ];

	}

	set posture( pos ) {

		this.rotation.set( rad( pos[ 0 ]), rad( pos[ 1 ]), rad( pos[ 2 ]), 'XYZ' );
		this.parentJoint.rotation.set( rad( pos[ 0 ]), rad( pos[ 1 ]), rad( pos[ 2 ]), 'XYZ' );

	} // Head.posture

} // Head




export { Head };
