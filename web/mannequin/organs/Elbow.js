import * as THREE from "three";
import { LimbShape } from '../shapes/LimbShape.js';
import { Joint } from "./Joint.js";
import { grad, rad } from "../globals.js";

class Elbow extends Joint {

	constructor( parentJoint ) {

		super( parentJoint, null, [ 2.5, 11, 2, -40, 150, 0.5, 0.45, 1.1 ], LimbShape );

		this.minRot = new THREE.Vector3( 0, 0, 0 );
		this.maxRot = new THREE.Vector3( 0, 0, 150 );

		this.name = 'Elbow';

	} // Elbow.constructor

	get bend() {

		return this.z;

	}

	set bend( angle ) {

		this.z = angle;

	}

	get posture() {

		this.rotation.reorder( 'XYZ' );
		return [ grad( this.rotation.z ) ];

	}

	set posture( pos ) {

		this.rotation.set( 0, 0, rad( pos[ 0 ]), 'XYZ' );

	}

} // Elbow



export { Elbow };
