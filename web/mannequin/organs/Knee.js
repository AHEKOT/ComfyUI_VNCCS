import * as THREE from "three";
import { LimbShape } from '../shapes/LimbShape.js';
import { Joint } from "./Joint.js";
import { grad, rad } from "../globals.js";


class Knee extends Joint {

	constructor( parentJoint ) {

		super( parentJoint, null, [ 4, 14, 4, -40, 290, 0.65, 0.25, 1.5 ], LimbShape );

		this.minRot = new THREE.Vector3( 0, 0, 0 );
		this.maxRot = new THREE.Vector3( 0, 0, 150 );

		this.name = 'Knee';

	} // Knee.constructor

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

} // Knee




export { Knee };
