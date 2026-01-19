import * as THREE from "three";
import { LimbShape } from '../shapes/LimbShape.js';
import { Joint } from "./Joint.js";


class Neck extends Joint {

	constructor( parentJoint ) {

		super( parentJoint, [ 0, 15, 0 ], [ 2, parentJoint.feminine ? 5 : 4, 2, 45, 60, 1, 0.2, 0 ], LimbShape );

		this.minRot = new THREE.Vector3( -45 / 2, -90 / 2, -60 );
		this.maxRot = new THREE.Vector3( 45 / 2, 90 / 2, 50 / 2 );

		this.name = 'Neck';

	} // Neck.constructor

} // Neck




export { Neck };
