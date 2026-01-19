import * as THREE from "three";
import { PelvisShape } from '../shapes/PelvisShape.js';
import { Joint } from "./Joint.js";


class Pelvis extends Joint {

	constructor( parentJoint ) {

		super( parentJoint, null, [ 3, 4, parentJoint.feminine ? 5.5 : 5 ], PelvisShape );

		this.minRot = new THREE.Vector3( Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY );
		this.maxRot = new THREE.Vector3( Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY );

		this.name = 'Pelvis';

	} // Pelvis.constructor

	recolor( color ) {

		if ( typeof color === 'string' )
			color = new THREE.Color( color );

		this.image.material.color = color;

	} // Pelvis.recolor

} // Pelvis




export { Pelvis };
