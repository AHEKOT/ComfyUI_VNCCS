import * as THREE from "three";
import { Joint } from "./Joint.js";


class Body extends Joint {

	constructor( feminine ) {

		super( null, null, [ 1, 1, 1 ], THREE.Group );

		this.feminine = feminine;

		this.minRot = new THREE.Vector3( Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY );
		this.maxRot = new THREE.Vector3( Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY );

		this.name = 'Body';

	} // Body.constructor

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

} // Body




export { Body };
