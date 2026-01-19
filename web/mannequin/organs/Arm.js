import { LimbShape } from '../shapes/LimbShape.js';
import { Joint } from "./Joint.js";


class Arm extends Joint {

	constructor( parentJoint, leftOrRight ) {

		super( parentJoint, [ 0, 14, leftOrRight * ( parentJoint.feminine ? 5 : 6 ) ], [ 3.5, 11, 2.5, -90, 360, 0.9, 0.2, 1.5 ], LimbShape );
		this.leftOrRight = leftOrRight;

		this.imageWrapper.rotation.set( 0, 0, Math.PI );

		this.name = 'Arm';

	} // Arm.constructor

	biologicallyImpossibleLevel() {

		var result = 0;

		this.image.updateWorldMatrix( true );

		var p = this.getBumper( 0, 15, -0 * 5 * this.leftOrRight );

		if ( p.z * this.leftOrRight < -3 ) result += -3 - p.z * this.leftOrRight;

		if ( p.x < -7 && p.y > 0 ) result = p.y;

		this.rotation.reorder( 'ZXY' );
		var r = this.rotation.y * 180 / Math.PI;
		var min = -90;
		var max = 90;
		//document.getElementById("name").innerHTML = (this.rotation.x*180/Math.PI).toFixed(0)+' '+(this.rotation.y*180/Math.PI).toFixed(0)+' '+(this.rotation.z*180/Math.PI).toFixed(0);
		//document.getElementById("name").innerHTML += '<br>'+(p.x).toFixed(1)+' '+(p.y).toFixed(1)+' '+(p.z).toFixed(1);

		if ( r > max ) result += r - max;
		if ( r < min ) result += min - r;
		return result;

	}

	get raise() {

		return this.z;

	}

	set raise( angle ) {

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

} // Arm




export { Arm };
