import { LimbShape } from '../shapes/LimbShape.js';
import { Joint } from "./Joint.js";


class Leg extends Joint {

	constructor( parentJoint, leftOrRight ) {

		super( parentJoint, [ -1, -3, 4 * leftOrRight ], [ 4, 15, 4, -70, 220, 1, 0.4, 2 ], LimbShape );
		this.leftOrRight = leftOrRight;

		this.imageWrapper.rotation.set( Math.PI, 0, 0 );

		this.name = 'Leg';

	} // Leg.constructor

	biologicallyImpossibleLevel() {

		// return 0 if the rotation is possible
		// return >0 if it is not possible, the higher the result, the more impossible it is

		var result = 0;

		this.image.updateWorldMatrix( true );

		var p = this.getBumper( 5, 0, 0 );
		if ( p.x < 0 ) result += -p.x;

		this.rotation.reorder( 'ZXY' );
		var y = this.y;
		if ( y > +60 ) result += y - 60;
		if ( y < -60 ) result += -60 - y;

		return result;

	} // Leg.biologicallyImpossibleLevel

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

} // Leg




export { Leg };
