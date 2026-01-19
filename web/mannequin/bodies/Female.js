import { Mannequin } from './Mannequin.js';


class Female extends Mannequin {

	constructor( height = 1.65 ) {

		super( true, height );

		this.name = 'Female';

		this.l_leg.straddle -= 4;
		this.r_leg.straddle -= 4;

		this.l_ankle.tilt -= 4;
		this.r_ankle.tilt -= 4;

		this.stepOnGround( );

	} // Female.constructor

} // Female




export { Female };
