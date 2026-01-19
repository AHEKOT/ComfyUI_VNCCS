import { Mannequin } from './Mannequin.js';


class Child extends Mannequin {

	constructor( height = 1.15 ) {

		super( false, height );

		this.name = 'Child';

		this.l_arm.straddle -= 2;
		this.r_arm.straddle -= 2;

		this.stepOnGround( );

	} // Child.constructor

} // Child




export { Child };
