import { BODY_COLORS, cos, cossers, sin } from '../globals.js';
import { ParametricShape } from './ParametricShape.js';


// pelvis shape as parametric surface
class PelvisShape extends ParametricShape {

	constructor( feminine, params ) {

		super( null, BODY_COLORS.PELVIS, function ( u, v, target ) {

			var r = cossers( u, v, [
				[ 0.6, 0.95, 0, 1, 4 ],
				[ 0.7, 1.0, 0.475, 0.525, -13 ],
				[ -0.2, 0.3, 0, 0.3, -4 ],
				[ -0.2, 0.3, -0.3, 0, -4 ]
			]);
			u = 360 * u - 90;
			v = 180 * v - 90;
			target.set( -1.5 + r * params[ 0 ] * cos( u ) * Math.pow( cos( v ), 0.6 ),
				r * params[ 1 ] * sin( u ) * Math.pow( cos( v ), 0.6 ),
				r * params[ 2 ] * sin( v ) );

		}, 20, 10 );

		this.name = 'PelvisShape';

	} // PelvisShape.constructor

} // PelvisShape


export { PelvisShape };
