import { BODY_COLORS, cos, cossers, sin } from '../globals.js';
import { ParametricShape } from './ParametricShape.js';


// shoe shape as parametric surface
class ShoeShape extends ParametricShape {

	constructor( feminine, params ) {

		super( null, BODY_COLORS.SHOES, function ( u, v, target ) {

			var r = cossers( u, v, [
				[ 0.6, 1.1, 0.05, 0.95, 1 ],
				[ 0.6, 0.68, 0.35, 0.65, feminine ? 1.2 : 1000 ]
			]);
			u = 360 * u;
			v = 180 * v - 90;
			target.set(
				( 3 * r - 2 ) * params.sx * ( cos( u ) * cos( v ) + ( feminine ? ( Math.pow( sin( u + 180 ), 2 ) * cos( v ) - 1 ) : 0 ) ) - ( feminine ? 0 : 2 ),
				params.sy * sin( u ) * cos( v ) + 2,
				params.sz * sin( v ) );

		}, 24, 12 );

		if ( feminine ) {

			this.add( new ParametricShape( null, BODY_COLORS.LIMBS, function ( u, v, target ) {

				var r = cossers( u, v, [
					[ 0.6, 1.1, 0.05, 0.95, 1 / 2 ]
				]);
				u = 360 * u;
				v = 180 * v - 90;
				target.set(
					0.3 * ( 3 * r - 2 ) * params.sx * ( cos( u ) * cos( v ) ),
					0.8 * params.sy * sin( u ) * cos( v ) + 2,
					0.6 * params.sz * sin( v ) );

			}, 12, 12 ) );

			this.geometry.rotateZ( 0.4 );
			this.children[ 0 ].rotation.set( 0, 0, 0.4 );

		} // if (feminine)

		this.rotation.z = -Math.PI / 2;
		this.name = 'ShoeShape';

	} // ShoeShape.constructor

} // ShoeShape


export { ShoeShape };
