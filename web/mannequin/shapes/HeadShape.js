import * as THREE from "three";

import { BODY_COLORS, cos, cossers, sin } from '../globals.js';
import { ParametricShape } from './ParametricShape.js';


// head texture
var texHead = new THREE.TextureLoader().load( "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABABAMAAABYR2ztAAAAGFBMVEX////Ly8v5+fne3t5GRkby8vK4uLi/v7/GbmKXAAAAZklEQVRIx2MYQUAQHQgQVkBtwEjICkbK3MAkQFABpj+R5ZkJKTAxImCFSSkhBamYVgiQrAADEHQkIW+iqiBCAfXjAkMHpgKqgyHgBiwBRfu4ECScYEZGvkD1JxEKhkA5OVTqi8EOAOyFJCGMDsu4AAAAAElFTkSuQmCC" );



// head shape as parametric surface
class HeadShape extends ParametricShape {

	constructor( feminine, params ) {

		super( texHead, BODY_COLORS.HEAD, function ( u, v, target ) {

			var r = cossers( u, v, [
				[ 0.4, 0.9, 0, 1, -3 ],
				[ 0, 1, 0, 0.1, 3 ],
				[ 0, 1, 0.9, 1, 3 ],
				[ 1.00, 1.05, 0.55, 0.85, -3 ],
				[ 1.00, 1.05, 0.15, 0.45, -3 ],
				[ 0.93, 1.08, 0.40, 0.60, 8 ],
				[ 0.0, 0.7, 0.05, 0.95, 3 ],
				[ -0.2, 0.2, -0.15, 1.15, -6 ],
				[ -0.07, 0.07, 0.45, 0.55, 20 ], // nose
				[ -0.07, 0.01, 0.35, 0.55, 10 ], // nostril
				[ -0.07, 0.01, 0.45, 0.65, 10 ], // nostril
			]);
			u = 360 * u;
			v = 180 * v - 90;
			var k = ( 1 + ( feminine ? 1 : 2 ) * sin( u ) * cos( v ) ) / 4;
			target.set(
				r * params.sx * cos( u ) * cos( v ),
				r * params.sy * sin( u ) * cos( v ),
				( r + k ) * params.sz * sin( v ) );

		}, 32, 32 );

		this.name = 'HeadShape';

	} // HeadShape.constructor

} // HeadShape


export { HeadShape };
