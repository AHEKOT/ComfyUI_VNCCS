import * as THREE from "three";
import { BODY_COLORS } from '../globals.js';
import { LimbShape } from '../shapes/LimbShape.js';
import { Joint } from "./Joint.js";



var NAIL_GEOMETRY = new THREE.IcosahedronGeometry( 1, 2 );



class Phalange extends Joint {

	constructor( parentJoint, params, nailSize ) {

		super( parentJoint, null, params, LimbShape );

		this.minRot = new THREE.Vector3( 0, 0, -10 );
		this.maxRot = new THREE.Vector3( 0, 0, 100 );

		if ( nailSize > 0 ) {

			this.nail = new THREE.Mesh( NAIL_GEOMETRY,
				new THREE.MeshStandardMaterial(
					{
						color: BODY_COLORS.NAILS,
					} ) );
			this.nail.name = 'nail';
			this.nail.castShadow = true;
			this.nail.receiveShadow = true;
			this.nail.scale.set( 0.05, 0.2*nailSize, 0.1*nailSize );
			this.nail.position.set( params[ 0 ]/4.5, params[ 1 ]*0.75, 0 );
			this.nail.rotation.set( 0, 0, 0.1 );
			this.nail.recolor = function ( color ) {

				if ( typeof color === 'string' )
					color = new THREE.Color( color );

				this.parent.nail.material.color = color;

			};

			this.add( this.nail );

		}

		this.name = 'Phalange';

	} // Phalange.constructor

	get bend() {

		return this.z;

	}

	set bend( angle ) {

		this.z = angle;

	}

	hide( recursive ) {

		super.hide( recursive );
		if ( this.nail ) this.nail.visible = false;

	}

	show( recursive ) {

		super.show( recursive );
		if ( this.nail ) this.show.visible = false;

	}

} // Phalange




export { Phalange };
