import * as THREE from "three";
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';
import { grad, rad } from "../globals.js";
import { font } from "../font.js";

var localScene = new THREE.Scene();

var fontStyle = new THREE.MeshPhongMaterial( { color: 'crimson' } ),
	fontParams = {
		font: font,
		size: 2.5,
		depth: 0.2,
		curveSegments: 8,
		bevelEnabled: false
	};


// flexible joint
class Joint extends THREE.Group {

	constructor( parentJoint, pos, params, shape ) {

		super();
		var yVal = params.sy || params[ 1 ];

		if ( shape ) {

			this.image = new shape( parentJoint ? parentJoint.feminine : false, params );
			if ( shape == THREE.Group ) this.image.name = 'Joint.image';

		} else {

			this.image = new THREE.Group();
			this.image.name = 'Joint.imageWrapper.image';

		}

		this.image.castShadow = true;
 		//if ( shape != PelvisShape && shape != ShoeShape ) {
		//
		//	this.image.position.set( 0, yVal / 2, 0 );
		//
		//}


		this.imageWrapper = new THREE.Group();
		this.imageWrapper.add( this.image );
		this.imageWrapper.castShadow = true;
		this.imageWrapper.name = 'Joint.imageWrapper';

		this.add( this.imageWrapper );

		this.castShadow = true;
		this.yVal = yVal;
		this.parentJoint = parentJoint;

		if ( parentJoint ) { // attaching to parent joint

			this.position.set( 0, shape?parentJoint.yVal:parentJoint.yVal/4, 0 );
			parentJoint.imageWrapper.add( this );
			this.feminine = parentJoint.feminine;

		}

		if ( pos ) { // initial joint position

			this.position.set( pos[ 0 ], pos[ 1 ], pos[ 2 ]);

		}

		this.minRot = new THREE.Vector3();
		this.maxRot = new THREE.Vector3();

		this.name = 'Joint';

	} // Joint.constructor

	get z() {

		this.rotation.reorder( 'YXZ' );
		return this.rotation.z * 180 / Math.PI;

	}

	set z( angle ) {

		this.rotation.reorder( 'YXZ' );
		this.rotation.z = angle * Math.PI / 180;

	} // Joint.z

	get x() {

		this.rotation.reorder( 'YZX' );
		return this.rotation.x * 180 / Math.PI;

	}

	set x( angle ) {

		this.rotation.reorder( 'YZX' );
		this.rotation.x = angle * Math.PI / 180;

	} // Joint.x

	get y() {

		this.rotation.reorder( 'ZXY' );
		return this.rotation.y * 180 / Math.PI;

	}

	set y( angle ) {

		this.rotation.reorder( 'ZXY' );
		this.rotation.y = angle * Math.PI / 180;

	} // Joint.y

	reset() {

		this.rotation.set( 0, 0, 0 );

	}

	get posture() {

		this.rotation.reorder( 'XYZ' );
		return [ grad( this.rotation.x ), grad( this.rotation.y ), grad( this.rotation.z ) ];

	}

	set posture( pos ) {

		this.rotation.set( rad( pos[ 0 ]), rad( pos[ 1 ]), rad( pos[ 2 ]), 'XYZ' );

	} // Joint.posture

	getBumper( x, y, z ) {

		var bumper = new THREE.Vector3( x, y, z );
		this.image.localToWorld( bumper );
		this.parentJoint.image.worldToLocal( bumper );
		return bumper;

	}

	hide( recursive ) {

		this.image.visible = false;

		if ( recursive )
			this.traverse( ( joint )=> {

				if ( joint.image )
					joint.hide();

			} );

	} // Joint.hide

	show( recursive ) {

		this.image.visible = true;

		if ( recursive )
			this.traverse( ( joint )=> {

				if ( joint.image )
					joint.show();

			} );

	} // Joint.show

	// attach Object3D instance to the joint
	attach( image ) {

		this.imageWrapper.add( image );

	} // Joint.attach

	detach( image ) {

		if ( this.imageWrapper.children.includes(
			this.imageWrapper.getObjectById( image.id ) ) ) {

			this.imageWrapper.remove(
				this.imageWrapper.getObjectById( image.id )
			);

		}

	} // Joint.detach

	// calculate global coordinates of point with coordinates relative to the joint
	point( x, y, z ) {

		return ( window.scene??localScene ).worldToLocal( this.localToWorld( new THREE.Vector3( x, y, z ) ) );

	} // Joint.point

	// change the colour of the joint
	recolor( color, secondaryColor = color ) {

		var joint = this.image;

		if ( typeof color === 'string' )
			color = new THREE.Color( color );

		if ( typeof secondaryColor === 'string' )
			secondaryColor = new THREE.Color( secondaryColor );

		if ( joint.material instanceof Array ) {

			joint.material[ 0 ].color = color;
			joint.material[ 1 ].color = secondaryColor;

		} else {

			joint.material.color = color;

			if ( joint.children.length > 0 ) {

				joint.children[ 0 ].material.color = secondaryColor;

			}

		}

	} // Joint.recolor

	select( state ) {

		this.traverse( function ( o ) {

			if ( ( o.material instanceof Array ) ) {

				if ( o.material[ 0 ] && o.material[ 0 ].emissive ) o.material[ 0 ].emissive.setRGB( 0, state ? -1 : 0, state ? -0.4 : 0 );
				if ( o.material[ 1 ] && o.material[ 1 ].emissive ) o.material[ 1 ].emissive.setRGB( 0, state ? -1 : 0, state ? -0.4 : 0 );

			} else
				if ( o.material && o.material.emissive ) o.material.emissive.setRGB( 0, state ? -1 : 0, state ? -0.4 : 0 );

		} );

	} // Joint.select


	label( text, y, z=0.5, x=0, rotY=1, rotZ=0, rotX=0 ) {

		var geometry = new TextGeometry( text, fontParams ),
			label = new THREE.Mesh( geometry, fontStyle );

		geometry.computeBoundingBox();
		var box = geometry.boundingBox;

		label.position.set( x, y, z*( box.max.x-box.min.x ) );
		label.rotation.z = Math.PI/2*rotZ;
		label.rotation.y = Math.PI/2*rotY;
		label.rotation.x = Math.PI/2*rotX;
		label.castShadow = true;

		this.attach( label );

		return label;

	} // Joint.label


} // Joint



export { Joint };
