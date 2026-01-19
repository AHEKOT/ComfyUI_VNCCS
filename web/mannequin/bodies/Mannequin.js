import * as THREE from "three";

import { GROUND_LEVEL, VERSION_POSTURE } from "../globals.js";
import { scene } from "../scene.js";
import { Body } from "../organs/Body.js";
import { Torso } from "../organs/Torso.js";
import { Neck } from "../organs/Neck.js";
import { Head } from "../organs/Head.js";
import { Leg } from "../organs/Leg.js";
import { Knee } from "../organs/Knee.js";
import { Pelvis } from "../organs/Pelvis.js";
import { Ankle } from "../organs/Ankle.js";
import { Arm } from "../organs/Arm.js";
import { Elbow } from "../organs/Elbow.js";
import { Wrist } from "../organs/Wrist.js";
import { Finger } from "../organs/Finger.js";
import { Fingers } from "../organs/Fingers.js";
import { Nails } from "../organs/Nails.js";

var box = new THREE.Box3();


class MannequinPostureVersionError extends Error {

	constructor( version ) {

		super( 'Posture data version ' + version + ' is incompatible with the currently supported version ' + VERSION_POSTURE + '.' );
		this.name = "IncompatibleMannequinError";

	}

}



class Mannequin extends THREE.Group {

	constructor( feminine, height = 1 ) {

		super();

		this.name = 'Mannequin';

		const LEFT = -1;
		const RIGHT = 1;
		this.rawHeight = 1;

		this.feminine = feminine;

		this.body = new Body( feminine );

		this.pelvis = new Pelvis( this.body );
		this.torso = new Torso( this.pelvis );
		this.neck = new Neck( this.torso );
		this.head = new Head( this.neck );

		this.l_leg = new Leg( this.pelvis, LEFT );
		this.l_knee = new Knee( this.l_leg );
		this.l_ankle = new Ankle( this.l_knee );

		this.r_leg = new Leg( this.pelvis, RIGHT );
		this.r_knee = new Knee( this.r_leg );
		this.r_ankle = new Ankle( this.r_knee );

		this.l_arm = new Arm( this.torso, LEFT );
		this.l_elbow = new Elbow( this.l_arm );
		this.l_wrist = new Wrist( this.l_elbow );
		this.l_finger_0 = new Finger( this.l_wrist, LEFT, 0 );
		this.l_finger_1 = new Finger( this.l_wrist, LEFT, 1 );
		this.l_finger_2 = new Finger( this.l_wrist, LEFT, 2 );
		this.l_finger_3 = new Finger( this.l_wrist, LEFT, 3 );
		this.l_finger_4 = new Finger( this.l_wrist, LEFT, 4 );

		this.l_fingers = new Fingers( this.l_finger_0, this.l_finger_1, this.l_finger_2, this.l_finger_3, this.l_finger_4 );
		this.l_nails = new Nails( this.l_finger_0, this.l_finger_1, this.l_finger_2, this.l_finger_3, this.l_finger_4 );

		this.r_arm = new Arm( this.torso, RIGHT );
		this.r_elbow = new Elbow( this.r_arm );
		this.r_wrist = new Wrist( this.r_elbow );
		this.r_finger_0 = new Finger( this.r_wrist, RIGHT, 0 );
		this.r_finger_1 = new Finger( this.r_wrist, RIGHT, 1 );
		this.r_finger_2 = new Finger( this.r_wrist, RIGHT, 2 );
		this.r_finger_3 = new Finger( this.r_wrist, RIGHT, 3 );
		this.r_finger_4 = new Finger( this.r_wrist, RIGHT, 4 );

		this.r_fingers = new Fingers( this.r_finger_0, this.r_finger_1, this.r_finger_2, this.r_finger_3, this.r_finger_4 );
		this.r_nails = new Nails( this.r_finger_0, this.r_finger_1, this.r_finger_2, this.r_finger_3, this.r_finger_4 );

		this.add( this.body );

		var s = 1.5 / ( 0.5 + height );
		this.head.scale.set( s, s, s );
		this.head.position.y = THREE.MathUtils.mapLinear( height, 1.8, 1.15, 4.7, 5.8 );
		this.castShadow = true;
		this.receiveShadow = true;
		scene.add( this );

		this.updateMatrix();
		this.updateWorldMatrix();

		// default general posture
		this.body.turn = -90;

		this.torso.bend = 2;

		this.head.nod = -10;

		this.l_arm.raise = -5;
		this.r_arm.raise = -5;

		this.l_arm.straddle = 7;
		this.r_arm.straddle = 7;

		this.l_elbow.bend = 15;
		this.r_elbow.bend = 15;

		this.l_wrist.bend = 5;
		this.r_wrist.bend = 5;

		this.l_finger_0.straddle = -20;
		this.r_finger_0.straddle = -20;

		this.l_finger_0.bend = -15;
		this.l_finger_1.bend = 10;
		this.l_finger_2.bend = 10;
		this.l_finger_3.bend = 10;
		this.l_finger_4.bend = 10;

		this.l_finger_0.mid.bend = 10;
		this.l_finger_1.mid.bend = 10;
		this.l_finger_2.mid.bend = 10;
		this.l_finger_3.mid.bend = 10;
		this.l_finger_4.mid.bend = 10;

		this.l_finger_0.tip.bend = 10;
		this.l_finger_1.tip.bend = 10;
		this.l_finger_2.tip.bend = 10;
		this.l_finger_3.tip.bend = 10;
		this.l_finger_4.tip.bend = 10;

		this.r_finger_0.bend = -15;
		this.r_finger_1.bend = 10;
		this.r_finger_2.bend = 10;
		this.r_finger_3.bend = 10;
		this.r_finger_4.bend = 10;

		this.r_finger_0.mid.bend = 10;
		this.r_finger_1.mid.bend = 10;
		this.r_finger_2.mid.bend = 10;
		this.r_finger_3.mid.bend = 10;
		this.r_finger_4.mid.bend = 10;

		this.r_finger_0.tip.bend = 10;
		this.r_finger_1.tip.bend = 10;
		this.r_finger_2.tip.bend = 10;
		this.r_finger_3.tip.bend = 10;
		this.r_finger_4.tip.bend = 10;

		this.updateMatrixWorld( true );
		this.stepOnGround();

		this.scale.setScalar( height/this.rawHeight );

		this.stepOnGround();

	} // Mannequin.constructor

	get bend() {

		return -this.body.z;

	}

	set bend( angle ) {

		this.body.z = -angle;

	}

	get tilt() {

		return this.body.x;

	}

	set tilt( angle ) {

		this.body.x = angle;

	}

	get turn() {

		return this.body.y;

	}

	set turn( angle ) {

		this.body.y = angle;

	}

	get posture() {

		var posture = [
			[
				Number( ( this.body.position.x /*+ this.position.x*/ ).toFixed( 1 ) ),
				Number( ( this.body.position.y /*+ this.position.y*/ ).toFixed( 1 ) ),
				Number( ( this.body.position.z /*+ this.position.z*/ ).toFixed( 1 ) ),
			],
			this.body.posture,
			this.torso.posture,
			this.head.posture,
			this.l_leg.posture,
			this.l_knee.posture,
			this.l_ankle.posture,
			this.r_leg.posture,
			this.r_knee.posture,
			this.r_ankle.posture,
			this.l_arm.posture,
			this.l_elbow.posture,
			this.l_wrist.posture,
			this.l_finger_0.posture,
			this.l_finger_1.posture,
			this.l_finger_2.posture,
			this.l_finger_3.posture,
			this.l_finger_4.posture,
			this.r_arm.posture,
			this.r_elbow.posture,
			this.r_wrist.posture,
			this.r_finger_0.posture,
			this.r_finger_1.posture,
			this.r_finger_2.posture,
			this.r_finger_3.posture,
			this.r_finger_4.posture,
		];
		return {
			version: VERSION_POSTURE,
			data: posture,
		};

	} // Mannequin.posture

	set posture( posture ) {

		if ( posture.version != VERSION_POSTURE )
			throw new MannequinPostureVersionError( posture.version );

		var i = 0;

		this.body.position.set( ...posture.data[ i++ ]);

		this.body.posture = posture.data[ i++ ];
		this.torso.posture = posture.data[ i++ ];
		this.head.posture = posture.data[ i++ ];

		this.l_leg.posture = posture.data[ i++ ];
		this.l_knee.posture = posture.data[ i++ ];
		this.l_ankle.posture = posture.data[ i++ ];

		this.r_leg.posture = posture.data[ i++ ];
		this.r_knee.posture = posture.data[ i++ ];
		this.r_ankle.posture = posture.data[ i++ ];

		this.l_arm.posture = posture.data[ i++ ];
		this.l_elbow.posture = posture.data[ i++ ];
		this.l_wrist.posture = posture.data[ i++ ];
		this.l_finger_0.posture = posture.data[ i++ ];
		this.l_finger_1.posture = posture.data[ i++ ];
		this.l_finger_2.posture = posture.data[ i++ ];
		this.l_finger_3.posture = posture.data[ i++ ];
		this.l_finger_4.posture = posture.data[ i++ ];

		this.r_arm.posture = posture.data[ i++ ];
		this.r_elbow.posture = posture.data[ i++ ];
		this.r_wrist.posture = posture.data[ i++ ];
		this.r_finger_0.posture = posture.data[ i++ ];
		this.r_finger_1.posture = posture.data[ i++ ];
		this.r_finger_2.posture = posture.data[ i++ ];
		this.r_finger_3.posture = posture.data[ i++ ];
		this.r_finger_4.posture = posture.data[ i++ ];

	} // Mannequin.posture

	get postureString() {

		return JSON.stringify( this.posture );

	}

	set postureString( string ) {

		this.posture = JSON.parse( string );

	}


	stepOnGround( ) {

		this.position.y = 0;
		this.updateMatrixWorld( true );
		box.setFromObject( this, true );
		this.position.y = -box.min.y + GROUND_LEVEL - 0.01;
		this.rawHeight = box.max.y-box.min.y;
		this.updateMatrixWorld( true );

	}


	recolor( head, shoes, pelvis, joints, limbs, torso, nails ) {

		if ( head ) {

			this.head.recolor( head );

		}

		if ( shoes ) {

			this.l_ankle.recolor( shoes );
			this.r_ankle.recolor( shoes );

		}

		if ( pelvis ) {

			this.pelvis.recolor( pelvis );

		}

		if ( torso ) {

			this.torso.recolor( torso, joints );

		}

		if ( nails ) {

			this.l_nails.recolor( nails );

		}

		if ( limbs ) {

			this.l_leg.recolor( limbs, joints );
			this.l_knee.recolor( limbs, joints );
			this.l_fingers.recolor( limbs, joints );
			this.l_arm.recolor( limbs, joints );
			this.l_elbow.recolor( limbs, joints );
			this.l_wrist.recolor( limbs, joints );

			this.r_leg.recolor( limbs, joints );
			this.r_knee.recolor( limbs, joints );
			this.r_fingers.recolor( limbs, joints );
			this.r_arm.recolor( limbs, joints );
			this.r_elbow.recolor( limbs, joints );
			this.r_wrist.recolor( limbs, joints );

		}

	}

} // Mannequin




export { Mannequin, MannequinPostureVersionError };
