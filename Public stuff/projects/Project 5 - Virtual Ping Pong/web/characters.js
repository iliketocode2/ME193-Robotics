// The three opponents, built from simple primitives with toon shading
// (Wii-style: chunky shapes, flat colors, black outlines). Gameplay numbers
// for each live in opponents.py under the same key.
import * as THREE from "three";

const gradient = (() => {
  const data = new Uint8Array([90, 170, 255]);           // 3-step toon ramp
  const tex = new THREE.DataTexture(data, 3, 1, THREE.RedFormat);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  return tex;
})();

export function toon(color, extra = {}) {
  return new THREE.MeshToonMaterial({ color, gradientMap: gradient, ...extra });
}

const outlineMat = new THREE.MeshBasicMaterial({ color: 0x111122, side: THREE.BackSide });

// Mesh + slightly larger black back-face copy = cartoon outline.
export function part(geo, mat, { outline = 0.04, shadow = true } = {}) {
  const m = new THREE.Mesh(geo, mat);
  m.castShadow = shadow;
  if (outline > 0) {
    const o = new THREE.Mesh(geo, outlineMat);
    o.scale.setScalar(1 + outline);
    m.add(o);
  }
  return m;
}

export function makePaddle(rubber = 0xd62828) {
  const g = new THREE.Group();
  const head = part(new THREE.CylinderGeometry(0.085, 0.085, 0.014, 28), toon(rubber), { outline: 0.06 });
  head.rotation.x = Math.PI / 2;
  const handle = part(new THREE.BoxGeometry(0.03, 0.1, 0.022), toon(0xc8955a), { outline: 0.08 });
  handle.position.y = -0.12;
  g.add(head, handle);
  return g;
}

// Every character: { group, armPivot (holds the paddle), head, update(dt, t, mode) }
// mode: "idle" | "ready" | "celebrate" | "sulk"
function rig(group, armPivot, head, opts = {}) {
  const c = { group, armPivot, head, swingT: 1, mode: "idle", baseY: 0, ...opts };
  c.swing = () => { c.swingT = 0; };
  c.update = (dt, t) => {
    // swing: quick forward arc of the paddle arm
    c.swingT = Math.min(1, c.swingT + dt / 0.35);
    const s = Math.sin(c.swingT * Math.PI);
    armPivot.rotation.x = -0.4 - s * 1.6;
    armPivot.rotation.z = (opts.armZ ?? -0.5) + s * 0.6;
    let y = 0, tilt = 0, headTilt = 0;
    if (c.mode === "idle") y = Math.abs(Math.sin(t * 2.2)) * 0.03;
    if (c.mode === "ready") y = Math.abs(Math.sin(t * 6)) * 0.02;
    if (c.mode === "celebrate") { y = Math.abs(Math.sin(t * 7)) * 0.25; armPivot.rotation.x = -2.8; }
    if (c.mode === "sulk") { headTilt = 0.45; tilt = 0.12; }
    group.position.y = c.baseY + y;
    group.rotation.x = tilt;
    head.rotation.x = headTilt + (opts.headBob ? Math.sin(t * 2) * 0.05 : 0);
    opts.extra?.(dt, t, c);
  };
  return c;
}

function eye(r = 0.035, pupil = 0.018) {
  const g = new THREE.Group();
  const w = part(new THREE.SphereGeometry(r, 16, 12), toon(0xffffff), { outline: 0.1 });
  const p = new THREE.Mesh(new THREE.SphereGeometry(pupil, 12, 10), new THREE.MeshBasicMaterial({ color: 0x111111 }));
  p.position.z = r * 0.75;
  g.add(w, p);
  return g;
}

// ---------------------------------------------------------------- Pip (easy)
function makePip() {
  const g = new THREE.Group();
  const body = part(new THREE.SphereGeometry(0.42, 32, 24), toon(0x22263a));
  body.scale.set(1, 1.25, 0.95);
  body.position.y = 0.6;
  const belly = part(new THREE.SphereGeometry(0.34, 28, 20), toon(0xf4f1ea), { outline: 0 });
  belly.scale.set(0.95, 1.15, 0.7);
  belly.position.set(0, 0.52, 0.17);
  const head = new THREE.Group();
  head.position.y = 1.0;
  const eL = eye(0.06, 0.03), eR = eye(0.06, 0.03);
  eL.position.set(-0.13, 0.08, 0.3); eR.position.set(0.13, 0.08, 0.3);
  const beak = part(new THREE.ConeGeometry(0.07, 0.18, 16), toon(0xff9f1c));
  beak.rotation.x = Math.PI / 2; beak.position.set(0, -0.04, 0.4);
  const blushL = new THREE.Mesh(new THREE.CircleGeometry(0.04, 16), new THREE.MeshBasicMaterial({ color: 0xff7b9c }));
  blushL.position.set(-0.22, -0.04, 0.3); blushL.rotation.y = -0.5;
  const blushR = blushL.clone(); blushR.position.x = 0.22; blushR.rotation.y = 0.5;
  head.add(eL, eR, beak, blushL, blushR);
  const footMat = toon(0xff9f1c);
  const fL = part(new THREE.SphereGeometry(0.1, 16, 10), footMat); fL.scale.set(1, 0.35, 1.5); fL.position.set(-0.15, 0.04, 0.12);
  const fR = fL.clone(); fR.position.x = 0.15;
  // flippers
  const flipGeo = new THREE.SphereGeometry(0.1, 16, 12);
  const leftFlip = part(flipGeo, toon(0x22263a)); leftFlip.scale.set(0.4, 1.4, 0.8); leftFlip.position.set(-0.45, 0.65, 0); leftFlip.rotation.z = -0.4;
  const armPivot = new THREE.Group(); armPivot.position.set(0.4, 0.8, 0);
  const rightFlip = part(flipGeo, toon(0x22263a)); rightFlip.scale.set(0.4, 1.4, 0.8); rightFlip.position.y = -0.13;
  const paddle = makePaddle(0x2a9d8f); paddle.position.set(0, -0.32, 0.05);
  armPivot.add(rightFlip, paddle);
  g.add(body, belly, head, fL, fR, leftFlip, armPivot);
  return rig(g, armPivot, head, {
    armZ: -0.2, headBob: true,
    extra: (dt, t) => { g.rotation.z = Math.sin(t * 3) * 0.06; },   // waddle
  });
}

// --------------------------------------------------------------- Rita (medium)
function makeRita() {
  const g = new THREE.Group();
  const skin = toon(0xf2c49b), suit = toon(0x1d9a8a), stripe = toon(0xffffff), shoe = toon(0xffffff);
  const legGeo = new THREE.CapsuleGeometry(0.07, 0.55, 6, 12);
  const lL = part(legGeo, toon(0x184e77)); lL.position.set(-0.1, 0.38, 0);
  const lR = lL.clone(); lR.position.x = 0.1;
  const sL = part(new THREE.BoxGeometry(0.12, 0.07, 0.22), shoe); sL.position.set(-0.1, 0.04, 0.04);
  const sR = sL.clone(); sR.position.x = 0.1;
  const torso = part(new THREE.CapsuleGeometry(0.19, 0.4, 8, 16), suit); torso.position.y = 1.02;
  torso.scale.z = 0.75;
  const zip = new THREE.Mesh(new THREE.BoxGeometry(0.015, 0.45, 0.01), stripe.clone());
  zip.position.set(0, 1.04, 0.145);
  const whistle = part(new THREE.CylinderGeometry(0.018, 0.018, 0.05, 10), toon(0xffd166), { outline: 0.15 });
  whistle.rotation.z = Math.PI / 2; whistle.position.set(0.05, 1.18, 0.16);
  const head = new THREE.Group(); head.position.y = 1.5;
  const skull = part(new THREE.SphereGeometry(0.16, 28, 20), skin); skull.scale.set(1, 1.08, 1);
  const hair = part(new THREE.SphereGeometry(0.17, 28, 20, 0, Math.PI * 2, 0, Math.PI * 0.55), toon(0x5a3825));
  hair.position.y = 0.02; hair.rotation.x = -0.25;
  const pony = part(new THREE.SphereGeometry(0.07, 16, 12), toon(0x5a3825)); pony.scale.set(0.8, 1.6, 0.8); pony.position.set(0, -0.02, -0.19);
  const band = part(new THREE.TorusGeometry(0.163, 0.02, 8, 32), toon(0xe63946), { outline: 0.1 });
  band.rotation.x = Math.PI / 2 - 0.25; band.position.y = 0.06;
  const eL = eye(0.025, 0.015), eR = eye(0.025, 0.015);
  eL.position.set(-0.055, 0.0, 0.14); eR.position.set(0.055, 0.0, 0.14);
  const brow = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.01, 0.01), new THREE.MeshBasicMaterial({ color: 0x3b2416 }));
  const bL = brow.clone(); bL.position.set(-0.055, 0.045, 0.15); bL.rotation.z = -0.25;
  const bR = brow.clone(); bR.position.set(0.055, 0.045, 0.15); bR.rotation.z = 0.25;   // determined brows
  const mouth = new THREE.Mesh(new THREE.TorusGeometry(0.035, 0.007, 6, 16, Math.PI), new THREE.MeshBasicMaterial({ color: 0x7a2e2e }));
  mouth.rotation.z = Math.PI; mouth.position.set(0, -0.06, 0.15);
  head.add(skull, hair, pony, band, eL, eR, bL, bR, mouth);
  const armGeo = new THREE.CapsuleGeometry(0.055, 0.42, 6, 12);
  const aL = part(armGeo, suit); aL.position.set(-0.27, 1.05, 0.02); aL.rotation.z = 0.25;
  const armPivot = new THREE.Group(); armPivot.position.set(0.25, 1.25, 0);
  const aR = part(armGeo, suit); aR.position.y = -0.24;
  const hand = part(new THREE.SphereGeometry(0.05, 12, 10), skin); hand.position.y = -0.5;
  const paddle = makePaddle(0xe63946); paddle.position.set(0, -0.6, 0.04);
  armPivot.add(aR, hand, paddle);
  g.add(lL, lR, sL, sR, torso, zip, whistle, head, aL, armPivot);
  return rig(g, armPivot, head, { armZ: -0.35 });
}

// ------------------------------------------------------------- Viktor (hard)
function makeViktor() {
  const g = new THREE.Group();
  const metal = toon(0x3d4451), dark = toon(0x1c2028), red = toon(0xd00000);
  const glow = new THREE.MeshBasicMaterial({ color: 0xff2a2a });
  const legGeo = new THREE.BoxGeometry(0.14, 0.7, 0.16);
  const lL = part(legGeo, dark); lL.position.set(-0.13, 0.4, 0);
  const lR = lL.clone(); lR.position.x = 0.13;
  const torso = part(new THREE.BoxGeometry(0.56, 0.62, 0.32), metal); torso.position.y = 1.12;
  const chest = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.05, 0.01), glow); chest.position.set(0, 1.22, 0.165);
  const core = new THREE.Mesh(new THREE.CircleGeometry(0.05, 20), glow); core.position.set(0, 1.05, 0.165);
  const head = new THREE.Group(); head.position.y = 1.62;
  const skull = part(new THREE.BoxGeometry(0.32, 0.3, 0.3), metal);
  const visor = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.07, 0.02), glow); visor.position.set(0, 0.02, 0.155);
  const jaw = part(new THREE.BoxGeometry(0.26, 0.06, 0.26), dark); jaw.position.y = -0.15;
  const antenna = part(new THREE.CylinderGeometry(0.01, 0.01, 0.18, 8), dark); antenna.position.set(0.1, 0.24, 0);
  const tip = new THREE.Mesh(new THREE.SphereGeometry(0.025, 10, 8), glow); tip.position.set(0.1, 0.34, 0);
  head.add(skull, visor, jaw, antenna, tip);
  const shoulderGeo = new THREE.SphereGeometry(0.1, 16, 12);
  const shL = part(shoulderGeo, red); shL.position.set(-0.36, 1.36, 0);
  const shR = part(shoulderGeo, red); shR.position.set(0.36, 1.36, 0);
  const armGeo = new THREE.BoxGeometry(0.1, 0.5, 0.1);
  const aL = part(armGeo, dark); aL.position.set(-0.4, 1.08, 0); aL.rotation.z = 0.15;
  const armPivot = new THREE.Group(); armPivot.position.set(0.38, 1.34, 0);
  const aR = part(armGeo, dark); aR.position.y = -0.28;
  const paddle = makePaddle(0x111111); paddle.position.set(0, -0.62, 0.04);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.086, 0.006, 6, 28), glow);
  paddle.children[0].add(rim); rim.rotation.x = Math.PI / 2;
  armPivot.add(aR, paddle);
  g.add(lL, lR, torso, chest, core, head, shL, shR, aL, armPivot);
  return rig(g, armPivot, head, {
    armZ: -0.3,
    extra: (dt, t) => {
      const k = 0.7 + 0.3 * Math.sin(t * 4);
      glow.color.setRGB(k, 0.12 * k, 0.12 * k);
    },
  });
}

export function makeCharacter(key) {
  if (key === "pip") return makePip();
  if (key === "rita") return makeRita();
  return makeViktor();
}

// Head height (m above the character's feet) for speech bubbles
export const HEAD_HEIGHT = { pip: 1.25, rita: 1.75, viktor: 1.95 };
