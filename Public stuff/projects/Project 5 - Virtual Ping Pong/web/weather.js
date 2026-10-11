// The arena's sky and weather, matching the real weather at Tufts (Python's
// weather.py sends scene parameters, 0..1, from the UNO Q's reading).
//
// (GLSL: smoothstep's edges must be ascending -- reversed is undefined and
// returns 0 on some GPUs/ANGLE. Use 1.0 - smoothstep(lo, hi, x) instead.)
//
// PERFORMANCE RULES (the camera, the AI and this page share one laptop chip):
//  * Everything is built ONCE here and pre-compiled (renderer.compile). Weather
//    changes only touch uniforms, colours, counts and visibility -- never new
//    lights, never fog type or shader #defines (each would recompile materials
//    and stall the GPU mid-rally).
//  * No per-frame CPU loops over particles: rain/snow is ONE instanced draw
//    animated entirely in the vertex shader. Per frame the CPU lerps ~25 numbers.
//  * Effects scale with the page's quality governor (setFx): weather is cut
//    before shadows are.
import * as THREE from "three";

const TAU = 6;                          // seconds: weather eases in, never pops
const PRECIP_MAX = { high: 6000, medium: 3000, low: 1200 };
const DEFAULTS = {                      // the original sunny afternoon, until real weather arrives
  cloud: 0.15, cloud_dark: 0, precip: 0, snow_mix: 0, hail: 0, lightning: 0, fog: 0, wind: 0.15, gust: 0.1,
  wind_x: 0.6, wind_z: -0.8, daylight: 1, sun_el: 50, stars: 0, floods: 0, wet: 0, snow_cover: 0,
  fog_near: 14, fog_far: 34, rain_alpha: 0.2, key_min: 0.6, moon_phase: 0.5,
};
const LERPED = Object.keys(DEFAULTS).filter((k) => k !== "moon_phase");

export function createWeather({ scene, camera, renderer, hemi, sun, floor, court, ball, audio }) {
  const cur = { ...DEFAULTS, sun_dir: new THREE.Vector3(0.43, 0.76, 0.48) };
  let target = { ...DEFAULTS, sun_dir: cur.sun_dir.clone() };
  let first = true;
  let fx = "high";
  let flash = 0, nextStrike = 4, strikeLeft = 0;

  // ------------------------------------------------------------------ sky dome
  const noiseTex = makeNoiseTexture(128);
  const skyU = {
    uZenith: { value: new THREE.Color() }, uHorizon: { value: new THREE.Color() }, uGround: { value: new THREE.Color() },
    uSunDir: { value: new THREE.Vector3(0, 1, 0) }, uSunColor: { value: new THREE.Color(1, 0.95, 0.8) }, uSunVis: { value: 1 },
    uMoonDir: { value: new THREE.Vector3(-0.4, 0.5, -0.75).normalize() }, uMoonVis: { value: 0 }, uMoonPhase: { value: 0.5 },
    uStars: { value: 0 }, uCloud: { value: 0.15 }, uCloudDark: { value: 0 }, uCloudLight: { value: 1 },
    uCloudOffset: { value: new THREE.Vector2() }, uFlash: { value: 0 }, uHaze: { value: 0 }, uTime: { value: 0 },
    uNoise: { value: noiseTex },
  };
  const sky = new THREE.Mesh(new THREE.SphereGeometry(80, 32, 16), new THREE.ShaderMaterial({
    uniforms: skyU, side: THREE.BackSide, depthWrite: false, fog: false,
    vertexShader: `varying vec3 vDir;
      void main() {
        vDir = position;
        vec4 p = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_Position = p.xyww;                                  // always at the far plane
      }`,
    fragmentShader: `
      uniform vec3 uZenith, uHorizon, uGround, uSunDir, uSunColor, uMoonDir;
      uniform float uSunVis, uMoonVis, uMoonPhase, uStars, uCloud, uCloudDark, uCloudLight, uFlash, uHaze, uTime;
      uniform vec2 uCloudOffset;
      uniform sampler2D uNoise;
      varying vec3 vDir;
      float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
      void main() {
        vec3 d = normalize(vDir);
        float h = d.y;
        vec3 col = h > 0.0 ? mix(uHorizon, uZenith, smoothstep(0.0, 0.55, h))
                           : mix(uHorizon, uGround, smoothstep(0.0, 0.25, -h));
        // stars (clear nights) -- one cell hash per pixel, skipped entirely by day
        if (uStars > 0.01 && h > 0.02) {
          vec2 g = d.xz / (h + 0.25) * 90.0;
          vec2 cell = floor(g);
          float s = step(0.994, hash(cell));
          float tw = 0.6 + 0.4 * sin(uTime * 2.0 + hash(cell + 7.0) * 30.0);
          float r = length(fract(g) - 0.5);
          col += vec3(s * tw * (1.0 - smoothstep(0.0, 0.35, r)) * uStars * smoothstep(0.02, 0.2, h));
        }
        // sun and moon
        float ds = max(dot(d, uSunDir), 0.0);
        col += uSunColor * (smoothstep(0.9993, 0.9997, ds) * 3.0 + pow(ds, 90.0) * 0.6 + pow(ds, 8.0) * 0.12) * uSunVis;
        float dm = dot(d, uMoonDir);
        float lit = 0.25 + 0.75 * (1.0 - abs(uMoonPhase - 0.5) * 2.0);
        col += vec3(0.85, 0.88, 0.95) * (smoothstep(0.9990, 0.9994, dm) * lit + pow(max(dm, 0.0), 40.0) * 0.08) * uMoonVis;
        // clouds: two scrolling noise layers projected on a dome
        if (h > 0.0 && uCloud > 0.01) {
          vec2 uv = d.xz / (h + 0.12) * 0.22 + uCloudOffset;
          float n = texture2D(uNoise, uv).r * 0.62 + texture2D(uNoise, uv * 2.7 + 0.37).r * 0.38;
          float edge = 1.0 - uCloud;
          float c = smoothstep(edge - 0.12, edge + 0.18, n) * smoothstep(0.0, 0.14, h) * min(1.0, uCloud * 1.6);
          vec3 cloudCol = mix(vec3(0.9), vec3(0.07, 0.08, 0.1), uCloudDark) * uCloudLight;   // (linear: storm clouds are dark)
          cloudCol += uSunColor * pow(ds, 6.0) * 0.25 * uSunVis;          // silver lining
          col = mix(col, cloudCol, c * 0.96);
          col += vec3(0.75, 0.8, 1.0) * uFlash * (0.35 + 0.9 * c);
        } else {
          col += vec3(0.75, 0.8, 1.0) * uFlash * 0.3;
        }
        col = mix(col, uHorizon, uHaze * (1.0 - smoothstep(0.0, 0.35, abs(h))));   // fog/mist near the horizon
        gl_FragColor = vec4(col, 1.0);
        #include <colorspace_fragment>
      }`,
  }));
  sky.renderOrder = 1;                  // after the opaque scene: hidden sky pixels are skipped by the depth test
  sky.frustumCulled = false;
  scene.add(sky);
  scene.background = null;

  // ------------------------------------------------------------- precipitation
  const BOX = new THREE.Vector3(18, 9, 18);
  const pGeo = new THREE.InstancedBufferGeometry();
  pGeo.index = new THREE.PlaneGeometry(1, 1).index;
  pGeo.setAttribute("position", new THREE.PlaneGeometry(1, 1).getAttribute("position"));
  pGeo.setAttribute("uv", new THREE.PlaneGeometry(1, 1).getAttribute("uv"));
  const seeds = new Float32Array(PRECIP_MAX.high * 4);
  for (let i = 0; i < seeds.length; i++) seeds[i] = Math.random();
  pGeo.setAttribute("aSeed", new THREE.InstancedBufferAttribute(seeds, 4));
  pGeo.instanceCount = 0;
  const pU = {
    uTime: { value: 0 }, uCenter: { value: new THREE.Vector3() }, uBox: { value: BOX },
    uFall: { value: 9 }, uFallSnow: { value: 1.1 }, uWind: { value: new THREE.Vector2() }, uSnowMix: { value: 0 },
    uRainAlpha: { value: 0.2 }, uBright: { value: 1 },
  };
  const precip = new THREE.Mesh(pGeo, new THREE.ShaderMaterial({
    uniforms: pU, transparent: true, depthWrite: false,
    side: THREE.DoubleSide,              // a falling streak points down-screen: its quad is mirrored (back-facing)
    vertexShader: `
      attribute vec4 aSeed;
      uniform float uTime, uFall, uFallSnow, uSnowMix, uRainAlpha;
      uniform vec3 uCenter, uBox;
      uniform vec2 uWind;
      varying vec2 vUv; varying float vAlpha; varying float vSnow;
      void main() {
        float snow = step(aSeed.w, uSnowMix);
        float fall = mix(uFall, uFallSnow, snow) * (0.85 + 0.3 * aSeed.z);
        float y = uBox.y - mod(aSeed.y * uBox.y + uTime * fall, uBox.y);
        vec2 drift = uWind * uTime * mix(1.0, 0.7, snow);
        float sway = snow * sin(uTime * (0.8 + aSeed.x) + aSeed.z * 40.0) * 0.35;
        vec2 xz = mod(vec2(aSeed.x, aSeed.z) * uBox.xz + drift + sway - uCenter.xz + 0.5 * uBox.xz, uBox.xz) - 0.5 * uBox.xz;
        vec3 p = vec3(xz.x + uCenter.x, y - 0.6, xz.y + uCenter.z);
        vec4 mv = viewMatrix * vec4(p, 1.0);
        vec2 vel = (viewMatrix * vec4(uWind.x, -fall, uWind.y, 0.0)).xy;
        vec2 along = normalize(vel + vec2(0.0, -1e-4));
        vec2 across = vec2(-along.y, along.x);
        float len = mix(0.28 + 0.02 * fall, 0.045, snow);
        float wid = mix(0.012, 0.045, snow);
        mv.xy += across * position.x * wid + along * position.y * len;
        gl_Position = projectionMatrix * mv;
        float dist = -mv.z;
        vAlpha = smoothstep(0.6, 2.2, dist) * (1.0 - smoothstep(10.0, 17.0, dist)) * mix(uRainAlpha, 0.9, snow);
        vUv = uv; vSnow = snow;
      }`,
    fragmentShader: `
      uniform float uBright;
      varying vec2 vUv; varying float vAlpha; varying float vSnow;
      void main() {
        vec2 q = vUv - 0.5;
        float a = vSnow > 0.5 ? (1.0 - smoothstep(0.15, 0.5, length(q)))
                              : (1.0 - abs(q.x) * 2.0) * (1.0 - smoothstep(0.2, 0.5, abs(q.y)));
        vec3 c = mix(vec3(0.72, 0.8, 0.92), vec3(1.0), vSnow) * uBright;
        gl_FragColor = vec4(c, a * vAlpha);
        #include <colorspace_fragment>
      }`,
  }));
  precip.frustumCulled = false;
  precip.renderOrder = 3;
  scene.add(precip);

  // ------------------------------------------------------------------ lightning
  const boltPts = new Float32Array(18 * 3);
  const boltGeo = new THREE.BufferGeometry();
  boltGeo.setAttribute("position", new THREE.BufferAttribute(boltPts, 3));
  const bolt = new THREE.Line(boltGeo, new THREE.LineBasicMaterial({
    color: 0xe8f0ff, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }));
  bolt.frustumCulled = false;
  scene.add(bolt);
  function newBolt() {
    let x = (Math.random() - 0.5) * 50, y = 34, z = -38 - Math.random() * 10;
    for (let i = 0; i < 18; i++) {
      boltPts.set([x, y, z], i * 3);
      x += (Math.random() - 0.5) * 3.5; y -= 2.1 + Math.random() * 0.6;
    }
    boltGeo.attributes.position.needsUpdate = true;
  }

  // ---------------------------------------------------------- floodlight towers
  const towers = new THREE.Group();
  const lampMat = new THREE.MeshBasicMaterial({ color: 0x55585e });
  const poleMat = new THREE.MeshToonMaterial({ color: 0x8d99ae });
  const glowPos = [];
  for (const [x, z] of [[-7.2, -8.2], [7.2, -8.2], [-8.6, 2.5], [8.6, 2.5]]) {
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.18, 9, 8), poleMat);
    pole.position.set(x, 4.5, z);
    const head = new THREE.Mesh(new THREE.BoxGeometry(1.8, 1.1, 0.25), lampMat);
    head.position.set(x, 9.2, z);
    head.lookAt(0, 0.8, 0);
    towers.add(pole, head);
    glowPos.push(x * 0.97, 9.2, z * 0.97);
  }
  scene.add(towers);
  const glowGeo = new THREE.BufferGeometry();
  glowGeo.setAttribute("position", new THREE.Float32BufferAttribute(glowPos, 3));
  const glow = new THREE.Points(glowGeo, new THREE.PointsMaterial({
    size: 4.5, map: radialTexture(), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    opacity: 0, color: 0xfff6e0, fog: false }));
  glow.frustumCulled = false;
  scene.add(glow);
  // light cones: visible in fog and rain (fx medium+)
  const coneMat = new THREE.MeshBasicMaterial({ map: coneTexture(), transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, side: THREE.DoubleSide, opacity: 0, fog: false });
  const cones = new THREE.Group();
  for (let i = 0; i < 4; i++) {
    const from = new THREE.Vector3(glowPos[i * 3], 9.2, glowPos[i * 3 + 2]);
    const to = new THREE.Vector3(0, 0, -0.6);
    const lenC = from.distanceTo(to);
    const cone = new THREE.Mesh(new THREE.CylinderGeometry(0.4, 3.2, lenC, 16, 1, true), coneMat);
    cone.position.copy(from).lerp(to, 0.5);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, -1, 0), to.clone().sub(from).normalize());
    cones.add(cone);
  }
  scene.add(cones);

  // -------------------------------------------------------------------- flags
  // Pennants on the stands: one merged geometry, waved in the vertex shader.
  const flagU = { uTime: { value: 0 }, uWind: { value: 0.15 }, uGust: { value: 0.1 }, uAngle: { value: 0 },
                  uLight: { value: new THREE.Color(1, 1, 1) } };
  const flagCols = [0xef476f, 0xffd23f, 0x06d6a0, 0x2d7ff9, 0xff8c42, 0x9b5de5];
  const fPos = [], fCol = [], fPole = [], fU = [], fIdx = [];
  const poles = [];
  flagCols.forEach((hex, k) => {
    const px = -5.5 + k * 2.2, py = 3.6, pz = -8.0;
    poles.push([px, pz]);
    const c = new THREE.Color(hex);
    const base = fPos.length / 3;
    for (let j = 0; j <= 2; j++) for (let i = 0; i <= 8; i++) {
      const u = i / 8, v = j / 2;
      const halfH = 0.32 * (1 - u * 0.85);            // tapered pennant
      fPos.push(0, py + (v - 0.5) * 2 * halfH, 0); fPole.push(px, py, pz); fU.push(u);
      fCol.push(c.r, c.g, c.b);
    }
    for (let j = 0; j < 2; j++) for (let i = 0; i < 8; i++) {
      const a = base + j * 9 + i, b = a + 1, c2 = a + 9, d = c2 + 1;
      fIdx.push(a, c2, b, b, c2, d);
    }
  });
  const flagGeo = new THREE.BufferGeometry();
  flagGeo.setAttribute("position", new THREE.Float32BufferAttribute(fPos, 3));
  flagGeo.setAttribute("color", new THREE.Float32BufferAttribute(fCol, 3));
  flagGeo.setAttribute("aPole", new THREE.Float32BufferAttribute(fPole, 3));
  flagGeo.setAttribute("aU", new THREE.Float32BufferAttribute(fU, 1));
  flagGeo.setIndex(fIdx);
  const flags = new THREE.Mesh(flagGeo, new THREE.ShaderMaterial({
    uniforms: flagU, side: THREE.DoubleSide, vertexColors: true,
    vertexShader: `
      attribute vec3 aPole; attribute float aU;
      uniform float uTime, uWind, uGust, uAngle;
      varying vec3 vCol; varying float vShade;
      void main() {
        float len = 1.1;
        float w = clamp(uWind + uGust * 0.5 * (0.5 + 0.5 * sin(uTime * 0.7 + aPole.x)), 0.0, 1.0);
        float droop = (1.0 - w) * 0.75;                                 // limp when calm
        float wave = sin(aU * 7.0 - uTime * (3.0 + 9.0 * w) + aPole.x) * aU * (0.04 + 0.16 * w);
        float ang = uAngle + wave * 1.5;
        vec3 off = vec3(cos(ang) * aU * len * (1.0 - droop * 0.6), position.y - aPole.y - droop * aU * len * 0.8,
                        sin(ang) * aU * len * (1.0 - droop * 0.6) + wave);
        vec3 p = aPole + off;
        vCol = color; vShade = 0.8 + 0.2 * sin(aU * 7.0 - uTime * 6.0);
        gl_Position = projectionMatrix * viewMatrix * vec4(p, 1.0);
      }`,
    fragmentShader: `uniform vec3 uLight; varying vec3 vCol; varying float vShade;
      void main() { gl_FragColor = vec4(vCol * uLight * vShade, 1.0);
        #include <colorspace_fragment>
      }`,
  }));
  flags.frustumCulled = false;
  const poleMesh = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.03, 0.03, 1.6, 6), poleMat, poles.length);
  poles.forEach(([x, z], i) => poleMesh.setMatrixAt(i, new THREE.Matrix4().makeTranslation(x, 3.3, z)));
  scene.add(flags, poleMesh);

  // -------------------------------------------------------- compile everything
  pGeo.instanceCount = 1; glow.material.opacity = 0.001; coneMat.opacity = 0.001; bolt.visible = true;
  renderer.compile(scene, camera);
  pGeo.instanceCount = 0; bolt.visible = false;

  // base colours of things weather tints
  const floorBase = floor.material.color.clone(), courtBase = court.material.color.clone();
  const ballEmissive = ball.material.emissive ? ball.material.emissive.clone() : null;

  // ------------------------------------------------------------- colour tables
  const C = (h) => new THREE.Color(h);
  const ZEN = { day: C(0x3d8ee8), over: C(0x8e9aab), dusk: C(0x4a5f9a), night: C(0x060b1c), nightOver: C(0x15171d) };
  const HOR = { day: C(0xbfe6ff), over: C(0xc4cad2), dusk: C(0xffa565), night: C(0x1a2338), nightOver: C(0x23262e) };
  const SUN = { day: C(0xffffff), dusk: C(0xffb26b), night: C(0xdde6ff) };
  const HEMI_SKY = { day: C(0xe3f4ff), over: C(0xc8d0dc), night: C(0x7d8db5) };
  const HEMI_GND = { day: C(0xc9a77a), night: C(0x3c3a44) };
  const tz = new THREE.Color(), th = new THREE.Color(), tc = new THREE.Color(), tv = new THREE.Vector3();
  const nightKey = new THREE.Vector3(0.15, 1, 0.35).normalize();

  function applyLook(dt, t, state) {
    const d = cur.daylight, c = cur.cloud, cd = cur.cloud_dark, f = cur.fog, fl = cur.floods;
    const duskK = Math.max(0, 1 - Math.abs(cur.sun_el - 4) / 14) * (1 - 0.7 * c) * (cur.sun_el > -8 ? 1 : 0);
    // sky colours: day/night x clear/overcast, plus the dusk glow
    tz.copy(ZEN.day).lerp(ZEN.over, c * 0.9);
    tc.copy(ZEN.night).lerp(ZEN.nightOver, c);
    tz.lerp(tc, 1 - d).lerp(ZEN.dusk, duskK * 0.5);
    th.copy(HOR.day).lerp(HOR.over, c);
    tc.copy(HOR.night).lerp(HOR.nightOver, c);
    th.lerp(tc, 1 - d).lerp(HOR.dusk, duskK * 0.75);
    th.lerp(HOR.over, f * 0.5 * d).multiplyScalar(1 - 0.6 * cd * d);
    skyU.uZenith.value.copy(tz).multiplyScalar(1 - 0.7 * cd);
    skyU.uHorizon.value.copy(th);
    skyU.uGround.value.copy(th).multiplyScalar(0.55);
    skyU.uSunDir.value.copy(cur.sun_dir);
    skyU.uSunColor.value.copy(SUN.day).lerp(SUN.dusk, duskK);
    skyU.uSunVis.value = THREE.MathUtils.smoothstep(cur.sun_el, -3, 2) * (1 - c * 0.85);
    skyU.uMoonVis.value = (1 - d) * (1 - c * 0.9);
    skyU.uMoonPhase.value = target.moon_phase;
    skyU.uStars.value = cur.stars;
    skyU.uCloud.value = c; skyU.uCloudDark.value = cd;
    skyU.uCloudLight.value = 0.12 + 0.88 * d + 0.12 * fl;
    skyU.uCloudOffset.value.x += cur.wind_x * (0.004 + 0.02 * cur.wind) * dt;
    skyU.uCloudOffset.value.y += cur.wind_z * (0.004 + 0.02 * cur.wind) * dt;
    skyU.uCloudOffset.value.set(skyU.uCloudOffset.value.x % 100, skyU.uCloudOffset.value.y % 100);
    skyU.uFlash.value = flash; skyU.uHaze.value = f;
    skyU.uTime.value = t % 600;
    renderer.setClearColor(th);
    // fog (linear, colour + distance only)
    scene.fog.color.copy(th).lerp(tc.set(0xffffff), flash * 0.3);
    scene.fog.near = cur.fog_near; scene.fog.far = cur.fog_far;
    // key light: the sun by day (warm and low at dusk), the floodlights by night
    const sunUp = THREE.MathUtils.smoothstep(cur.sun_el, -2, 8);
    tv.copy(nightKey).lerp(cur.sun_dir, sunUp);
    if (tv.y < 0.25) tv.y = 0.25;
    sun.position.copy(tv.normalize()).multiplyScalar(9);
    sun.color.copy(SUN.night).lerp(SUN.day, sunUp).lerp(SUN.dusk, duskK * sunUp);
    const dayI = 2.3 * (1 - 0.5 * c) * (1 - 0.3 * f);
    const nightI = 1.7 * Math.max(fl, 0.6);
    sun.intensity = Math.max(2.3 * cur.key_min, dayI * sunUp + nightI * (1 - sunUp)) + flash * 1.5;
    hemi.color.copy(HEMI_SKY.day).lerp(HEMI_SKY.over, c).lerp(HEMI_SKY.night, 1 - d);
    hemi.groundColor.copy(HEMI_GND.day).lerp(HEMI_GND.night, 1 - d);
    hemi.intensity = 0.75 + 0.95 * d + 0.25 * fl * (1 - d) + flash * 1.2;
    // floodlights
    lampMat.color.setRGB(0.33 + 0.67 * fl, 0.34 + 0.66 * fl, 0.37 + 0.6 * fl);
    glow.material.opacity = fl * 0.9;
    coneMat.opacity = fx === "low" ? 0 : fl * (0.05 + 0.25 * Math.max(f, cur.precip * 0.6));
    cones.visible = coneMat.opacity > 0.003;
    // the ground: darker when wet, whiter under snow (the table itself is never tinted)
    const wetK = 1 - 0.35 * cur.wet;
    floor.material.color.copy(floorBase).multiplyScalar(wetK).lerp(tc.set(0xf4f8ff), cur.snow_cover * 0.75);
    court.material.color.copy(courtBase).multiplyScalar(wetK).lerp(tc.set(0xe8eef8), cur.snow_cover * 0.5);
    if (ballEmissive) ball.material.emissive.copy(ballEmissive).lerp(tc.set(0x8a7a60), (1 - d) * 0.6);
    // precipitation
    pGeo.instanceCount = Math.round(PRECIP_MAX[fx] * Math.min(1, cur.precip));
    precip.visible = pGeo.instanceCount > 0;
    pU.uTime.value = t % 600;
    pU.uCenter.value.set(camera.position.x, 0, camera.position.z - 5);
    pU.uSnowMix.value = cur.snow_mix;
    pU.uFallSnow.value = target.hail ? 7 : 1.1;
    const windMs = 1 + 14 * cur.wind;
    pU.uWind.value.set(cur.wind_x * windMs, cur.wind_z * windMs);
    pU.uRainAlpha.value = cur.rain_alpha;
    pU.uBright.value = 0.35 + 0.65 * Math.max(d, fl * 0.8);
    // flags
    flagU.uTime.value = t % 600;
    flagU.uWind.value = fx === "low" ? 0.2 : cur.wind;
    flagU.uGust.value = fx === "low" ? 0 : cur.gust;
    flagU.uAngle.value = Math.atan2(cur.wind_z, cur.wind_x);
    flagU.uLight.value.setScalar(0.35 + 0.65 * Math.max(d, fl * 0.7));
    sky.position.copy(camera.position);
  }

  function lightning(dt, t, state) {
    if (cur.lightning < 0.05) { flash = 0; bolt.visible = false; return; }
    if (strikeLeft > 0) {                         // a strike: at most 3 flickers, ~250 ms
      strikeLeft -= dt;
      const k = strikeLeft / 0.25;
      flash = Math.max(0, Math.sin(k * Math.PI * 3)) * (state === "rally" ? 0.35 : 0.8);
      bolt.visible = state !== "rally" && flash > 0.1;
      bolt.material.opacity = flash;
      if (strikeLeft <= 0) { flash = 0; bolt.visible = false; }
      return;
    }
    nextStrike -= dt;
    if (nextStrike <= 0) {
      strikeLeft = 0.25;
      newBolt();
      audio?.()?.thunder(0.6 + Math.random() * 2.4, 0.5 + Math.random() * 0.5);
      // at least 4 s apart (photosensitivity), mean gap shrinks with intensity
      nextStrike = 4 + (6 + 9 * Math.random()) / Math.max(0.3, cur.lightning);
    }
  }

  return {
    setTarget(params) {
      target = { ...target, ...params, sun_dir: new THREE.Vector3(...(params.sun_dir ?? [0.43, 0.76, 0.48])).normalize() };
      if (first) {                               // first reading: snap, don't fade in
        first = false;
        for (const k of LERPED) cur[k] = target[k];
        cur.sun_dir.copy(target.sun_dir);
      }
    },
    update(dt, t, state) {
      const a = 1 - Math.exp(-dt / TAU);
      for (const k of LERPED) cur[k] += (target[k] - cur[k]) * a;
      cur.sun_dir.lerp(target.sun_dir, a).normalize();
      lightning(dt, t, state);
      applyLook(dt, t, state);
    },
    setFx(level) { fx = level; },
    debug: { precip, pU, pGeo, sky, skyU, cur, renderer },
    info() {
      return `cloud ${cur.cloud.toFixed(2)} rain ${cur.precip.toFixed(2)}${cur.snow_mix > 0.5 ? " (snow)" : ""} ` +
        `fog ${cur.fog.toFixed(2)} wind ${cur.wind.toFixed(2)} day ${cur.daylight.toFixed(2)} floods ${cur.floods.toFixed(2)} ` +
        `| fx ${fx}, ${pGeo.instanceCount} drops${cur.lightning > 0.05 ? ", lightning" : ""}`;
    },
  };
}

// ------------------------------------------------------------------ textures
// Tileable value-noise fbm, built once (~10 ms).
function makeNoiseTexture(n) {
  const grid = (s) => { const g = new Float32Array(s * s); for (let i = 0; i < g.length; i++) g[i] = Math.random(); return g; };
  const data = new Uint8Array(n * n);
  const octs = [[4, 0.5], [8, 0.25], [16, 0.15], [32, 0.1]].map(([s, a]) => [s, a, grid(s)]);
  const sm = (x) => x * x * (3 - 2 * x);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let v = 0;
    for (const [s, a, g] of octs) {
      const fx = (x / n) * s, fy = (y / n) * s;
      const x0 = Math.floor(fx), y0 = Math.floor(fy), tx = sm(fx - x0), ty = sm(fy - y0);
      const at = (i, j) => g[((j + s) % s) * s + ((i + s) % s)];
      const top = at(x0, y0) * (1 - tx) + at(x0 + 1, y0) * tx;
      const bot = at(x0, y0 + 1) * (1 - tx) + at(x0 + 1, y0 + 1) * tx;
      v += (top * (1 - ty) + bot * ty) * a;
    }
    data[y * n + x] = Math.min(255, v * 255);
  }
  const t = new THREE.DataTexture(data, n, n, THREE.RedFormat);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.needsUpdate = true;
  return t;
}

function radialTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const gr = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  gr.addColorStop(0, "rgba(255,255,255,1)"); gr.addColorStop(0.25, "rgba(255,250,230,.6)"); gr.addColorStop(1, "rgba(255,240,200,0)");
  g.fillStyle = gr; g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}

function coneTexture() {
  const c = document.createElement("canvas");
  c.width = 4; c.height = 64;
  const g = c.getContext("2d");
  const gr = g.createLinearGradient(0, 0, 0, 64);
  gr.addColorStop(0, "rgba(255,248,225,.9)"); gr.addColorStop(1, "rgba(255,248,225,0)");
  g.fillStyle = gr; g.fillRect(0, 0, 4, 64);
  return new THREE.CanvasTexture(c);
}
