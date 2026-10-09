// Rogers Cup -- the renderer. Python (pingpong.py) runs the game and streams
// state at 60 Hz plus your webcam cut-out at ~20 Hz; this file only draws it
// and sends controls back: on-screen buttons, or keys ENTER start / home,
// R rematch, M announcer, C crowd, SPACE swing in --sim, D debug,
// [ ] swing sensitivity, ESC quit.
import * as THREE from "three";
import { makeCharacter, makePaddle, toon, part, HEAD_HEIGHT } from "./characters.js";
import { GameAudio } from "./audio.js";

const TABLE_Y = 0.76;                          // table top height (game y=0)
let T = { half_w: 0.7625, half_l: 1.37, net_h: 0.1525, hit_z: 1.55, opp_z: -1.55, hit_rx: 0.38 };
let BODY = { shoulder_world: 0.36, shoulder_y: 0.45, body_x_gain: 1.6 };
let OPPONENTS = [];
let SIM = false;
let GAME_OVER_LOCK = 2.5;
let S = null;                                   // latest state from Python
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ renderer
const renderer = new THREE.WebGLRenderer({ canvas: $("scene"), antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.05, 100);
function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

function canvasTex(w, h, draw, repeat) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(...repeat); }
  return t;
}

scene.background = canvasTex(4, 256, (g, w, h) => {
  const gr = g.createLinearGradient(0, 0, 0, h);
  gr.addColorStop(0, "#5fb4ff"); gr.addColorStop(0.6, "#bfe6ff"); gr.addColorStop(1, "#fff4d6");
  g.fillStyle = gr; g.fillRect(0, 0, w, h);
});
scene.fog = new THREE.Fog(0xcfeaff, 14, 34);

scene.add(new THREE.HemisphereLight(0xe3f4ff, 0xc9a77a, 1.7));
const sun = new THREE.DirectionalLight(0xffffff, 2.3);
sun.position.set(3, 7, 4);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -5, right: 5, top: 5, bottom: -5, near: 1, far: 20 });
sun.shadow.bias = -0.0005;
scene.add(sun);

// ------------------------------------------------------------------- arena
const floorTex = canvasTex(512, 512, (g, w, h) => {
  const cols = ["#d9a066", "#cf955a", "#e0aa70", "#d49b60"];
  for (let i = 0; i < 8; i++) {
    g.fillStyle = cols[i % 4]; g.fillRect(0, i * h / 8, w, h / 8);
    g.fillStyle = "rgba(90,50,20,.25)"; g.fillRect(0, i * h / 8, w, 3);
    g.fillRect(((i * 197) % w), i * h / 8, 3, h / 8);
  }
}, [10, 10]);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), new THREE.MeshToonMaterial({ map: floorTex }));
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const court = new THREE.Mesh(new THREE.PlaneGeometry(6.5, 9.5), toon(0x3f6fb5));
court.rotation.x = -Math.PI / 2; court.position.y = 0.005; court.receiveShadow = true;
scene.add(court);

const boardTex = canvasTex(1024, 64, (g, w, h) => {
  const segs = ["ROGERS CUP", "ME193", "PING-PONG", "LEGO EDUCATION"];
  const colors = ["#ef476f", "#ffd23f", "#06d6a0", "#2d7ff9"];
  for (let i = 0; i < 4; i++) {
    g.fillStyle = colors[i]; g.fillRect(i * w / 4, 0, w / 4, h);
    g.fillStyle = "#fff"; g.font = "bold 34px Fredoka, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(segs[i], i * w / 4 + w / 8, h / 2 + 2);
  }
}, [2, 1]);
function barrier(len, x, z, rotY) {
  const b = part(new THREE.BoxGeometry(len, 0.55, 0.08), new THREE.MeshToonMaterial({ map: boardTex.clone() }), { outline: 0.01 });
  b.material.map.repeat.set(len / 4, 1); b.material.map.needsUpdate = true;
  b.position.set(x, 0.275, z); b.rotation.y = rotY;
  scene.add(b);
}
barrier(6.5, 0, -4.75, 0);
barrier(9.5, -3.25, 0, Math.PI / 2);
barrier(9.5, 3.25, 0, -Math.PI / 2);

// crowd: tiered stands behind the far end and down both sides
const crowd = [];
{
  const n = 220;
  const bodyMesh = new THREE.InstancedMesh(new THREE.CapsuleGeometry(0.17, 0.3, 4, 8), toon(0xffffff), n);
  const headMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.13, 12, 10), toon(0xffffff), n);
  const shirts = [0xef476f, 0xffd23f, 0x06d6a0, 0x118ab2, 0xff8c42, 0x9b5de5, 0xf15bb5, 0xffffff];
  const skins = [0xf2c49b, 0xd9a07a, 0xa86f4c, 0x6e4630, 0xffdbb8];
  const col = new THREE.Color();
  let i = 0;
  const seat = (x, z, tier) => {
    if (i >= n) return;
    crowd.push({ x, z, y: 0.35 + tier * 0.45, phase: Math.random() * 6.28, i });
    bodyMesh.setColorAt(i, col.set(shirts[(Math.random() * shirts.length) | 0]));
    headMesh.setColorAt(i, col.set(skins[(Math.random() * skins.length) | 0]));
    i++;
  };
  for (let tier = 0; tier < 4; tier++) {
    for (let x = -5.5; x <= 5.5; x += 0.55) seat(x + (tier % 2) * 0.27, -5.4 - tier * 0.6, tier);
    for (let z = -4.2; z <= 3.0; z += 0.6) { seat(-4.0 - tier * 0.6, z, tier); seat(4.0 + tier * 0.6, z, tier); }
  }
  bodyMesh.count = headMesh.count = i;
  scene.add(bodyMesh, headMesh);
  for (let tier = 0; tier < 4; tier++) {   // stand steps
    const mat = toon(tier % 2 ? 0x8d99ae : 0x9eaabf);
    const back = new THREE.Mesh(new THREE.BoxGeometry(13, 0.45 * (tier + 1), 0.6), mat);
    back.position.set(0, 0.225 * (tier + 1) - 0.1, -5.4 - tier * 0.6); scene.add(back);
    for (const s of [-1, 1]) {
      const side = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.45 * (tier + 1), 8.6), mat);
      side.position.set(s * (4.0 + tier * 0.6), 0.225 * (tier + 1) - 0.1, -0.6); scene.add(side);
    }
  }
  crowd.bodyMesh = bodyMesh; crowd.headMesh = headMesh; crowd.cheer = 0;
}
const _m = new THREE.Matrix4();
function updateCrowd(t, dt) {
  crowd.cheer = Math.max(0, crowd.cheer - dt * 0.6);
  for (const c of crowd) {
    const hop = Math.abs(Math.sin(t * (3 + crowd.cheer * 6) + c.phase)) * (0.02 + crowd.cheer * 0.18);
    _m.makeTranslation(c.x, c.y + hop, c.z); crowd.bodyMesh.setMatrixAt(c.i, _m);
    _m.makeTranslation(c.x, c.y + hop + 0.36, c.z); crowd.headMesh.setMatrixAt(c.i, _m);
  }
  crowd.bodyMesh.instanceMatrix.needsUpdate = crowd.headMesh.instanceMatrix.needsUpdate = true;
}

// ------------------------------------------------------------------- table
const table = new THREE.Group();
scene.add(table);
function buildTable() {
  table.clear();
  const W = T.half_w * 2, L = T.half_l * 2;
  const top = part(new THREE.BoxGeometry(W, 0.04, L), toon(0x1f5fbf), { outline: 0.006 });
  top.position.y = TABLE_Y - 0.02; top.receiveShadow = true;
  table.add(top);
  const white = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const line = (w, l, x, z) => { const m = new THREE.Mesh(new THREE.BoxGeometry(w, 0.002, l), white); m.position.set(x, TABLE_Y + 0.001, z); table.add(m); };
  line(W, 0.02, 0, T.half_l - 0.01); line(W, 0.02, 0, -T.half_l + 0.01);
  line(0.02, L, T.half_w - 0.01, 0); line(0.02, L, -T.half_w + 0.01, 0);
  line(0.006, L, 0, 0);
  const legMat = toon(0x2b2d42);
  for (const [x, z] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    const leg = part(new THREE.BoxGeometry(0.06, TABLE_Y - 0.04, 0.06), legMat);
    leg.position.set(x * (T.half_w - 0.15), (TABLE_Y - 0.04) / 2, z * (T.half_l - 0.3));
    table.add(leg);
  }
  const netTex = canvasTex(256, 32, (g, w, h) => {
    g.strokeStyle = "rgba(255,255,255,.85)"; g.lineWidth = 1.5;
    for (let x = 0; x <= w; x += 6) { g.beginPath(); g.moveTo(x, 0); g.lineTo(x, h); g.stroke(); }
    for (let y = 0; y <= h; y += 6) { g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  });
  const net = new THREE.Mesh(new THREE.PlaneGeometry(W + 0.3, T.net_h),
    new THREE.MeshBasicMaterial({ map: netTex, transparent: true, side: THREE.DoubleSide, color: 0x222244 }));
  net.position.set(0, TABLE_Y + T.net_h / 2, 0);
  table.add(net);
  const band = part(new THREE.BoxGeometry(W + 0.3, 0.018, 0.012), toon(0xffffff), { outline: 0.05 });
  band.position.set(0, TABLE_Y + T.net_h, 0); table.add(band);
  for (const s of [-1, 1]) {
    const post = part(new THREE.CylinderGeometry(0.015, 0.015, T.net_h + 0.03, 8), toon(0x2b2d42));
    post.position.set(s * (T.half_w + 0.15), TABLE_Y + T.net_h / 2, 0); table.add(post);
  }
}
buildTable();

// your paddle column marker: green when you're lined up with the incoming ball
const marker = new THREE.Mesh(new THREE.PlaneGeometry(1, 0.6),
  new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35, depthWrite: false }));
marker.rotation.x = -Math.PI / 2;
scene.add(marker);

// --------------------------------------------------------------------- ball
const BALL_R = 0.03;                             // a bit bigger than real (0.02) so it reads on screen
const ball = part(new THREE.SphereGeometry(BALL_R, 20, 14), toon(0xfffaf0, { emissive: 0x332211 }), { outline: 0.12 });
scene.add(ball);
const ballShadow = new THREE.Mesh(new THREE.CircleGeometry(BALL_R * 1.1, 16),
  new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.3, depthWrite: false }));
ballShadow.rotation.x = -Math.PI / 2;
scene.add(ballShadow);
const trail = [];
for (let i = 0; i < 10; i++) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(BALL_R * (1 - i * 0.08), 10, 8),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 * (1 - i / 10), depthWrite: false }));
  scene.add(m); trail.push(m);
}
const trailPos = [];

// hit sparks
const sparks = [];
for (let i = 0; i < 40; i++) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.012, 6, 5), new THREE.MeshBasicMaterial({ color: 0xffd23f }));
  m.visible = false; scene.add(m);
  sparks.push({ m, v: new THREE.Vector3(), life: 0 });
}
function burst(pos, color) {
  let n = 14;
  for (const s of sparks) {
    if (s.life > 0) continue;
    s.m.material.color.set(color); s.m.position.copy(pos); s.m.visible = true; s.life = 0.6;
    s.v.set((Math.random() - 0.5) * 3, Math.random() * 2.5, (Math.random() - 0.5) * 3);
    if (--n === 0) break;
  }
}
function updateSparks(dt) {
  for (const s of sparks) {
    if (s.life <= 0) continue;
    s.life -= dt; s.v.y -= 9 * dt;
    s.m.position.addScaledVector(s.v, dt);
    s.m.scale.setScalar(Math.max(0.01, s.life / 0.6));
    if (s.life <= 0) s.m.visible = false;
  }
}

// ---------------------------------------------------------------- you
const camCanvas = document.createElement("canvas");
camCanvas.width = 320; camCanvas.height = 240;
const camCtx = camCanvas.getContext("2d");
const camTex = new THREE.CanvasTexture(camCanvas);
camTex.colorSpace = THREE.NoColorSpace;          // passed straight through by the shader below
// Person mask from MediaPipe (raw bytes, top row first). The GPU upsamples it
// and uses it as transparency -- no transparent-image encoding in Python.
let MASK_W = 160, MASK_H = 120;
let maskTex = makeMaskTex();
function makeMaskTex() {
  const t = new THREE.DataTexture(new Uint8Array(MASK_W * MASK_H), MASK_W, MASK_H, THREE.RedFormat);
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.unpackAlignment = 1;
  t.needsUpdate = true;
  return t;
}
const youMat = new THREE.ShaderMaterial({
  uniforms: { map: { value: camTex }, mask: { value: maskTex }, opacity: { value: 0.85 } },
  vertexShader: `varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform sampler2D map; uniform sampler2D mask; uniform float opacity; varying vec2 vUv;
    void main() {
      float m = texture2D(mask, vec2(vUv.x, 1.0 - vUv.y)).r;         // mask rows are top-first
      float a = smoothstep(0.25, 0.65, m)                              // crisp person edge
              * smoothstep(0.0, 0.4, vUv.y)                            // fade out where the webcam frame ends
              * opacity;
      gl_FragColor = vec4(texture2D(map, vUv).rgb, a);
    }`,
  transparent: true, depthWrite: false, side: THREE.DoubleSide,
});
const you = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), youMat);
you.renderOrder = 5;
scene.add(you);
const YOU_Z = 1.7;                              // just behind your end of the table
const youFit = { cx: 0, cy: 1.1, w: 1.4 };

const myPaddle = makePaddle(0xd62828);
myPaddle.scale.setScalar(1.3);
scene.add(myPaddle);
const PADDLE_Z = 1.8;                           // in front of the cut-out, so it covers your real hand
let mySwingT = 1;

// ------------------------------------------------------------ opponents
const PEDESTAL_X = { pip: -1.75, rita: 0, viktor: 1.75 };
const STAGE_Z = -2.75;
const chars = {};
const pedestals = {};
for (const key of ["pip", "rita", "viktor"]) {
  const c = makeCharacter(key);
  chars[key] = c;
  c.group.position.set(PEDESTAL_X[key], 0.22, STAGE_Z);
  scene.add(c.group);
  const ped = part(new THREE.CylinderGeometry(0.5, 0.55, 0.22, 32), toon({ pip: 0x4cc9f0, rita: 0xffd23f, viktor: 0xef476f }[key]));
  ped.position.set(PEDESTAL_X[key], 0.11, STAGE_Z); ped.receiveShadow = true;
  scene.add(ped); pedestals[key] = ped;
}
const spot = new THREE.SpotLight(0xfff3c4, 0, 9, 0.35, 0.5);
spot.position.set(0, 5, STAGE_Z + 1.5);
scene.add(spot, spot.target);

// ------------------------------------------------------------ scoreboard
const sbCanvas = document.createElement("canvas");
sbCanvas.width = 1024; sbCanvas.height = 560;
const sbTex = new THREE.CanvasTexture(sbCanvas);
sbTex.colorSpace = THREE.SRGBColorSpace;
const scoreboard = new THREE.Group();
{
  const frame = part(new THREE.BoxGeometry(1.62, 0.92, 0.08), toon(0x1b2340), { outline: 0.01 });
  const face = new THREE.Mesh(new THREE.PlaneGeometry(1.5, 0.82), new THREE.MeshBasicMaterial({ map: sbTex }));
  face.position.z = 0.045;
  const pole = part(new THREE.CylinderGeometry(0.04, 0.04, 1.6, 10), toon(0x8d99ae));
  pole.position.y = -1.1;
  scoreboard.add(frame, face, pole);
  scoreboard.position.set(-2.3, 1.85, -1.3);
  scoreboard.rotation.y = 0.42;
  scene.add(scoreboard);
}
let sbKey = "";
let sbFlip = 1;
function roundRect(g, x, y, w, h, r) { g.beginPath(); g.roundRect(x, y, w, h, r); g.fill(); }
function drawScoreboard() {
  const g = sbCanvas.getContext("2d"), W = sbCanvas.width, H = sbCanvas.height;
  g.fillStyle = "#1b2340"; g.fillRect(0, 0, W, H);
  const name = (S?.opp_name || "").toUpperCase();
  const cols = [["YOU", S?.score.you ?? 0, S?.server === "player"], [name, S?.score.opp ?? 0, S?.server === "opp"]];
  cols.forEach(([label, pts, serving], i) => {
    const x = 40 + i * 492;
    g.fillStyle = i ? "#ef476f" : "#2d7ff9"; roundRect(g, x, 30, 452, 80, 24);
    g.fillStyle = "#fff"; g.font = "bold 50px Fredoka, sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(label.length > 14 ? label.slice(0, 13) + "…" : label, x + 226, 72);
    g.fillStyle = "#0e1428"; roundRect(g, x + 76, 130, 300, 260, 30);
    g.fillStyle = "#ffd23f"; g.font = "bold 230px Fredoka, sans-serif";
    g.fillText(String(pts), x + 226, 272);
    if (serving && S?.state !== "game_over") {
      g.fillStyle = "#06d6a0"; g.beginPath(); g.arc(x + 40, 260, 18, 0, 7); g.fill();
    }
  });
  g.fillStyle = "#fff"; g.font = "600 46px Fredoka, sans-serif"; g.textAlign = "center";
  g.fillText(`STREAK ${S?.streak ?? 0}   ·   RECORD ${(S?.record ?? 0).toFixed(1)}`, W / 2, 470);
  sbTex.needsUpdate = true;
}

// ------------------------------------------------------------------- audio
// Browsers only allow sound after a click/key, so audio starts on the first one.
let audio = null;
function ensureAudio() {
  if (audio) { if (audio.ctx.state === "suspended") audio.ctx.resume(); return; }
  audio = new GameAudio();
  audio.onCaption = showCaption;
  audio.setMood(S && S.state !== "select" ? "idle" : "quiet");
  refreshSoundButtons();
}
addEventListener("pointerdown", ensureAudio);
const blip = (...a) => audio?.blip(...a);
const arp = (notes, step = 0.09, type = "triangle") =>
  notes.forEach((f, i) => setTimeout(() => blip(f, 0.16, type, 0.12), i * step * 1000));

// ------------------------------------------------------------------ UI bits
const banner = $("banner"), subBanner = $("subBanner"), bubble = $("bubble"), hint = $("hint");
let bannerUntil = 0, subUntil = 0, bubbleUntil = 0;
function showBanner(text, cls = "", secs = 1.4) {
  banner.textContent = text;
  banner.className = "banner " + cls;
  void banner.offsetWidth; banner.classList.add("show");
  bannerUntil = performance.now() + secs * 1000;
}
function showSub(text, secs = 2) {
  subBanner.textContent = text; subBanner.classList.remove("hidden");
  subUntil = performance.now() + secs * 1000;
}
function say(text) {
  bubble.textContent = text; bubble.classList.remove("hidden");
  bubbleUntil = performance.now() + 2800;
}
function pop(id) { const el = $(id); el.classList.remove("pop"); void el.offsetWidth; el.classList.add("pop"); }

const MISS_TEXT = {
  no_swing: "Swing the paddle!",
  wrong_place: "Move your paddle to the ball!",
  no_pose: "Step into the camera view!",
  bad_timing: "So close: time your swing!",
};

function buildCards() {
  $("cards").innerHTML = OPPONENTS.map((o) => `
    <div class="card" id="card-${o.key}">
      <div class="lvl ${o.level}">${"★".repeat(o.stars)}${"☆".repeat(3 - o.stars)} ${o.level}</div>
      <h3>${o.name}</h3>
      <p>${o.tagline}</p>
      <div class="tag">AprilTag #${o.tag_id}</div>
    </div>`).join("");
}

// ---------------------------------------------------------------- network
let ws = null;
let lastState = "";
let lastScoreKey = "";
function send(cmd) { if (ws?.readyState === 1) ws.send(JSON.stringify({ cmd })); }

function connect() {
  ws = new WebSocket(`ws://${location.hostname}:${Number(location.port) + 1}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => $("conn").classList.add("hidden");
  ws.onclose = () => {
    $("conn").classList.remove("hidden");
    setTimeout(connect, 1000);
  };
  ws.onmessage = (e) => {
    if (typeof e.data === "string") return onJson(JSON.parse(e.data));
    const kind = String.fromCharCode(new Uint8Array(e.data, 0, 1)[0]);
    if (kind === "F") {
      const img = $("camInset"), old = img.src;
      img.src = URL.createObjectURL(new Blob([e.data.slice(1)], { type: "image/jpeg" }));
      if (old.startsWith("blob:")) URL.revokeObjectURL(old);
    } else if (kind === "C") {
      pendingCut = e.data;                       // newest wins; never queue frames behind a slow decode
      if (!decoding) decodeCut();
    }
  };
}

// "C" message = [1 byte "C"][MASK_W*MASK_H mask bytes][JPEG]
let pendingCut = null, decoding = false;
function decodeCut() {
  const buf = pendingCut;
  pendingCut = null;
  if (!buf) return;
  decoding = true;
  const n = MASK_W * MASK_H;
  createImageBitmap(new Blob([buf.slice(1 + n)], { type: "image/jpeg" })).then((bmp) => {
    camCtx.drawImage(bmp, 0, 0, 320, 240); bmp.close();
    camTex.needsUpdate = true;
    maskTex.image.data.set(new Uint8Array(buf, 1, n));      // same frame as the image -> edges line up
    maskTex.needsUpdate = true;
  }).catch(() => {}).finally(() => { decoding = false; decodeCut(); });
}

function onJson(msg) {
  if (msg.type === "hello") {
    T = { ...T, ...msg.table }; BODY = msg.body; OPPONENTS = msg.opponents; SIM = msg.sim;
    GAME_OVER_LOCK = msg.game_over_lock ?? GAME_OVER_LOCK;
    if (msg.mask_size && (msg.mask_size[0] !== MASK_W || msg.mask_size[1] !== MASK_H)) {
      [MASK_W, MASK_H] = msg.mask_size;
      maskTex.dispose(); maskTex = makeMaskTex(); youMat.uniforms.mask.value = maskTex;
    }
    $("topic").textContent = msg.topic;
    buildTable(); buildCards();
    if (SIM) hint.textContent = "SIM MODE: SPACE = swing";
    return;
  }
  if (msg.type !== "state") return;
  if (!S) {                                       // first state: jump straight to the right view
    const v = msg.state === "select" ? VIEWS.select : VIEWS.game;
    camPos.copy(v.pos); camLook.copy(v.look);
  }
  S = msg;
  for (const ev of msg.events) onEvent(ev);
  if (msg.swung) mySwingT = 0;                   // the IMU saw a swing (shows even if you miss)
  if (msg.state !== lastState) onStateChange(msg.state, lastState);
  lastState = msg.state;
  updateHud();
}

let rally = 0;                                  // shots since the serve (sizes the crowd's reaction)
function onEvent(ev) {
  const opp = S?.opp && chars[S.opp];
  switch (ev.type) {
    case "start":
      audio?.applause(2.5, 0.8); audio?.cheer(0.35);
      break;
    case "call":
      if (audio) audio.say(ev.text, ev.excite, ev.interrupt);
      else showCaption(ev.text);
      break;
    case "hit":
      rally++;
      if (ev.streak && ev.streak % 5 === 0) audio?.cheer(0.3);
      mySwingT = 0;
      burst(ball.position, 0xffd23f);
      shake = 0.05;
      blip(640 + Math.min(ev.streak, 20) * 20, 0.07, "square", 0.1);
      pop("streakPill");
      crowd.cheer = Math.min(1, crowd.cheer + 0.15);
      break;
    case "opp_hit": rally++; opp?.swing(); blip(480, 0.07, "square", 0.08); break;
    case "bounce": blip(1100, 0.03, "sine", 0.08); break;
    case "net": blip(160, 0.2, "sawtooth", 0.08, -60); audio?.ooh(0.6); break;
    case "record":
      pop("recordPill");
      if (ev.record > 1) showSub(`NEW RECORD: ${ev.record.toFixed(1)}`, 1.5);
      break;
    case "point":
      if (ev.winner === "player") {
        showBanner("POINT!", "good", 1.2); arp([523, 659, 784]); crowd.cheer = 1;
        audio?.cheer(0.35 + Math.min(rally, 20) / 25); audio?.applause(2, 0.7);
      } else {
        showBanner(ev.reason === "miss" ? "MISS" : "POINT", "lose", 1.2);
        blip(300, 0.35, "sawtooth", 0.08, -180);
        audio?.ooh(0.55, true); audio?.applause(1.2, 0.3);         // "aww", then polite applause
      }
      break;
    case "game_over":
      if (ev.winner === "player") {
        arp([523, 659, 784, 1047, 1319], 0.12); crowd.cheer = 1;
        audio?.cheer(1); audio?.applause(5, 1);
      } else {
        arp([392, 330, 262, 196], 0.18, "sawtooth");
        audio?.applause(3, 0.5);
      }
      break;
    case "miss":
      showSub(MISS_TEXT[ev.reason] || "Miss!", 2);
      if (ev.reason === "bad_timing") audio?.ooh(0.7);                 // so close!
      break;
    case "say": say(ev.text); break;
  }
}

function onStateChange(state, prev) {
  const inGame = state !== "select";
  $("select").classList.toggle("hidden", inGame);
  $("hud").classList.toggle("hidden", !inGame);
  $("homeBtn").classList.toggle("hidden", !inGame || state === "game_over");
  disarmHome();
  if (state === "select") { banner.className = "banner hidden"; subBanner.classList.add("hidden"); }
  $("over").classList.toggle("hidden", state !== "game_over");
  if (state === "game_over") showGameOver();
  if (state === "serve") rally = 0;
  audio?.setMood({ select: "quiet", countdown: "buzz", rally: "rally", game_over: "buzz" }[state] ?? "idle");
  if (state === "serve" && prev === "countdown") { showBanner("PLAY!", "good", 0.8); blip(1046, 0.25, "triangle", 0.12); }
}

let lastCount = 0;
function updateHud() {
  $("streak").textContent = S.streak;
  $("fire").textContent = S.streak >= 5 ? "🔥" : "";
  $("record").textContent = S.record.toFixed(1);
  $("mqttDot").classList.toggle("on", S.mqtt_ok);

  if (S.state === "select") {
    for (const o of OPPONENTS) $("card-" + o.key)?.classList.toggle("on", S.highlight === o.key);
    const c = $("startBtn");
    const o = OPPONENTS.find((o) => o.key === S.highlight);
    c.classList.toggle("ready", !!o);
    c.disabled = !o;
    const html = o ? `▶ Start match vs ${o.name} <kbd>Enter</kbd>` : "Hold up an AprilTag to choose your opponent";
    if (c.innerHTML !== html) c.innerHTML = html;
  }
  if (S.state === "countdown") {
    const n = Math.ceil(S.countdown);
    if (n !== lastCount && n > 0) { showBanner(String(n), "", 0.9); blip(660, 0.1, "triangle", 0.12); }
    lastCount = n;
  } else lastCount = 0;

  if (S.state === "game_over") {
    const ready = S.state_age >= GAME_OVER_LOCK;    // short lock so a stray swing/keypress can't skip the result
    $("overHome").disabled = $("overRematch").disabled = !ready;
  }

  let h = "";
  if (S.state === "serve" && S.server === "player") h = "Swing the paddle to serve!";
  else if (S.state !== "select" && !S.paddle.visible) h = "Step into the camera view!";
  else if (SIM && S.state !== "select" && S.state !== "game_over") h = "SIM MODE: SPACE = swing";
  hint.textContent = h;

  const key = `${S.score.you}-${S.score.opp}-${S.server}-${S.streak}-${S.record}-${S.opp_name}-${S.state === "game_over"}`;
  if (key !== sbKey) {
    if (key.split("-").slice(0, 2).join() !== sbKey.split("-").slice(0, 2).join()) sbFlip = 0;
    sbKey = key;
    drawScoreboard();
  }

  if (!$("debug").classList.contains("hidden")) {
    const d = S.debug, frac = Math.min(1, d.gyro / Math.max(1, d.threshold * 2));
    $("debug").innerHTML =
      `state      ${S.state}\nvision fps ${d.fps}   frame age ${d.frame_age_ms ?? "-"} ms\npaddle     x ${S.paddle.x.toFixed(2)}  y ${S.paddle.y.toFixed(2)}  ${S.paddle.visible ? "seen" : "NOT SEEN"}\n` +
      `gyro       ${d.gyro}   swing threshold ${d.threshold}  ([ / ])\nmotor      ${d.paddle_connected ? "connected" : SIM ? "sim" : "OFFLINE"}` +
      `<span class="bar"><span style="position:absolute;left:0;top:0;bottom:0;width:${frac * 100}%;background:${d.gyro >= d.threshold ? "#06d6a0" : "#ffd23f"};border-radius:4px"></span>` +
      `<span style="position:absolute;left:50%;top:-3px;bottom:-3px;width:2px;background:#fff"></span></span>`;
  }
}

// ------------------------------------------------------------- controls
// Every button blurs after a click, so a later SPACE (swing) or ENTER can't
// re-trigger it by accident.
function button(id, fn) {
  $(id).addEventListener("click", (e) => { ensureAudio(); fn(e); e.currentTarget.blur(); });
}
button("startBtn", () => send("confirm"));
button("overHome", () => send("home"));
button("overRematch", () => send("rematch"));
button("voiceBtn", () => toggleVoice());
button("crowdBtn", () => toggleCrowd());

// Home mid-match takes two clicks so a stray click can't throw the game away.
let homeArmed = 0;
button("homeBtn", () => {
  if (performance.now() < homeArmed) { disarmHome(); send("home"); return; }
  homeArmed = performance.now() + 3000;
  $("homeBtn").classList.add("armed");
  $("homeBtn").textContent = "Quit match? Click again";
  setTimeout(() => { if (performance.now() >= homeArmed) disarmHome(); }, 3100);
});
function disarmHome() {
  homeArmed = 0;
  $("homeBtn").classList.remove("armed");
  $("homeBtn").textContent = "🏠 Home";
}

function toggleVoice() { if (audio) { audio.setVoice(!audio.voiceOn); refreshSoundButtons(); } }
function toggleCrowd() { if (audio) { audio.setCrowd(!audio.crowdOn); refreshSoundButtons(); } }
function refreshSoundButtons() {
  $("voiceBtn").classList.toggle("off", !!audio && !audio.voiceOn);
  $("crowdBtn").classList.toggle("off", !!audio && !audio.crowdOn);
}

function showGameOver() {
  const won = S.winner === "player";
  const t = $("overTitle");
  t.textContent = won ? "YOU WIN!" : "YOU LOSE";
  t.className = won ? "win" : "lose";
  $("overScore").textContent = `${S.score.you} – ${S.score.opp}  vs  ${S.opp_name}`;
  $("overRecord").textContent = `Session record: ${S.record.toFixed(1)} hits in a row  ·  posted to ${$("topic").textContent}`;
  $("overHome").disabled = $("overRematch").disabled = true;
}

let captionTimer = 0;
function showCaption(text) {
  const c = $("caption");
  c.textContent = text; c.classList.remove("fade");
  clearTimeout(captionTimer);
  captionTimer = setTimeout(() => c.classList.add("fade"), 2500 + text.length * 45);
}

// ------------------------------------------------------------------- input
addEventListener("keydown", (e) => {
  ensureAudio();
  if (e.repeat) return;
  if (e.key === "Enter") send("confirm");
  else if (e.key === "r" || e.key === "R") send("rematch");
  else if (e.key === "m" || e.key === "M") toggleVoice();
  else if (e.key === "c" || e.key === "C") toggleCrowd();
  else if (e.code === "Space") { e.preventDefault(); if (SIM) { send("swing"); mySwingT = 0; } }
  else if (e.key === "d" || e.key === "D") $("debug").classList.toggle("hidden");
  else if (e.key === "[") send("sens_up");
  else if (e.key === "]") send("sens_down");
  else if (e.key === "Escape") { send("quit"); $("conn").innerHTML = "Game closed.<small>Run pingpong.py to play again</small>"; }
});

// -------------------------------------------------------------------- loop
const camPos = new THREE.Vector3(0, 1.6, 1.5), camLook = new THREE.Vector3(0, 1, -2.5);
const VIEWS = {
  select: { pos: new THREE.Vector3(0, 1.6, 2.6), look: new THREE.Vector3(0, 0.75, STAGE_Z) },
  game: { pos: new THREE.Vector3(0, 1.95, 3.75), look: new THREE.Vector3(0, 0.8, -0.6) },
};
let shake = 0;
const clock = new THREE.Clock();
const tmp = new THREE.Vector3();
const lerp = (a, b, k) => a + (b - a) * k;

function frame() {
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  const state = S?.state ?? "select";
  const inGame = state !== "select";

  // camera
  const view = inGame ? VIEWS.game : VIEWS.select;
  const k = 1 - Math.pow(0.02, dt);
  camPos.lerp(view.pos, k); camLook.lerp(view.look, k);
  camera.position.copy(camPos);
  shake = Math.max(0, shake - dt * 0.25);
  if (shake > 0) camera.position.add(tmp.set((Math.random() - 0.5) * shake, (Math.random() - 0.5) * shake, 0));
  camera.lookAt(camLook);

  // opponents
  for (const [key, c] of Object.entries(chars)) {
    const chosen = S?.opp === key;
    c.group.visible = !inGame || chosen;
    pedestals[key].visible = !inGame;
    if (!inGame) {
      const hi = S?.highlight === key;
      c.baseY = 0.22;
      c.group.position.x = lerp(c.group.position.x, PEDESTAL_X[key], k);
      c.group.position.z = STAGE_Z;
      c.group.rotation.y = hi ? Math.sin(t * 1.5) * 0.5 : 0;
      c.group.scale.setScalar(lerp(c.group.scale.x, hi ? 1.12 : 0.92, k));
      c.mode = hi ? "celebrate" : "idle";
    } else if (chosen) {
      c.baseY = 0;
      // stand so the PADDLE (held out to the side) is where Python says it is
      c.group.position.x = lerp(c.group.position.x, (S.opp_x ?? 0) - c.armPivot.position.x, 0.3);
      c.group.position.z = T.opp_z - 0.42;
      c.group.rotation.y = 0;
      c.group.scale.setScalar(1);
      c.mode = state === "game_over" ? (S.winner === "opp" ? "celebrate" : "sulk") : state === "select" ? "idle" : "ready";
    }
    c.update(dt, t);
  }
  const hiKey = S?.highlight;
  spot.intensity = !inGame && hiKey ? 60 : 0;
  if (hiKey) { spot.target.position.set(PEDESTAL_X[hiKey], 0, STAGE_Z); spot.position.x = PEDESTAL_X[hiKey]; }

  // ball
  const b = S?.ball;
  const showBall = inGame && b?.visible;
  ball.visible = ballShadow.visible = !!showBall;
  if (showBall) {
    ball.position.set(b.x, TABLE_Y + b.y + BALL_R * 0.5, b.z);
    const overTable = Math.abs(b.x) <= T.half_w && Math.abs(b.z) <= T.half_l && b.y >= -0.02;
    ballShadow.position.set(b.x, overTable ? TABLE_Y + 0.003 : 0.01, b.z);
    const hgt = Math.max(0, b.y + (overTable ? 0 : TABLE_Y));
    ballShadow.scale.setScalar(1 + hgt * 1.5);
    ballShadow.material.opacity = Math.max(0.08, 0.35 - hgt * 0.25);
    trailPos.unshift(ball.position.clone()); trailPos.length = Math.min(trailPos.length, trail.length);
  } else trailPos.length = 0;
  trail.forEach((m, i) => { m.visible = !!showBall && i < trailPos.length && i > 0; if (m.visible) m.position.copy(trailPos[i]); });

  // you: cut-out video lined up so your real hand sits on the virtual paddle
  you.visible = myPaddle.visible = marker.visible = inGame;
  if (inGame && S) {
    const bd = S.body;
    const W = BODY.shoulder_world / Math.max(bd.w, 0.05), H = W * 0.75;
    const sx = (bd.u - 0.5) * BODY.body_x_gain, sy = TABLE_Y + BODY.shoulder_y;
    youFit.w = lerp(youFit.w, W, 0.25);
    youFit.cx = lerp(youFit.cx, sx - (bd.u - 0.5) * W, 0.35);
    youFit.cy = lerp(youFit.cy, sy - (0.5 - bd.v) * H, 0.35);
    you.scale.set(youFit.w, youFit.w * 0.75, 1);
    you.position.set(youFit.cx, youFit.cy, YOU_Z);
    const op = youMat.uniforms.opacity;
    op.value = lerp(op.value, S.paddle.visible ? 0.85 : 0.3, 0.1);

    mySwingT = Math.min(1, mySwingT + dt / 0.3);
    const sw = Math.sin(mySwingT * Math.PI);
    myPaddle.position.set(S.paddle.x, TABLE_Y + S.paddle.y, PADDLE_Z - sw * 0.25);
    myPaddle.rotation.set(-0.3 - sw * 1.2, 0, -0.4 + sw * 0.5);

    const incoming = b && b.visible && (state === "rally") && b.z > -0.2;
    const aligned = Math.abs(S.paddle.x - (b?.x ?? 0)) <= T.hit_rx;
    marker.position.set(Math.max(-T.half_w, Math.min(T.half_w, S.paddle.x)), TABLE_Y + 0.004, T.half_l - 0.32);
    marker.scale.x = T.hit_rx * 2;
    marker.material.color.set(!S.paddle.visible ? 0x888888 : incoming ? (aligned ? 0x06d6a0 : 0xef476f) : 0xffffff);
    marker.material.opacity = incoming ? 0.55 : 0.25;
  }

  // scoreboard flip (squash + re-grow when the score changes)
  scoreboard.visible = inGame;
  sbFlip = Math.min(1, sbFlip + dt / 0.4);
  scoreboard.scale.y = 0.15 + 0.85 * Math.abs(Math.cos(sbFlip * Math.PI)) ** 0.5;

  // speech bubble follows the opponent's head
  const now = performance.now();
  if (now > bubbleUntil || !inGame || !S?.opp) bubble.classList.add("hidden");
  else {
    const c = chars[S.opp];
    tmp.set(c.group.position.x, c.group.position.y + HEAD_HEIGHT[S.opp] + 0.15, c.group.position.z).project(camera);
    bubble.style.left = `${(tmp.x * 0.5 + 0.5) * innerWidth}px`;
    bubble.style.top = `${(-tmp.y * 0.5 + 0.5) * innerHeight}px`;
  }
  if (now > bannerUntil) banner.classList.add("hidden");
  if (now > subUntil) subBanner.classList.add("hidden");

  updateSparks(dt);
  updateCrowd(t, dt);
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

buildCards();
drawScoreboard();
connect();
requestAnimationFrame(frame);
