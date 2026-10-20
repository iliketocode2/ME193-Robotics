// Sumo Control dashboard. Talks to sumo_brain.py over a websocket on page port + 1:
//   in:  JSON {type: "hello" | "state" | "chat"}, binary "F"+JPEG video frames
//   out: JSON {cmd: ...} -- estop, start, pause, reset, strategy, manual, target,
//        clear_target, snapshot_arena, say, frame_ack (flow control), and from the
//        Advanced drawer: calibrate_drive, dry_run, refit_ring, clear_arena, test_song, hsv

const $ = (id) => document.getElementById(id);
const canvas = $("video");
const ctx = canvas.getContext("2d");

let ws = null;
let state = null;
let frameBitmap = null;
let dirty = true;                          // redraw only when a frame or state actually changed
let treeNames = [];
const MANUAL_SPEED = 70;                   // % for WASD driving
let topicRoot = "";
let refereeTopic = "ME193/Rogers";

// ------------------------------------------------------------------ websocket
function connect() {
  ws = new WebSocket(`ws://${location.hostname}:${Number(location.port) + 1}`);
  ws.binaryType = "arraybuffer";
  ws.onclose = () => { setPill("mqtt", false, "Brain offline"); setTimeout(connect, 1000); };
  ws.onmessage = async (ev) => {
    if (typeof ev.data !== "string") {
      const bytes = new Uint8Array(ev.data);
      if (bytes[0] === 70) {               // "F"
        try {
          const bmp = await createImageBitmap(new Blob([bytes.subarray(1)], { type: "image/jpeg" }));
          if (frameBitmap) frameBitmap.close();
          frameBitmap = bmp;
          dirty = true;
          $("videoEmpty").classList.add("hidden");
        } finally {
          send({ cmd: "frame_ack" });      // ready for the next frame
        }
      }
      return;
    }
    const msg = JSON.parse(ev.data);
    if (msg.type === "hello") onHello(msg);
    else if (msg.type === "state") onState(msg);
    else if (msg.type === "chat") addChat(msg);
  };
}
function send(obj) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); }

function onHello(msg) {
  topicRoot = msg.topic_root;
  refereeTopic = msg.referee;
  $("team").textContent = `${msg.team} · ${msg.topic_root}`;
  treeNames = msg.tree;
  $("chat").innerHTML = "";
  msg.chat.forEach(addChat);
}

// ------------------------------------------------------------------ state -> DOM
function setPill(name, ok, label) {
  const el = document.querySelector(`.pill[data-link="${name}"]`);
  if (!el) return;
  el.classList.toggle("ok", !!ok);
  if (label !== undefined) el.textContent = label;
}

const STRAT_NAMES = [[0.2, "Defensive"], [0.45, "Careful"], [0.6, "Balanced"], [0.85, "Bold"], [1.01, "Aggressive"]];

function onState(s) {
  state = s;
  dirty = true;
  const L = s.links;
  setPill("mqtt", L.mqtt, "MQTT");
  const missing = [!L.motor && "motor", !L.color && "color sensor"].filter(Boolean);
  setPill("robot", L.robot && !missing.length,
          !L.robot ? "Robot · offline" : missing.length ? `Robot · no ${missing.join(", ")}` : "Robot");
  setPill("camera", L.camera, L.camera ? `Camera · ${L.fps} fps` : "Camera · no field");
  setPill("fan", s.fan_rpm > 0, s.fan_rpm == null ? "Fan · ?" : `Fan · ${s.fan_rpm} rpm`);
  const w = s.world;
  const field = w.field ? (w.field_kind === "closed" ? "field ✓" : `field: partial ${Math.round((w.field_quality || 0) * 100)}%`)
                        : (w.field_pending ? "field: confirming…" : "no field");
  $("camMeta").textContent = `${field} · ${L.mode.replace(/^YOLO /, "")}${L.vision_ms != null ? ` · ${L.vision_ms} ms` : ""}`;

  const m = $("match");
  m.textContent = s.match === "idle" ? "WAITING" : s.match.toUpperCase();
  m.className = `match ${s.match}`;

  $("action").textContent = s.action || "—";
  const active = treeNames[s.bt.indexOf("active")];
  $("branch").textContent = [active && `branch: ${active}`, s.dry_run && "DRY RUN"].filter(Boolean).join(" · ");
  const card = $("actionCard");
  card.classList.toggle("danger", /E-STOP|PUSHED OUT|ESCAPE|FAILED/.test(s.action));
  card.classList.toggle("win", s.match === "won");
  setBar("L", s.sent[0]);
  setBar("R", s.sent[1]);

  const sn = s.sensor;
  $("swatchName").textContent = sn.ok ? (sn.name || "?") : "no data";
  if (sn.ok && sn.rgb && sn.rgb[0] != null) {
    const mx = Math.max(1, ...sn.rgb);
    const [r, g, b] = sn.rgb.map((v) => Math.round((v / mx) * 255 * Math.min(1, (sn.reflection || 50) / 60 + 0.3)));
    $("swatch").style.background = `rgb(${r},${g},${b})`;
  }
  $("edgeWarn").classList.toggle("on", !!sn.on_blue);

  $("estop").classList.toggle("on", s.estop);
  $("estop").textContent = s.estop ? "E-STOPPED (click to release)" : "E-STOP";
  if (document.activeElement !== $("strategy")) $("strategy").value = s.strategy;
  $("stratLabel").textContent = STRAT_NAMES.find(([t]) => s.strategy < t)[1];
  const age = w.arena_ref_age;
  $("arenaStatus").textContent = age == null ? "no snapshot yet (take one with the field empty)" : `snapshot ✓ ${age}s ago`;
  $("arenaStatus").classList.toggle("ok", age != null);
  $("calStatus").textContent = s.calibrating ? "calibrating… keep clear" : `drive: ${s.drive_cal}`;
  $("btnCalib").disabled = !!s.calibrating;
  $("dry").checked = s.dry_run;
  syncHsv(s.hsv);
}

function setBar(side, v) {
  const fill = $(`bar${side}`);
  const pct = Math.min(100, Math.abs(v)) / 2;
  fill.style.width = `${pct}%`;
  fill.style.left = v >= 0 ? "50%" : `${50 - pct}%`;
  fill.classList.toggle("neg", v < 0);
  $(`val${side}`).textContent = v > 0 ? `+${v}` : `${v}`;
}

// ------------------------------------------------------------------ video + overlays
function draw() {
  requestAnimationFrame(draw);
  if (!frameBitmap) return;
  const pulsing = state && state.world.target_px;
  if (!dirty && !pulsing) return;
  dirty = false;
  // canvas uses VISION pixel coordinates (state.frame); the video JPEG is smaller and gets scaled up
  const [W, H] = state && state.frame ? state.frame : [frameBitmap.width, frameBitmap.height];
  if (canvas.width !== W || canvas.height !== H) {
    canvas.width = W; canvas.height = H;
    const wrap = $("videoWrap");
    wrap.style.aspectRatio = `${W} / ${H}`;
    wrap.style.width = `min(100%, calc(52vh * ${W / H}))`;
  }
  ctx.drawImage(frameBitmap, 0, 0, W, H);
  if (!state) return;
  const w = state.world;
  const k = W / 960;                       // line widths scale with frame size

  if (w.field) {
    // the field's real outline (any shape); the edge-limit and "out" lines are the
    // same outline scaled toward the center -- exactly the r = const lines the brain uses
    const F = w.field;
    const outline = (scale) => {
      ctx.beginPath();
      F.poly.forEach(([x, y], i) => {
        const px = F.cx + (x - F.cx) * scale, py = F.cy + (y - F.cy) * scale;
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      });
      ctx.closePath();
    };
    const line = (scale, color, width, dash) => {
      ctx.save();
      ctx.setLineDash(dash.map((d) => d * k));
      ctx.strokeStyle = color; ctx.lineWidth = width * k;
      outline(scale); ctx.stroke(); ctx.restore();
    };
    ctx.save();
    ctx.fillStyle = "rgba(59,130,246,0.07)";
    outline(1); ctx.fill();
    ctx.restore();
    line(1, "rgba(59,130,246,0.95)", 3, []);
    line(w.edge, "rgba(251,191,36,0.85)", 2, [10, 8]);
    line(w.out_r, "rgba(244,63,94,0.5)", 1.5, [4, 8]);
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.beginPath(); ctx.arc(F.cx, F.cy, 4 * k, 0, Math.PI * 2); ctx.fill();
  } else {
    const msg = w.field_pending ? "Field found - confirming…" : "NO FIELD - need blue tape enclosing the arena";
    ctx.save();
    ctx.font = `800 ${22 * k}px Inter, sans-serif`;
    const tw = ctx.measureText(msg).width;
    ctx.fillStyle = "rgba(7,9,15,0.7)";
    ctx.beginPath(); ctx.roundRect(W / 2 - tw / 2 - 16 * k, 18 * k, tw + 32 * k, 40 * k, 10 * k); ctx.fill();
    ctx.fillStyle = w.field_pending ? "#fbbf24" : "#f43f5e";
    ctx.fillText(msg, W / 2 - tw / 2, 46 * k);
    ctx.restore();
  }

  for (const r of w.robots) {
    if (r.ours) continue;
    box(r.box, "rgba(148,163,214,0.6)", `candidate ${Math.round(r.conf * 100)}%`, 1.5);
  }
  // every non-our AprilTag: red outline if it's the opponent's, grey otherwise (e.g. a robot waiting outside)
  for (const t of w.tags || []) {
    const isOpp = w.opp && w.opp.tag_id === t.id;
    ctx.save();
    ctx.strokeStyle = isOpp ? "#f43f5e" : "rgba(148,163,214,0.8)"; ctx.lineWidth = 2 * k;
    ctx.beginPath(); t.corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.stroke();
    if (!isOpp) {
      ctx.font = `700 ${12 * k}px Inter, sans-serif`; ctx.fillStyle = "rgba(148,163,214,0.95)";
      ctx.fillText(`tag #${t.id}`, t.corners[0][0], t.corners[0][1] - 6 * k);
    }
    ctx.restore();
  }
  if (w.opp) {
    const tagTxt = w.opp.tag_id != null ? ` · tag #${w.opp.tag_id}` : "";
    box(w.opp.box, "#f43f5e", `OPPONENT${tagTxt}${w.opp.r != null ? `  r=${w.opp.r.toFixed(2)}` : ""}`, 3);
  }

  if (w.us) {
    const u = w.us;
    const usLabel = `US${u.r != null ? `  r=${u.r.toFixed(2)}` : ""}`;
    if (u.box) box(u.box, "#22d3ee", usLabel, 3);
    else {
      const xs = u.corners.map((p) => p[0]), ys = u.corners.map((p) => p[1]);
      tag(Math.min(...xs), Math.min(...ys) - 6 * k, "#22d3ee", usLabel);
    }
    ctx.save();
    ctx.strokeStyle = "#22d3ee"; ctx.lineWidth = 2 * k;
    ctx.beginPath(); u.corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.stroke();
    const len = Math.max(50 * k, u.size * 1.6);
    arrow(u.x, u.y, u.x + u.hx * len, u.y + u.hy * len, "#22d3ee", 4 * k);
    ctx.restore();
    if (w.goal_px) {
      ctx.save(); ctx.setLineDash([8 * k, 8 * k]);
      arrow(u.x, u.y, w.goal_px[0], w.goal_px[1], "rgba(255,255,255,0.85)", 2 * k);
      ctx.restore();
    }
  }
  if (w.target_px) {
    const [x, y] = w.target_px, t = performance.now() / 300;
    ctx.save(); ctx.strokeStyle = "#a78bfa"; ctx.lineWidth = 2.5 * k;
    ctx.beginPath(); ctx.arc(x, y, (12 + 3 * Math.sin(t)) * k, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - 20 * k, y); ctx.lineTo(x + 20 * k, y); ctx.moveTo(x, y - 20 * k); ctx.lineTo(x, y + 20 * k); ctx.stroke();
    ctx.restore();
  }

  function box([x1, y1, x2, y2], color, label, width) {
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = width * k;
    roundRect(x1, y1, x2 - x1, y2 - y1, 8 * k); ctx.stroke();
    ctx.restore();
    tag(x1, y1, color, label);
  }
  function tag(x1, y1, color, label) {
    ctx.save();
    ctx.font = `700 ${13 * k}px Inter, sans-serif`;
    const tw = ctx.measureText(label).width + 12 * k;
    ctx.fillStyle = color; roundRect(x1, y1 - 22 * k, tw, 20 * k, 6 * k); ctx.fill();
    ctx.fillStyle = "#06080e"; ctx.fillText(label, x1 + 6 * k, y1 - 7 * k);
    ctx.restore();
  }
  function roundRect(x, y, w2, h2, r) {
    ctx.beginPath(); ctx.roundRect(x, y, w2, h2, r);
  }
  function arrow(x1, y1, x2, y2, color, width) {
    const ang = Math.atan2(y2 - y1, x2 - x1), head = 12 * k;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(ang - 0.45), y2 - head * Math.sin(ang - 0.45));
    ctx.lineTo(x2 - head * Math.cos(ang + 0.45), y2 - head * Math.sin(ang + 0.45));
    ctx.closePath(); ctx.fill();
  }
}

function canvasPoint(e) {
  const r = canvas.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * canvas.width, y: ((e.clientY - r.top) / r.height) * canvas.height };
}
canvas.addEventListener("click", (e) => send({ cmd: "target", ...canvasPoint(e) }));
canvas.addEventListener("contextmenu", (e) => { e.preventDefault(); send({ cmd: "clear_target" }); });

// ------------------------------------------------------------------ controls
$("estop").onclick = () => send({ cmd: "estop", on: !(state && state.estop) });
$("btnStart").onclick = () => send({ cmd: "start", source: "dashboard" });
$("btnPause").onclick = () => send({ cmd: "pause" });
$("btnReset").onclick = () => send({ cmd: "reset" });
$("btnRefit").onclick = () => send({ cmd: "refit_ring" });
$("btnArena").onclick = () => send({ cmd: "snapshot_arena" });
$("btnCalib").onclick = () => send({ cmd: "calibrate_drive" });
$("btnArenaClear").onclick = () => send({ cmd: "clear_arena" });
$("btnWin").onclick = () => send({ cmd: "test_song", song: "win" });
$("btnLose").onclick = () => send({ cmd: "test_song", song: "lose" });
$("strategy").oninput = (e) => send({ cmd: "strategy", value: Number(e.target.value) });
$("dry").onchange = (e) => send({ cmd: "dry_run", on: e.target.checked });

// HSV calibration sliders (OpenCV ranges: H 0-179, S/V 0-255)
const HSV_ROWS = [["H low", "low", 0, 179], ["H high", "high", 0, 179], ["S low", "low", 1, 255],
                  ["S high", "high", 1, 255], ["V low", "low", 2, 255], ["V high", "high", 2, 255]];
let hsvLocal = null, hsvTouched = 0;
$("hsv").innerHTML = HSV_ROWS.map(([label, , , max], i) =>
  `<span>${label}</span><input type="range" min="0" max="${max}" data-i="${i}"><b id="hsvv${i}">-</b>`).join("");
$("hsv").querySelectorAll("input").forEach((inp) => {
  inp.oninput = () => {
    const [, which, ch] = HSV_ROWS[inp.dataset.i];
    hsvLocal[which][ch] = Number(inp.value);
    $(`hsvv${inp.dataset.i}`).textContent = inp.value;
    hsvTouched = performance.now();
    send({ cmd: "hsv", low: hsvLocal.low, high: hsvLocal.high });
  };
});
function syncHsv(hsv) {
  if (performance.now() - hsvTouched < 1500) return;     // don't fight the user's drag
  hsvLocal = { low: [...hsv.low], high: [...hsv.high] };
  $("hsv").querySelectorAll("input").forEach((inp) => {
    const [, which, ch] = HSV_ROWS[inp.dataset.i];
    inp.value = hsvLocal[which][ch];
    $(`hsvv${inp.dataset.i}`).textContent = inp.value;
  });
}

// ------------------------------------------------------------------ keyboard teleop
const held = new Set();
let manualTimer = null;
function typing() { return document.activeElement && document.activeElement.tagName === "INPUT" && document.activeElement.type === "text"; }
const KEYMAP = { w: "w", arrowup: "w", s: "s", arrowdown: "s", a: "a", arrowleft: "a", d: "d", arrowright: "d" };

document.addEventListener("keydown", (e) => {
  if (typing()) return;
  const key = e.key.toLowerCase();
  if (key === " ") { e.preventDefault(); $("estop").click(); return; }
  const k = KEYMAP[key];
  if (!k) return;
  e.preventDefault();
  held.add(k);
  if (!manualTimer) { sendManual(); manualTimer = setInterval(sendManual, 100); }
});
document.addEventListener("keyup", (e) => {
  const k = KEYMAP[e.key.toLowerCase()];
  if (!k) return;
  held.delete(k);
  if (!held.size) stopManual();
});
window.addEventListener("blur", () => { held.clear(); stopManual(); });

function sendManual() {
  const sp = MANUAL_SPEED;
  const fwd = (held.has("w") ? 1 : 0) - (held.has("s") ? 1 : 0);
  const turn = (held.has("d") ? 1 : 0) - (held.has("a") ? 1 : 0);
  let l = fwd * sp + turn * sp * 0.6, r = fwd * sp - turn * sp * 0.6;
  const m = Math.max(100, Math.abs(l), Math.abs(r)) / 100;
  send({ cmd: "manual", l: Math.round(l / m), r: Math.round(r / m) });
}
function stopManual() {
  clearInterval(manualTimer); manualTimer = null;
  send({ cmd: "manual", release: true });
}

// ------------------------------------------------------------------ chat
function addChat(c) {
  const list = $("chat");
  const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 40;
  const li = document.createElement("li");
  if (c.hi) li.classList.add("hi");
  let kind = "", label = c.topic;
  if (c.topic === "system") kind = "system";
  else if (c.topic === refereeTopic) kind = "referee";
  else if (topicRoot && c.topic.startsWith(topicRoot)) { kind = "ours"; label = c.topic.slice(topicRoot.length + 1) || c.topic; }
  if (/MATCH START/.test(c.text) || (kind === "referee" && /start/i.test(c.text))) li.classList.add("start");
  const t = new Date(c.ts * 1000).toLocaleTimeString([], { hour12: false });
  li.innerHTML = `<div class="meta"><span class="chip ${kind}"></span><time>${t}</time></div><div class="txt"></div>`;
  li.querySelector(".chip").textContent = label;
  li.querySelector(".txt").textContent = c.text;
  list.appendChild(li);
  while (list.children.length > 300) list.firstChild.remove();
  if (atBottom) list.scrollTop = list.scrollHeight;
}
$("sayForm").onsubmit = (e) => {
  e.preventDefault();
  const text = $("sayText").value.trim();
  if (text) send({ cmd: "say", text });
  $("sayText").value = "";
};

connect();
requestAnimationFrame(draw);
