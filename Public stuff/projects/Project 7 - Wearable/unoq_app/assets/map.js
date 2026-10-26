// Feature map: every training window as a faint dot, your current window as a
// big moving dot. The x-axis starts as Model 1's only input; the y-axis starts
// as the Model 2 feature that best separates the pair Model 1 mixes up most.
// The strip underneath is the same points on x alone -- Model 1's whole view.

const LABELS = ['still', 'walk', 'run', 'bike'];
const NAME = { still: 'Still', walk: 'Walk', run: 'Run', bike: 'Bike' };
const SHAPE = { still: 'circle', walk: 'square', run: 'triangle', bike: 'diamond' };
const TRAIL = 12;                          // live points kept (6 s at one per 0.5 s)

const $ = (sel) => document.querySelector(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

let map = null;                            // {names, points, labels, x, y, pair}
let xi = 0;
let yi = 0;
const trail = [];                          // [{vec, label}] newest last
let drawn = [];                            // screen positions of training points, for hover

// ---------- drawing ----------

function marker(ctx, shape, x, y, r) {
  ctx.beginPath();
  if (shape === 'circle') ctx.arc(x, y, r, 0, Math.PI * 2);
  else if (shape === 'square') ctx.rect(x - r * 0.85, y - r * 0.85, r * 1.7, r * 1.7);
  else if (shape === 'triangle') {
    ctx.moveTo(x, y - r * 1.1);
    ctx.lineTo(x + r, y + r * 0.8);
    ctx.lineTo(x - r, y + r * 0.8);
    ctx.closePath();
  } else {
    ctx.moveTo(x, y - r * 1.15);
    ctx.lineTo(x + r, y);
    ctx.lineTo(x, y + r * 1.15);
    ctx.lineTo(x - r, y);
    ctx.closePath();
  }
}

function setup(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function range(values) {
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (hi - lo < 1e-6) { lo -= 1; hi += 1; }
  const pad = (hi - lo) * 0.06;
  return [lo - pad, hi + pad];
}

function niceTicks(lo, hi, n) {
  const step0 = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= step0);
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) ticks.push(+t.toFixed(6));
  return ticks;
}

function draw() {
  requestAnimationFrame(draw);
  const plot = setup($('#map'));
  const strip = setup($('#strip'));
  if (!map || !map.points) return;

  const P = map.points;
  const live = trail.map((t) => t.vec);
  const [x0, x1] = range([...P.map((p) => p[xi]), ...live.map((v) => v[xi])]);
  const [y0, y1] = range([...P.map((p) => p[yi]), ...live.map((v) => v[yi])]);
  const L = 52, R = 12, T = 10, B = 40;
  const sx = (v) => L + ((v - x0) / (x1 - x0)) * (plot.w - L - R);
  const sy = (v) => plot.h - B - ((v - y0) / (y1 - y0)) * (plot.h - T - B);
  const ctx = plot.ctx;

  // grid + axes
  ctx.font = '11px system-ui';
  ctx.lineWidth = 1;
  ctx.strokeStyle = css('--line');
  ctx.fillStyle = css('--muted');
  for (const t of niceTicks(x0, x1, 6)) {
    ctx.beginPath(); ctx.moveTo(sx(t), T); ctx.lineTo(sx(t), plot.h - B); ctx.stroke();
    ctx.textAlign = 'center'; ctx.fillText(t, sx(t), plot.h - B + 14);
  }
  for (const t of niceTicks(y0, y1, 5)) {
    ctx.beginPath(); ctx.moveTo(L, sy(t)); ctx.lineTo(plot.w - R, sy(t)); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(t, L - 6, sy(t) + 4);
  }
  ctx.textAlign = 'center';
  ctx.fillText(`${map.names[xi]}${xi === map.x ? '  (Model 1’s input)' : ''}`, L + (plot.w - L - R) / 2, plot.h - 8);
  ctx.save();
  ctx.translate(13, T + (plot.h - T - B) / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(map.names[yi], 0, 0);
  ctx.restore();

  // training points
  drawn = [];
  ctx.globalAlpha = 0.45;
  P.forEach((p, i) => {
    const lab = map.labels[i];
    const x = sx(p[xi]);
    const y = sy(p[yi]);
    drawn.push({ x, y, i });
    marker(ctx, SHAPE[lab], x, y, 4);
    ctx.fillStyle = css(`--${lab}`);
    ctx.fill();
  });
  ctx.globalAlpha = 1;

  // cluster labels at each activity's median
  ctx.font = '600 13px system-ui';
  ctx.textAlign = 'left';
  for (const lab of LABELS) {
    const idx = map.labels.map((l, i) => (l === lab ? i : -1)).filter((i) => i >= 0);
    if (!idx.length) continue;
    const med = (k) => idx.map((i) => P[i][k]).sort((a, b) => a - b)[Math.floor(idx.length / 2)];
    ctx.fillStyle = css('--text');
    ctx.fillText(NAME[lab], sx(med(xi)) + 10, sy(med(yi)) - 10);
  }

  // live trail: a line through the last few windows, newest dot biggest
  if (trail.length > 1) {
    ctx.beginPath();
    trail.forEach((t, k) => {
      if (k === 0) ctx.moveTo(sx(t.vec[xi]), sy(t.vec[yi]));
      else ctx.lineTo(sx(t.vec[xi]), sy(t.vec[yi]));
    });
    ctx.strokeStyle = css('--text');
    ctx.globalAlpha = 0.35;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  trail.forEach((t, k) => {
    const newest = k === trail.length - 1;
    const x = sx(t.vec[xi]);
    const y = sy(t.vec[yi]);
    ctx.globalAlpha = newest ? 1 : 0.15 + 0.5 * (k / trail.length);
    ctx.beginPath();
    ctx.arc(x, y, newest ? 11 : 5, 0, Math.PI * 2);
    ctx.fillStyle = t.label ? css(`--${t.label}`) : css('--text');
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = css('--card');
    ctx.stroke();
    if (newest) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = css('--text');
      ctx.beginPath(); ctx.arc(x, y, 14, 0, Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.font = '600 13px system-ui';
      ctx.fillStyle = css('--text');
      ctx.fillText('you', x + 18, y + 4);
    }
  });
  ctx.globalAlpha = 1;

  // Model 1's view: one row per activity, x only
  const rows = LABELS.filter((l) => map.labels.includes(l));
  const rowH = (strip.h - 6) / rows.length;
  const s = strip.ctx;
  s.font = '11px system-ui';
  rows.forEach((lab, r) => {
    const cy = 3 + rowH * (r + 0.5);
    s.fillStyle = css('--muted');
    s.textAlign = 'right';
    s.fillText(NAME[lab], L - 6, cy + 4);
    s.globalAlpha = 0.35;
    s.fillStyle = css(`--${lab}`);
    P.forEach((p, i) => {
      if (map.labels[i] === lab) s.fillRect(sx(p[xi]) - 1, cy - rowH * 0.32, 2, rowH * 0.64);
    });
    s.globalAlpha = 1;
  });
  if (trail.length) {
    const x = sx(trail[trail.length - 1].vec[xi]);
    s.strokeStyle = css('--text');
    s.lineWidth = 2;
    s.beginPath(); s.moveTo(x, 0); s.lineTo(x, strip.h); s.stroke();
  }
}
requestAnimationFrame(draw);

// ---------- hover ----------

$('#map').addEventListener('mousemove', (e) => {
  const rect = e.target.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  let best = null;
  let bestD = 144;                         // 12 px hit radius, bigger than the 4 px mark
  for (const d of drawn) {
    const dd = (d.x - mx) ** 2 + (d.y - my) ** 2;
    if (dd < bestD) { best = d; bestD = dd; }
  }
  const tip = $('#tip');
  if (!best) { tip.hidden = true; return; }
  const p = map.points[best.i];
  tip.innerHTML = `<b>${NAME[map.labels[best.i]]}</b> training window<br>${map.names[xi]}: ${p[xi]}<br>${map.names[yi]}: ${p[yi]}`;
  tip.style.left = `${best.x + 12}px`;
  tip.style.top = `${best.y + 12}px`;
  tip.hidden = false;
});
$('#map').addEventListener('mouseleave', () => { $('#tip').hidden = true; });

// ---------- controls ----------

function fillSelects() {
  for (const id of ['#sel-x', '#sel-y']) {
    $(id).innerHTML = map.names.map((n, i) => `<option value="${i}">${n}</option>`).join('');
  }
  $('#sel-x').value = xi;
  $('#sel-y').value = yi;
}

function explain() {
  const pair = map.pair ? `${NAME[map.pair[0]]} and ${NAME[map.pair[1]]}` : 'the activities';
  $('#explain').innerHTML = (xi === map.x && yi === map.y)
    ? `Each faint shape is one 2-second training window. The x-axis, <b>${map.names[xi]}</b>, is the only thing Model 1 ever looks at. ` +
      `The y-axis, <b>${map.names[yi]}</b>, is the one Model 2 feature that best pulls apart ${pair}, the pair Model 1 mixes up most. ` +
      'Clusters that overlap left-to-right but separate top-to-bottom are exactly where Model 2 beats Model 1. The big dot is you, right now.'
    : 'Custom axes. Each faint shape is one training window; the big dot is you. Press “Reset axes” for the Model 1 vs Model 2 view.';
}

$('#sel-x').addEventListener('change', (e) => { xi = +e.target.value; explain(); });
$('#sel-y').addEventListener('change', (e) => { yi = +e.target.value; explain(); });
$('#reset').addEventListener('click', () => { xi = map.x; yi = map.y; fillSelects(); explain(); });

function buildLegend() {
  $('#legend').innerHTML = LABELS.filter((l) => map.labels.includes(l)).map((l) => {
    const n = map.labels.filter((x) => x === l).length;
    return `<span class="lg-${SHAPE[l]}" style="--c: var(--${l})">${NAME[l]} · ${n}</span>`;
  }).join('') + '<span class="lg-you">you (colored by Model 2)</span>';
}

// ---------- socket ----------

const ui = new WebUI();

ui.on_connect(() => {
  $('#pill-link').textContent = 'Board: connected';
  $('#pill-link').className = 'pill good';
  ui.send_message('get_map');
});
ui.on_disconnect(() => {
  $('#pill-link').textContent = 'Board: disconnected';
  $('#pill-link').className = 'pill bad';
});

ui.on_message('map', (m) => {
  if (!m.points) {
    map = null;
    $('#explain').textContent = 'No training data yet: record some activities on the Dashboard and press Train.';
    return;
  }
  const first = map === null;
  map = m;
  if (first || xi >= m.names.length || yi >= m.names.length) { xi = m.x; yi = m.y; }
  fillSelects();
  buildLegend();
  explain();
});

ui.on_message('prediction', (p) => {
  if (!p.vec) return;
  trail.push({ vec: p.vec, label: p.m2 ? p.m2.label : null });
  if (trail.length > TRAIL) trail.shift();
  const m1 = p.m1 ? NAME[p.m1.label] : '—';
  const m2 = p.m2 ? NAME[p.m2.label] : '—';
  $('#now').innerHTML = `Right now: Model 1 says <b>${m1}</b>, Model 2 says <b>${m2}</b>` +
    (p.m1 && p.m2 && p.m1.label !== p.m2.label ? ' (they disagree: look where the dot sits on each view)' : '');
});
