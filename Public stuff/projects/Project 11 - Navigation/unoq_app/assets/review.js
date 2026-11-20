// Review page: list walks, fix each walk's maneuver labels on a heading plot, train, export.

const COLORS = { left: '--left', right: '--right', stairs: '--stairs', arrive: '--arrive' };
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

let walks = [];
let walk = null;            // the walk being edited (from the board)
let marks = [];             // its maneuvers being edited: [{t, action}]
let selected = -1;
let dragging = false;
let dirty = false;

// ---------- walk list ----------

document.addEventListener('board-connected', () => board.send_message('list_walks', {}));

board.on_message('walks', (d) => {
  walks = d.walks;
  const b = d.budget;
  $('#budget').innerHTML = `Route <b>${escapeHtml(d.route)}</b>: <b>${b.used}/${b.limit}</b> training walks used` +
    (b.recorded > b.used ? ` <span class="muted">(${b.recorded - b.used} excluded)</span>` : '') +
    (b.used > b.limit ? ' <b style="color: var(--bad)">over the limit</b>' : '');
  $('#no-walks').classList.toggle('hidden', walks.length > 0);
  $('#walks').innerHTML = walks.slice().reverse().map((w) => {
    const when = new Date(w.start * 1000);
    const training = ['demo', 'dagger'].includes(w.kind)
      ? (!w.reviewed ? '<span style="color: var(--warn)">needs review</span>' : w.include ? 'yes' : 'excluded')
      : '—';
    return `<tr data-id="${escapeHtml(w.id)}" class="${walk && walk.id === w.id ? 'selected' : ''}">
      <td>${when.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} ${when.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</td>
      <td>${escapeHtml(w.walker)}</td><td><span class="tag ${w.kind}">${w.kind}${w.activity ? ': ' + escapeHtml(w.activity) : ''}</span></td>
      <td>${w.meters} m · ${w.steps} steps</td><td>${w.maneuvers}</td>
      <td>${w.reviewed ? '✓' : ''}</td><td>${training}</td></tr>`;
  }).join('');
  $$('#walks tr').forEach((tr) => tr.addEventListener('click', () => {
    if (dirty && !confirm('Discard unsaved changes to this walk?')) return;
    board.send_message('get_walk', { id: tr.dataset.id });
  }));
});

// ---------- one walk ----------

board.on_message('walk', (w) => {
  walk = w;
  marks = w.maneuvers.map((m) => ({ ...m }));
  selected = -1;
  dirty = false;
  $('#editor').classList.remove('hidden');
  const when = new Date(w.start * 1000).toLocaleString();
  $('#walk-title').textContent = `${w.walker} · ${w.kind} · ${when} · sensing ${w.sensing}`;
  $('#include').checked = w.include;
  $('#mps').value = w.meters_per_step;
  const btn = $('#use-walker-mps');
  if (w.walker_mps && Math.abs(w.walker_mps - w.meters_per_step) > 0.005) {
    btn.textContent = `Use ${w.walker}'s calibration (${w.walker_mps.toFixed(2)})`;
    btn.classList.remove('hidden');
  } else btn.classList.add('hidden');
  $$('#walks tr').forEach((tr) => tr.classList.toggle('selected', tr.dataset.id === w.id));
  updateButtons();
  draw();
  if (!w.steps.length) toast('This walk has no steps. Check the IMU and the step detector.', true);
});

$('#use-walker-mps').addEventListener('click', () => {
  $('#mps').value = walk.walker_mps;
  dirty = true;
});
$('#mps').addEventListener('change', () => (dirty = true));
$('#include').addEventListener('change', () => (dirty = true));

$('#save').addEventListener('click', () => {
  if (!walk) return;
  board.send_message('save_review', {
    id: walk.id, maneuvers: marks, include: $('#include').checked,
    meters_per_step: parseFloat($('#mps').value) || null,
  });
  dirty = false;
  walk.reviewed = true;
});

$('#reset-taps').addEventListener('click', () => {
  marks = walk.taps.map((m) => ({ ...m }));
  selected = -1;
  dirty = true;
  updateButtons();
  draw();
});

$('#delete-marker').addEventListener('click', deleteSelected);
document.addEventListener('keydown', (e) => {
  if ((e.key === 'Delete' || e.key === 'Backspace') && selected >= 0 && document.activeElement.tagName !== 'INPUT') {
    e.preventDefault();
    deleteSelected();
  }
});

function deleteSelected() {
  if (selected < 0) return;
  marks.splice(selected, 1);
  selected = -1;
  dirty = true;
  updateButtons();
  draw();
}

function updateButtons() {
  $('#delete-marker').disabled = selected < 0;
}

// meters walked at time t (interpolated between steps)
function metersAt(t) {
  const s = walk.steps;
  if (!s.length) return 0;
  if (t <= s[0].t) return s[0].m;
  for (let i = 1; i < s.length; i++) {
    if (s[i].t >= t) {
      const f = (t - s[i - 1].t) / (s[i].t - s[i - 1].t || 1);
      return s[i - 1].m + f * (s[i].m - s[i - 1].m);
    }
  }
  return s[s.length - 1].m;
}

$('#snap-all').addEventListener('click', () => {
  let moved = 0;
  for (const m of marks) {
    if (m.action !== 'left' && m.action !== 'right') continue;
    const want = m.action === 'left' ? 1 : -1;
    const at = metersAt(m.t);
    let best = null;
    for (const d of walk.detected_turns) {
      if (d.sign !== want) continue;
      const dist = Math.abs(metersAt(d.t) - at);
      if (dist <= walk.snap_window_m && (!best || dist < best.dist)) best = { dist, t: d.t };
    }
    if (best && best.t !== m.t) {
      m.t = best.t;
      moved++;
    }
  }
  dirty = dirty || moved > 0;
  toast(moved ? `Moved ${moved} turn(s) to where the heading starts to change.` : 'Nothing to snap: no detected turn of the right direction nearby.');
  draw();
});

// ---------- plot ----------

const canvas = $('#plot');
const PAD = { l: 44, r: 12, t: 26, b: 34 };

function geometry() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  const t0 = walk.start;
  const t1 = Math.max(walk.end || 0, walk.steps.length ? walk.steps[walk.steps.length - 1].t : t0 + 1, t0 + 1);
  let hmin = 0;
  let hmax = 0;
  for (const s of walk.steps) {
    hmin = Math.min(hmin, s.h);
    hmax = Math.max(hmax, s.h);
  }
  const span = Math.max(90, hmax - hmin);
  hmin -= span * 0.08;
  hmax += span * 0.08;
  return {
    w, h, t0, t1,
    x: (t) => PAD.l + ((t - t0) / (t1 - t0)) * (w - PAD.l - PAD.r),
    t: (x) => t0 + ((x - PAD.l) / (w - PAD.l - PAD.r)) * (t1 - t0),
    y: (v) => PAD.t + (1 - (v - hmin) / (hmax - hmin)) * (h - PAD.t - PAD.b),
    hmin, hmax,
  };
}

function draw() {
  if (!walk) return;
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth;
  const ch = canvas.clientHeight;
  if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
    canvas.width = cw * dpr;
    canvas.height = ch * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const g = geometry();
  ctx.font = '11px system-ui';
  ctx.textBaseline = 'middle';

  // heading grid every 90 degrees
  ctx.strokeStyle = css('--line');
  ctx.fillStyle = css('--muted');
  ctx.lineWidth = 1;
  for (let v = Math.ceil(g.hmin / 90) * 90; v <= g.hmax; v += 90) {
    ctx.beginPath();
    ctx.moveTo(PAD.l, g.y(v));
    ctx.lineTo(g.w - PAD.r, g.y(v));
    ctx.stroke();
    ctx.textAlign = 'right';
    ctx.fillText(`${v}°`, PAD.l - 6, g.y(v));
  }

  // time axis (seconds) and meters every 20 m
  ctx.textAlign = 'center';
  const total = g.t1 - g.t0;
  const every = total > 300 ? 60 : total > 120 ? 30 : 10;
  for (let s = 0; s <= total; s += every) ctx.fillText(fmtTime(s), g.x(g.t0 + s), g.h - 9);
  let nextM = 20;
  for (const s of walk.steps) {
    if (s.m >= nextM) {
      ctx.fillText(`${nextM} m`, g.x(s.t), g.h - 23);
      nextM += 20;
    }
  }

  // steps as ticks
  ctx.strokeStyle = css('--muted');
  for (const s of walk.steps) {
    const x = g.x(s.t);
    ctx.beginPath();
    ctx.moveTo(x, g.h - PAD.b);
    ctx.lineTo(x, g.h - PAD.b + 4);
    ctx.stroke();
  }

  // detected turns
  ctx.fillStyle = css('--warn');
  ctx.globalAlpha = 0.25;
  for (const d of walk.detected_turns) ctx.fillRect(g.x(d.t) - 3, PAD.t, 6, g.h - PAD.t - PAD.b);
  ctx.globalAlpha = 1;

  // heading line
  ctx.strokeStyle = css('--text');
  ctx.lineWidth = 2;
  ctx.beginPath();
  walk.steps.forEach((s, i) => (i ? ctx.lineTo(g.x(s.t), g.y(s.h)) : ctx.moveTo(g.x(s.t), g.y(s.h))));
  ctx.stroke();

  // original taps (dashed)
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  for (const m of walk.taps) {
    ctx.strokeStyle = css(COLORS[m.action]);
    const x = g.x(m.t);
    ctx.beginPath();
    ctx.moveTo(x, PAD.t);
    ctx.lineTo(x, g.h - PAD.b);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // policy cues and wrong-cue taps along the top
  for (const e of walk.events) {
    const x = g.x(e.t);
    if (e.type === 'cue') {
      const [m, when] = e.action.split('_');
      ctx.fillStyle = css(COLORS[m] || '--muted');
      ctx.globalAlpha = when === 'soon' ? 0.5 : 1;
      ctx.beginPath();
      ctx.moveTo(x - 5, 4);
      ctx.lineTo(x + 5, 4);
      ctx.lineTo(x, 13);
      ctx.fill();
      ctx.globalAlpha = 1;
    } else if (e.type === 'wrong_cue') {
      ctx.fillStyle = css('--bad');
      ctx.textAlign = 'center';
      ctx.font = 'bold 13px system-ui';
      ctx.fillText('✕', x, 9);
      ctx.font = '11px system-ui';
    }
  }

  // maneuvers being edited
  marks.forEach((m, i) => {
    const x = g.x(m.t);
    ctx.strokeStyle = css(COLORS[m.action]);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = i === selected ? 4 : 2;
    ctx.beginPath();
    ctx.moveTo(x, PAD.t - 6);
    ctx.lineTo(x, g.h - PAD.b);
    ctx.stroke();
    ctx.textAlign = x > g.w - 70 ? 'right' : 'left';
    ctx.font = (i === selected ? 'bold ' : '') + '11px system-ui';
    ctx.fillText(MANEUVER_NAMES[m.action], x + (x > g.w - 70 ? -5 : 5), PAD.t + 8);
  });
}

function hit(x) {
  const g = geometry();
  let best = -1;
  let bestD = 12;
  marks.forEach((m, i) => {
    const d = Math.abs(g.x(m.t) - x);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

function eventX(e) {
  return e.clientX - canvas.getBoundingClientRect().left;
}

canvas.addEventListener('pointerdown', (e) => {
  if (!walk) return;
  const x = eventX(e);
  const i = hit(x);
  const action = $('#add-action').value;
  if (i >= 0) {
    selected = i;
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
  } else if (action) {
    const g = geometry();
    marks.push({ t: Math.min(Math.max(g.t(x), g.t0), g.t1), action });
    marks.sort((a, b) => a.t - b.t);
    selected = marks.findIndex((m) => m.action === action && Math.abs(g.x(m.t) - x) < 1);
    dirty = true;
  } else {
    selected = -1;
  }
  updateButtons();
  draw();
});

canvas.addEventListener('pointermove', (e) => {
  if (!dragging || selected < 0) return;
  const g = geometry();
  marks[selected].t = Math.min(Math.max(g.t(eventX(e)), g.t0), g.t1);
  dirty = true;
  draw();
});

canvas.addEventListener('pointerup', () => {
  if (!dragging) return;
  dragging = false;
  const moved = marks[selected];
  marks.sort((a, b) => a.t - b.t);
  selected = marks.indexOf(moved);
  draw();
});

window.addEventListener('resize', draw);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
window.addEventListener('beforeunload', (e) => {
  if (dirty) {
    e.preventDefault();
    e.returnValue = '';
  }
});

// ---------- train / reprocess / export ----------

$('#train').addEventListener('click', () => {
  if (dirty) return toast('Save the review of this walk first.', true);
  board.send_message('train', {});
  toast('Training…');
});

$('#reprocess').addEventListener('click', () => {
  if (confirm('Re-run step and heading detection over every walk\'s raw IMU with the current sensing.py? Labels are kept; check them afterwards.')) {
    board.send_message('reprocess', {});
  }
});

$('#export').addEventListener('click', () => {
  board.send_message('export', {});
  toast('Preparing export…');
});
$('#export-imu').addEventListener('click', () => {
  board.send_message('export', { imu: true });
  toast('Preparing export with raw IMU (can take a few seconds)…');
});

board.on_message('export', (d) => {
  const blob = new Blob([JSON.stringify(d)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `navigator_export_${new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-')}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
});

board.on_message('train_result', (d) => {
  $('#report-card').classList.remove('hidden');
  const r = d.report;
  const counts = Object.entries(d.info.class_counts).filter(([, n]) => n).map(([c, n]) => `${c} ${n}`).join(', ');
  let html = `<p class="muted">Trained on ${d.info.walks.length} walks, ${d.info.steps} steps (${counts}).</p>`;
  if (!r) {
    html += '<p class="muted">Record and review at least 2 walks to get a held-out score.</p>';
  } else {
    const pct = (v) => (v === null ? '—' : `${Math.round(v * 100)}%`);
    html += `<p>Each walk was held out in turn, the policy trained on the others, and the held-out walk replayed through the live cue logic.</p>
      <div class="stats" style="margin-bottom: 12px">
        <div><b>${pct(r.hit_rate)}</b><span>maneuvers cued</span></div>
        <div><b>${pct(r.heads_up_rate)}</b><span>with heads-up</span></div>
        <div><b>${r.mean_error_m ?? '—'} m</b><span>mean timing</span></div>
        <div><b>${r.false_alarms}</b><span>false "now"s</span></div>
      </div>
      <p class="muted">Timing: negative = cue came before the maneuver. Mean |error| ${r.mean_abs_error_m ?? '—'} m, standard deviation ${r.std_error_m ?? '—'} m.</p>
      <div class="tablewrap"><table><thead><tr><th>Held-out walk</th><th>Maneuvers (error m)</th><th>False alarms</th></tr></thead><tbody>` +
      r.per_walk.map((w) => `<tr><td>${escapeHtml(w.walk)}</td><td>${w.maneuvers.map((m) =>
        m.hit ? `${MANEUVER_NAMES[m.action]} ${m.error_m > 0 ? '+' : ''}${m.error_m}` : `<b style="color: var(--bad)">${MANEUVER_NAMES[m.action]} missed</b>`).join(' · ')}</td><td>${w.false_alarms}</td></tr>`).join('') +
      '</tbody></table></div>';
  }
  $('#report').innerHTML = html;
  board.send_message('list_walks', {});
});
