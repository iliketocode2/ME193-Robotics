// Phone page: Navigate / Record / DAgger / Calibrate. Talks to nav_core.py.

const TABS = ['navigate', 'record', 'dagger', 'calibrate'];
const ARROWS = { straight: '↑', left: '↰', right: '↱', stairs: '🪜', arrive: '📍' };

let status = null;
let clockOffset = 0;          // board time - phone time, seconds
let lastCue = null;           // {action, text}
let tab = store.get('tab', 'navigate');
let taps = store.get('taps', []);   // [{id, walk, t, type, action, acked}] for the current walk

// ---------- tabs ----------

function showTab(name) {
  if (!TABS.includes(name)) name = 'navigate';
  tab = name;
  store.set('tab', name);
  $$('nav.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  $$('[data-panel]').forEach((p) => p.classList.toggle('hidden', p.dataset.panel !== name));
}
$$('nav.tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
showTab(tab);

// ---------- speech ----------

function speak(text) {
  if (!('speechSynthesis' in window) || !text) return;
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    window.speechSynthesis.speak(u);
  } catch (e) {
    /* no speech on this browser */
  }
}

// ---------- walker / route ----------

$('#walker').addEventListener('change', (e) => {
  if (e.target.value === '__add') {
    $('#new-walker-row').classList.remove('hidden');
    $('#new-walker').focus();
    return;
  }
  board.send_message('set_walker', { name: e.target.value });
});
$('#add-walker').addEventListener('click', () => {
  const name = $('#new-walker').value.trim();
  if (!name) return;
  board.send_message('set_walker', { name });
  $('#new-walker').value = '';
  $('#new-walker-row').classList.add('hidden');
  showTab('calibrate');
});
$('#route').addEventListener('change', (e) => board.send_message('set_route', { name: e.target.value }));

function renderWalkers(s) {
  const sel = $('#walker');
  const names = Object.keys(s.walkers || {});
  if (s.walker && !names.includes(s.walker)) names.push(s.walker);
  const html = (names.length ? '' : '<option value="">(none yet)</option>') +
    names.map((n) => `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`).join('') +
    '<option value="__add">+ Add walker…</option>';
  if (sel.dataset.html !== html) {
    sel.innerHTML = html;
    sel.dataset.html = html;
  }
  if (document.activeElement !== sel) sel.value = s.walker || '';
  const cal = s.walkers && s.walkers[s.walker];
  $('#walker-info').innerHTML = !s.walker ? 'Add a walker to start.'
    : cal ? `${s.meters_per_step.toFixed(2)} m per step (calibrated)`
    : `<span style="color: var(--warn)">Not calibrated: using ${s.meters_per_step.toFixed(2)} m per step. Do a calibration walk.</span>`;
  if (document.activeElement !== $('#route')) $('#route').value = s.route;
}

// ---------- start / stop ----------

$$('[data-start]').forEach((b) =>
  b.addEventListener('click', () => {
    speak(' ');      // iOS only allows speech after a tap; this unlocks it for the cues later
    taps = [];
    store.set('taps', taps);
    renderTaps();
    lastCue = null;
    renderCue(null);
    board.send_message('start', { mode: b.dataset.start, test: $('#save-test').checked });
  }),
);
$$('[data-stop]').forEach((b) => b.addEventListener('click', () => board.send_message('stop', {})));
$('#cal-stop').addEventListener('click', () =>
  board.send_message('stop', { distance_m: parseFloat($('#cal-distance').value) }),
);
$$('[data-repeat]').forEach((b) => b.addEventListener('click', () => lastCue && speak(lastCue.text)));

// ---------- taps (queued until the board confirms them) ----------

function newId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function sendTap(type, action) {
  if (!status || !status.walk) return toast('Start a walk first.', true);
  const tap = { id: newId(), walk: status.walk.id, t: Date.now() / 1000 + clockOffset, type, acked: false };
  if (action) tap.action = action;
  taps.push(tap);
  store.set('taps', taps);
  if (navigator.vibrate) navigator.vibrate(25);
  board.send_message('tap', tap);
  renderTaps();
}

function resendTaps() {
  for (const tap of taps) if (!tap.acked) board.send_message('tap', tap);
}

board.on_message('tap_ok', (d) => {
  const tap = taps.find((x) => x.id === d.id);
  if (tap) {
    tap.acked = true;
    store.set('taps', taps);
    renderTaps();
  }
});

$$('[data-taps] button').forEach((b) =>
  b.addEventListener('click', () => {
    sendTap('label', b.dataset.action);
    b.classList.add('flash');
    setTimeout(() => b.classList.remove('flash'), 150);
  }),
);
$$('[data-undo]').forEach((b) => b.addEventListener('click', () => sendTap('undo')));
$$('[data-wrong]').forEach((b) => b.addEventListener('click', () => sendTap('wrong_cue')));

function renderTaps() {
  const start = status && status.walk ? Date.now() / 1000 + clockOffset - status.walk.seconds : null;
  const rows = taps.slice(-6).reverse().map((t) => {
    const what = t.type === 'label' ? MANEUVER_NAMES[t.action] : t.type === 'undo' ? 'Undo' : 'Wrong cue';
    const when = start ? fmtTime(t.t - start) : '';
    return `<li><span>${when} · ${what}</span><span class="${t.acked ? '' : 'pending'}">${t.acked ? 'saved' : 'sending…'}</span></li>`;
  });
  $$('[data-taplog]').forEach((ul) => (ul.innerHTML = rows.join('')));
}

// ---------- cues ----------

function renderCue(action, detail) {
  const [maneuver, when] = action ? action.split('_') : ['straight', null];
  let text = 'Continue straight';
  if (action) {
    const name = MANEUVER_NAMES[maneuver] || maneuver;
    text = lastCue && lastCue.action === action ? lastCue.text : when === 'soon' ? `${name} coming up` : name;
  }
  for (const el of [$('#cue-nav'), $('#cue-dagger')]) {
    el.className = 'cue' + (action ? ` ${maneuver} ${when}` : '');
    $('.arrow', el).textContent = ARROWS[maneuver] || '↑';
    $('.text', el).textContent = status && status.walk ? text : 'Ready';
    if (detail !== undefined) $('.detail', el).textContent = detail;
  }
}

board.on_message('cue', (d) => {
  lastCue = d;
  speak(d.text);
  if (navigator.vibrate) navigator.vibrate(d.action.endsWith('_now') ? [200, 100, 200] : 120);
  renderCue(d.action);
});

board.on_message('step', (d) => {
  $('#s-steps').textContent = d.n;
  $('#s-meters').textContent = Math.round(d.meters);
  $('#s-heading').textContent = Math.round(d.heading) + '°';
  if (d.cue !== undefined) {
    const active = d.cue && d.cue !== 'straight' ? d.cue : null;
    if (!active) lastCue = null;
    const conf = d.prediction ? ` · policy: ${d.prediction.replace('_', ' ')} (${Math.round(d.confidence * 100)}%)` : '';
    renderCue(active, `${Math.round(d.meters_since_turn)} m since last turn · ${d.turns} turns${conf}`);
  }
});

// ---------- status ----------

board.on_message('status', (s) => {
  const firstStatus = status === null;
  status = s;
  clockOffset = s.board_time - Date.now() / 1000;
  showImu(s.imu);
  renderWalkers(s);

  if (s.walk) {
    if (firstStatus || TABS.includes(s.mode)) {
      if (tab !== s.mode && TABS.includes(s.mode)) showTab(s.mode);
    }
    $('#s-time').textContent = fmtTime(s.walk.seconds);
    // drop queued taps from an older walk
    const before = taps.length;
    taps = taps.filter((t) => t.walk === s.walk.id);
    if (taps.length !== before) store.set('taps', taps);
  } else {
    $('#s-time').textContent = '—';
  }
  if (s.mode === 'idle' || !s.walk) {
    $('#s-steps').textContent = s.steps;
    $('#s-meters').textContent = Math.round(s.meters);
    $('#s-heading').textContent = Math.round(s.heading) + '°';
  }

  const walking = !!s.walk || s.mode !== 'idle';
  document.body.classList.toggle('walking', walking);
  $$('[data-start]').forEach((b) => (b.disabled = walking));
  $$('[data-stop]').forEach((b) => (b.disabled = !walking));
  $('#cal-stop').disabled = s.mode !== 'calibrate';
  $$('[data-panel="record"] [data-taps] button, [data-panel="record"] [data-undo]').forEach(
    (b) => (b.disabled = s.mode !== 'record'));
  $$('[data-panel="dagger"] [data-taps] button, [data-panel="dagger"] [data-undo], [data-wrong]').forEach(
    (b) => (b.disabled = s.mode !== 'dagger'));

  const b = s.budget;
  const budget = `Training walks for "${s.route}": ${b.used}/${b.limit} used` +
    (b.recorded > b.used ? ` (${b.recorded - b.used} excluded)` : '') +
    (b.used >= b.limit ? ' · limit reached' : '');
  $('#budget-record').textContent = budget;
  $('#budget-dagger').textContent = budget;
  const p = s.policy;
  $('#policy-info').textContent = !p ? 'No policy trained for this route yet.'
    : `Policy: ${p.walks} walks, ${p.steps} steps` + (p.report && p.report.hit_rate !== null
      ? ` · held-out walks: ${Math.round(p.report.hit_rate * 100)}% of maneuvers cued, ±${p.report.mean_abs_error_m} m`
      : '');
  if (!s.walk) renderCue(null, s.policy ? 'Press Start at the start of the route.' : 'Train a policy on the Review page first.');
  renderTaps();
});

document.addEventListener('board-connected', resendTaps);
