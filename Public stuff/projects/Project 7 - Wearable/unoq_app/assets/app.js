// Activity wearable dashboard: live IMU charts, both models' predictions, and
// the record / train controls. Talks to python/main.py over the web_ui socket.

const LABELS = ['still', 'walk', 'run', 'bike'];
const CHART_SECONDS = 5;
const CHART_HZ = 50;                       // main.py sends every 2nd sample of 100 Hz
const CHART_POINTS = CHART_SECONDS * CHART_HZ;

const $ = (sel) => document.querySelector(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const history = [];                        // rows of [ax, ay, az, gx, gy, gz]
let lastStatus = null;

// ---------- charts ----------

function drawChart(canvas, cols, fixedRange) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== w * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // symmetric range that grows to fit the data, never smaller than fixedRange
  let range = fixedRange;
  for (const row of history) for (const c of cols) range = Math.max(range, Math.abs(row[c]));
  range *= 1.1;
  const y = (v) => h / 2 - (v / range) * (h / 2 - 4);
  const x = (i) => (i / (CHART_POINTS - 1)) * w;

  ctx.strokeStyle = css('--line');
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
  ctx.fillStyle = css('--muted');
  ctx.font = '11px system-ui';
  ctx.fillText(`+${range.toFixed(range < 10 ? 1 : 0)}`, 4, 12);
  ctx.fillText(`−${range.toFixed(range < 10 ? 1 : 0)}`, 4, h - 4);

  const offset = CHART_POINTS - history.length;
  ['--x', '--y', '--z'].forEach((color, k) => {
    ctx.strokeStyle = css(color);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    history.forEach((row, i) => {
      const px = x(i + offset);
      const py = y(row[cols[k]]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  });
}

function redraw() {
  drawChart($('#chart-acc'), [0, 1, 2], 2);
  drawChart($('#chart-gyro'), [3, 4, 5], 200);
  requestAnimationFrame(redraw);
}
requestAnimationFrame(redraw);

// ---------- predictions ----------

function showPrediction(id, pred, detail) {
  const card = $(`#${id}`);
  const state = card.querySelector('.state');
  state.textContent = pred ? pred.label : '—';
  state.dataset.label = pred ? pred.label : '';
  card.querySelector('.bar').style.width = pred ? `${pred.conf * 100}%` : '0';
  card.querySelector('.detail').textContent = pred ? detail : 'Not trained yet';
}

function drawPeakLine(models, value) {
  const line = $('#peak-line');
  if (!models) {
    line.innerHTML = '';
    return;
  }
  const stats = models.peak_stats;
  const max = Math.max(...Object.values(stats).map(([m, s]) => m + 2 * s), value || 0) * 1.05;
  const pos = (v) => `${Math.min(100, (v / max) * 100)}%`;
  let html = '';
  for (const [label, [mean]] of Object.entries(stats)) {
    html += `<span class="tick" style="left:${pos(mean)};color:${css('--' + label)}">${label}<br>${mean.toFixed(2)} g</span>`;
  }
  if (value != null) html += `<span class="now" style="left:${pos(value)}"></span>`;
  line.innerHTML = html;
}

// ---------- training panel ----------

function buildLabelButtons() {
  for (const label of LABELS) {
    const btn = document.createElement('button');
    btn.className = 'label-btn';
    btn.id = `rec-${label}`;
    btn.innerHTML = `<span class="del" title="Delete all ${label} recordings">delete</span>
      <strong>${label === 'bike' ? 'bike / air-bike' : label}</strong><small>0 s recorded</small>`;
    btn.addEventListener('click', (e) => {
      if (e.target.classList.contains('del')) {
        if (confirm(`Delete every ${label} recording?`)) ui.send_message('delete', { label });
        return;
      }
      ui.send_message('record', { label });
    });
    $('#labels').appendChild(btn);
  }
}
buildLabelButtons();

$('#stop').addEventListener('click', () => ui.send_message('stop'));
$('#train').addEventListener('click', () => {
  $('#notice').textContent = 'Training…';
  ui.send_message('train');
});

function confusionTable(title, result) {
  if (!result) return '';
  const head = result.labels.map((l) => `<th>${l}</th>`).join('');
  const rows = result.matrix.map((row, i) =>
    `<tr><th>${result.labels[i]}</th>${row.map((n, j) => `<td class="${i === j ? 'hit' : ''}">${n}</td>`).join('')}</tr>`
  ).join('');
  return `<div><h3>${title}: ${(result.accuracy * 100).toFixed(0)}% on held-out data</h3>
    <table><tr><th>true ↓ / predicted →</th>${head}</tr>${rows}</table></div>`;
}

function showStatus(s) {
  lastStatus = s;
  const imu = $('#pill-imu');
  imu.textContent = `IMU: ${s.imu}`;
  imu.className = `pill ${s.imu === 'ok' ? 'good' : 'bad'}`;

  for (const label of LABELS) {
    const btn = $(`#rec-${label}`);
    btn.querySelector('small').textContent = `${s.seconds[label]} s recorded`;
    btn.classList.toggle('recording', s.recording?.label === label);
  }
  $('#stop').disabled = !s.recording;
  $('#rec-status').textContent = s.recording ? `● Recording ${s.recording.label} · ${s.recording.seconds.toFixed(0)} s` : '';

  const m = s.models;
  $('#m1-axis').textContent = m ? `axis ${m.peak_axis}` : '';
  const r = m?.report;
  $('#report').innerHTML = r
    ? confusionTable('Model 1 (peak accel)', r.model1) + confusionTable('Model 2 (all IMU, k-NN)', r.model2)
    : '';
  if (!m) drawPeakLine(null);
}

// ---------- socket ----------

const ui = new WebUI();

ui.on_connect(() => {
  const pill = $('#pill-link');
  pill.textContent = 'Board: connected';
  pill.className = 'pill good';
});
ui.on_disconnect(() => {
  const pill = $('#pill-link');
  pill.textContent = 'Board: disconnected';
  pill.className = 'pill bad';
});

ui.on_message('samples', (data) => {
  history.push(...data.s);
  if (history.length > CHART_POINTS) history.splice(0, history.length - CHART_POINTS);
});

ui.on_message('prediction', (p) => {
  showPrediction('m1', p.m1, p.m1 && `peak ${p.m1.value.toFixed(2)} g · ${(p.m1.conf * 100).toFixed(0)}% likely`);
  showPrediction('m2', p.m2, p.m2 && `cadence ${p.m2.cadence.toFixed(1)} Hz · ${(p.m2.conf * 100).toFixed(0)}% of neighbours agree`);
  drawPeakLine(lastStatus?.models, p.m1 ? p.m1.value : null);
});

ui.on_message('status', showStatus);
ui.on_message('notice', (n) => { $('#notice').textContent = n.text; });
