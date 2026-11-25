// Shared by index.html and review.html: the board connection, toasts, storage.

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const board = new WebUI();
let connected = false;

function setPill(el, text, cls) {
  el.textContent = text;
  el.className = 'pill ' + (cls || '');
}

board.on_connect(() => {
  connected = true;
  setPill($('#pill-link'), 'Board: connected', 'good');
  board.send_message('get_status', {});
  document.dispatchEvent(new Event('board-connected'));
});
board.on_disconnect(() => {
  connected = false;
  setPill($('#pill-link'), 'Board: reconnecting…', 'bad');
});

let toastTimer = null;
function toast(text, error = false) {
  const el = $('#toast');
  el.textContent = text;
  el.className = 'show' + (error ? ' error' : '');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.className = ''), error ? 6000 : 3500);
}
board.on_message('notice', (d) => toast(d.text, d.error));

function showImu(status) {
  const el = $('#pill-imu');
  if (!el) return;
  if (status === 'ok') setPill(el, 'IMU: ok', 'good');
  else if (status === 'stalled') setPill(el, 'IMU: stalled', 'warn');
  else if (status === 'waiting') setPill(el, 'IMU: waiting', 'warn');
  else setPill(el, 'IMU: ' + String(status).split('|')[0].trim(), 'bad');
}

// localStorage can be missing or throw (private mode); everything must work without it
const store = {
  get(key, fallback) {
    try {
      const v = localStorage.getItem(key);
      return v === null ? fallback : JSON.parse(v);
    } catch (e) {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
      /* ignore */
    }
  },
};

function fmtTime(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const MANEUVER_NAMES = { left: 'Left', right: 'Right', stairs: 'Stairs', arrive: 'Arrived' };
// combined maneuvers ("left+stairs") are named "Left + Stairs" and drawn like their first part
const maneuverName = (a) => a.split('+').map((m) => MANEUVER_NAMES[m] || m).join(' + ');
const firstManeuver = (a) => a.split('+')[0];
