// Game audio: the match announcer (browser text-to-speech) and a crowd that
// is synthesized live with WebAudio (no sound files). Python decides WHAT the
// announcer says (commentary.py); this file decides how it sounds.
//
// Browsers only allow audio after a click or key press, so GameAudio is
// created on the first one (see ensureAudio() in game.js).

const PREFS_KEY = "rogersCupAudio";

// Best announcer voice available, most preferred first. Edge ships natural
// neural voices ("... Online (Natural)"); Chrome has "Google UK English Male".
const VOICE_PREFS = [
  (v) => /(Ryan|Thomas).*Natural/i.test(v.name),                          // British male, neural (Edge)
  (v) => /(Guy|Christopher|Eric|Andrew|Brian|Davis|Roger).*Natural/i.test(v.name),
  (v) => /Natural/i.test(v.name) && v.lang.startsWith("en"),
  (v) => /Google UK English Male/i.test(v.name),
  (v) => v.lang === "en-GB",
  (v) => v.lang.startsWith("en"),
];

export class GameAudio {
  constructor() {
    const ctx = (this.ctx = new AudioContext());
    this.master = ctx.createGain();
    this.master.connect(ctx.destination);
    this.sfx = ctx.createGain();
    this.sfx.connect(this.master);
    this.crowdOut = ctx.createGain();          // on/off toggle
    this.duck = ctx.createGain();              // dips while the announcer talks
    this.crowdBus = ctx.createGain();
    this.crowdBus.connect(this.duck).connect(this.crowdOut).connect(this.master);

    const prefs = loadPrefs();
    this.voiceOn = prefs.voice ?? true;
    this.crowdOn = prefs.crowd ?? true;
    this.crowdOut.gain.value = this.crowdOn ? 1 : 0;

    this.noise = noiseBuffer(ctx, 4, "pink");
    this.grain = noiseBuffer(ctx, 0.03, "white", true);
    this._startMurmur();

    this.voice = null;
    this.speaking = 0;
    this.onCaption = () => {};
    const pick = () => { this.voice = pickVoice(); };
    pick();
    speechSynthesis.addEventListener?.("voiceschanged", pick);
  }

  // ------------------------------------------------------------ toggles
  setVoice(on) {
    this.voiceOn = on;
    if (!on) speechSynthesis.cancel();
    savePrefs({ voice: this.voiceOn, crowd: this.crowdOn });
  }
  setCrowd(on) {
    this.crowdOn = on;
    this.crowdOut.gain.setTargetAtTime(on ? 1 : 0, this.ctx.currentTime, 0.1);
    savePrefs({ voice: this.voiceOn, crowd: this.crowdOn });
  }

  // ------------------------------------------------------------ announcer
  say(text, excite = 0.5, interrupt = false) {
    this.onCaption(text);
    if (!this.voiceOn || !("speechSynthesis" in window)) return;
    if (interrupt) speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    if (this.voice) u.voice = this.voice;
    u.lang = this.voice?.lang || "en-GB";
    u.pitch = 0.95 + 0.3 * excite;              // hyped = higher and faster
    u.rate = 1.0 + 0.3 * excite;
    u.volume = 1;
    const t = () => this.ctx.currentTime;
    u.onstart = () => { this.speaking++; this.duck.gain.setTargetAtTime(0.45, t(), 0.08); };
    const done = () => {
      this.speaking = Math.max(0, this.speaking - 1);
      if (!this.speaking) this.duck.gain.setTargetAtTime(1, t(), 0.3);
    };
    u.onend = u.onerror = done;
    speechSynthesis.speak(u);
  }

  // ------------------------------------------------------------ crowd
  // mood: "quiet" (menus), "idle" (between points), "rally" (hushed), "buzz" (excited)
  setMood(mood) {
    this.murmurLevel = { quiet: 0.05, idle: 0.11, rally: 0.045, buzz: 0.17 }[mood] ?? 0.1;
  }

  _startMurmur() {
    const ctx = this.ctx;
    this.murmur = ctx.createGain();
    this.murmur.gain.value = 0;
    this.murmur.connect(this.crowdBus);
    for (const [rate, freq, q] of [[0.9, 550, 0.7], [1.13, 1250, 0.9], [0.75, 320, 0.8]]) {
      const src = ctx.createBufferSource();
      src.buffer = this.noise; src.loop = true; src.playbackRate.value = rate;
      const bp = ctx.createBiquadFilter();
      bp.type = "bandpass"; bp.frequency.value = freq; bp.Q.value = q;
      src.connect(bp).connect(this.murmur);
      src.start();
    }
    this.setMood("quiet");
    // babble: the level wanders like a room full of conversations
    setInterval(() => {
      const target = this.murmurLevel * (0.75 + Math.random() * 0.5);
      this.murmur.gain.setTargetAtTime(target, ctx.currentTime, 0.35);
    }, 300);
  }

  cheer(size = 0.6) {
    const ctx = this.ctx, t = ctx.currentTime;
    const src = ctx.createBufferSource();
    src.buffer = this.noise; src.playbackRate.value = 1.2;
    const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 450;
    const bp = ctx.createBiquadFilter(); bp.type = "bandpass"; bp.frequency.value = 1700; bp.Q.value = 0.5;
    const g = ctx.createGain();
    const peak = 0.55 * size, hold = 0.5 + size * 1.2;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(peak, t + 0.25);
    g.gain.setValueAtTime(peak, t + hold);
    g.gain.exponentialRampToValueAtTime(0.001, t + hold + 2);
    src.connect(hp).connect(bp).connect(g).connect(this.crowdBus);
    src.start(t, Math.random() * 2); src.stop(t + hold + 2.1);
    // whoops and whistles from the stands
    for (let i = 0; i < Math.round(size * 5); i++) this._whoop(t + Math.random() * hold);
  }

  _whoop(t) {
    const ctx = this.ctx, o = ctx.createOscillator(), g = ctx.createGain();
    const f = 800 + Math.random() * 700;
    o.type = "sine";
    o.frequency.setValueAtTime(f, t);
    o.frequency.exponentialRampToValueAtTime(f * (1.3 + Math.random() * 0.4), t + 0.25);
    o.frequency.exponentialRampToValueAtTime(f * 0.9, t + 0.55);
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.025, t + 0.05);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.6);
    o.connect(g).connect(this.crowdBus);
    o.start(t); o.stop(t + 0.65);
  }

  applause(seconds = 2, intensity = 0.7) {
    const ctx = this.ctx, t0 = ctx.currentTime;
    const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 1100;
    hp.connect(this.crowdBus);
    const n = Math.round(70 * seconds * intensity);
    for (let i = 0; i < n; i++) {
      const at = Math.pow(Math.random(), 1.6) * seconds;      // denser at the start, tapering off
      const src = ctx.createBufferSource(), g = ctx.createGain();
      src.buffer = this.grain; src.playbackRate.value = 0.6 + Math.random() * 0.9;
      g.gain.value = 0.35 * intensity * (1 - 0.7 * at / seconds) * (0.5 + Math.random() * 0.5);
      src.connect(g).connect(hp);
      src.start(t0 + at);
    }
  }

  // "ooh" (rising, a near miss) or "aww" (falling, a disappointment)
  ooh(size = 0.6, falling = false) {
    const ctx = this.ctx, t = ctx.currentTime, dur = 1.1;
    const f1 = ctx.createBiquadFilter(); f1.type = "bandpass"; f1.frequency.value = falling ? 700 : 450; f1.Q.value = 3;
    const f2 = ctx.createBiquadFilter(); f2.type = "bandpass"; f2.frequency.value = falling ? 1150 : 850; f2.Q.value = 4;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.5 * size, t + 0.15);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    f1.connect(g); f2.connect(g); g.connect(this.crowdBus);
    for (let i = 0; i < 9; i++) {
      const o = ctx.createOscillator(), f = 140 + Math.random() * 130;
      o.type = "sawtooth";
      o.frequency.setValueAtTime(f, t);
      if (falling) o.frequency.exponentialRampToValueAtTime(f * 0.72, t + dur);
      else { o.frequency.exponentialRampToValueAtTime(f * 1.15, t + 0.35); o.frequency.exponentialRampToValueAtTime(f * 0.9, t + dur); }
      o.connect(f1); o.connect(f2);
      o.start(t + Math.random() * 0.08); o.stop(t + dur);
    }
  }

  // ------------------------------------------------------------ sfx
  blip(freq, dur = 0.08, type = "square", vol = 0.12, slide = 0) {
    const ctx = this.ctx, o = ctx.createOscillator(), g = ctx.createGain(), t = ctx.currentTime;
    o.type = type; o.frequency.setValueAtTime(freq, t);
    if (slide) o.frequency.exponentialRampToValueAtTime(Math.max(40, freq + slide), t + dur);
    g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g).connect(this.sfx); o.start(t); o.stop(t + dur);
  }
}

function pickVoice() {
  const voices = speechSynthesis.getVoices();
  for (const pref of VOICE_PREFS) {
    const v = voices.find(pref);
    if (v) return v;
  }
  return null;
}

function noiseBuffer(ctx, seconds, color, windowed = false) {
  const n = Math.max(1, Math.floor(ctx.sampleRate * seconds));
  const buf = ctx.createBuffer(1, n, ctx.sampleRate), d = buf.getChannelData(0);
  let b0 = 0, b1 = 0, b2 = 0;
  for (let i = 0; i < n; i++) {
    const w = Math.random() * 2 - 1;
    if (color === "pink") {                       // Paul Kellet's cheap pink filter
      b0 = 0.99765 * b0 + w * 0.099; b1 = 0.963 * b1 + w * 0.2965; b2 = 0.57 * b2 + w * 1.0526;
      d[i] = (b0 + b1 + b2 + w * 0.1848) * 0.2;
    } else d[i] = w;
    if (windowed) d[i] *= Math.exp(-i / (n * 0.25));   // a single clap: sharp attack, fast decay
  }
  return buf;
}

function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(PREFS_KEY)) || {}; } catch { return {}; }
}
function savePrefs(p) {
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch { /* private mode etc. */ }
}
