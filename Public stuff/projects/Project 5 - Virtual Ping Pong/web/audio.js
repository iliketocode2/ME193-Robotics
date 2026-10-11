// Game audio: the two-person commentary booth (browser text-to-speech) and
// the crowd. Python decides WHAT is said (commentary.py: Ray scripted,
// Sonia AI-written); this file decides how it sounds.
//
// Crowd: real stadium recordings (audio/crowd/, built by make_crowd_audio.py,
// credits in audio/CREDITS.md) -- looping ambience beds, cheers, roars,
// applause, gasps and football chants. If the files can't load, a
// synthesized crowd (SynthCrowd) stands in.
//
// Browsers only allow audio after a click or key press, so GameAudio is
// created on the first one (see ensureAudio() in game.js).

const PREFS_KEY = "rogersCupAudio";
const CROWD_DIR = "audio/crowd/";

// Voices, most preferred first. Edge ships natural neural voices
// ("... Online (Natural)"); Chrome has "Google UK English Male/Female".
const RAY_PREFS = [
  (v) => /(Ryan|Thomas).*Natural/i.test(v.name),                        // British male, neural (Edge)
  (v) => /(Guy|Christopher|Eric|Andrew|Brian|Davis|Roger).*Natural/i.test(v.name),
  (v) => /Natural/i.test(v.name) && v.lang.startsWith("en"),
  (v) => /Google UK English Male/i.test(v.name),
  (v) => v.localService && /(George|David|Mark)/.test(v.name),   // installed Windows voices (offline fallback)
  (v) => v.lang === "en-GB",
  (v) => v.lang.startsWith("en"),
];
const SONIA_PREFS = [
  (v) => /(Sonia|Libby|Maisie|Bella|Hollie|Abbi|Olivia).*Natural/i.test(v.name),   // British female, neural
  (v) => /(Aria|Jenny|Ava|Emma|Michelle|Natasha|Clara|Emily).*Natural/i.test(v.name),
  (v) => /Natural/i.test(v.name) && v.lang.startsWith("en"),
  (v) => /Google UK English Female/i.test(v.name),
  (v) => v.localService && /(Hazel|Susan|Zira)/.test(v.name),   // installed Windows voices (offline fallback)
  (v) => v.lang === "en-GB",
  (v) => v.lang.startsWith("en"),
];
// Browser-voice fallback delivery (excite 0..1 from Python). Pitch stays at
// 1: shifting it is what made the neural voices sound processed and robotic.
const DELIVERY = {
  ray: (e) => ({ pitch: 1, rate: 1.0 + 0.15 * e }),       // play-by-play: a touch quicker when it matters
  sonia: (e) => ({ pitch: 1, rate: 0.98 + 0.08 * e }),    // analyst: calmer, conversational
};
// Natural voices (Kokoro clips, see tts.py): a call's sentences play back to back.
const CLIP_GAP_S = 0.12;
// A queued line older than this is skipped. Sonia's is long because her line
// usually has to wait for Ray's call to finish (his match welcome alone is ~7 s);
// once the next point ends her queued line is dropped anyway (see say()).
const LINE_TTL_MS = { ray: 8000, sonia: 12000 };
// Edge's "Natural" voices are streamed from an online service. If it's slow,
// blocked or stuck, speak() just never starts -- silently. A line that hasn't
// started by then is retried with an installed (offline) voice instead.
const START_TIMEOUT_MS = 2500;
const MS_PER_CHAR = 62;                 // rough speaking speed at rate 1 (matches commentary.py)
const SONIA_GRACE_MS = 4500;            // Ray waits for Sonia to finish her sentence if it ends within this

export class GameAudio {
  constructor() {
    const ctx = (this.ctx = new AudioContext());
    this.master = ctx.createGain();
    this.master.connect(ctx.destination);
    this.sfx = ctx.createGain();
    this.sfx.connect(this.master);
    this.crowdOut = ctx.createGain();          // on/off toggle
    this.duck = ctx.createGain();              // dips while a commentator talks
    this.crowdBus = ctx.createGain();
    this.crowdBus.gain.value = 0.85;
    this.crowdBus.connect(this.duck).connect(this.crowdOut).connect(this.master);

    const prefs = loadPrefs();
    this.voiceOn = prefs.voice ?? true;
    this.crowdOn = prefs.crowd ?? true;
    this.crowdOut.gain.value = this.crowdOn ? 1 : 0;

    this.mood = "quiet";
    this.synth = new SynthCrowd(ctx, this.crowdBus);
    this.samples = new SampleCrowd(ctx, this.crowdBus);
    this.samplesLoaded = this.samples.load();          // resolves once the beds are playing
    this.samplesLoaded.then((ok) => {
      if (ok) { this.synth.stop(); this.samples.setMood(this.mood); }
    });

    this.voiceBus = ctx.createGain();          // the natural-voice clips (not ducked)
    this.voiceBus.connect(this.master);
    this.clipCache = new Map();                // url -> Promise<AudioBuffer>

    this.voices = { ray: null, sonia: null };
    this.offlineOnly = false;                  // set once an online voice has failed: use installed voices
    this.speech = { spoken: 0, natural: 0, failed: 0, lastError: "" };   // for the D overlay
    this.queue = [];
    this.current = null;
    this.onCaption = () => {};
    const pick = () => { this.voices = pickVoices(this.offlineOnly); };
    pick();
    speechSynthesis.addEventListener?.("voiceschanged", pick);
  }

  get crowd() { return this.samples.ready ? this.samples : this.synth; }
  get crowdKind() { return this.samples.ready ? "recorded" : "synth"; }

  // ------------------------------------------------------------ toggles
  setVoice(on) {
    this.voiceOn = on;
    if (!on) { this.queue = []; this._cancelCurrent(); }
    savePrefs({ voice: this.voiceOn, crowd: this.crowdOn });
  }
  setCrowd(on) {
    this.crowdOn = on;
    this.crowdOut.gain.setTargetAtTime(on ? 1 : 0, this.ctx.currentTime, 0.1);
    savePrefs({ voice: this.voiceOn, crowd: this.crowdOn });
  }

  // ------------------------------------------------------------ the booth
  // One queue for both voices so Ray and Sonia never talk over each other.
  // interrupt (Ray's point calls) clears the queue and cuts in immediately --
  // except that Sonia, if she's nearly done, finishes her sentence first.
  // kind (Sonia only): "intro" | "point" | "game_over".
  // clips: natural-voice audio for the line (one per sentence), or null for the browser voice.
  say(text, excite = 0.5, interrupt = false, speaker = "ray", kind = "", clips = null) {
    if (!this.voiceOn || (!clips && !("speechSynthesis" in window))) { this.onCaption(text, speaker); return; }
    const item = { text, excite, speaker, kind, clips, expires: performance.now() + (LINE_TTL_MS[speaker] ?? 6000) };
    clips?.forEach((u) => this._load(u));          // fetch + decode now, so it's ready when its turn comes
    if (!interrupt) {
      this.queue.push(item);
      setTimeout(() => this._next(), 0);
      return;
    }
    // Anything still queued is about the last point -- except Sonia's match
    // intro, which is still worth hearing after Ray's first call.
    this.queue = [item, ...this.queue.filter((i) => i.kind === "intro")];
    const c = this.current;
    if (c?.speaker === "sonia" && c.startedAt && c.endsAt - performance.now() <= SONIA_GRACE_MS) {
      return;                                      // she's nearly done: Ray goes straight after her
    }
    const wasTalking = this._cancelCurrent();
    setTimeout(() => this._next(), wasTalking ? 40 : 0);   // Chrome drops a speak() issued right after cancel()
  }

  // Drop a speaker's queued (not yet spoken) lines -- e.g. Sonia's once the next rally is under way.
  dropQueued(speaker) { this.queue = this.queue.filter((i) => i.speaker !== speaker); }

  _next() {
    if (this.current) return;
    let item;
    while ((item = this.queue.shift()) && performance.now() > item.expires) { /* stale: skip */ }
    if (!item) { this._unduck(); return; }
    if (item.clips) this._playClips(item);
    else this._speak(item);
  }

  // A line in the natural voice: its clips, back to back. Falls back to the
  // browser voice if they can't be loaded.
  _playClips(item) {
    const cur = (this.current = { speaker: item.speaker, startedAt: 0, endsAt: 0, timer: 0, stop: null });
    const fallback = (why) => {
      if (this.current !== cur) return;
      clearTimeout(cur.timer);
      this.current = null;
      this.speech.failed++;
      this.speech.lastError = why;
      item.clips = null;
      this._speak(item);
    };
    cur.timer = setTimeout(() => fallback("voice clip load timed out"), START_TIMEOUT_MS);
    this.onCaption(item.text, item.speaker);
    Promise.all(item.clips.map((u) => this._load(u))).then((bufs) => {
      if (this.current !== cur) return;           // cut off while loading
      clearTimeout(cur.timer);
      const ctx = this.ctx, t0 = ctx.currentTime + 0.03;
      let t = t0;
      const srcs = bufs.map((buf) => {
        const src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(this.voiceBus);
        src.start(t);
        t += buf.duration + CLIP_GAP_S;
        return src;
      });
      const ms = (t - t0 - CLIP_GAP_S) * 1000;
      cur.stop = () => srcs.forEach((s) => { try { s.stop(); } catch { /* not started */ } });
      cur.startedAt = performance.now();
      cur.endsAt = cur.startedAt + ms;
      this.speech.spoken++;
      this.speech.natural++;
      this._duck();
      cur.timer = setTimeout(() => {
        if (this.current !== cur) return;
        this.current = null;
        this._next();
      }, ms + 30);
    }, () => fallback("voice clip failed to load"));
  }

  _load(url) {
    let p = this.clipCache.get(url);
    if (!p) {
      p = fetch(url)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.arrayBuffer(); })
        .then((b) => this.ctx.decodeAudioData(b));
      p.catch(() => this.clipCache.delete(url));   // let a later call try again
      this.clipCache.set(url, p);
      if (this.clipCache.size > 300) this.clipCache.delete(this.clipCache.keys().next().value);
    }
    return p;
  }

  _speak(item) {
    const ss = speechSynthesis;
    // A previous utterance can be left stuck in the engine (no end event ever
    // came) and blocks everything after it; a paused engine never speaks.
    // Clear both before speaking -- otherwise the booth goes silent for good.
    if ((ss.speaking || ss.pending) && !item.cleared) {
      item.cleared = true;                         // once: never loop on an engine that stays "busy"
      ss.cancel();
      const hold = (this.current = { speaker: item.speaker, timer: 0 });   // keeps _next() from double-speaking
      hold.timer = setTimeout(() => { if (this.current === hold) { this.current = null; this._speak(item); } }, 40);
      return;
    }
    if (ss.paused) ss.resume();

    const u = new SpeechSynthesisUtterance(item.text);
    const sonia = item.speaker === "sonia";
    const voice = sonia ? this.voices.sonia : this.voices.ray;
    if (voice) { u.voice = voice; u.lang = voice.lang; } else u.lang = "en-GB";
    const d = (DELIVERY[item.speaker] ?? DELIVERY.ray)(item.excite);
    // only one usable voice? keep the two commentators apart by pitch
    const same = sonia && (!this.voices.sonia || this.voices.sonia === this.voices.ray);
    u.pitch = Math.min(2, d.pitch + (same ? 0.35 : 0));
    u.rate = d.rate;
    u.volume = 1;

    const cur = (this.current = { u, speaker: item.speaker, startedAt: 0, endsAt: 0, timer: 0 });
    const finish = () => {
      if (this.current !== cur) return;            // a cancelled line reporting late
      clearTimeout(cur.timer);
      this.current = null;
      this._next();
    };
    // The voice never started (online voice service slow/blocked) or broke:
    // switch to installed voices for the rest of the session and say it again.
    const fail = (why) => {
      if (this.current !== cur) return;
      clearTimeout(cur.timer);
      this.current = null;
      this.speech.failed++;
      this.speech.lastError = `${why} (${voice?.name ?? "default voice"})`;
      console.warn("Commentary voice failed:", this.speech.lastError);
      ss.cancel();
      if (!this.offlineOnly && voice && !voice.localService) {
        this.offlineOnly = true;
        this.voices = pickVoices(true);
        if (!item.retried) { item.retried = true; this.queue.unshift(item); }
      }
      setTimeout(() => this._next(), 40);
    };
    u.onstart = () => {
      if (this.current !== cur) return;
      clearTimeout(cur.timer);
      const ms = (item.text.length * MS_PER_CHAR) / u.rate;
      cur.startedAt = performance.now();
      cur.endsAt = cur.startedAt + ms;
      cur.timer = setTimeout(finish, 2000 + ms * 1.8);   // in case onend never fires
      this.speech.spoken++;
      this._duck();
    };
    u.onend = finish;
    u.onerror = (e) => {
      if (e.error === "interrupted" || e.error === "canceled") finish();   // we cut it off ourselves
      else fail(e.error || "error");
    };
    cur.timer = setTimeout(() => fail("never started"), START_TIMEOUT_MS);
    // Caption now, not on start: the line is visible even if speech fails.
    this.onCaption(item.text, item.speaker);
    ss.speak(u);
  }

  // Cut off the current line. Returns true if the browser's speech engine had
  // to be cancelled (its next speak() then needs a short pause).
  _cancelCurrent() {
    const c = this.current;
    this.current = null;
    if (c) { clearTimeout(c.timer); c.stop?.(); }
    const busy = "speechSynthesis" in window && (speechSynthesis.speaking || speechSynthesis.pending);
    if (busy) speechSynthesis.cancel();
    return busy;
  }

  // One line for the D overlay: which voices, how many lines spoken, any failures.
  speechInfo() {
    if (!this.voiceOn) return "MUTED (press M)";
    const v = (x) => (x ? x.name.replace(/^Microsoft /, "").replace(/ - .*$/, "") : "default");
    const s = this.speech;
    return `natural ${s.natural} of ${s.spoken} lines   fallback: Ray ${v(this.voices.ray)} / Sonia ${v(this.voices.sonia)}` +
      `${this.offlineOnly ? " [offline voices]" : ""}${s.failed ? `  FAILED ${s.failed}: ${s.lastError}` : ""}`;
  }
  _duck() { this.duck.gain.setTargetAtTime(0.4, this.ctx.currentTime, 0.08); }
  _unduck() { this.duck.gain.setTargetAtTime(1, this.ctx.currentTime, 0.35); }

  // ------------------------------------------------------------ crowd
  // mood: "quiet" (menus), "idle" (between points), "rally" (hushed), "buzz" (excited)
  setMood(mood) {
    this.mood = mood;
    this.crowd.setMood(mood);
  }
  cheer(size = 0.6) { this.crowd.cheer(size); }
  applause(seconds = 2, intensity = 0.7) { this.crowd.applause(seconds, intensity); }
  ooh(size = 0.6, falling = false) { this.crowd.ooh(size, falling); }
  chant() { return this.crowd.chant(); }
  stopChant() { this.crowd.stopChant(); }

  // ------------------------------------------------------------ sfx
  blip(freq, dur = 0.08, type = "square", vol = 0.12, slide = 0) {
    const ctx = this.ctx, o = ctx.createOscillator(), g = ctx.createGain(), t = ctx.currentTime;
    o.type = type; o.frequency.setValueAtTime(freq, t);
    if (slide) o.frequency.exponentialRampToValueAtTime(Math.max(40, freq + slide), t + dur);
    g.gain.setValueAtTime(vol, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    o.connect(g).connect(this.sfx); o.start(t); o.stop(t + dur);
  }
}

// =========================================================================
// Recorded crowd
// =========================================================================
// Bed levels per mood: [stadium + walla beds, excited "buzz" bed]
const BED_LEVELS = { quiet: [0.35, 0], idle: [0.85, 0.3], rally: [0.4, 0], buzz: [0.9, 0.75] };
const CHANT_GAP_S = 15;
// If a clip is missing, use its nearest neighbour instead of going silent.
const FALLBACK = { applause_l: "applause_m", applause_m: "applause_s", applause_s: "applause_m",
                   roar: "cheer", cheer: "cheer_small", cheer_small: "cheer" };

class SampleCrowd {
  constructor(ctx, out) {
    this.ctx = ctx;
    this.out = out;
    this.banks = {};                 // role -> [AudioBuffer]
    this.ready = false;
    this.mood = "quiet";
    this.lastPick = {};
    this.chantPlaying = null;
    this.lastChant = -1e9;
    this.bedGain = ctx.createGain(); this.bedGain.gain.value = 0; this.bedGain.connect(out);
    this.buzzGain = ctx.createGain(); this.buzzGain.gain.value = 0; this.buzzGain.connect(out);
  }

  // Each clip loads on its own: the recorded crowd switches on as soon as the
  // ambience beds are in, and one bad file only loses that one clip.
  async load() {
    let manifest;
    try {
      manifest = await (await fetch(CROWD_DIR + "manifest.json")).json();
    } catch (e) {
      console.warn("Recorded crowd unavailable, using the synthesized one:", e);
      return false;
    }
    const loadClip = async (role, f) => {
      try {
        const r = await fetch(CROWD_DIR + f);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const buf = await this.ctx.decodeAudioData(await r.arrayBuffer());
        (this.banks[role] ??= []).push(buf);
      } catch (e) {
        console.warn(`crowd clip ${f} failed:`, e);
      }
    };
    const entries = Object.entries(manifest);
    const beds = entries.filter(([role]) => role.startsWith("bed"));
    await Promise.all(beds.flatMap(([role, files]) => files.map((f) => loadClip(role, f))));
    if (!this.banks.bed?.length) return false;
    this._startBeds();
    this.ready = true;
    this.allLoaded = Promise.all(entries.filter(([role]) => !role.startsWith("bed"))   // the rest stream in
      .flatMap(([role, files]) => files.map((f) => loadClip(role, f))));
    return true;
  }

  _startBeds() {
    const ctx = this.ctx;
    const loops = [...this.banks.bed.map((b) => [b, this.bedGain]), ...(this.banks.bed_buzz ?? []).map((b) => [b, this.buzzGain])];
    const drifters = loops.map(([buf, bus]) => {
      const src = ctx.createBufferSource(), g = ctx.createGain();
      src.buffer = buf; src.loop = true;
      src.connect(g).connect(bus);
      src.start(0, Math.random() * buf.duration);           // random offset: never the same start twice
      return g;
    });
    // the crowd's level wanders a little, like a real stadium
    setInterval(() => {
      for (const g of drifters) g.gain.setTargetAtTime(0.8 + Math.random() * 0.35, ctx.currentTime, 0.6);
    }, 900);
  }

  setMood(mood) {
    this.mood = mood;
    const [bed, buzz] = BED_LEVELS[mood] ?? BED_LEVELS.idle;
    const t = this.ctx.currentTime, k = mood === "rally" ? 0.5 : 0.35;   // a slow hush as the rally starts
    this.bedGain.gain.setTargetAtTime(bed, t, k);
    this.buzzGain.gain.setTargetAtTime(buzz, t, k);
  }

  _pick(role) {
    let bank = this.banks[role];
    for (let hops = 0; !bank?.length && FALLBACK[role] && hops < 3; hops++) {
      role = FALLBACK[role];
      bank = this.banks[role];
    }
    if (!bank?.length) return null;
    let i = Math.floor(Math.random() * bank.length);
    if (bank.length > 1 && i === this.lastPick[role]) i = (i + 1) % bank.length;   // no back-to-back repeats
    this.lastPick[role] = i;
    return bank[i];
  }

  play(role, { gain = 1, rate = 1, delay = 0, pan = 0, fadeIn = 0 } = {}) {
    const buf = this._pick(role);
    if (!buf) return null;
    const ctx = this.ctx, t = ctx.currentTime + delay;
    const src = ctx.createBufferSource(), g = ctx.createGain(), p = ctx.createStereoPanner();
    src.buffer = buf;
    src.playbackRate.value = rate;
    p.pan.value = pan;
    if (fadeIn) { g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(gain, t + fadeIn); }
    else g.gain.value = gain;
    src.connect(g).connect(p).connect(this.out);
    src.start(t);
    return { src, g };
  }

  // Bigger reactions stack a second, slightly detuned copy: sounds like more people.
  _layered(role, gain, size) {
    const jitter = () => 0.94 + Math.random() * 0.12;
    const pan = (Math.random() - 0.5) * 0.6;
    this.play(role, { gain, rate: jitter(), pan });
    if (size > 0.5) this.play(role, { gain: gain * 0.55, rate: jitter(), pan: -pan, delay: 0.04 + Math.random() * 0.08 });
  }

  cheer(size = 0.6) {
    const role = size >= 0.75 && this.banks.roar ? "roar" : size >= 0.4 ? "cheer" : "cheer_small";
    this._layered(role, 0.55 + 0.5 * size, size);
  }

  applause(seconds = 2, intensity = 0.7) {
    const role = intensity >= 0.85 ? "applause_l" : intensity >= 0.5 ? "applause_m" : "applause_s";
    this.play(role, { gain: 0.45 + 0.5 * intensity, rate: 0.96 + Math.random() * 0.08, delay: 0.25,
                      pan: (Math.random() - 0.5) * 0.4 });
  }

  // "ooh" (near miss) or "aww" (falling, pitched down: disappointment)
  ooh(size = 0.6, falling = false) {
    const base = falling ? 0.86 : 1.03;
    this.play("ooh", { gain: 0.6 + 0.5 * size, rate: base, pan: -0.2 });
    this.play("ooh", { gain: 0.4 + 0.3 * size, rate: base * 1.07, pan: 0.25, delay: 0.06 });
  }

  // A football chant -- at most one every CHANT_GAP_S seconds. Returns true if one started.
  chant() {
    const now = this.ctx.currentTime;
    if (!this.banks.chant || now - this.lastChant < CHANT_GAP_S) return false;
    this.stopChant(0.3);
    const h = this.play("chant", { gain: 0.8, fadeIn: 0.8 });
    if (!h) return false;
    this.chantPlaying = h;
    this.lastChant = now;
    h.src.onended = () => { if (this.chantPlaying === h) this.chantPlaying = null; };
    return true;
  }

  // Fade the chant out (e.g. when the next rally starts, so it never covers the ball sounds).
  stopChant(fade = 0.6) {
    const h = this.chantPlaying;
    if (!h) return;
    this.chantPlaying = null;
    const t = this.ctx.currentTime;
    h.g.gain.cancelScheduledValues(t);
    h.g.gain.setTargetAtTime(0, t, fade / 3);
    h.src.stop(t + fade + 0.1);
  }
}

// =========================================================================
// Synthesized crowd (fallback if the recordings can't load)
// =========================================================================
class SynthCrowd {
  constructor(ctx, out) {
    this.ctx = ctx;
    this.out = out;
    this.noise = noiseBuffer(ctx, 4, "pink");
    this.grain = noiseBuffer(ctx, 0.03, "white", true);
    this.murmurLevel = 0.05;
    this._startMurmur();
  }

  stop() {
    clearInterval(this.babble);
    this.murmur.gain.setTargetAtTime(0, this.ctx.currentTime, 0.5);
  }

  setMood(mood) {
    this.murmurLevel = { quiet: 0.05, idle: 0.11, rally: 0.045, buzz: 0.17 }[mood] ?? 0.1;
  }

  chant() { return false; }
  stopChant() {}

  _startMurmur() {
    const ctx = this.ctx;
    this.murmur = ctx.createGain();
    this.murmur.gain.value = 0;
    this.murmur.connect(this.out);
    for (const [rate, freq, q] of [[0.9, 550, 0.7], [1.13, 1250, 0.9], [0.75, 320, 0.8]]) {
      const src = ctx.createBufferSource();
      src.buffer = this.noise; src.loop = true; src.playbackRate.value = rate;
      const bp = ctx.createBiquadFilter();
      bp.type = "bandpass"; bp.frequency.value = freq; bp.Q.value = q;
      src.connect(bp).connect(this.murmur);
      src.start();
    }
    // babble: the level wanders like a room full of conversations
    this.babble = setInterval(() => {
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
    src.connect(hp).connect(bp).connect(g).connect(this.out);
    src.start(t, Math.random() * 2); src.stop(t + hold + 2.1);
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
    o.connect(g).connect(this.out);
    o.start(t); o.stop(t + 0.65);
  }

  applause(seconds = 2, intensity = 0.7) {
    const ctx = this.ctx, t0 = ctx.currentTime;
    const hp = ctx.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 1100;
    hp.connect(this.out);
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

  ooh(size = 0.6, falling = false) {
    const ctx = this.ctx, t = ctx.currentTime, dur = 1.1;
    const f1 = ctx.createBiquadFilter(); f1.type = "bandpass"; f1.frequency.value = falling ? 700 : 450; f1.Q.value = 3;
    const f2 = ctx.createBiquadFilter(); f2.type = "bandpass"; f2.frequency.value = falling ? 1150 : 850; f2.Q.value = 4;
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(0.5 * size, t + 0.15);
    g.gain.exponentialRampToValueAtTime(0.001, t + dur);
    f1.connect(g); f2.connect(g); g.connect(this.out);
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
}

// =========================================================================
// helpers
// =========================================================================
// offlineOnly: only installed voices (after an online "Natural" voice failed).
function pickVoices(offlineOnly = false) {
  const voices = speechSynthesis.getVoices()
    .filter((v) => !/undefined/i.test(v.name))                     // Edge 150 bug
    .filter((v) => !offlineOnly || v.localService);
  const first = (prefs, exclude) => {
    for (const pref of prefs) {
      const v = voices.find((v) => pref(v) && v !== exclude);
      if (v) return v;
    }
    return null;
  };
  const ray = first(RAY_PREFS, null);
  return { ray, sonia: first(SONIA_PREFS, ray) };
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

// The 🎙 setting from earlier sessions, readable before audio is allowed to start.
export function savedVoicePref() { return loadPrefs().voice ?? true; }

function loadPrefs() {
  try { return JSON.parse(localStorage.getItem(PREFS_KEY)) || {}; } catch { return {}; }
}
function savePrefs(p) {
  try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch { /* private mode etc. */ }
}
