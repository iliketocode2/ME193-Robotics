// A small real-time oscilloscope on a <canvas>: scrolling traces, auto-sized to the window.
//
//   const s = new Scope("scope-light", {min: 0, max: 100, window: 10,
//                       traces: [{name: "light", color: "#fbbf24"}]});
//   s.push(0, 42);          // trace 0, value 42, stamped with the current time
//
// It redraws itself every animation frame, so the trace keeps scrolling (and flat-lines
// honestly) even when no new data arrives.
class Scope {
    constructor(canvasId, opts) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext("2d");
        this.min = opts.min;
        this.max = opts.max;
        this.window = opts.window || 10;           // seconds of history on screen
        this.pausedAt = null;
        this.traces = opts.traces.map((t) => ({ ...t, t: [], v: [] }));
        const loop = () => { this.draw(); requestAnimationFrame(loop); };
        requestAnimationFrame(loop);
    }

    // pause() freezes the display (and ignores new data) so a finished run can be studied;
    // resume() carries on live. A paused scope keeps showing the moment it was paused.
    pause() { if (this.pausedAt === null) this.pausedAt = performance.now() / 1000; }
    resume() { this.pausedAt = null; }

    push(i, value) {
        if (this.pausedAt !== null) return;
        if (value === null || value === undefined || Number.isNaN(value)) return;
        const tr = this.traces[i];
        const now = performance.now() / 1000;
        tr.t.push(now);
        tr.v.push(value);
        const cutoff = now - this.window - 1;
        while (tr.t.length && tr.t[0] < cutoff) { tr.t.shift(); tr.v.shift(); }
    }

    clear() { this.traces.forEach((tr) => { tr.t = []; tr.v = []; }); }

    draw() {
        const c = this.canvas, ctx = this.ctx;
        const dpr = window.devicePixelRatio || 1;
        const w = c.clientWidth, h = c.clientHeight;
        if (w < 60 || h < 40) return;              // hidden (e.g. on another tab)
        if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
            c.width = Math.round(w * dpr);
            c.height = Math.round(h * dpr);
        }
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        const css = getComputedStyle(document.documentElement);
        ctx.fillStyle = css.getPropertyValue("--scope-bg"); ctx.fillRect(0, 0, w, h);

        const left = 38, top = 8, right = 8, bottom = 18;
        const pw = w - left - right, ph = h - top - bottom;
        const y = (v) => top + ph * (1 - (v - this.min) / (this.max - this.min));

        // grid + axis labels
        ctx.font = "11px ui-monospace, Menlo, monospace";
        ctx.textBaseline = "middle"; ctx.textAlign = "right";
        ctx.strokeStyle = css.getPropertyValue("--scope-grid"); ctx.lineWidth = 1;
        for (let k = 0; k <= 4; k++) {
            const v = this.min + (this.max - this.min) * k / 4;
            ctx.beginPath(); ctx.moveTo(left, y(v) + .5); ctx.lineTo(left + pw, y(v) + .5); ctx.stroke();
            ctx.fillStyle = css.getPropertyValue("--scope-text"); ctx.fillText(Math.round(v), left - 5, y(v));
        }
        ctx.textAlign = "center"; ctx.textBaseline = "top";
        for (let s = 0; s <= this.window; s += 2) {
            const x = left + pw * (1 - s / this.window);
            ctx.beginPath(); ctx.moveTo(x + .5, top); ctx.lineTo(x + .5, top + ph); ctx.stroke();
            ctx.fillStyle = css.getPropertyValue("--scope-text"); ctx.fillText(s === 0 ? "now" : "-" + s + "s", x, top + ph + 3);
        }

        // traces
        const now = this.pausedAt !== null ? this.pausedAt : performance.now() / 1000;
        ctx.save();
        ctx.beginPath(); ctx.rect(left, top, pw, ph); ctx.clip();
        for (const tr of this.traces) {
            ctx.strokeStyle = tr.color; ctx.lineWidth = 2; ctx.lineJoin = "round";
            ctx.setLineDash(tr.dash ? [6, 4] : []);
            ctx.beginPath();
            tr.t.forEach((t, j) => {
                const x = left + pw * (1 - (now - t) / this.window);
                const yy = y(Math.max(this.min, Math.min(this.max, tr.v[j])));
                j === 0 ? ctx.moveTo(x, yy) : ctx.lineTo(x, yy);
            });
            ctx.stroke();
        }
        ctx.restore();
    }
}
window.Scope = Scope;
