// The Bang bang run plot: a Plotly chart, so people can zoom, pan and hover after a run.
//
//   const p = new RunPlot("plot-bb");
//   p.start();                          // new run: clear, and redraw ~10x a second
//   p.push(target, reflection, left, right);   // one sample, stamped with the time
//   p.stop();                           // freeze; the chart stays fully interactive
//
// Zoom is kept while a run is live and reset when the next run starts.
class RunPlot {
    constructor(divId) {
        this.div = document.getElementById(divId);
        this.series = [
            { name: "Target reflection", color: "#57534e", dash: "dash" },
            { name: "Measured reflection", color: "#15803d", dash: "dash" },
            { name: "Left motor", color: "#0369a1" },
            { name: "Right motor", color: "#be185d" },
        ];
        this.runs = 0;
        this.timer = null;
        this.reset();
        this.draw(true);
        // the chart is created while its tab is hidden (zero width): resize it when shown
        new ResizeObserver(() => { if (this.div.clientWidth > 0) Plotly.Plots.resize(this.div); })
            .observe(this.div);
    }

    reset() {
        this.t0 = performance.now();
        this.x = [];
        this.y = this.series.map(() => []);
    }

    start() {
        this.runs += 1;
        this.reset();
        this.draw();
        clearInterval(this.timer);
        this.timer = setInterval(() => this.draw(), 100);
    }

    stop() {
        if (this.timer === null) return;
        clearInterval(this.timer);
        this.timer = null;
        this.draw();
    }

    push(...values) {
        if (this.timer === null) return;
        this.x.push((performance.now() - this.t0) / 1000);
        values.forEach((v, i) => this.y[i].push(v));
    }

    draw(first = false) {
        const css = getComputedStyle(this.div);       // each tab sets its own --scope-* colors
        const bg = css.getPropertyValue("--scope-bg").trim();
        const grid = css.getPropertyValue("--scope-grid").trim();
        const text = css.getPropertyValue("--scope-text").trim();
        // copies, because Plotly.react decides whether anything changed by comparing array
        // identity: handing it the same (growing) arrays would look like "no change"
        const data = this.series.map((s, i) => ({
            x: this.x.slice(), y: this.y[i].slice(), name: s.name, type: "scatter", mode: "lines",
            // stepped ("hv"): each value is held flat until the next sample, like the motors hold a speed
            line: { color: s.color, width: s.width || 2, dash: s.dash || "solid", shape: "hv" },
        }));
        const axis = (title, extra) => ({
            title: { text: title }, gridcolor: grid, zerolinecolor: grid, linecolor: grid, ...extra,
        });
        const layout = {
            paper_bgcolor: bg, plot_bgcolor: bg, font: { color: text, size: 12 },
            margin: { l: 50, r: 16, t: 10, b: 44 },
            xaxis: axis("seconds into the run", {}),
            yaxis: axis("reflection / motor speed", { range: [-100, 100] }),
            legend: { orientation: "h", y: 1.12, x: 0 },
            uirevision: "run" + this.runs,         // keep the user's zoom until the next run
        };
        const config = { responsive: true, displaylogo: false };
        first ? Plotly.newPlot(this.div, data, layout, config) : Plotly.react(this.div, data, layout, config);
    }
}
window.RunPlot = RunPlot;
