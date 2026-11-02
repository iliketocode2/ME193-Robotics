"""Uncertainty analysis: known-count trials, false positives vs misses, and the
accelerometer-only ablation. Writes plots/, results/results.md and the README's
results section.

    python evaluate.py
"""

import json
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
import dataset
import stepcount
from model import load_checkpoint

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
       10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
       18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
FALSE_POSITIVE_ACTIVITIES = ["still", "shake", "turn"]
README = config.ROOT / "README.md"


def t_critical(df):
    return T95[max(k for k in T95 if k <= df)] if df >= 1 else math.nan


def trial_stats(rows):
    err = np.array([r["pred"] - r["true"] for r in rows], dtype=float)
    pct = np.array([100 * (r["pred"] - r["true"]) / r["true"] for r in rows])
    n = len(rows)
    sd = pct.std(ddof=1) if n > 1 else math.nan
    half = t_critical(n - 1) * sd / math.sqrt(n) if n > 1 else math.nan
    return {"n": n, "mae": float(np.abs(err).mean()), "mean_pct": float(pct.mean()),
            "mape": float(np.abs(pct).mean()), "sd_pct": float(sd),
            "ci95": [float(pct.mean() - half), float(pct.mean() + half)]}


def evaluate_model(path, sessions):
    model, ckpt = load_checkpoint(path)
    trials = [{"name": s.name, "true": int(s.meta["true_steps"]),
               "pred": len(stepcount.count_session(model, ckpt, s))}
              for s in sessions.values() if s.is_trial and s.meta.get("true_steps")]
    events = {"hits": 0, "misses": 0, "fp_walking": 0, "fp_nonstep": 0, "nonstep_seconds": 0.0}
    per_activity = {}
    for name in ckpt["split"]["test"]:
        s = sessions.get(name)
        if s is None:
            continue
        h, f, m = stepcount.match_events(stepcount.count_session(model, ckpt, s), s.t[s.events > 0])
        events["hits"] += h
        events["misses"] += m
        if s.activity in FALSE_POSITIVE_ACTIVITIES:
            events["fp_nonstep"] += f
            events["nonstep_seconds"] += s.t[-1] - s.t[0]
        else:
            events["fp_walking"] += f
        a = per_activity.setdefault(s.activity, {"true": 0, "hits": 0, "false_pos": 0, "missed": 0})
        a["true"] += h + m
        a["hits"] += h
        a["false_pos"] += f
        a["missed"] += m
    fp = events["fp_walking"] + events["fp_nonstep"]
    events["f1"] = 2 * events["hits"] / max(2 * events["hits"] + fp + events["misses"], 1)
    return {"model": path.name, "channel_set": ckpt["channel_set"], "threshold": ckpt["threshold"],
            "trials": trials, "stats": trial_stats(trials) if trials else None,
            "events": events, "per_activity": per_activity}, model, ckpt


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)


def plot_pred_vs_actual(results, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    top = max(max(r["true"], r["pred"]) for res in results for r in res["trials"]) * 1.1
    ax.plot([0, top], [0, top], color="#8b8d98", lw=1, ls="--", label="perfect")
    for res, color, marker in zip(results, ["#3e63dd", "#f76b15"], ["o", "^"]):
        ax.scatter([r["true"] for r in res["trials"]], [r["pred"] for r in res["trials"]],
                   color=color, marker=marker, s=40, label=f"{res['channel_set']} channels", zorder=3)
    ax.set(xlabel="actual steps (counted by hand)", ylabel="predicted steps", xlim=(0, top), ylim=(0, top),
           title="Predicted vs actual, per trial")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_error_distribution(results, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    pcts = [[100 * (r["pred"] - r["true"]) / r["true"] for r in res["trials"]] for res in results]
    lo, hi = min(min(p) for p in pcts), max(max(p) for p in pcts)
    bins = np.linspace(min(lo, -1) - 1, max(hi, 1) + 1, 16)
    for p, res, color in zip(pcts, results, ["#3e63dd", "#f76b15"]):
        ax.hist(p, bins=bins, color=color, alpha=0.6, label=f"{res['channel_set']} channels")
    ax.axvline(0, color="black", lw=1)
    ax.set(xlabel="percent error per trial (predicted - actual) / actual", ylabel="trials",
           title="Error distribution")
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sample_trace(model, ckpt, session, path):
    times, probs = stepcount.session_probs(model, ckpt, session)
    pred = stepcount.detect_steps(times, probs, ckpt["threshold"])
    true = session.t[session.events > 0]
    end = min(session.t[0] + 15, session.t[-1])
    keep = session.t <= end
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=(11, 5))
    t = session.t[keep] - session.t[0]
    for c, color in zip(range(3), ["#e5484d", "#30a46c", "#3e63dd"]):
        a.plot(t, session.x[keep, c], color=color, lw=0.8, label=config.CHANNELS[c])
    for s in true[true <= end]:
        a.axvline(s - session.t[0], color="black", lw=1, alpha=0.5)
    a.set(ylabel="acceleration (g)", title=f"{session.name}: true steps (black lines) vs predicted (orange)")
    a.legend(frameon=False, ncol=3, loc="upper right", fontsize=8)
    pk = times <= end
    b.plot(times[pk] - session.t[0], probs[pk], color="#3e63dd", lw=1.2, label="network output")
    b.axhline(ckpt["threshold"], color="#8b8d98", ls="--", lw=1, label="threshold")
    p = pred[pred <= end] - session.t[0]
    b.scatter(p, np.interp(p, times - session.t[0], probs), color="#f76b15", zorder=3, label="predicted step")
    for s in true[true <= end]:
        b.axvline(s - session.t[0], color="black", lw=1, alpha=0.5)
    b.set(xlabel="time (s)", ylabel="P(step)", ylim=(-0.05, 1.3))
    b.legend(frameon=False, ncol=3, loc="upper center", fontsize=8)
    for ax in (a, b):
        style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt(v, digits=1):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


def report(results):
    main = results[0]
    lines = []
    if main["stats"]:
        st = main["stats"]
        lines += [f"**{st['n']} known-count trials** with the all-sensor model"
                  + (" (the assignment asks for at least 10: record more with `collect.py --trial-steps N`)."
                     if st["n"] < 10 else "."), "",
                  "| Trial | Actual steps | Predicted | Error | % error |", "| --- | --- | --- | --- | --- |"]
        for r in main["trials"]:
            lines.append(f"| {r['name']} | {r['true']} | {r['pred']} | {r['pred'] - r['true']:+d} | "
                         f"{100 * (r['pred'] - r['true']) / r['true']:+.1f}% |")
        lines += ["", "| Statistic | Value |", "| --- | --- |",
                  f"| Mean absolute error | {fmt(st['mae'])} steps |",
                  f"| Mean percent error (signed) | {fmt(st['mean_pct'])}% |",
                  f"| Mean absolute percent error | {fmt(st['mape'])}% |",
                  f"| Standard deviation of percent error | {fmt(st['sd_pct'])}% |",
                  f"| 95% confidence interval on mean percent error | "
                  f"{fmt(st['ci95'][0])}% to {fmt(st['ci95'][1])}% |", ""]
    else:
        lines += ["No uncertainty trials recorded yet: run `collect.py --trial-steps N` at least 10 times.", ""]

    ev = main["events"]
    lines += ["**False positives vs missed steps** on held-out test sessions (step matched within "
              f"{int(config.MATCH_TOLERANCE_S * 1000)} ms):", "",
              "| Activity | True steps | Detected | False positives | Missed |", "| --- | --- | --- | --- | --- |"]
    for act, a in main["per_activity"].items():
        lines.append(f"| {act} | {a['true']} | {a['hits']} | {a['false_pos']} | {a['missed']} |")
    rate = ev["fp_nonstep"] / (ev["nonstep_seconds"] / 60) if ev["nonstep_seconds"] else None
    lines += ["", f"- False positives while standing / shaking / turning: **{ev['fp_nonstep']}**"
              + (f" ({fmt(rate)} per minute)" if rate is not None else ""),
              f"- False positives while walking (extra steps): **{ev['fp_walking']}**",
              f"- Missed steps: **{ev['misses']}**",
              f"- Step-event F1: **{ev['f1']:.3f}**", ""]

    if len(results) > 1:
        acc = results[1]
        lines += ["**Ablation: accelerometer only vs all six channels**", "",
                  "| Model | Trial MAE (steps) | Mean abs % error | Test step F1 | False pos. (non-step) | Missed |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in results:
            st, ev = r["stats"] or {}, r["events"]
            lines.append(f"| {r['channel_set']} | {fmt(st.get('mae'))} | {fmt(st.get('mape'))}% | "
                         f"{ev['f1']:.3f} | {ev['fp_nonstep']} | {ev['misses']} |")
        d = main["events"]["f1"] - acc["events"]["f1"]
        lines += ["", f"Adding the gyroscope changes test F1 by **{d:+.3f}**"
                  + (f" and trial MAE by **{main['stats']['mae'] - acc['stats']['mae']:+.1f} steps**."
                     if main["stats"] and acc["stats"] else "."), ""]
    else:
        lines += ["Ablation: run `python train.py --channels accel` and re-run this script.", ""]

    plots = [("Predicted vs actual", "predicted_vs_actual.png"), ("Error distribution", "error_distribution.png"),
             ("Sample trace", "sample_trace.png"), ("Training curves", "training_curves_all.png")]
    for title, f in plots:
        if (config.PLOTS_DIR / f).exists():
            lines += [f"![{title}](plots/{f})", ""]
    return "\n".join(lines)


def update_readme(text):
    if not README.exists():
        return
    body = README.read_text(encoding="utf-8")
    new = re.sub(r"(<!-- RESULTS:START -->\n).*?(\n<!-- RESULTS:END -->)",
                 lambda m: m.group(1) + text + m.group(2), body, flags=re.S)
    README.write_text(new, encoding="utf-8")


def main():
    paths = [config.MODELS_DIR / f"step_cnn_{c}.pt" for c in ("all", "accel")]
    paths = [p for p in paths if p.exists()]
    if not paths or paths[0].name != "step_cnn_all.pt":
        raise SystemExit("Train first: python train.py")
    sessions = {s.name: s for s in dataset.load_sessions()}
    results, models = [], []
    for p in paths:
        res, model, ckpt = evaluate_model(p, sessions)
        results.append(res)
        models.append((model, ckpt))

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if results[0]["trials"]:
        plot_pred_vs_actual(results, config.PLOTS_DIR / "predicted_vs_actual.png")
        plot_error_distribution(results, config.PLOTS_DIR / "error_distribution.png")
    model, ckpt = models[0]
    candidates = [sessions[n] for n in ckpt["split"]["test"] if sessions[n].activity == "walk"]
    candidates += [s for s in sessions.values() if s.is_trial and s.labeled]
    candidates += [sessions[n] for n in ckpt["split"]["test"]]
    if candidates:
        plot_sample_trace(model, ckpt, candidates[0], config.PLOTS_DIR / "sample_trace.png")

    text = report(results)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "results.md").write_text(text, encoding="utf-8")
    (config.RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    update_readme(text)
    print(text)


if __name__ == "__main__":
    main()
