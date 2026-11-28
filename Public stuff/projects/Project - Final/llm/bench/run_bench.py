#!/usr/bin/env python3
"""Benchmark candidate LLMs as jim's command planner, on jim itself.

For each model: start llama-server with the same flags as the production
service, measure load time and memory, then send every command in
cases.jsonl through the app's own planner.py (same prompt, same grammar,
same validation) and score the plan against the expected one.

    python3 run_bench.py --planner ~/ArduinoApps/jim/python --models ~/models/*.gguf
    python3 run_bench.py ... --no-grammar        # same, without the JSON grammar

Writes results/<tag>_<model>.json (every case) and prints a summary table.
Standard library only (runs on the board's own Python 3.13).

Scoring per case:
  json_ok    the model's text parsed as JSON
  valid      planner.validate() accepted it (known skills, sane arguments)
  correct    the validated steps match the expected ones: same skills in
             the same order, and for each expected argument either an exact
             (case/punctuation-insensitive) match, {"contains": [...]} all
             present, or {"any": true}. A refuse step matches any reason.
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Same forbidden set as the app (main.FORBIDDEN_TOPICS).
FORBIDDEN = {"ME193/q/will/cmd", "ME193/q/will/status", "ME193/q/will/log", "ME193/q/will/online"}
SERVER_FLAGS =["-c", "2048", "-t", "4", "-np", "1", "-cram", "256", "--jinja"]


def norm(s):
    return str(s).strip().strip(".!?\"'“”").strip().lower()


def arg_ok(want, got):
    if isinstance(want, dict):
        if want.get("any"):
            return got is not None and str(got).strip() != ""
        return all(c.lower() in str(got).lower() for c in want["contains"])
    if isinstance(want, (int, float)):
        try:
            return abs(float(got) - want) < 1e-6
        except (TypeError, ValueError):
            return False
    return norm(want) == norm(got)


def plan_ok(expect, steps):
    if [e["skill"] for e in expect] != [s["skill"] for s in steps]:
        return False
    for e, s in zip(expect, steps):
        if e["skill"] == "refuse":
            continue
        for k, v in e.items():
            if k != "skill" and not arg_ok(v, s.get(k)):
                return False
    return True


def mem_kb(pid, field):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def mem_available_mb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024


def wait_healthy(url, proc, timeout=300):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return time.monotonic() - t0
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("llama-server never became healthy")


def run_model(model_path, args, planner, cases):
    name = Path(model_path).stem
    url = f"http://127.0.0.1:{args.port}"
    log = open(HERE / "results" / f"server_{name}.log", "w")
    mem_before = mem_available_mb()
    proc = subprocess.Popen([args.server, "-m", model_path, "--host", "127.0.0.1",
                             "--port", str(args.port), *SERVER_FLAGS],
                            stdout=log, stderr=subprocess.STDOUT)
    out = {"model": name, "file_mb": os.path.getsize(model_path) // 2**20,
           "grammar": not args.no_grammar, "cases": []}
    try:
        out["load_s"] = round(wait_healthy(url, proc), 1)
        llm = planner.LLMPlanner(url=url, timeout=300, use_grammar=not args.no_grammar,
                                 prompt_version=args.prompt)
        out["prompt"] = args.prompt

        # Cold request: the whole system prompt + examples get processed once.
        # (A phrase that isn't one of the cases, so no case gets a free cache hit.)
        _, info = llm.plan("hello there")
        out["cold_prompt_tokens"] = info["prompt_tokens"]
        out["cold_prompt_ms"] = round(info["prompt_ms"] or 0)
        out["cold_total_ms"] = round(info["ms"])

        for case in cases:
            rec = {"id": case["id"], "cat": case["cat"], "cmd": case["cmd"],
                   "heldout": bool(case.get("heldout"))}
            try:
                plan, info = llm.plan(case["cmd"])
                rec.update(json_ok=True, raw=info["raw"], ms=round(info["ms"]),
                           prompt_tokens=info["prompt_tokens"], prompt_ms=info["prompt_ms"],
                           gen_tokens=info["gen_tokens"], gen_ms=info["gen_ms"])
                try:
                    steps = planner.validate(plan, FORBIDDEN)
                    rec.update(valid=True, steps=steps)
                except planner.PlanError as e:
                    # The app turns a rejected plan into a refusal; score what the user would see.
                    steps = [{"skill": "refuse", "reason": f"(validation) {e}"}]
                    rec.update(valid=False, steps=steps, error=str(e))
                rec["correct"] = plan_ok(case["expect"], steps)
            except planner.PlanError as e:
                rec.update(json_ok=False, valid=False, correct=False, error=str(e))
            except OSError as e:
                rec.update(json_ok=False, valid=False, correct=False, error=f"request failed: {e}")
            out["cases"].append(rec)
            mark = "ok " if rec["correct"] else "BAD"
            print(f"  {mark} #{case['id']:>2} {rec.get('ms', 0):>6} ms  {case['cmd'][:50]!r} -> "
                  f"{json.dumps(rec.get('steps', rec.get('error')))[:110]}", flush=True)
        out["rss_mb"] = (mem_kb(proc.pid, "VmRSS") or 0) // 1024
        out["peak_rss_mb"] = (mem_kb(proc.pid, "VmHWM") or 0) // 1024
        out["mem_available_drop_mb"] = mem_before - mem_available_mb()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    return out


def summarize(r):
    cs = r["cases"]
    ind = [c for c in cs if c["cat"] != "ood"]
    ood = [c for c in cs if c["cat"] == "ood"]
    timed = [c for c in cs if c.get("gen_ms")]

    def pct(xs, key):
        return round(100 * sum(bool(c.get(key)) for c in xs) / max(1, len(xs)))

    false_refusals = sum(1 for c in ind if c.get("steps") and c["steps"][0]["skill"] == "refuse")
    tuned = [c for c in cs if not c.get("heldout")]
    held = [c for c in cs if c.get("heldout")]
    return {
        "model": r["model"], "prompt": r.get("prompt", "v1"), "grammar": r["grammar"],
        "file_mb": r["file_mb"],
        "first40_correct_pct": pct(tuned, "correct"),
        "heldout_correct_pct": pct(held, "correct") if held else None,
        "load_s": r.get("load_s"), "peak_rss_mb": r.get("peak_rss_mb"),
        "mem_drop_mb": r.get("mem_available_drop_mb"),
        "cold_prompt_tok_s": round(r["cold_prompt_tokens"] / (r["cold_prompt_ms"] / 1000), 1)
        if r.get("cold_prompt_ms") and r.get("cold_prompt_tokens") else None,
        "cold_prompt_s": round(r.get("cold_prompt_ms", 0) / 1000, 1),
        "gen_tok_s": round(sum(c["gen_tokens"] for c in timed) / (sum(c["gen_ms"] for c in timed) / 1000), 2)
        if timed else None,
        "median_s": round(statistics.median(c["ms"] for c in cs if c.get("ms")) / 1000, 1)
        if any(c.get("ms") for c in cs) else None,
        "max_s": round(max((c.get("ms") or 0) for c in cs) / 1000, 1),
        "json_ok_pct": pct(cs, "json_ok"),
        "valid_pct": pct(cs, "valid"),
        "correct_pct": pct(cs, "correct"),
        "in_domain_correct_pct": pct(ind, "correct"),
        "ood_refused_pct": pct(ood, "correct"),
        "false_refusals": false_refusals,
        "instructor_cases_ok": sum(1 for c in cs if c["cat"] == "instructor" and c["correct"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default=os.path.expanduser("~/ArduinoApps/jim/python"))
    ap.add_argument("--server", default=os.path.expanduser("~/llama/current/llama-server"))
    ap.add_argument("--models", nargs="+", default=sorted(glob.glob(os.path.expanduser("~/models/*.gguf"))))
    ap.add_argument("--cases", default=str(HERE / "cases.jsonl"))
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    args = ap.parse_args()

    sys.path.insert(0, args.planner)
    import planner  # the app's own planner.py

    cases = [json.loads(l) for l in open(args.cases) if l.strip()]
    if args.limit:
        cases = cases[:args.limit]
    (HERE / "results").mkdir(exist_ok=True)
    tag = ("nogrammar" if args.no_grammar else "grammar") + "_" + args.prompt
    summaries = []
    for m in args.models:
        print(f"\n=== {Path(m).stem} ({tag}) ===", flush=True)
        try:
            r = run_model(m, args, planner, cases)
        except Exception as e:
            print(f"  failed: {e}", flush=True)
            continue
        (HERE / "results" / f"{tag}_{r['model']}.json").write_text(json.dumps(r, indent=1))
        try:
            s = summarize(r)
        except Exception as e:
            print(f"  couldn't summarize: {e!r}", flush=True)
            continue
        summaries.append(s)
        print("  " + json.dumps(s), flush=True)
    (HERE / "results" / f"summary_{tag}.json").write_text(json.dumps(summaries, indent=1))

    cols = ["model", "prompt", "first40_correct_pct", "heldout_correct_pct", "file_mb", "peak_rss_mb", "load_s", "cold_prompt_s", "gen_tok_s", "median_s",
            "max_s", "json_ok_pct", "valid_pct", "correct_pct", "in_domain_correct_pct",
            "ood_refused_pct", "false_refusals", "instructor_cases_ok"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for s in summaries:
        print("| " + " | ".join(str(s.get(c)) for c in cols) + " |")


if __name__ == "__main__":
    main()
