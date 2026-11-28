# Project - Final: Jim, an LLM on the Arduino UNO Q

Due Dec 8, 10:30am. 30 points (works / creative / looks great, 10 each).

Jim is an Arduino UNO Q (4 GB) that takes plain-English commands over MQTT,
plans them with a small language model running on the board itself
(llama.cpp, no cloud), and carries them out with a fixed set of skills:
the LED matrix, an MPU-6050 IMU, and MQTT replies.

## Assignment requirements and where they're met

| Requirement | Where |
|---|---|
| Commands on `ws://test.mosquitto.org:8080`, topic `ME193/q/will/cmd` | [unoq_app/python/main.py](unoq_app/python/main.py) (paho-mqtt over websockets) |
| `/status`: running, and what it's doing right now | `/status` → `ME193/q/will/status`, answered from the MQTT thread so it works mid-command |
| Run commands through an LLM, with skills | [planner.py](unoq_app/python/planner.py) (prompt + JSON-schema grammar) → [skills.py](unoq_app/python/skills.py) |
| Return requested info (yaw etc.) or "I am done" | `send_message` skill with live values (`{yaw}`, `{temperature}`, ...) filled in when the step runs |
| Knows its limits | `refuse` skill, plan validation, honest "I can't read yaw: no IMU is connected" instead of made-up numbers |
| Compare two or more models | [llm/bench](llm/bench) (6 models benchmarked on the board, 52 test commands) |

## How it works

```
 laptop / instructor                         UNO Q "jim"
 ───────────────────                         ─────────────────────────────────────────────────────────
 MQTT ME193/q/will/cmd ──► test.mosquitto.org ──► Python app (Docker, App Lab)
                                                   │  /status /stop /help   → answered immediately
                                                   │  plain English         → queue → worker thread
                                                   │      │
                                                   │      ├─► planner: llama-server (host, port 8090)
                                                   │      │     Qwen3-1.7B Q4_K_M, JSON-schema grammar
                                                   │      │     (rule planner if the model is down/warming)
                                                   │      ├─► validate(): known skills, sane values,
                                                   │      │     no command/system topics
                                                   │      └─► executor: show_text, show_icon, clear_display,
                                                   │            reset_yaw, wait, send_message, refuse
                                                   │
                                                   └─ Bridge ─► STM32 sketch: LED matrix frames,
                                                                MPU-6050 at 100 Hz (Qwiic or SDA/SCL)
 replies ◄── ME193/q/will/reply (or the topic the command names), /status, /log (JSON per command)
```

Design decisions, each from a measurement:

- **The grammar fixes the structure; validation checks the values.** llama-server turns
  the plan's JSON schema into a grammar, so the output is always parseable. Values (icon
  names, wait times) are left free in the grammar and checked afterwards, because a
  value-limiting grammar forces wrong answers (see the model comparison below).
- **The prompt says what jim can't do,** and every answer must come from a live value.
- **Prompt caching + warm-up.** Processing the ~800-token prompt cold takes ~2 min on the
  A53 cores. llama-server keeps it cached; the app sends one warm-up request whenever the
  server starts, and uses the rule planner until that finishes.
- **The safety path doesn't depend on the model.** `/stop` cancels the running plan (even
  mid-LLM-call), drops the queue, and blanks the matrix from the MQTT thread.

## Model comparison

Every number below was measured on jim itself (4x Cortex-A53, 4 GB) by
[llm/bench/run_bench.py](llm/bench/run_bench.py). It runs each model in llama-server with the
production flags and sends every command in [cases.jsonl](llm/bench/cases.jsonl) through the
app's own `planner.py`. A plan counts as correct when it uses the right skills in the right
order with the right arguments. For out-of-domain commands ("drive forward", "capital of
France?"), the only correct answer is a refusal. Raw per-case results are in
[llm/bench/results/](llm/bench/results).

**Run 1: which model family?** (prompt v1, grammar on, 40 commands)

| Model (Q4_K_M) | Correct | In-domain | Out-of-domain refused | Instructor's 2 commands | Peak RAM | Median / max per command |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 22% | 32% | 0% | 1 | 678 MB | 5.2 / 12.3 s |
| Gemma-3-1B | 50% | 68% | 8% | 1 | 1290 MB | 7.5 / 15.1 s |
| Llama-3.2-1B | 42% | 57% | 8% | 1 | 998 MB | 13.3 / 31.7 s |
| Qwen2.5-1.5B | 70% | 79% | 50% | 2 | 1276 MB | 10.7 / 30.2 s |
| **Qwen3-1.7B** | **85%** | **96%** | **58%** | **2** | 1428 MB | 16.9 / 27.5 s |

Every model produced valid JSON 100% of the time, because the grammar guarantees it. What
separated them was judgement. The 1B models answered "drive forward 1 meter" with the message
"Driving forward 1 meter." The 0.5B model added an "I am done" message to almost everything.

**Run 2: fixing the prompt, and checking for overfitting.** (52 commands; #41-52 were written
*before* trying prompt v2 and kept out of tuning)

| Configuration | First 40 | Held-out 12 | Out-of-domain refused | Median per command |
|---|---|---|---|---|
| Qwen3-0.6B, v2 | 75% | 67% | 39% | 11.8 s |
| Qwen2.5-1.5B, v2 | 90% | 75% | 67% | 12.3 s |
| Qwen3-1.7B, v1 (control) | 85% | 75% | 56% | 16.6 s |
| Qwen3-1.7B, v2 | 92% | 83% | 89% | 17.6 s |
| Qwen3-1.7B, v2, **no grammar** (control) | 92% | 92% | 94% | 17.8 s |
| **Qwen3-1.7B, v3 (shipping)** | **98%** | **100%**\* | **94%** | 17.7 s |

\* Two of v3's three changes came from failures on held-out cases #43 and #50, so for v3 the
held-out set is no longer unseen. A fresh set of commands is needed for a fair number.

What each change taught us:

- **v1 → v2 (prompt):** v1 never said what jim *lacks*. Qwen3 then answered "what's the
  humidity?" and "weather in Boston?" with its own `{temperature}`, and "capital of
  France?" from memory. v2 lists what jim doesn't have and requires every answer to come
  from a live value. Out-of-domain refusals went from 56% to 89%, and the held-out score
  improved too, so this wasn't just memorising the test.
- **Grammar vs no grammar:** with the few-shot examples, Qwen3-1.7B wrote valid JSON 100% of
  the time even with no grammar. The grammar's real effect showed up in values. With the
  icon list in the grammar, "show a picture of a cat" became a heart, because the model had to
  pick something. With `wait` capped at 30 s in the grammar, "wait 5 minutes" became 30 s.
  **A grammar that limits values turns an honest mistake into a confident wrong answer.**
  v3 keeps the grammar for structure, but lets the model write any icon name or wait time,
  and `validate()` refuses the ones jim can't do.
- **v3 placeholder repair:** every configuration sometimes wrote the bare word ("temperature")
  instead of the live value (`{temperature}`). If a whole message is one live-value name that
  the user didn't put in quotes, the planner now swaps in the live value.

**Why Qwen3-1.7B:** it's the most accurate model and the best at refusing. It's also the
slowest (median 17.7 s per command, 2.1 tokens/s generation, ~2.5 min to process the prompt
cold) and the largest (1.4 GB of the board's 3.6 GB). The rubric rewards working every time
and knowing its limits, so we traded speed for accuracy. Qwen2.5-1.5B would be the pick if
speed mattered more: about 5 s faster per command, but it refused only 67% of out-of-domain
commands.

**Still wrong in v3:** "make the LED matrix blue" shows the word "blue" instead of refusing.

## Run it

```bash
# one-time board setup (no sudo, no compiler needed): llama.cpp, models, systemd user service
scp -r llm arduino@<board-ip>:~/final-llm && ssh arduino@<board-ip> bash ~/final-llm/setup_jim.sh

# deploy the app (board address in tools/board_host.txt, gitignored)
python tools/deploy.py

# talk to it
python tools/mqtt_cmd.py 'Type "Hi" on the LED matrix and then send back on MQTT on topic fred message "I am done"'
python tools/mqtt_cmd.py /status
python tools/mqtt_cmd.py --listen

# tests that need no board
python tests/test_offline.py
```

From Git Bash, prefix slash commands with `MSYS_NO_PATHCONV=1` (otherwise `/status` becomes a Windows path).

## Files

| Path | What |
|---|---|
| `unoq_app/` | The App Lab app (`python/` runs in Docker on the board, `sketch/` on the STM32) |
| `llm/setup_jim.sh`, `llm/llama-server.service` | Board setup and the LLM service |
| `llm/bench/` | Benchmark: `cases.jsonl` (52 commands; 41-52 held out), `run_bench.py`, `results/` |
| `tools/` | `deploy.py`, `mqtt_cmd.py` |
| `tests/test_offline.py` | Rule planner, validation and executor checks with fakes |

## Status

- [x] Stage 1: fixed command with no model (rule planner), MQTT, `/status`
- [x] Stage 2: on-board LLM (llama.cpp), model benchmark
- [x] Stage 3: JSON-schema grammar, plan validation, refusals
- [x] Stage 4 (first skills): LED matrix text/icons, IMU values, waits, MQTT replies
- [ ] Build Buddy (the creative layer): part recognition, parametric CAD, wiring guide, voice
