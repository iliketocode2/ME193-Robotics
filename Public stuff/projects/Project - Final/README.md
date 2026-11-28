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

- **Grammar-constrained output.** llama-server turns the plan's JSON schema into a grammar,
  so even the 0.5B model produced valid JSON 100% of the time. The grammar must not be *too*
  tight: capping `wait` at 30 s in the grammar made every model turn "wait 5 minutes" into
  "wait 30". Now the grammar allows up to an hour and `validate()` refuses anything over 30 s.
- **The prompt says what jim can't do.** Prompt v1 led the best model to answer "what's the
  humidity?" with its temperature, and "capital of France?" from memory. v2 lists what jim
  lacks and requires every answer to come from a live value.
- **Prompt caching + warm-up.** Processing the ~800-token prompt cold takes ~2 min on the
  A53 cores. llama-server keeps it cached; the app sends one warm-up request whenever the
  server starts, and uses the rule planner until that finishes.
- **The safety path doesn't depend on the model.** `/stop` cancels the running plan (even
  mid-LLM-call), drops the queue, and blanks the matrix from the MQTT thread.

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
