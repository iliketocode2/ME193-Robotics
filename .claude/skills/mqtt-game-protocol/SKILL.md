---
name: mqtt-game-protocol
description: Patterns for coordinating two or more independent robot scripts over MQTT via mqttlib.py -- event schema design, self-echo filtering on the public test.mosquitto.org broker, and gating game state (start/stop/win/lose) from message callbacks. Use when a task needs multiple robots/scripts to coordinate a shared game or competition state over MQTT, not just simple chat/telemetry.
---

# MQTT multi-robot game coordination — ground truth from Project 3 (World Cup)

This documents the pattern used in
`Public stuff/projects/Project 3 - World Cup/world_cup.py` to coordinate a
two-robot (ball vs. goalie) match over MQTT, on top of `mqttlib.py` (see
`MQTTLIB.md` for the base client API — this skill is about the *protocol
design* layered on top of it, not the client itself).

## Start from `mqttlib.MQTTClient`, don't reinvent pub/sub

```python
from mqttlib import MQTTClient

mqtt = MQTTClient()          # defaults to test.mosquitto.org, qos=0
mqtt.connect()
mqtt.subscribe(TOPIC, on_message)   # on_message(topic: str, payload: str)
mqtt.publish(TOPIC, json.dumps({...}))
```

Everything below is about what to put in `on_message` and what to publish
— `mqttlib.py` itself needs no changes for a game protocol.

## The public broker echoes your own publishes back to you

`test.mosquitto.org` is the default broker and it is genuinely
unauthenticated and shared with everyone else using it — including anyone
else in the class on the same topic. **Every subscriber, including the
publisher itself, receives every message published on a topic it's
subscribed to.** `MQTTLIB.md` documents this ("Reading back your own
publish") and `mqtt_chat.py` already works around it for a chat UI
(comparing `sender == self.username`). A game protocol needs the same
filter, or a robot will re-process its own event from its own echo:

```python
def on_message(topic, payload):
    data = json.loads(payload)
    if data.get("team") == MY_ROLE:      # or "sender" == my own id, etc.
        return                            # our own publish, echoed back
    ...
```

Pick a field that identifies *who the event is about* (not just who
published it — for World Cup, `"team": "ball"` on both the ball's own
publish and everyone's receipt of it) and filter on that, rather than
trying to suppress your own `publish()` calls from reaching your own
`subscribe()` callback (there's no clean way to do that with `mqttlib.py`
as written, and you don't want to — the broker round-trip is also a useful
confirmation the message actually sent, per `MQTTLIB.md`'s "Basic usage —
subscribe" example).

## Design the message schema explicitly, and get the other side to agree on it

Unlike a single-robot script, a game protocol only works if **every
participating script agrees on the exact field names and values** — there
is no schema negotiation at the protocol level. Concretely for a
multi-robot competition:

- Pick one **plain, unambiguous trigger message** for shared events that
  come from a referee/instructor rather than a robot (e.g. literal
  `"start"`, not JSON) — simpler to type/publish by hand from any
  MQTT client, and avoids every robot needing to agree on a JSON shape for
  something that isn't robot-to-robot.
- Pick a **small JSON schema** for robot-to-robot events, e.g.
  `{"event": "fail", "team": "ball"}` — an `event` type field plus whatever
  identifies who/what it's about. Keep it minimal; every extra field is
  another thing to get wrong across two independently-written scripts.
- **State the schema explicitly in the project's README** and treat
  "agree with the opponent team before the match" as a hard requirement,
  not an assumption — a script that silently assumes its own schema is
  the only one in play will just never see the opponent's messages if
  they diverge even slightly (a different key name, a different string
  value, wrapping in an extra layer of JSON, etc.).
- Prefer **`qos=0`, no `retain`** for live game state (matches
  `MQTTLIB.md`'s own guidance) — a stale retained "start" or "goal"
  message delivered to a robot that (re)subscribes mid-match is actively
  wrong, not just unhelpful.

## Gate local behavior on message-driven state, from any thread

`mqttlib`'s `subscribe()` callback fires on paho's own background network
thread (`loop_start()`), same as `mqtt_chat.py`'s comment about Tkinter
threading applies here too: don't do hardware/BLE calls or unguarded shared
state writes directly from inside `on_message` if another thread (an audio
callback, a camera loop, a main loop) is also reading/writing that state.
The Project 3 pattern: a lock-guarded dict of flags (`game_active`,
`game_over`) that `on_message` sets, and the actual control loop (on its
own thread) checks before acting:

```python
with state_lock:
    if text.lower() == "start" and not shared["game_over"]:
        shared["game_active"] = True
```

## Common mistakes to catch in review

- No self-echo filter — a robot's own `publish()` immediately re-triggers
  its own `on_message` handler as if the opponent sent it.
- Assuming the opponent's script uses the same message schema without it
  being written down and agreed on anywhere.
- Using `retain=True` for time-sensitive game state (start/stop/win/lose)
  — a late-joining or reconnecting subscriber gets stale state instead of
  waiting for the next real event.
- Doing hardware calls directly inside the MQTT callback thread without
  going through whatever synchronization the rest of the script already
  uses for cross-thread state.
- Publishing to an overly generic topic name (e.g. plain `"ME193"`, the
  chat topic) instead of a specific one — see `MQTTLIB.md`'s crosstalk
  warning; a game protocol especially doesn't want unrelated class traffic
  (or worse, another team's match) triggering event handlers.
