"""Command text -> a plan: a short list of skill steps.

Two planners share one output format, {"steps": [{"skill": ..., ...}, ...]}:

  RulePlanner  regular expressions, no model. Milestone 1 ("type Hi on the
               matrix and send I am done") and the fallback when the LLM
               server is down.
  LLMPlanner   asks llama-server (running on jim's Linux side, outside the
               app container) with a JSON-schema grammar, so the model can
               only ever produce a well-formed plan made of known skills.

validate() is applied to every plan whatever produced it: a plan that names
an unknown skill, a wildcard MQTT topic, an absurd wait, etc. never reaches
the hardware.

Standard library only, so llm/bench/run_bench.py imports this exact file and
benchmarks what the app really runs.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

ICONS = ["heart", "smile", "sad", "check", "x",
         "arrow_up", "arrow_down", "arrow_left", "arrow_right"]
PLACEHOLDERS = ["yaw", "pitch", "roll", "temperature", "accel", "gyro",
                "time", "date", "status"]
MAX_STEPS = 6
MAX_WAIT_S = 30
MAX_TEXT = 60
MAX_MESSAGE = 200

# --- JSON schema (llama-server turns it into a GBNF grammar) ---------------


def _step(skill, props=None, required=()):
    properties = {"skill": {"const": skill}}
    properties.update(props or {})
    return {"type": "object", "properties": properties,
            "required": ["skill", *required], "additionalProperties": False}


def _schema(grammar_max_wait, icon_enum=True):
    icon = {"enum": ICONS} if icon_enum else {"type": "string", "minLength": 1, "maxLength": 20}
    steps = [
        _step("show_text", {"text": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT}}, ["text"]),
        _step("show_icon", {"icon": icon}, ["icon"]),
        _step("clear_display"),
        _step("reset_yaw"),
        _step("wait", {"seconds": {"type": "integer", "minimum": 0, "maximum": grammar_max_wait}}, ["seconds"]),
        _step("send_message", {"topic": {"type": "string", "maxLength": 60},
                               "message": {"type": "string", "minLength": 1, "maxLength": MAX_MESSAGE}},
              ["topic", "message"]),
        _step("refuse", {"reason": {"type": "string", "minLength": 1, "maxLength": 120}}, ["reason"]),
    ]
    return {
        "type": "object",
        "properties": {"steps": {"type": "array", "minItems": 1, "maxItems": MAX_STEPS,
                                 "items": {"anyOf": steps}}},
        "required": ["steps"],
        "additionalProperties": False,
    }


# v1 capped wait at 30 s in the grammar itself. Benchmarking showed why that's
# wrong: asked to "wait 5 minutes", every model wrote 30, because the grammar
# left it no way to say 300. v2 lets the model write what it means (up to an
# hour) and validate() refuses anything over MAX_WAIT_S, so the user is told
# instead of being quietly given a different wait.
#
# v3 applies the same lesson to icons: with the icon list in the grammar, "show
# a picture of a cat" became a heart (the model had to pick *something* from
# the list). Without it the model writes "cat", validate() rejects it, and jim
# says it has no cat icon. The grammar still fixes the structure.
PLAN_SCHEMAS = {"v1": _schema(MAX_WAIT_S), "v2": _schema(3600), "v3": _schema(3600, icon_enum=False)}
SKILLS = [s["properties"]["skill"]["const"]
          for s in PLAN_SCHEMAS["v2"]["properties"]["steps"]["items"]["anyOf"]]

# --- prompt ------------------------------------------------------------------
# Kept identical for every command so llama-server can reuse the processed
# prompt (KV cache) and only has to read the new command each time.

PROMPT_VERSION = "v3"     # what the app uses; the benchmark can pick any version

SYSTEM_PROMPT_V1 = f"""You plan commands for "jim", an Arduino UNO Q robot. Turn the user's command into a JSON plan using ONLY these skills:
- show_text {{text}}: show text on the 8x13 LED matrix. Short text stays still, long text scrolls.
- show_icon {{icon}}: show a picture. icon is one of: {", ".join(ICONS)}.
- clear_display {{}}: turn the LED matrix off.
- reset_yaw {{}}: make the current heading yaw = 0.
- wait {{seconds}}: pause for 0 to {MAX_WAIT_S} seconds.
- send_message {{topic, message}}: publish an MQTT message. topic is "" unless the user names a topic. Use it to answer questions and to say "I am done" when the user asks to be told.
- refuse {{reason}}: use it ALONE when any part of the command needs something these skills cannot do, such as driving motors, other hardware, the internet, or opinions.
Text and messages may contain live values that are filled in when the step runs: {" ".join("{" + p + "}" for p in PLACEHOLDERS)}. yaw, pitch and roll are in degrees, temperature in C.
Rules: copy the user's exact words for text and messages. Keep the order the user asked for. Never invent numbers; use the live values. Only send a message when the user asks for information or asks to be told something."""

FEW_SHOT_V1 = [
    ('Write "Yo" on the LED matrix and then send back on MQTT on topic alice message "finished"',
     {"steps": [{"skill": "show_text", "text": "Yo"},
                {"skill": "send_message", "topic": "alice", "message": "finished"}]}),
    ("what is your yaw right now?",
     {"steps": [{"skill": "send_message", "topic": "", "message": "My yaw is {yaw} degrees"}]}),
    ("show a smiley face, wait 3 seconds, then turn the display off and tell me when you're done",
     {"steps": [{"skill": "show_icon", "icon": "smile"}, {"skill": "wait", "seconds": 3},
                {"skill": "clear_display"},
                {"skill": "send_message", "topic": "", "message": "I am done"}]}),
    ("spin the left wheel twice",
     {"steps": [{"skill": "refuse", "reason": "I have no motors connected."}]}),
]

# v2: written after the first benchmark (results in llm/bench/results). The
# failures were (a) answering from the wrong sensor ("humidity" -> {temperature},
# "weather in Boston" -> {temperature}) or from memory ("capital of France"),
# (b) faking things it can't do ("make the matrix blue" -> show_text "LED matrix
# is blue"), and (c) adding an "I am done" message nobody asked for. So v2 says
# plainly what jim does NOT have, ties every answer to a live value, and adds a
# refusal example for a knowledge question and a no-reply example.
SYSTEM_PROMPT_V2 = f"""You plan commands for "jim", an Arduino UNO Q robot. Turn the user's command into a JSON plan using ONLY these skills:
- show_text {{text}}: show text on the 8x13 LED matrix. Short text stays still, long text scrolls.
- show_icon {{icon}}: show a picture. icon is one of: {", ".join(ICONS)}.
- clear_display {{}}: turn the LED matrix off.
- reset_yaw {{}}: make the current heading yaw = 0.
- wait {{seconds}}: pause for a number of seconds.
- send_message {{topic, message}}: publish an MQTT message. topic is "" unless the user names a topic.
- refuse {{reason}}: use it ALONE when any part of the command needs something jim does not have.
jim can only sense these live values, which are filled in when the step runs: {" ".join("{" + p + "}" for p in PLACEHOLDERS)}. yaw, pitch and roll are in degrees, temperature is the board's own temperature in C.
jim does NOT have: motors or wheels, a camera, a speaker, colors (the matrix has one color and fixed brightness), sensors for humidity, light, sound or distance, or the internet. jim does not know facts, news, weather or trivia, and has no opinions.
Rules:
1. If any part of the command needs something jim does not have, or asks a question that none of the live values answer, use refuse alone.
2. Answer questions only with send_message and the matching live value. Never use one value in place of another and never answer from memory.
3. Copy the user's exact words for text and messages, and keep the order the user asked for.
4. Only send a message when the user asks a question or asks to be told something. If the user asks to be told that you are done or finished, the message is "I am done". Do not add "I am done" unless asked."""

FEW_SHOT_V2 = FEW_SHOT_V1 + [
    ("how tall is the Eiffel Tower?",
     {"steps": [{"skill": "refuse", "reason": "I don't know facts; I can only report my own sensors."}]}),
    ('put "on" on the screen',
     {"steps": [{"skill": "show_text", "text": "on"}]}),
]

# v3 = v2's prompt with the looser icon grammar and repair_placeholders().
PROMPTS = {"v1": (SYSTEM_PROMPT_V1, FEW_SHOT_V1), "v2": (SYSTEM_PROMPT_V2, FEW_SHOT_V2),
           "v3": (SYSTEM_PROMPT_V2, FEW_SHOT_V2)}


def repair_placeholders(plan, command):
    """Every configuration in benchmark run 2 sometimes wrote the bare word
    instead of the live value: "tell me the temperature" -> message
    "temperature", "show the yaw on the matrix" -> text "yaw". If a whole
    text/message is exactly one live-value name, and the user didn't quote
    that word (asking to *show the word* "yaw"), use the live value."""
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        return plan
    quoted = {q.strip().lower() for q in re.findall(
        "[" + _Q_OPEN + "]([^" + _Q_CLOSE + "]+)[" + _Q_CLOSE + "]", command)}
    for step in plan["steps"]:
        if not isinstance(step, dict):
            continue
        for key in ("text", "message"):
            word = str(step.get(key, "")).strip().strip(".!?").lower()
            if word in PLACEHOLDERS and word not in quoted:
                step[key] = "{" + word + "}"
    return plan


def chat_messages(command, version=PROMPT_VERSION):
    system, few_shot = PROMPTS[version]
    messages = [{"role": "system", "content": system}]
    for user, plan in few_shot:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": json.dumps(plan)})
    messages.append({"role": "user", "content": command})
    return messages


# --- validation --------------------------------------------------------------


class PlanError(ValueError):
    pass


def clean_topic(topic, forbidden=()):
    """forbidden: jim's own topics (cmd, status, log, online). Any robot's
    .../cmd topic is also off limits: two bots answering each other's
    command topics would loop forever."""
    topic = (topic or "").strip().strip('"\'“”').strip()
    if not topic:
        return ""
    if any(c in topic for c in "#+\x00") or len(topic) > 60 or " " in topic:
        raise PlanError(f"bad MQTT topic {topic!r}")
    if topic in forbidden or topic.rstrip("/").endswith("/cmd"):
        raise PlanError(f"I won't publish on {topic!r}: that's a command or system topic")
    return topic


def validate(plan, forbidden_topics=()):
    """Return a cleaned list of steps, or raise PlanError."""
    if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
        raise PlanError("plan has no steps list")
    steps = plan["steps"]
    if not 1 <= len(steps) <= MAX_STEPS:
        raise PlanError(f"plan has {len(steps)} steps (1-{MAX_STEPS} allowed)")
    out = []
    for raw in steps:
        if not isinstance(raw, dict) or raw.get("skill") not in SKILLS:
            raise PlanError(f"unknown step {raw!r}")
        skill = raw["skill"]
        step = {"skill": skill}
        if skill == "show_text":
            text = str(raw.get("text", "")).strip()
            if not text:
                raise PlanError("show_text with no text")
            step["text"] = text[:MAX_TEXT]
        elif skill == "show_icon":
            if raw.get("icon") not in ICONS:
                raise PlanError(f"I have no {raw.get('icon')!r} picture (I have: {', '.join(ICONS)})")
            step["icon"] = raw["icon"]
        elif skill == "wait":
            try:
                seconds = float(raw.get("seconds"))
            except (TypeError, ValueError):
                raise PlanError("wait needs a number of seconds")
            if not 0 <= seconds <= MAX_WAIT_S:
                raise PlanError(f"I can only wait up to {MAX_WAIT_S} seconds at a time, "
                                f"and that needs {seconds:g}")
            step["seconds"] = seconds
        elif skill == "send_message":
            message = str(raw.get("message", "")).strip()
            if not message:
                raise PlanError("send_message with no message")
            step["topic"] = clean_topic(raw.get("topic", ""), forbidden_topics)
            step["message"] = message[:MAX_MESSAGE]
        elif skill == "refuse":
            step["reason"] = str(raw.get("reason", "")).strip() or "I can't do that."
        out.append(step)
    refusals = [s for s in out if s["skill"] == "refuse"]
    if refusals:
        # Half-doing a command is worse than not doing it: refuse wins.
        return refusals[:1]
    return out


# --- rule planner (no model) -------------------------------------------------

_Q_OPEN = "\"“'‘"
_Q_CLOSE = "\"”'’"
_SHOW_VERB = r"(?:type|write|show|display|put|print|scroll|spell)"
_MATRIX = r"(?:the\s+)?(?:led\s+)?(?:matrix|display|screen|leds?)"

_RULES = [
    # show_text: quoted text ('type "Hi" on the LED matrix')
    ("show_text", re.compile(
        _SHOW_VERB + r"\s+(?:the\s+(?:word|text|message)\s+)?[" + _Q_OPEN + r"](?P<text>[^" + _Q_CLOSE + r"]+)[" + _Q_CLOSE + r"]"
        r"(?:\s+(?:on|to|across)\s+" + _MATRIX + r")?",
        re.I)),
    # show_text: unquoted text ending in "on the matrix" ('write Hi on the LED matrix')
    ("show_text", re.compile(
        _SHOW_VERB + r"\s+(?:the\s+(?:word|text|message)\s+)?(?P<text>[^\s" + _Q_OPEN + r"][^" + _Q_OPEN + r"]*?)\s+(?:on|to|across)\s+" + _MATRIX,
        re.I)),
    # send_message with topic and quoted message
    ("send_message", re.compile(
        r"send\s+(?:it\s+|that\s+)?(?:back\s+)?(?:(?:on|over|via)\s+mqtt\s+)?(?:on|to)\s+(?:the\s+)?topic\s+[" + _Q_OPEN + r"]?(?P<topic>[^\s" + _Q_CLOSE + r"]+)[" + _Q_CLOSE + r"]?"
        r"\s+(?:with\s+)?(?:the\s+|a\s+)?message\s+(?:of\s+)?[" + _Q_OPEN + r"](?P<message>[^" + _Q_CLOSE + r"]+)[" + _Q_CLOSE + r"]",
        re.I)),
    # "tell me (over MQTT) (that) it is done"
    ("send_message", re.compile(
        r"(?:tell|let)\s+me\s+(?:know\s+)?(?:(?:on|over|via)\s+mqtt\s+)?(?:that\s+|when\s+)?(?:it\s+is|it's|you\s+are|you're|you\s+have|you've)\s+(?:done|finished)",
        re.I)),
    ("clear_display", re.compile(r"\b(?:clear|blank)\s+" + _MATRIX + r"|turn\s+(?:off\s+" + _MATRIX + r"|" + _MATRIX + r"\s+off)", re.I)),
]


class RulePlanner:
    name = "rules"

    def plan(self, command):
        t0 = time.monotonic()
        found = []
        taken = []  # character spans already matched, so two rules can't claim the same words
        for skill, rx in _RULES:
            for m in rx.finditer(command):
                if any(m.start() < e and s < m.end() for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                step = {"skill": skill}
                if skill == "show_text":
                    step["text"] = m.group("text").strip()
                elif skill == "send_message":
                    gd = m.groupdict()
                    step["topic"] = gd.get("topic") or ""
                    step["message"] = gd.get("message") or "I am done"
                found.append((m.start(), step))
        found.sort(key=lambda p: p[0])
        steps = [s for _, s in found]
        ms = (time.monotonic() - t0) * 1000
        # Whatever no rule matched must be filler ("and then", "please"...).
        # Otherwise part of the command would be silently skipped, e.g.
        # 'type "Hi" ... and then spin the motor' would only type Hi.
        rest = command
        for s, e in sorted(taken, reverse=True):
            rest = rest[:s] + " " + rest[e:]
        leftover = [w for w in re.findall(r"[a-z0-9']+", rest.lower()) if w not in _FILLER]
        if not steps or leftover:
            why = f" (I don't know how to: {' '.join(leftover)})" if steps and leftover else ""
            return {"steps": [{"skill": "refuse", "reason":
                               "Without the language model I only understand simple display and "
                               f"reply commands{why}."}]}, {"ms": ms}
        return {"steps": steps}, {"ms": ms}


_FILLER = set("""a an the and then also after that finally first next now please pls can could would
you jim hey hi ok okay i want to me for it on""".split())


# --- LLM planner -------------------------------------------------------------

LLM_PORT = 8090


def default_llm_url():
    """llama-server runs on jim's Linux side. From inside the app's Docker
    container that's the container's default gateway; on the host it's
    localhost."""
    if os.environ.get("LLM_URL"):
        return os.environ["LLM_URL"].rstrip("/")
    if os.path.exists("/.dockerenv"):
        try:
            with open("/proc/net/route") as f:
                for line in f.readlines()[1:]:
                    fields = line.split()
                    if fields[1] == "00000000":          # default route
                        gw = bytes.fromhex(fields[2])[::-1]
                        return f"http://{'.'.join(map(str, gw))}:{LLM_PORT}"
        except (OSError, IndexError, ValueError):
            pass
    return f"http://127.0.0.1:{LLM_PORT}"


class LLMPlanner:
    def __init__(self, url=None, timeout=120, use_grammar=True, prompt_version=PROMPT_VERSION):
        self.url = (url or default_llm_url()).rstrip("/")
        self.timeout = timeout
        self.use_grammar = use_grammar
        self.prompt_version = prompt_version
        self.model = None

    @property
    def name(self):
        return f"llm:{self.model or '?'}"

    def _get(self, path, timeout=3):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def healthy(self):
        try:
            ok = self._get("/health").get("status") == "ok"
            if ok and self.model is None:
                path = self._get("/props").get("model_path", "")
                self.model = os.path.basename(path).replace(".gguf", "") or "?"
            return ok
        except Exception:  # connection refused, half-restarted server, odd body: all mean "not usable"
            return False

    def plan(self, command, timeout=None):
        """Returns (plan_dict, info). Raises PlanError if the model's output
        isn't a plan, OSError if the server can't be reached."""
        body = {
            "messages": chat_messages(command, self.prompt_version),
            "temperature": 0,
            "max_tokens": 256,
            "cache_prompt": True,
            # Qwen3 would otherwise start with a <think> block.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.use_grammar:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "plan",
                                                       "schema": PLAN_SCHEMAS[self.prompt_version]}}
        req = urllib.request.Request(self.url + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            resp = json.loads(r.read().decode())
        ms = (time.monotonic() - t0) * 1000
        text = resp["choices"][0]["message"].get("content") or ""
        timings = resp.get("timings", {})
        info = {"ms": ms, "raw": text,
                "prompt_tokens": timings.get("prompt_n"),
                "prompt_ms": timings.get("prompt_ms"),
                "gen_tokens": timings.get("predicted_n"),
                "gen_ms": timings.get("predicted_ms")}
        try:
            plan = json.loads(_strip_fences(text))
        except json.JSONDecodeError as e:
            raise PlanError(f"model output is not JSON: {text[:120]!r}") from e
        if self.prompt_version == "v3":
            plan = repair_placeholders(plan, command)
        return plan, info


def _strip_fences(text):
    """Without a grammar, small models like to wrap JSON in ```json fences."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else text
