import asyncio
import numpy as np
from collections import deque
from concurrent.futures import Future
from pyscript import document, window, when
from pyodide.ffi import create_proxy
import sys

import legoeducation as le
import legoeducation.background_worker as bw
from pyscript.js_modules.ble import BLEDevice as _BLEDeviceJS
from legoeducation import DoubleMotor as _DM
from legoeducation import ColorSensor as _CS
from legoeducation.rpc_message import InfoRequest, InfoResponse, RPC_VERSION
from qlearn import QTable

# ── WASM WORKER PATCH ──
def _wasm_start_thread(self):
    if getattr(self, '_wasm_loop_started', False): return
    self._wasm_loop_started = True
    if not hasattr(self, '_js_ble_registry'): self._js_ble_registry = {}
    try:
        self.loop = asyncio.get_running_loop()
        self.loop_ready.set()
        asyncio.ensure_future(_wasm_worker_loop(self))
    except RuntimeError: pass

def _wasm_put_request(self, request):
    if not self.loop_ready.is_set():
        try:
            self.loop = asyncio.get_running_loop()
            self.loop_ready.set()
            if not hasattr(self, '_js_ble_registry'): self._js_ble_registry = {}
            asyncio.ensure_future(_wasm_worker_loop(self))
        except RuntimeError: return
    asyncio.ensure_future(self.async_put_request(request))

bw.Worker.start_thread = _wasm_start_thread
bw.Worker.put_request = _wasm_put_request

async def _wasm_worker_loop(worker):
    while True:
        try:
            req = await worker.request_queue.get()
            if req is None: break
            topic = req.get('topic')
            if topic == 'send':
                device = req.get('msg')
                message = req.get('msg2')
                js_ble = worker._js_ble_registry.get(id(device))
                if js_ble and message:
                    await js_ble.send(list(message))
            elif topic == 'connect':
                cb = req.get('msg3')
                if cb: cb(True)
            elif topic == 'disconnect':
                device = req.get('msg')
                js_ble = worker._js_ble_registry.pop(id(device), None)
                if js_ble: js_ble.disconnect()
        except Exception as e: print(f"Worker error: {e}")

SERVICE_UUID = '0000fd02-0000-1000-8000-00805f9b34fb'
WRITE_UUID   = '0000fd02-0001-1000-8000-00805f9b34fb'
NOTIFY_UUID  = '0000fd02-0002-1000-8000-00805f9b34fb'
# Shared across every LEGO Education hub type (Single/Double Motor, Color
# Sensor, Controller) — the legoeducation package hardcodes these same three
# UUIDs package-wide, since they all speak the same LWP3 RPC protocol. Safe
# to reuse as-is for both dm_device and cs_device below.

# ── LOGGING ──
def log(msg):
    term = document.getElementById('terminal')
    div = document.createElement('div')
    div.innerText = msg
    term.appendChild(div)
    term.scrollTop = term.scrollHeight
    print(msg)

# ── FIRMWARE VERSION CHECK ──
async def _check_firmware_version(device, timeout=3.0):
    """Ask the hub for its RPC/firmware version and warn in the log if it's
    out of sync with this app's legoeducation package.

    Reimplements what device.connect()/info() already do natively, but
    awaiting the response via asyncio instead of blocking on a
    concurrent.futures.Future.result() — that call (used by info()) would
    freeze the browser's single-threaded event loop, same reason every
    motor/sensor call elsewhere in this file passes blocking=False.
    """
    msg = InfoRequest().Serialize()
    response_id = msg[0] + 1
    future = Future()
    device._pending_responses.setdefault(response_id, deque()).append(future)
    device._ask('send', device.device, msg)
    try:
        payload = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout)
    except asyncio.TimeoutError:
        log(f"{device.prefix.upper()}: could not read firmware version (no response).")
        return

    info = InfoResponse.Deserialize(payload)
    version_rpc = (info.rpcMajor, info.rpcMinor, info.rpcBuild)
    if version_rpc == RPC_VERSION:
        return
    if version_rpc < RPC_VERSION:
        log(f"{device.prefix.upper()}: firmware update needed (hub RPC version {version_rpc} "
            f"< app's {RPC_VERSION}). Go to https://code.legoeducation.com/ to update your hardware.")
    else:
        log(f"{device.prefix.upper()}: app update needed (hub RPC version {version_rpc} "
            f"> app's {RPC_VERSION}). Run: pip install --upgrade legoeducation.")


# ── DEVICE MANAGEMENT ──
class WebDevice:
    def __init__(self, prefix):
        self.prefix = prefix
        self.connected = False
        self.js_ble = None

    async def connect_web(self):
        js_ble = _BLEDeviceJS.new()
        self._notification_proxy = create_proxy(lambda data: asyncio.ensure_future(self._on_notification(bytes(data.to_py()))))
        self._disconnect_proxy = create_proxy(self._on_disconnect)
        js_ble.callback = self._notification_proxy
        js_ble.disconnectCallback = self._disconnect_proxy

        success = await js_ble.connect(SERVICE_UUID, WRITE_UUID, NOTIFY_UUID)
        if success:
            self.js_ble = js_ble
            self.connected = True
            self.device = self
            import legoeducation.basic_device as bd
            bd.my_worker._js_ble_registry[id(self)] = self.js_ble
            try:
                self.device_notification_request(100, blocking=False)
            except: pass

            dot = document.getElementById(f'{self.prefix}-dot')
            if dot: dot.classList.add('connected')
            btn = document.getElementById(f'btn-connect-{self.prefix}')
            if btn:
                btn.innerText = 'Connected'
                btn.disabled = True
            btn_dis = document.getElementById(f'btn-disconnect-{self.prefix}')
            if btn_dis:
                btn_dis.style.display = 'inline-block'
            log(f"{self.prefix.upper()} connected.")
            check_ready()
            asyncio.ensure_future(_check_firmware_version(self))

    async def _on_notification(self, data: bytes):
        await self._device_callback(NOTIFY_UUID, data)

    def _on_disconnect(self, event):
        self.connected = False
        dot = document.getElementById(f'{self.prefix}-dot')
        if dot: dot.classList.remove('connected')
        btn = document.getElementById(f'btn-connect-{self.prefix}')
        if btn:
            btn.innerText = 'Connect'
            btn.disabled = False
        btn_dis = document.getElementById(f'btn-disconnect-{self.prefix}')
        if btn_dis:
            btn_dis.style.display = 'none'
        log(f"{self.prefix.upper()} disconnected.")
        check_ready()

    def disconnect_web(self):
        if self.js_ble:
            self.js_ble.disconnect()
            self._on_disconnect(None)

    def send_command(self, packet):
        if self.js_ble is None:
            return
        if isinstance(packet, (bytes, bytearray)):
            packet = list(packet)
        asyncio.ensure_future(self.js_ble.send(packet))

class DoubleMotorDevice(WebDevice, _DM):
    def __init__(self):
        _DM.__init__(self)
        WebDevice.__init__(self, 'dm')

dm_device = DoubleMotorDevice()

def connect_dm(e):
    asyncio.ensure_future(dm_device.connect_web())

def disconnect_dm(e):
    dm_device.disconnect_web()


# ── COLOR SENSOR DEVICE (used only by the Line Following tab) ──
class ColorSensorDevice(WebDevice, _CS):
    def __init__(self):
        _CS.__init__(self)
        WebDevice.__init__(self, 'cs')

cs_device = ColorSensorDevice()

def connect_cs(e):
    asyncio.ensure_future(cs_device.connect_web())

def disconnect_cs(e):
    cs_device.disconnect_web()


_running_task = None

def _is_line_page_active():
    """True if the Line Following tab is currently showing. Same dispatch
    role as _is_dqn_page_active(), for the line-following training/policy loops."""
    try:
        page_line = document.getElementById('page-line')
        return not page_line.classList.contains('hidden')
    except Exception:
        return False

def _is_maze_page_active():
    """True if the Maze Solver tab is currently showing."""
    try:
        page_maze = document.getElementById('page-maze')
        return not page_maze.classList.contains('hidden')
    except Exception:
        return False

def _set_running_ui(active):
    """Enables/disables the single shared Start/Stop/Step/Policy buttons,
    since only one training or policy loop (tabular, DQN, or line-following)
    can safely drive the physical motors at a time."""
    connected = dm_device.connected
    # Line-following and maze-solving additionally require the Color Sensor;
    # Silly Walk only needs the Double Motor.
    needs_cs = _is_line_page_active() or _is_maze_page_active()
    connected_run = connected and (cs_device.connected if needs_cs else True)
    for btn_id in ('btn-train', 'btn-policy', 'btn-step'):
        el = document.getElementById(btn_id)
        if el:
            el.disabled = active or not connected_run
    el = document.getElementById('btn-stop')
    if el:
        el.disabled = not active
    # Reset Yaw exists in two places (sidebar + Current Yaw modal), synced
    # together via a shared class rather than a single hardcoded id.
    try:
        reset_btns = document.querySelectorAll('.reset-yaw-btn')
        for i in range(reset_btns.length):
            reset_btns.item(i).disabled = active or not connected
    except Exception:
        pass
    # Per-action "physically execute this action" buttons in the Q-table
    # header — dynamically created, so re-synced here rather than set once.
    try:
        exec_btns = document.querySelectorAll('.execute-action-btn')
        for i in range(exec_btns.length):
            exec_btns.item(i).disabled = active or not connected
    except Exception:
        pass

def check_ready():
    _set_running_ui(_running_task is not None)
    btn_yaw = document.getElementById('btn-show-yaw')
    if btn_yaw:
        btn_yaw.disabled = not dm_device.connected
    if not dm_device.connected:
        hide_yaw_modal()
    btn_reflection = document.getElementById('btn-show-reflection')
    if btn_reflection:
        btn_reflection.disabled = not cs_device.connected
    if not cs_device.connected:
        hide_reflection_modal()
    btn_maze_ready = document.getElementById('btn-maze-ready')
    if btn_maze_ready:
        btn_maze_ready.disabled = not (dm_device.connected and cs_device.connected)


# ── Q-LEARNING LOGIC ──

# Number of actions currently in play: 2 (LEFT/RIGHT) or 3 (+ FORWARD).
# Switched via the "Actions" dropdown -> on_action_count_change().
NUM_ACTIONS = 2

def action_names():
    return ['LEFT', 'RIGHT', 'FORWARD'][:NUM_ACTIONS]

direction_mapping = {}

def _rebuild_direction_mapping():
    global direction_mapping
    direction_mapping = {i: name for i, name in enumerate(action_names())}

_rebuild_direction_mapping()

# Names for each state, keyed by how many states are currently active (3 or 5).
# Must mirror STATE_CONFIGS.names in index.html.
STATE_NAMES = {
    3: ['Drifted Left', 'Straight', 'Drifted Right'],
    5: ['Hard Left', 'Soft Left', 'Straight', 'Soft Right', 'Hard Right'],
}

# Number of discrete yaw-drift states currently in use. Switched via the
# "States" dropdown, which calls on_state_count_change().
NUM_STATES = 3

# Yaw boundaries (degrees), descending, length NUM_STATES - 1.
# Read from the Q-table's "thresh-b*" inputs in apply_rewards().
# State s (for s < middle) is entered when yaw > BOUNDARIES[s].
# The last state is entered when yaw doesn't exceed any boundary.
BOUNDARIES = [10, -10]

def state_names():
    return STATE_NAMES[NUM_STATES]

def discretize_yaw(yaw):
    if yaw is None:
        return (NUM_STATES - 1) // 2  # default to the middle "Straight" state
    for i, b in enumerate(BOUNDARIES):
        if yaw > b:
            return i
    return NUM_STATES - 1

# The IMU reports yaw in tenths of a degree; converting to actual degrees
# right here means every boundary, log message, and downstream calculation
# works in degrees without needing to remember the raw hardware unit.
def raw_yaw_to_degrees(raw_yaw):
    return round(raw_yaw / 10.0, 1)

def get_yaw_state():
    raw_yaw = dm_device.imu_device.yaw
    if raw_yaw is None or (isinstance(raw_yaw, float) and np.isnan(raw_yaw)):
        raw_yaw = 0
    yaw = raw_yaw_to_degrees(raw_yaw)
    return discretize_yaw(yaw), yaw


# ── Current Yaw modal ──
# Independent of the Start/Stop/Step/Policy run-guard: this only *reads* the
# IMU (no motor commands), so it's safe to run concurrently with training,
# policy runs, or manual stepping.
_yaw_monitor_task = None

async def _yaw_monitor_loop():
    while True:
        raw_yaw = dm_device.imu_device.yaw
        if raw_yaw is None or (isinstance(raw_yaw, float) and np.isnan(raw_yaw)):
            raw_yaw = 0
        yaw = raw_yaw_to_degrees(raw_yaw)
        window.updateYawGauge(float(yaw))
        await asyncio.sleep(0.2)

def show_yaw_modal(e=None):
    global _yaw_monitor_task
    window.openYawModal()
    if _yaw_monitor_task is None:
        _yaw_monitor_task = asyncio.ensure_future(_yaw_monitor_loop())

def hide_yaw_modal(e=None):
    global _yaw_monitor_task
    window.closeYawModal()
    if _yaw_monitor_task is not None:
        _yaw_monitor_task.cancel()
        _yaw_monitor_task = None

TURN_DEGREES = 135  # overwritten by apply_rewards() from the "Turn Degrees" slider
TURN_WAIT = 0.3      # overwritten by apply_rewards() from the "Turn Wait" slider
TRAIN_ITERS = 10     # overwritten by apply_rewards() from the "Number of Steps" slider
NUM_EPISODES = 10    # overwritten by apply_rewards() from the "Number of Episodes" slider

async def take_action(a):
    """Use the standard legoeducation motor_run_for_degrees call, same as the
    hardware (non-browser) version of this script — but non-blocking, since a
    blocking call here would freeze the browser's single-threaded event loop
    (there's no real background thread to service it while we wait)."""
    if a == 'LEFT':
        dm_device.motor_run_for_degrees(TURN_DEGREES, direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, motor=le.MOTOR_RIGHT, blocking=False)
    elif a == 'RIGHT':
        dm_device.motor_run_for_degrees(TURN_DEGREES, direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_LEFT, blocking=False)
    elif a == 'FORWARD':
        dm_device.movement_move_for_degrees(TURN_DEGREES, blocking=False)
    else:
        return

    # Yield control back to the event loop so the BLE write is dispatched
    await asyncio.sleep(0)
    # Wait for the physical turn to complete before we read the next state
    await asyncio.sleep(TURN_WAIT)

async def reset_yaw():
    dm_device.imu_reset_yaw_axis(0, blocking=False)
    # let a fresh notification arrive so imu_device.yaw reflects the reset
    await asyncio.sleep(0.3)

async def _reset_yaw_now():
    log("Resetting yaw...")
    await reset_yaw()
    log("Yaw reset to 0.")

def reset_yaw_now(e=None):
    """Manual "Reset Yaw" button handler. Blocked while a training/policy/
    step run is active, since that loop resetting yaw itself (at episode
    boundaries, etc.) would conflict with an externally-triggered reset."""
    if _running_task is not None:
        log("Stop the current run before manually resetting yaw.")
        return
    asyncio.ensure_future(_reset_yaw_now())


# ── Manual per-action execution (does NOT update the Q-table) ──
# Triggered by the small ▶ button next to each action in the Q-table header.
# Bridged via a CustomEvent + document.addEventListener rather than py-click,
# since py-click bindings on dynamically-injected elements (the header is
# rebuilt whenever states/actions change) aren't reliably picked up by
# PyScript — this listener is registered once, here, at static load time.
async def _manual_action_run(action_name):
    log(f"Manually executing: {action_name} (Q-table not updated).")
    await take_action(a=action_name)

def _on_manual_action_event(event):
    if _running_task is not None:
        log("Stop the current run before manually executing an action.")
        return
    if not dm_device.connected:
        return
    try:
        action_name = str(event.detail.action)
    except Exception:
        return
    asyncio.ensure_future(_manual_action_run(action_name))

_manual_action_proxy = create_proxy(_on_manual_action_event)
document.addEventListener('manual-action', _manual_action_proxy)

def update_qtable_ui(qtable):
    for s in range(NUM_STATES):
        for a in range(NUM_ACTIONS):
            val = qtable.table[s, a]
            window.updateCell(s, a, float(val))

def apply_rewards(e=None):
    global TURN_DEGREES, TURN_WAIT, TRAIN_ITERS, NUM_EPISODES, ALPHA, GAMMA, BOUNDARIES
    try:
        TURN_DEGREES = int(document.getElementById('slider-turn-degrees').value)
        TURN_WAIT = float(document.getElementById('slider-turn-wait').value)
        TRAIN_ITERS = int(document.getElementById('slider-train-iters').value)
        NUM_EPISODES = int(document.getElementById('slider-num-episodes').value)
        ALPHA = float(document.getElementById('slider-alpha').value)
        GAMMA = float(document.getElementById('slider-gamma').value)

        # Read the (NUM_STATES - 1) yaw boundary inputs: thresh-b0, thresh-b1, ...
        new_boundaries = []
        for b in range(NUM_STATES - 1):
            el = document.getElementById(f'thresh-b{b}')
            new_boundaries.append(int(el.value))
        for i in range(1, len(new_boundaries)):
            if new_boundaries[i] >= new_boundaries[i - 1]:
                log("Warning: yaw boundaries should strictly decrease from left to right.")
                break
        BOUNDARIES = new_boundaries

        # Read the per-state reward sliders for every active action:
        # reward-s0-left, reward-s0-right, (reward-s0-forward if enabled), ...
        rewards = np.full((NUM_STATES, NUM_ACTIONS), -1)
        for s in range(NUM_STATES):
            for ai, aname in enumerate(action_names()):
                el = document.getElementById(f'reward-s{s}-{aname.lower()}')
                rewards[s, ai] = int(el.value)
    except Exception as ex:
        log(f"Could not read all settings, keeping previous values ({ex}).")
        return

    qtable_instance.set_rewards(rewards)
    qtable_instance.alpha = ALPHA
    qtable_instance.gamma = GAMMA
    if e is not None:
        log(f"Settings updated. Turn = {TURN_DEGREES}°, Turn Wait = {TURN_WAIT:.2f}s, Training Iterations = {TRAIN_ITERS}, "
            f"Episodes = {NUM_EPISODES}, alpha = {ALPHA:.2f}, gamma = {GAMMA:.2f}, "
            f"yaw boundaries = {BOUNDARIES}.")

def reset_qtable(e=None):
    global qtable_instance, _step_i, _step_s, _step_yaw
    qtable_instance = init_qtable()
    apply_rewards()
    update_qtable_ui(qtable_instance)
    window.resetBellmanReadout()
    _step_i = 0
    _step_s = None
    _step_yaw = None
    document.getElementById('step-counter').innerText = '0'
    _set_running_ui(_running_task is not None)
    log("Q-Table has been reset to zeros.")

def init_qtable():
    states = np.arange(NUM_STATES)
    actions = np.arange(NUM_ACTIONS)
    return QTable(states=states, actions=actions, gamma=GAMMA, alpha=ALPHA)

@when("change", "#state-count-select")
def on_state_count_change(event=None):
    """Fired when the "States" dropdown changes. Rebuilds the Q-table rows
    and reward sliders in JS for the new count, then reinitializes the
    Q-table in Python to match."""
    global NUM_STATES
    if _running_task is not None:
        log("Stop the current run before switching the number of states.")
        # revert the dropdown back to the current NUM_STATES
        document.getElementById('state-count-select').value = str(NUM_STATES)
        return

    new_n = int(document.getElementById('state-count-select').value)
    NUM_STATES = new_n
    window.rebuildStateUI(new_n, action_names())
    reset_qtable()
    log(f"Switched to {new_n}-state mode ({', '.join(state_names())}).")

@when("change", "#action-count-select")
def on_action_count_change(event=None):
    """Fired when the "Actions" dropdown changes. Rebuilds the Q-table
    columns and reward sliders for the new action count, then reinitializes
    the Q-table in Python to match."""
    global NUM_ACTIONS
    if _running_task is not None:
        log("Stop the current run before changing the action space.")
        document.getElementById('action-count-select').value = str(NUM_ACTIONS)
        return

    new_n = int(document.getElementById('action-count-select').value)
    NUM_ACTIONS = new_n
    _rebuild_direction_mapping()
    window.rebuildStateUI(NUM_STATES, action_names())
    reset_qtable()
    log(f"Action space changed to: {', '.join(action_names())}.")

ALPHA = 0.1  # overwritten by apply_rewards() from the "Alpha" slider
GAMMA = 0.9  # overwritten by apply_rewards() from the "Gamma" slider
qtable_instance = init_qtable()
apply_rewards()
update_qtable_ui(qtable_instance)
_set_running_ui(False)  # ensure freshly-created buttons (execute-action, etc.) start disabled

async def _train_loop():
    log("Resetting yaw...")
    await reset_yaw()
    log("Starting training...")
    _set_running_ui(True)

    s, yaw = get_yaw_state()
    n = max(TRAIN_ITERS, 1)          # steps per episode
    num_episodes = max(NUM_EPISODES, 1)
    total_steps = n * num_episodes   # epsilon decays continuously across all episodes
    names = state_names()
    step_counter = 0
    try:
        for episode in range(num_episodes):
            if episode > 0:
                log(f"Episode {episode + 1}/{num_episodes}: resetting yaw.")
                await reset_yaw()
                s, yaw = get_yaw_state()

            for i in range(n):
                eps = max(1 - step_counter / total_steps, 0.05)
                a = qtable_instance.choose_action(s=s, eps=eps)
                action_name = direction_mapping[a]

                window.highlightStateRow(s, 'row-from', 900)
                log(f"Episode {episode + 1}/{num_episodes} Step {i + 1}/{n}: "
                    f"State={names[s]} (yaw={yaw}), Action={action_name}, eps={eps:.2f}")
                await take_action(a=action_name)

                # Short extra pause to let the IMU notification stabilise after motion
                await asyncio.sleep(0.2)

                next_s, yaw = get_yaw_state()

                # Capture the Bellman update's terms before/after so the readout
                # can show the literal numbers, not just the final value.
                old_q = float(qtable_instance.table[s, a])
                reward = float(qtable_instance.rewards[s, a])
                alpha = float(qtable_instance.alpha)
                gamma = float(qtable_instance.gamma)
                next_max_q = float(np.max(qtable_instance.table[next_s, :]))

                qtable_instance.bellman_update(s, a, next_s)
                new_q = float(qtable_instance.table[s, a])

                update_qtable_ui(qtable_instance)
                window.pulseCell(s, a, 900)
                window.highlightStateRow(next_s, 'row-to', 900)
                window.highlightActionHeader(a, 900)
                window.updateBellmanReadout(names[s], action_name, names[next_s],
                                             old_q, alpha, reward, gamma, next_max_q, new_q)

                s = next_s
                step_counter += 1

        dm_device.motor_stop(motor=le.MOTOR_RIGHT, blocking=False)
        dm_device.motor_stop(motor=le.MOTOR_LEFT, blocking=False)
        log("DONE TRAINING. Reset yaw before running the policy if needed.")
    except asyncio.CancelledError:
        log("Training stopped by user.")
    except Exception as e:
        log(f"Error: {e}")
    finally:
        global _running_task
        _running_task = None
        _set_running_ui(False)

async def _policy_loop():
    log("Resetting yaw...")
    await reset_yaw()
    log("Running learned policy...")
    _set_running_ui(True)
    names = state_names()
    try:
        for i in range(100):
            s, yaw = get_yaw_state()
            window.highlightStateRow(s, 'row-from', 900)
            a = qtable_instance.choose_action(s=s, eps=0)
            await take_action(a=direction_mapping[a])
            log(f"Policy Step {i+1}: State={names[s]} (yaw={yaw}), Action={direction_mapping[a]}")
            await asyncio.sleep(0.2)

        dm_device.motor_stop(motor=le.MOTOR_RIGHT, blocking=False)
        dm_device.motor_stop(motor=le.MOTOR_LEFT, blocking=False)
        log("DONE RUNNING POLICY.")
    except asyncio.CancelledError:
        log("Policy stopped by user.")
    except Exception as e:
        log(f"Error: {e}")
    finally:
        global _running_task
        _running_task = None
        _set_running_ui(False)


# ══════════════════════════════════════════════════════════════
# LINE FOLLOWING (separate tab). Fixed 3-state / 3-action Q-table —
# no state/action-count dropdowns here, unlike the tabular tab, since the
# state and action space for line following doesn't need to be adjustable.
# Ported from a standalone hardware script: same states/actions/rewards,
# but motor calls are non-blocking (await asyncio.sleep instead of
# time.sleep) so the browser's event loop isn't frozen during a step.
# ══════════════════════════════════════════════════════════════

LINE_STATE_NAMES = ['Off Line (Left)', 'On Line', 'Off Line (Right)']
LINE_ACTION_NAMES = ['LEFT', 'RIGHT', 'FORWARD']
direction_mapping_line = {0: 'LEFT', 1: 'RIGHT', 2: 'FORWARD'}

TURN_DEGREES_LINE = 15     # overwritten by apply_line_rewards() from the "Turn Degrees" slider
FORWARD_DEGREES_LINE = 45  # overwritten by apply_line_rewards() from the "Forward Degrees" slider
LINE_BOUNDARIES = [33, 66] # [low, high] reflection thresholds; overwritten by apply_line_rewards()
TRAIN_STEPS_LINE = 50      # fixed step count, mirroring the original hardware script's `for i in range(50)`
REST_TIME_LINE = 0.4       # overwritten by apply_line_rewards() from the "Rest Time" slider

# ── Current Reflection modal ──
# Independent of the Start/Stop/Policy run-guard: this only *reads* the
# Color Sensor (no motor commands), so it's safe to run concurrently with
# training or policy runs, same as the yaw modal on the Silly Walk tab.
_reflection_monitor_task = None

async def _reflection_monitor_loop():
    while True:
        reflection = cs_device.sensor.reflection
        if reflection is None or (isinstance(reflection, float) and np.isnan(reflection)):
            reflection = 0
        window.updateReflectionGauge(float(reflection))
        await asyncio.sleep(0.2)

def show_reflection_modal(e=None):
    global _reflection_monitor_task
    window.openReflectionModal()
    if _reflection_monitor_task is None:
        _reflection_monitor_task = asyncio.ensure_future(_reflection_monitor_loop())

def hide_reflection_modal(e=None):
    global _reflection_monitor_task
    window.closeReflectionModal()
    if _reflection_monitor_task is not None:
        _reflection_monitor_task.cancel()
        _reflection_monitor_task = None


def discretize_reflection(reflection):
    if reflection is None:
        return 1  # default to "On Line"
    lo, hi = LINE_BOUNDARIES
    if reflection < lo:
        return 0
    elif reflection > hi:
        return 2
    return 1

def get_line_state():
    reflection = cs_device.sensor.reflection
    if reflection is None or (isinstance(reflection, float) and np.isnan(reflection)):
        reflection = 50  # neutral default if no reading has arrived yet
    return discretize_reflection(reflection), reflection

async def take_action_line(a):
    """Same non-blocking pattern as take_action(): drive is via the Double
    Motor (dm_device) even though the *state* comes from the Color Sensor
    (cs_device) — matches the two-separate-connected-devices structure of
    the original hardware script (dm and cs connected independently)."""
    if a == 'LEFT':
        dm_device.movement_turn_for_degrees(TURN_DEGREES_LINE, direction=le.MOVEMENT_TURN_DIRECTION_LEFT, blocking=False)
    elif a == 'RIGHT':
        dm_device.movement_turn_for_degrees(TURN_DEGREES_LINE, direction=le.MOVEMENT_TURN_DIRECTION_RIGHT, blocking=False)
    elif a == 'FORWARD':
        dm_device.movement_move_for_degrees(FORWARD_DEGREES_LINE, direction=le.MOVEMENT_MOVE_DIRECTION_FORWARD, blocking=False)
    else:
        return
    await asyncio.sleep(0)
    await asyncio.sleep(REST_TIME_LINE)

def init_qtable_line():
    return QTable(states=np.arange(3), actions=np.arange(3), gamma=0.9, alpha=0.1)

qtable_line = init_qtable_line()

def update_qtable_ui_line(qtable):
    for s in range(3):
        for a in range(3):
            val = qtable.table[s, a]
            window.updateLineCell(s, a, float(val))

def apply_line_rewards(e=None):
    global TURN_DEGREES_LINE, FORWARD_DEGREES_LINE, LINE_BOUNDARIES, REST_TIME_LINE
    try:
        TURN_DEGREES_LINE = int(document.getElementById('slider-line-turn-degrees').value)
        FORWARD_DEGREES_LINE = int(document.getElementById('slider-line-forward-degrees').value)
        REST_TIME_LINE = float(document.getElementById('slider-line-rest-time').value)
        lo = int(document.getElementById('slider-line-thresh-lo').value)
        hi = int(document.getElementById('slider-line-thresh-hi').value)
        if lo >= hi:
            log("Warning: left boundary should be less than right boundary.")
        LINE_BOUNDARIES = [lo, hi]

        rewards = np.full((3, 3), -10)
        for s in range(3):
            for a, aname in enumerate(LINE_ACTION_NAMES):
                el = document.getElementById(f'reward-line-s{s}-{aname.lower()}')
                rewards[s, a] = int(el.value)
    except Exception as ex:
        log(f"Could not read all line-following settings, keeping previous values ({ex}).")
        return

    qtable_line.set_rewards(rewards)
    if e is not None:
        log(f"Line settings updated. Turn = {TURN_DEGREES_LINE}°, Forward = {FORWARD_DEGREES_LINE}°, "
            f"Rest Time = {REST_TIME_LINE:.2f}s, "
            f"boundaries = {LINE_BOUNDARIES}.")

def reset_qtable_line(e=None):
    global qtable_line
    qtable_line = init_qtable_line()
    apply_line_rewards()
    update_qtable_ui_line(qtable_line)
    log("Line-following Q-Table has been reset to zeros.")

async def _train_loop_line():
    log("Place the robot on the line. Training starts in a few seconds...")
    _set_running_ui(True)
    await asyncio.sleep(3)  # gives a human time to position the robot; there's no
                             # automatic reset available for line following the way
                             # reset_yaw() resets the tabular task between episodes.
    log("Starting line-following training...")

    s, reflection = get_line_state()
    n = max(TRAIN_STEPS_LINE, 1)
    try:
        for i in range(n):
            eps = max(1 - i / n, 0.05)
            a = qtable_line.choose_action(s=s, eps=eps)
            action_name = direction_mapping_line[a]
            log(f"Step {i + 1}/{n}: State={LINE_STATE_NAMES[s]} (reflection={reflection}), "
                f"Action={action_name}, eps={eps:.2f}")
            await take_action_line(a=action_name)

            next_s, reflection = get_line_state()
            qtable_line.bellman_update(s, a, next_s)
            update_qtable_ui_line(qtable_line)
            s = next_s

        dm_device.motor_stop(motor=le.MOTOR_RIGHT, blocking=False)
        dm_device.motor_stop(motor=le.MOTOR_LEFT, blocking=False)
        log("DONE TRAINING (line following). Reposition the robot on the line before running the policy.")
    except asyncio.CancelledError:
        log("Line-following training stopped by user.")
    except Exception as e:
        log(f"Error: {e}")
    finally:
        global _running_task
        _running_task = None
        _set_running_ui(False)

async def _policy_loop_line():
    log("Running learned line-following policy...")
    _set_running_ui(True)
    try:
        for i in range(100):
            s, reflection = get_line_state()
            a = qtable_line.choose_action(s=s, eps=0)
            action_name = direction_mapping_line[a]
            log(f"Policy Step {i + 1}: State={LINE_STATE_NAMES[s]} (reflection={reflection}), Action={action_name}")
            await take_action_line(a=action_name)

        dm_device.motor_stop(motor=le.MOTOR_RIGHT, blocking=False)
        dm_device.motor_stop(motor=le.MOTOR_LEFT, blocking=False)
        log("DONE RUNNING LINE-FOLLOWING POLICY.")
    except asyncio.CancelledError:
        log("Line-following policy stopped by user.")
    except Exception as e:
        log(f"Error: {e}")
    finally:
        global _running_task
        _running_task = None
        _set_running_ui(False)

apply_line_rewards()
update_qtable_ui_line(qtable_line)


# ══════════════════════════════════════════════════════════════
# MAZE SOLVER (separate tab). Fixed 11-state / 3-action Q-table, combining
# color (from the line-maze tape) and yaw-drift into one discrete state,
# ported from a standalone hardware script. Two things differ from that
# script, both because this runs in a browser tab instead of a blocking
# terminal loop:
#   - motor calls are non-blocking, same pattern as every other tab here.
#   - the original used input() to pause between episodes while a human
#     repositioned the robot at the maze start. A browser event loop can't
#     block like that, so a "Robot Ready" button + asyncio.Event stands in
#     for it: training genuinely pauses until the button is clicked, same
#     as the original pausing until Enter was pressed.
# Q-table always starts fresh (no save/load) — no persistence across
# sessions, so every group starts from a blank table.
# ══════════════════════════════════════════════════════════════

color_mapping = {
    0: 'No color',
    1: 'Red',
    2: 'Yellow',
    3: 'Blue',
    4: 'Teal',
    5: 'Green',
    6: 'Purple',
    7: 'White',
    8: 'Magenta',
    9: 'Orange',
    10: 'Azure',
}

MAZE_STATE_NAMES = [
    'Black-Left', 'Black-Right', 'Black-Center',
    'Red-Left', 'Red-Right', 'Red-Center',
    'Blue-Left', 'Blue-Right', 'Blue-Center',
    'Green (Goal)', 'White',
]
(STATE_LEFT_BLACK, STATE_RIGHT_BLACK, STATE_CENTER_BLACK,
 STATE_LEFT_RED, STATE_RIGHT_RED, STATE_CENTER_RED,
 STATE_LEFT_BLUE, STATE_RIGHT_BLUE, STATE_CENTER_BLUE,
 STATE_GREEN, STATE_WHITE) = range(11)

MAZE_ACTION_NAMES = ['LEFT', 'RIGHT', 'FORWARD']
direction_mapping_maze = {0: 'LEFT', 1: 'RIGHT', 2: 'FORWARD'}

TURN_DEGREES_MAZE = 30      # overwritten by apply_maze_rewards() from its "Turn Degrees" slider
FORWARD_DEGREES_MAZE = 90   # overwritten by apply_maze_rewards() from its "Forward Degrees" slider
YAW_DEADBAND_MAZE = 10      # degrees; overwritten by apply_maze_rewards() from its "Yaw Deadband" slider
NUM_EPISODES_MAZE = 10      # overwritten by apply_maze_rewards() from its "Number of Episodes" slider
MAX_STEPS_MAZE = 100        # overwritten by apply_maze_rewards() from its "Max Steps per Episode" slider
REST_TIME_MAZE = 0.4        # overwritten by apply_maze_rewards() from its "Rest Time" slider

def discretize_yaw_maze(yaw_degrees):
    if yaw_degrees > YAW_DEADBAND_MAZE:
        return 0  # drifted left
    elif yaw_degrees < -YAW_DEADBAND_MAZE:
        return 2  # drifted right
    return 1  # roughly straight

def discretize_maze_state(color_name, dis_yaw):
    if color_name == 'Green':
        return STATE_GREEN
    if color_name == 'White':
        return STATE_WHITE
    if color_name == 'Red':
        if dis_yaw == 0:
            return STATE_LEFT_RED
        if dis_yaw == 2:
            return STATE_RIGHT_RED
        return STATE_CENTER_RED
    if color_name == 'Blue':
        if dis_yaw == 0:
            return STATE_LEFT_BLUE
        if dis_yaw == 2:
            return STATE_RIGHT_BLUE
        return STATE_CENTER_BLUE
    # Everything else (No color, Yellow, Teal, Purple, Magenta, Orange, Azure)
    # is treated as "black," matching the original script's fall-through.
    if dis_yaw == 0:
        return STATE_LEFT_BLACK
    if dis_yaw == 2:
        return STATE_RIGHT_BLACK
    return STATE_CENTER_BLACK

def get_maze_state():
    raw_color = cs_device.sensor.color
    if raw_color is None:
        raw_color = 0
    color_name = color_mapping.get(int(raw_color), 'No color')

    raw_yaw = dm_device.imu_device.yaw
    if raw_yaw is None or (isinstance(raw_yaw, float) and np.isnan(raw_yaw)):
        raw_yaw = 0
    yaw = raw_yaw_to_degrees(raw_yaw)
    dis_yaw = discretize_yaw_maze(yaw)

    return discretize_maze_state(color_name, dis_yaw), color_name, yaw

async def take_action_maze(a):
    if a == 'LEFT':
        dm_device.movement_turn_for_degrees(TURN_DEGREES_MAZE, direction=le.MOVEMENT_TURN_DIRECTION_LEFT, blocking=False)
    elif a == 'RIGHT':
        dm_device.movement_turn_for_degrees(TURN_DEGREES_MAZE, direction=le.MOVEMENT_TURN_DIRECTION_RIGHT, blocking=False)
    elif a == 'FORWARD':
        dm_device.movement_move_for_degrees(FORWARD_DEGREES_MAZE, direction=le.MOVEMENT_MOVE_DIRECTION_FORWARD, blocking=False)
    else:
        return
    await asyncio.sleep(0)
    await asyncio.sleep(REST_TIME_MAZE)

MAZE_ALPHA = 0.1  # overwritten by apply_maze_rewards() from its "Alpha" slider
MAZE_GAMMA = 0.9  # overwritten by apply_maze_rewards() from its "Gamma" slider

def init_qtable_maze():
    return QTable(states=np.arange(11), actions=np.arange(3), gamma=MAZE_GAMMA, alpha=MAZE_ALPHA)

qtable_maze = init_qtable_maze()

def update_qtable_ui_maze(qtable):
    for s in range(11):
        for a in range(3):
            window.updateMazeCell(s, a, float(qtable.table[s, a]))

def apply_maze_rewards(e=None):
    global TURN_DEGREES_MAZE, FORWARD_DEGREES_MAZE, YAW_DEADBAND_MAZE, NUM_EPISODES_MAZE, MAX_STEPS_MAZE, REST_TIME_MAZE, MAZE_ALPHA, MAZE_GAMMA
    try:
        TURN_DEGREES_MAZE = int(document.getElementById('slider-maze-turn-degrees').value)
        FORWARD_DEGREES_MAZE = int(document.getElementById('slider-maze-forward-degrees').value)
        YAW_DEADBAND_MAZE = int(document.getElementById('slider-maze-yaw-deadband').value)
        NUM_EPISODES_MAZE = int(document.getElementById('slider-maze-num-episodes').value)
        MAX_STEPS_MAZE = int(document.getElementById('slider-maze-max-steps').value)
        REST_TIME_MAZE = float(document.getElementById('slider-maze-rest-time').value)
        MAZE_ALPHA = float(document.getElementById('slider-maze-alpha').value)
        MAZE_GAMMA = float(document.getElementById('slider-maze-gamma').value)

        rewards = np.full((11, 3), -1)
        for s in range(11):
            for a, aname in enumerate(MAZE_ACTION_NAMES):
                el = document.getElementById(f'reward-maze-s{s}-{aname.lower()}')
                rewards[s, a] = int(el.value)
    except Exception as ex:
        log(f"Could not read all maze settings, keeping previous values ({ex}).")
        return

    qtable_maze.set_rewards(rewards)
    qtable_maze.alpha = MAZE_ALPHA
    qtable_maze.gamma = MAZE_GAMMA
    if e is not None:
        log(f"Maze settings updated. Turn = {TURN_DEGREES_MAZE}°, Forward = {FORWARD_DEGREES_MAZE}°, "
            f"Rest Time = {REST_TIME_MAZE:.2f}s, alpha = {MAZE_ALPHA:.2f}, gamma = {MAZE_GAMMA:.2f}, "
            f"Yaw Deadband = {YAW_DEADBAND_MAZE}°, Episodes = {NUM_EPISODES_MAZE}, Max Steps = {MAX_STEPS_MAZE}.")

def reset_qtable_maze(e=None):
    global qtable_maze
    qtable_maze = init_qtable_maze()
    apply_maze_rewards()
    update_qtable_ui_maze(qtable_maze)
    log("Maze Q-Table has been reset to zeros.")

apply_maze_rewards()
update_qtable_ui_maze(qtable_maze)

# "Robot Ready" gate: stands in for the original script's input() between
# episodes. _wait_for_maze_ready() blocks the episode loop until the button
# fires maze_robot_ready(), same as the original blocking until Enter.
_maze_ready_event = asyncio.Event()

def maze_robot_ready(e=None):
    _maze_ready_event.set()

async def _wait_for_maze_ready():
    _maze_ready_event.clear()
    log("Place the robot at the maze start, then click 'Robot Ready'.")
    await _maze_ready_event.wait()

async def run_episode_maze(eps, max_steps):
    """One maze attempt: resets yaw (the robot should already be physically
    positioned at the maze start by this point — see _wait_for_maze_ready),
    then steps until the Green goal tile is reached or max_steps runs out."""
    await reset_yaw()

    s, color, yaw = get_maze_state()
    for i in range(max_steps):
        a = qtable_maze.choose_action(s=s, eps=eps)
        action_name = direction_mapping_maze[a]
        log(f"Maze step {i + 1}/{max_steps}: State={MAZE_STATE_NAMES[s]} (color={color}, yaw={yaw}), "
            f"Action={action_name}, eps={eps:.2f}")
        await take_action_maze(a=action_name)

        next_s, color, yaw = get_maze_state()
        qtable_maze.bellman_update(s, a, next_s)
        update_qtable_ui_maze(qtable_maze)
        s = next_s

        if s == STATE_GREEN:
            # bellman_update reads rewards[current_state, a], so the terminal
            # reward is never applied by the update above (s there is the
            # state we arrived FROM). Seed it here via a self-loop so it
            # actually enters the table and can back up to earlier states.
            for terminal_action in qtable_maze.actions:
                qtable_maze.bellman_update(s, terminal_action, s)
            update_qtable_ui_maze(qtable_maze)
            dm_device.motor_stop(motor=le.MOTOR_RIGHT, blocking=False)
            dm_device.motor_stop(motor=le.MOTOR_LEFT, blocking=False)
            return 'GOAL'

    dm_device.motor_stop(motor=le.MOTOR_RIGHT, blocking=False)
    dm_device.motor_stop(motor=le.MOTOR_LEFT, blocking=False)
    return 'TIMEOUT'

async def _train_loop_maze():
    log("Starting maze training...")
    _set_running_ui(True)
    n_episodes = max(NUM_EPISODES_MAZE, 1)
    try:
        for episode in range(n_episodes):
            eps = max(1 - episode / max(1, n_episodes - 1), 0.05)
            await _wait_for_maze_ready()
            outcome = await run_episode_maze(eps=eps, max_steps=MAX_STEPS_MAZE)
            log(f"Episode {episode + 1}/{n_episodes} ended: {outcome}")

        log("DONE TRAINING (maze). Reset the robot to the maze start before running the policy.")
    except asyncio.CancelledError:
        log("Maze training stopped by user.")
    except Exception as e:
        log(f"Error: {e}")
    finally:
        global _running_task
        _running_task = None
        _set_running_ui(False)

async def _policy_loop_maze():
    _set_running_ui(True)
    try:
        await _wait_for_maze_ready()
        outcome = await run_episode_maze(eps=0, max_steps=MAX_STEPS_MAZE)
        log(f"Greedy maze run ended: {outcome}")
    except asyncio.CancelledError:
        log("Maze policy run stopped by user.")
    except Exception as e:
        log(f"Error: {e}")
    finally:
        global _running_task
        _running_task = None
        _set_running_ui(False)


def start_training(e):
    """Shared Start Training handler: dispatches to the Silly Walk,
    Line Follower, or Maze Solver training loop depending on which tab is
    currently active."""
    global _running_task, _step_s
    if _is_maze_page_active():
        if _running_task is not None:
            log("Stop the current run before starting a new one.")
            return
        _running_task = asyncio.ensure_future(_train_loop_maze())
        return
    if _is_line_page_active():
        if _running_task is not None:
            log("Stop the current run before starting a new one.")
            return
        _running_task = asyncio.ensure_future(_train_loop_line())
        return
    if _running_task is not None:
        log("Stop the current run before starting a new one.")
        return
    _step_s = None  # invalidate any in-progress manual-step session; it'll re-reset yaw on next Step click
    _running_task = asyncio.ensure_future(_train_loop())

def run_policy(e):
    """Shared Run Learned Policy handler: dispatches to the Silly Walk,
    Line Follower, or Maze Solver policy loop depending on which tab is
    currently active."""
    global _running_task, _step_s
    if _is_maze_page_active():
        if _running_task is not None:
            log("Stop the current run before starting a new one.")
            return
        _running_task = asyncio.ensure_future(_policy_loop_maze())
        return
    if _is_line_page_active():
        if _running_task is not None:
            log("Stop the current run before starting a new one.")
            return
        _running_task = asyncio.ensure_future(_policy_loop_line())
        return
    if _running_task is not None:
        log("Stop the current run before starting a new one.")
        return
    _step_s = None
    _running_task = asyncio.ensure_future(_policy_loop())

def stop_all(e):
    """Shared Stop handler for the Silly Walk, Line Follower, and Maze
    Solver pages: cancels whichever loop (if any) is currently driving the
    motors."""
    global _running_task
    if _running_task:
        _running_task.cancel()
        _running_task = None
    try:
        dm_device.motor_stop(motor=le.MOTOR_RIGHT, blocking=False)
        dm_device.motor_stop(motor=le.MOTOR_LEFT, blocking=False)
    except: pass
    log("Stopped motors.")
    _set_running_ui(False)


# ── Manual step-through ──
# Persists across "Step Once" clicks so each click advances exactly one
# transition from wherever the previous step left off. _step_s is None
# whenever there's no session in progress yet (fresh start, or invalidated
# by starting a continuous run) — the next step click will reset yaw and
# begin a new session from there.
_step_i = 0
_step_s = None
_step_yaw = None

async def _step_once():
    global _step_i, _step_s, _step_yaw, _running_task
    _set_running_ui(True)
    try:
        if _step_s is None:
            log("Resetting yaw for manual stepping...")
            await reset_yaw()
            _step_s, _step_yaw = get_yaw_state()
            _step_i = 0
            document.getElementById('step-counter').innerText = '0'

        s, yaw = _step_s, _step_yaw
        names = state_names()
        n = max(TRAIN_ITERS, 1)
        eps = max(1 - _step_i / n, 0.05)
        a = qtable_instance.choose_action(s=s, eps=eps)
        action_name = direction_mapping[a]

        window.highlightStateRow(s, 'row-from', 900)
        log(f"Manual step {_step_i + 1}: State={names[s]} (yaw={yaw}), Action={action_name}, eps={eps:.2f}")
        await take_action(a=action_name)
        await asyncio.sleep(0.2)

        next_s, next_yaw = get_yaw_state()

        old_q = float(qtable_instance.table[s, a])
        reward = float(qtable_instance.rewards[s, a])
        alpha = float(qtable_instance.alpha)
        gamma = float(qtable_instance.gamma)
        next_max_q = float(np.max(qtable_instance.table[next_s, :]))

        qtable_instance.bellman_update(s, a, next_s)
        new_q = float(qtable_instance.table[s, a])

        update_qtable_ui(qtable_instance)
        window.pulseCell(s, a, 900)
        window.highlightStateRow(next_s, 'row-to', 900)
        window.highlightActionHeader(a, 900)
        window.updateBellmanReadout(names[s], action_name, names[next_s],
                                     old_q, alpha, reward, gamma, next_max_q, new_q)

        _step_i += 1
        _step_s, _step_yaw = next_s, next_yaw
        document.getElementById('step-counter').innerText = str(_step_i)
    except Exception as ex:
        log(f"Error during manual step: {ex}")
    finally:
        _running_task = None
        _set_running_ui(False)

def step_training(e):
    global _running_task
    if _is_maze_page_active():
        log("Step Once is only available on the Silly Walk page.")
        return
    if _is_line_page_active():
        log("Step Once is only available on the Tabular Q-Learning page.")
        return
    if _running_task is not None:
        log("Stop the current run before stepping manually.")
        return
    _running_task = asyncio.ensure_future(_step_once())
