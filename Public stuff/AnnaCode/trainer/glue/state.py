"""Module-level state shared across the neural-network trainer UI.

Every other module reaches into this one via `import state` and reads/writes
`state.x` (never `from state import x`) so mutations are visible everywhere --
PyScript's cross-module rebinding only works through attribute access.
"""
import math
from pyscript import document

inputs:  list[dict] = []   # {id, name, prev_device, prev_channel}
layers:  list[dict] = []   # {id, neurons: [...], act_fn, custom_activation: {expr, pieces}}
outputs: list[dict] = []   # {id, prev_device, prev_channel}

input_counter  = 0
neuron_counter = 0
layer_counter  = 0
output_counter = 0

devices: list = []
is_running       = False
run_tick         = 0   # ticks since play_network() started, used as the x-axis for the live output plot
all_plots: dict[str, object] = {}

debug_mode = False

# Activation columns start hidden so first-touch students see a plain
# weighted-sum network; toggled on via the "+ Activation function" link.
show_activation = False

# Zoom only affects the network diagram's spacing/box sizes (via the CSS
# --zoom variable) -- never font sizes or arrow stroke widths, UNTIL zoom
# drops below TEXT_ZOOM_THRESHOLD, at which point text also starts
# shrinking (via --text-zoom) so the diagram can get extra compact.
ZOOM_MIN  = 0.2
ZOOM_MAX  = 1.0
ZOOM_STEP = 0.1
TEXT_ZOOM_THRESHOLD = 0.5
zoom_level = 1.0
input_values:       dict[int, float] = {}   # iid -> raw input reading
neuron_pre_values:  dict[int, float] = {}   # nid -> weighted sum + bias
neuron_post_values: dict[int, float] = {}   # nid -> post-activation value
output_values:      dict[int, float] = {}   # oid -> value routed to that output

# Training (manual test / backpropagate loop) --------------------------------
DEFAULT_LEARNING_RATE = 0.05   # seeds the on-page learning-rate box; the box wins after that
# layer_sources_snapshot[i] = the list of source values (raw inputs for layer 0,
# previous layer's post-activations otherwise) that fed layers[i] during the
# most recent Test() call. Backpropagate() needs this to compute weight
# gradients, since forward() overwrites neuron_pre/post_values continuously
# while the network is Running.
# Dataset-based training (Phase 1+) -------------------------------------------
training_data: list[dict] = []   # [{"id": int, "xs": {iid: float}, "ys": {oid: float}}, ...]
data_point_counter = 0
loss_history: list[float] = []
normalize_enabled = True
is_generating_data = False   # True while the "Generate data" hardware-sampling loop is running

# Crop window (Phase: dataset time-window cropping) -----------------------
# Points outside [dataset_crop_start, dataset_crop_end] stay in training_data
# and are still editable, but are hidden from the fit plot and highlighted
# red in the table. dataset_crop_end of None means "no tail crop" -- it
# tracks the newest point as more are added/streamed in, rather than
# freezing at whatever the last index happened to be when it was set.
dataset_crop_start = 0
dataset_crop_end: int | None = None

ACTIVATION_OPTIONS = [
    ("None",     ""),
    ("ReLU",     "relu"),
    ("Sigmoid",  "sigmoid"),
    ("Tanh",     "tanh"),
    ("Softplus", "softplus"),
    ("Custom",   "custom"),
]

ARROW_COLOR = "#1e40af"

piece_counter = 0   # global, so piece DOM ids never collide across layers

CUSTOM_ACT_NAMES = {
    "abs": abs, "max": max, "min": min, "round": round,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e, "inf": math.inf,
}

def get_id(id_: str):
    return document.getElementById(id_)
