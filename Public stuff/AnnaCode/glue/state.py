"""
AnnaCode — state.py

All module-level globals for the app, plus the get_id() DOM helper.
No logic beyond get_id lives here, and this file imports nothing from any
other new split-out module -- everything else imports from it.
"""
import math
from pyscript import document

# ── Topology state ──────────────────────────────────────────────────────────────

inputs:  list[dict] = []   # {id, name, prev_device, prev_channel}
layers:  list[dict] = []   # {id, neurons: [...], act_fn, custom_activation: {expr, pieces}}
outputs: list[dict] = []   # {id, prev_device, prev_channel}

input_counter  = 0
neuron_counter = 0
layer_counter  = 0
output_counter = 0

devices: list = []   # list[Device.Element]
is_running       = False
all_plots: dict[str, object] = {}

debug_mode = True
mode = "beginner"      # "beginner" | "advanced" | "fitting" | "training"
beginner_mode = True    # True (Beginner and Training) uses the simplified layout
use_activations = False       # Beginner ignores activations; Advanced and Training apply them
show_activation_blocks = False   # whether the activation columns are on screen (Advanced always; Training via its toggle)
show_activation = False       # Training mode's "+ Activation layers" toggle

def in_training_family() -> bool:
    """Fitting and Training share the brown data section and the simplified network layout."""
    return mode in ("fitting", "training")

# Training mode: recorded samples [{"id", "xs": {input id: v}, "ys": {output id: v}}], loss per epoch
training_data: list[dict] = []
data_point_counter = 0
loss_history: list[float] = []
undo_stack: list[list[dict]] = []   # batches of removed points, newest last
train_hint_closed = False   # the student closed the example hint; the next example brings it back
example_key = None   # which example curve filled the plot (None for recorded data)
is_recording = False
is_training = False

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

# ── DOM helper ───────────────────────────────────────────────────────────────

def get_id(id_: str):
    return document.getElementById(id_)
