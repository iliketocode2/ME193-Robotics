"""Settings shared by every stage of the finger-count lock."""

from pathlib import Path

CAMERA_INDEX = 1          # the USB webcam; run `python collect.py --list` to see which index it is
FRAME_W, FRAME_H = 640, 480
CROP_FRAC = 0.9             # the network sees the center square of the frame, this fraction of the frame height

SAVE_SIZE = 160             # collected images are saved as 160 x 160 grayscale squares
IMG_SIZE = 64               # the network input is 64 x 64 grayscale
CLASSES = [0, 1, 2, 3, 4, 5]  # fingers showing; 0 also covers a fist or an empty view

# Lock policy (lock_policy.py)
LOCK_CODE = (3, 1, 5)       # the secret: three finger counts, each 1 to 5
MIN_CONF = 0.80            # a reading counts only when the network is at least this sure
HOLD_S = 1.0                # ... and keeps giving the same count for this long
CLEAR_S = 0.4               # the hand must be gone (count 0) this long between digits
DIGIT_TIMEOUT_S = 10.0      # no digit within this time cancels the attempt
OPEN_S = 5.0                # how long the door stays unlocked
MAX_FAILS = 3               # wrong codes in a row before the lockout
LOCKOUT_S = 30.0            # how long the lockout lasts
RESULT_S = 1.5              # how long the "wrong code" signal shows

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "images"     # data/images/<label>/<session>_<n>.png
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "finger_cnn.pt"
NUMPY_WEIGHTS_PATH = MODELS_DIR / "finger_cnn.npz"
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"
