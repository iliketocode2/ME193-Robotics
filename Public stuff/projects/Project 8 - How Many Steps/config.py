"""Settings shared by every stage of the step counter."""

from pathlib import Path

BOARD_HOST = "Leroy.local"          # your UNO Q's IP or name; override with --board
BOARD_PORT = 5005
GAME_HOST = "127.0.0.1"
GAME_PORT = 9000

SAMPLE_HZ = 100
ACC_COUNTS_PER_G = 2048.0           # sketch sets +-16 g
GYRO_COUNTS_PER_DPS = 16.4          # sketch sets +-2000 deg/s
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
CHANNEL_SETS = {"all": [0, 1, 2, 3, 4, 5], "accel": [0, 1, 2]}

WINDOW = 100                        # 1 s of samples per network input
CENTER = 70                         # the network judges sample 70: 0.7 s of history, 0.3 s of lookahead
TRAIN_HOP = 2
LIVE_HOP = 4
EVENT_TOLERANCE = 7                 # a window is "step" if a step lies within 70 ms of sample CENTER
MATCH_TOLERANCE_S = 0.2             # a predicted step counts as correct within 200 ms of a true one
MIN_STEP_GAP_S = 0.25               # debounce: no two steps closer than this

ACTIVITIES = ["walk", "still", "turn", "stairs", "shake", "carry"]
NON_STEP_ACTIVITIES = ["still", "shake"]

ROOT = Path(__file__).resolve().parent
SESSIONS_DIR = ROOT / "data" / "sessions"
SAMPLE_DIR = ROOT / "data" / "sample"
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "plots"
MEDIA_DIR = ROOT / "media"
RESULTS_DIR = ROOT / "results"
CALIBRATION_PATH = ROOT / "calibration.json"
