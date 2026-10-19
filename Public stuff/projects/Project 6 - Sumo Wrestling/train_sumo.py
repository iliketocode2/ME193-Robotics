"""
train_sumo.py -- train the robot detector on our Roboflow export.

Same approach as Project 4 (train_minifig.py): Roboflow is only used to draw
boxes and export a "YOLOv8" dataset; training runs locally with Ultralytics,
fine-tuning COCO-pretrained yolo11n. Reuses Project 4's GPU picking and its
Intel Arc optimizer workaround rather than copying them.

Expects ONE class (any name, e.g. "robot") boxed on every robot -- ours too.
vision.py tells ours apart by the AprilTag, so the model doesn't need to.

Output: <repo>/models/sumo_yolo.pt (gitignored), which sumo_brain.py loads
automatically on its next start (otherwise it runs in tag-only mode).

Run with my_env_cpu (CPU-only PyTorch; Windows Smart App Control blocks
my_env's Intel-GPU PyTorch DLLs). 30 images took ~37 min:
    my_env_cpu\\Scripts\\python "Public stuff/projects/Project 6 - Sumo Wrestling/train_sumo.py"
It also writes models/sumo_yolo.onnx, which sumo_brain.py (in my_env) runs
with OpenVINO on the NPU.
"""

import importlib.util
import os
import shutil

import yaml

from config import HERE, MODEL_PATH, REPO_ROOT, YOLO_IMG_SIZE

# TODO: the folder name of the unzipped Roboflow export, inside this project folder
DATASET_DIR = "Lego Car Detection.v1.yolov8"
BASE_MODEL = "yolo11n.pt"
EPOCHS = 120
BATCH = 8
WORKERS = 4
RUNS_DIR = os.path.join(REPO_ROOT, "runs")


def _project4_helpers():
    path = os.path.join(HERE, "..", "Project 4 - Door to Door", "train_minifig.py")
    spec = importlib.util.spec_from_file_location("train_minifig", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.pick_device, mod.disable_foreach_optimizers


def write_data_yaml():
    """Roboflow's data.yaml uses relative paths that break when training from
    elsewhere; write an absolute-path copy under runs/."""
    src = os.path.join(HERE, DATASET_DIR)
    if not os.path.isdir(src):
        raise SystemExit(f"Dataset folder not found: {src}\n"
                         "Unzip the Roboflow 'YOLOv8' export there, or fix DATASET_DIR.")
    with open(os.path.join(src, "data.yaml")) as f:
        data = yaml.safe_load(f)
    out = {"path": src, "names": data["names"], "nc": len(data["names"])}
    for split, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if os.path.isdir(os.path.join(src, folder, "images")):
            out[split] = f"{folder}/images"
    if "val" not in out:
        out["val"] = out["train"]                 # tiny export with no valid split
        print("WARNING: no valid/ split -- validating on train (accuracy will look too good)")
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, "sumo_data.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(out, f)
    print(f"classes: {out['names']}")
    return path


def main():
    from ultralytics import YOLO
    pick_device, disable_foreach_optimizers = _project4_helpers()
    data_yaml = write_data_yaml()
    device = pick_device()
    if device == "xpu":
        disable_foreach_optimizers()
    print(f"training on {device}")
    model = YOLO(os.path.join(REPO_ROOT, BASE_MODEL) if os.path.exists(os.path.join(REPO_ROOT, BASE_MODEL))
                 else BASE_MODEL)
    model.train(data=data_yaml, epochs=EPOCHS, imgsz=YOLO_IMG_SIZE, batch=BATCH, device=device,
                project=RUNS_DIR, name="sumo", exist_ok=True, patience=30, workers=WORKERS,
                fliplr=0.5, flipud=0.5, degrees=180,     # overhead view: robots face any direction
                mosaic=1.0, plots=True)
    best = os.path.join(RUNS_DIR, "sumo", "weights", "best.pt")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    shutil.copy(best, MODEL_PATH)
    print(f"saved {MODEL_PATH}")
    # ONNX copy for the brain: vision.py runs it with OpenVINO on the NPU/GPU,
    # no PyTorch needed (my_env's PyTorch is blocked by Smart App Control)
    onnx = YOLO(MODEL_PATH).export(format="onnx", imgsz=YOLO_IMG_SIZE, dynamic=False, simplify=True)
    print(f"saved {onnx} -- restart sumo_brain.py to use it")


if __name__ == "__main__":   # required on Windows -- the dataloader re-imports this file
    main()
