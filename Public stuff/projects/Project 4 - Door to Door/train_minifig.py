"""
train_minifig.py -- train a YOLO object detector on the minifig dataset.

Trains locally with Ultralytics (not Roboflow's hosted training --
Roboflow was only used to draw the boxes and export the YOLOv8-format
dataset). Starts from COCO-pretrained YOLO nano weights and fine-tunes
them on our images, then reports precision / recall / mAP on the
held-out test split and copies the best weights to

    <repo root>/models/minifig_yolo.pt

which is what minifig_tracker.py loads. Everything this writes (runs/,
models/) lives OUTSIDE "Public stuff/", so it's gitignored on purpose --
model checkpoints and training plots don't belong in the class repo.

Adding the blue minifig later: add a "blue LEGO minifig" class in the
same Roboflow project, label blue images, export a new version into
this folder, point DATASET_DIR at it, and re-run this script. The
tracker picks up the new class automatically.

Run (from anywhere):
    my_env\\Scripts\\python "Public stuff/projects/Project 4 - Door to Door/train_minifig.py"
"""

import os
import shutil

import yaml
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# The Roboflow export folder (contains data.yaml + train/valid/test).
DATASET_DIR = os.path.join(HERE, "Green LEGO Minifig Detection.v2-v2.yolov8")

BASE_MODEL = "yolo11n.pt"   # COCO-pretrained nano model -- small enough to train/run on a laptop CPU
IMG_SIZE = 512              # matches Roboflow's 512x512 export resize
EPOCHS = 100
BATCH = 8
DEVICE = "cpu"              # no NVIDIA GPU on this laptop; set to 0 for CUDA

RUNS_DIR = os.path.join(REPO_ROOT, "runs")                          # gitignored
OUTPUT_WEIGHTS = os.path.join(REPO_ROOT, "models", "minifig_yolo.pt")  # gitignored


def write_fixed_data_yaml():
    """Roboflow's data.yaml uses '../train/images'-style paths that
    resolve differently depending on the working directory. Write a copy
    with an absolute `path:` root so training works from anywhere."""
    with open(os.path.join(DATASET_DIR, "data.yaml")) as f:
        src = yaml.safe_load(f)
    fixed = {
        "path": DATASET_DIR,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": dict(enumerate(src["names"])),
    }
    os.makedirs(RUNS_DIR, exist_ok=True)
    out = os.path.join(RUNS_DIR, "minifig_data.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(fixed, f, sort_keys=False)
    print(f"Classes: {fixed['names']}")
    return out


def main():
    data_yaml = write_fixed_data_yaml()

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=RUNS_DIR,
        name="minifig",
        exist_ok=True,
        patience=30,      # stop early if val mAP hasn't improved in 30 epochs
        workers=0,        # Windows: avoid dataloader multiprocessing issues
        # Augmentation matters a lot with only ~26 training images.
        # hsv_h is kept small: hue IS the feature that separates the
        # green minifig from the blue one, so don't randomize it much.
        hsv_h=0.005,
        fliplr=0.5,
        mosaic=1.0,
        degrees=10,
        plots=True,
    )

    best = os.path.join(str(results.save_dir), "weights", "best.pt")
    os.makedirs(os.path.dirname(OUTPUT_WEIGHTS), exist_ok=True)
    shutil.copy(best, OUTPUT_WEIGHTS)
    print(f"\nBest weights copied to {OUTPUT_WEIGHTS}")

    # Held-out test split: images the model never trained or early-stopped on.
    metrics = YOLO(OUTPUT_WEIGHTS).val(data=data_yaml, split="test", imgsz=IMG_SIZE,
                                       device=DEVICE, project=RUNS_DIR, name="minifig_test",
                                       exist_ok=True, workers=0)
    print("\n=== Test-split results ===")
    print(f"precision : {metrics.box.mp:.3f}")
    print(f"recall    : {metrics.box.mr:.3f}")
    print(f"mAP@50    : {metrics.box.map50:.3f}")
    print(f"mAP@50-95 : {metrics.box.map:.3f}")
    print(f"Plots (confusion matrix, PR curve, sample predictions): {results.save_dir}")


if __name__ == "__main__":   # required on Windows -- the dataloader re-imports this file
    main()
