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

Datasets: DATASETS lists one or more Roboflow YOLOv8 exports. They're
merged into one dataset (runs/dataset/) with classes matched BY NAME,
so a green-only export and a separate blue-only export train ONE model
that knows both colors. Before training, every label is checked and a
per-class box count is printed for each split -- training stops on
broken labels, and warns if a class is missing from valid/test (then
its accuracy simply isn't being measured). The tracker picks up a new
class automatically as long as its name contains "green" or "blue";
use the rename dict to fix one that doesn't (e.g. a class called "object").

Speed: trains on the laptop's Intel Arc GPU when PyTorch's Intel-GPU
build is installed (~8 s/epoch vs. 35-120 s/epoch on the CPU, which
throttles under sustained load). Falls back to the CPU automatically:

    my_env\\Scripts\\python -m pip install torch==2.14.1+xpu torchvision==0.29.1+xpu ^
        --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple

Run (from anywhere):
    my_env\\Scripts\\python "Public stuff/projects/Project 4 - Door to Door/train_minifig.py"
"""

import os
import shutil
from collections import Counter

import cv2
import torch
import yaml
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# Roboflow YOLOv8 export folders (each has data.yaml + train/valid/test),
# relative to this script's folder, each with an optional class rename:
#   ("Blue LEGO Minifig.v1.yolov8", {"object": "blue LEGO minifig"}),
DATASETS = [
    ("Green LEGO Minifig Detection.v3-blue_and_green.yolov8", {}),
]
TRACKER_COLORS = ("green", "blue")   # a class name must contain one of these for the tracker to use it
SPLITS = {"train": "train", "val": "valid", "test": "test"}   # YOLO split name -> Roboflow folder name

BASE_MODEL = "yolo11n.pt"   # COCO-pretrained nano model -- small enough to train/run on a laptop
IMG_SIZE = 512              # matches Roboflow's 512x512 export resize
EPOCHS = 100
BATCH = 8
WORKERS = 4                 # parallel data loading (the __main__ guard below makes this safe on Windows)


def pick_device():
    """Intel Arc GPU ("xpu") if PyTorch's Intel-GPU build is installed,
    NVIDIA GPU if present, else CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def disable_foreach_optimizers():
    """Work around an Intel GPU driver bug: PyTorch's multi-tensor
    ("foreach") optimizer kernels crash on the Arc 140V with
    UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY / DEVICE_LOST, even though
    the model only uses ~1 GB. Ultralytics doesn't expose a foreach
    switch, so make PyTorch's own default pick the plain per-tensor
    loop instead. Same math, a little slower per step, but works."""
    # `import ... as` (not attribute access): torch.optim hides its `optimizer` submodule name.
    import torch.optim.adam as adam_mod
    import torch.optim.adamw as adamw_mod
    import torch.optim.optimizer as optimizer_mod
    import torch.optim.sgd as sgd_mod
    for mod in (optimizer_mod, adam_mod, adamw_mod, sgd_mod):
        if hasattr(mod, "_default_to_fused_or_foreach"):
            mod._default_to_fused_or_foreach = lambda *args, **kwargs: (False, False)

RUNS_DIR = os.path.join(REPO_ROOT, "runs")                          # gitignored
MERGED_DIR = os.path.join(RUNS_DIR, "dataset")                      # rebuilt every run
OUTPUT_WEIGHTS = os.path.join(REPO_ROOT, "models", "minifig_yolo.pt")  # gitignored
PREVIOUS_WEIGHTS = os.path.join(REPO_ROOT, "models", "minifig_yolo_prev.pt")


def read_labels(path, n_classes, where):
    """YOLO label file -> list of (class_id, x, y, w, h). Raises on anything malformed."""
    rows = []
    if not os.path.exists(path):
        return rows                                    # no label file = background image
    with open(path) as f:
        for n, line in enumerate(f.read().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{where}:{n}: expected 5 values (a box), got {len(parts)} -- "
                                 "polygon labels? Re-export as 'YOLOv8' object detection.")
            cls, box = int(parts[0]), [float(v) for v in parts[1:]]
            if not 0 <= cls < n_classes:
                raise ValueError(f"{where}:{n}: class id {cls}, but data.yaml only has {n_classes} classes")
            if not all(0.0 <= v <= 1.0 for v in box):
                raise ValueError(f"{where}:{n}: box values must be 0..1, got {box}")
            rows.append((cls, *box))
    return rows


def build_dataset():
    """Merge every export in DATASETS into MERGED_DIR (classes unified by
    name), check every label, print per-class counts, and write the
    data.yaml Ultralytics trains on. Returns that yaml's path."""
    if os.path.isdir(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)
    names = []                                         # unified class list, in first-seen order
    counts = {split: Counter() for split in SPLITS}    # split -> {class name: boxes}
    images = Counter()                                 # split -> image count
    resized = Counter()                                # export folder -> images stretched to IMG_SIZE

    for i, (folder, rename) in enumerate(DATASETS):
        src = os.path.join(HERE, folder)
        with open(os.path.join(src, "data.yaml")) as f:
            src_names = yaml.safe_load(f)["names"]
        if isinstance(src_names, dict):
            src_names = [src_names[k] for k in sorted(src_names)]
        src_names = [rename.get(n, n) for n in src_names]
        remap = {}
        for old_id, name in enumerate(src_names):
            if name not in names:
                names.append(name)
            remap[old_id] = names.index(name)

        for split, rf_split in SPLITS.items():
            img_dir = os.path.join(src, rf_split, "images")
            if not os.path.isdir(img_dir):
                continue
            out_img = os.path.join(MERGED_DIR, split, "images")
            out_lbl = os.path.join(MERGED_DIR, split, "labels")
            os.makedirs(out_img, exist_ok=True)
            os.makedirs(out_lbl, exist_ok=True)
            for img in sorted(os.listdir(img_dir)):
                stem = os.path.splitext(img)[0]
                rows = read_labels(os.path.join(src, rf_split, "labels", stem + ".txt"),
                                   len(src_names), f"{folder}/{rf_split}/labels/{stem}.txt")
                # d0_, d1_ prefix: two exports can reuse the same file name
                src_img, dst_img = os.path.join(img_dir, img), os.path.join(out_img, f"d{i}_{img}")
                frame = cv2.imread(src_img)
                if frame is None:
                    raise ValueError(f"{folder}/{rf_split}/images/{img}: not a readable image")
                if frame.shape[:2] == (IMG_SIZE, IMG_SIZE):
                    shutil.copy2(src_img, dst_img)
                else:
                    # Stretch (not letterbox) to IMG_SIZE x IMG_SIZE -- the same
                    # resize Roboflow applied to the green export and the tracker
                    # applies to every webcam frame. Labels are normalized 0..1,
                    # so they stay valid unchanged.
                    cv2.imwrite(dst_img, cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))
                    resized[folder] += 1
                with open(os.path.join(out_lbl, f"d{i}_{stem}.txt"), "w") as f:
                    for cls, x, y, w, h in rows:
                        f.write(f"{remap[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                        counts[split][names[remap[cls]]] += 1
                images[split] += 1

    # ----- report + checks -----
    print("\nDataset check (boxes per class):")
    print(f"  {'class':28s}" + "".join(f"{s:>8s}" for s in SPLITS))
    for name in names:
        print(f"  {name:28s}" + "".join(f"{counts[s][name]:8d}" for s in SPLITS))
    print(f"  {'(images)':28s}" + "".join(f"{images[s]:8d}" for s in SPLITS))
    for folder, n in resized.items():
        print(f"  note: stretched {n} images from '{folder}' to {IMG_SIZE}x{IMG_SIZE} to match the tracker")
    for name in names:
        if counts["train"][name] == 0:
            raise SystemExit(f"Class '{name}' has no training boxes -- the model can't learn it.")
        for split in ("val", "test"):
            if counts[split][name] == 0:
                print(f"  WARNING: no '{name}' boxes in {split} -- its accuracy won't be measured "
                      f"there. Put some {name} images in that split in Roboflow.")
        if not any(c in name.lower() for c in TRACKER_COLORS):
            print(f"  WARNING: '{name}' has no color word {TRACKER_COLORS} -- the tracker will "
                  "ignore it. Rename it in DATASETS.")
    if images["val"] == 0:
        raise SystemExit("No validation images -- the Roboflow export needs a valid split.")

    data = {"path": MERGED_DIR, "train": "train/images", "val": "val/images",
            "names": dict(enumerate(names))}
    if images["test"]:
        data["test"] = "test/images"
    out = os.path.join(RUNS_DIR, "minifig_data.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def main():
    data_yaml = build_dataset()

    device = pick_device()
    print(f"Training on: {torch.xpu.get_device_name(0) if device == 'xpu' else device}")
    if device == "xpu":
        disable_foreach_optimizers()

    model = YOLO(BASE_MODEL)
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=device,
        project=RUNS_DIR,
        name="minifig",
        exist_ok=True,
        patience=30,      # stop early if val mAP hasn't improved in 30 epochs
        workers=WORKERS,
        # Augmentation matters a lot with only ~100 training images.
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
    if os.path.exists(OUTPUT_WEIGHTS):
        shutil.copy(OUTPUT_WEIGHTS, PREVIOUS_WEIGHTS)   # keep the last model in case this one is worse
        print(f"\nPrevious model kept as {PREVIOUS_WEIGHTS}")
    shutil.copy(best, OUTPUT_WEIGHTS)
    print(f"Best weights copied to {OUTPUT_WEIGHTS}")

    # Final scores are always computed on the CPU. On the Intel Arc GPU the
    # first inference pass after loading a model sometimes returns garbage
    # boxes (seen as precision 0.05 instead of 0.79, and an IndexError from
    # a class id that can't exist), so Ultralytics' own "Validating best.pt"
    # printout above (which ran on the GPU) can be wrong -- trust these.
    with open(data_yaml) as f:
        splits = ["val"] + (["test"] if "test" in yaml.safe_load(f) else [])
    for split in splits:
        metrics = YOLO(OUTPUT_WEIGHTS).val(data=data_yaml, split=split, imgsz=IMG_SIZE, device="cpu",
                                           project=RUNS_DIR, name=f"minifig_{split}", exist_ok=True,
                                           workers=0, verbose=False)
        print(f"\n=== {split} split (CPU) ===")
        print(f"  {'class':28s}{'P':>7s}{'R':>7s}{'mAP50':>8s}{'mAP50-95':>10s}")
        print(f"  {'all':28s}{metrics.box.mp:7.3f}{metrics.box.mr:7.3f}"
              f"{metrics.box.map50:8.3f}{metrics.box.map:10.3f}")
        for i, c in enumerate(metrics.box.ap_class_index):
            p, r, ap50, ap = metrics.box.class_result(i)
            print(f"  {metrics.names[int(c)]:28s}{p:7.3f}{r:7.3f}{ap50:8.3f}{ap:10.3f}")
    print(f"\nPlots (confusion matrix, PR curve, sample predictions): {results.save_dir}")


if __name__ == "__main__":   # required on Windows -- the dataloader re-imports this file
    main()
