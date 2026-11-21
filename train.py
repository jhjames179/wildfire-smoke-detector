"""Fine-tune an Ultralytics detector on exported Pyro-SDIS data."""
from __future__ import annotations
import shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "data" / "processed" / "pyro_sdis" / "data.yaml"
RUNS = ROOT / "runs" / "wildfire_smoke"
MODEL_OUT = ROOT / "models" / "best.pt"

def main() -> None:
    if not DATA_YAML.exists():
        raise SystemExit("Dataset not found. Run: python data/download_pyro_sdis.py")
    model = YOLO("yolo11n.pt")
    model.train(data=str(DATA_YAML), epochs=30, imgsz=640, batch=8, project=str(RUNS), name="pyro_sdis", exist_ok=True)
    best = RUNS / "pyro_sdis" / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished without best.pt at {best}")
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, MODEL_OUT)
    metrics = YOLO(str(MODEL_OUT)).val(data=str(DATA_YAML), split="val")
    print(f"Saved model: {MODEL_OUT}")
    print(f"Validation precision: {metrics.box.mp:.4f}")
    print(f"Validation recall:    {metrics.box.mr:.4f}")
    print(f"Validation mAP50:     {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95:  {metrics.box.map:.4f}")
    print("The source dataset has no held-out test split; these are validation results.")

if __name__ == "__main__":
    main()
