"""Download Pyro-SDIS v2.0 and export supplied boxes in YOLO format."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "processed" / "pyro_sdis"
REPO_ID = "pyronear/pyro-sdis"
REVISION = "v2.0"


def main() -> None:
    # Prefer the Parquet shards downloaded by the user. Fall back to the
    # pinned Hugging Face revision when local shards are not available.
    downloads = Path.home() / "Downloads"
    train_files = sorted(downloads.glob("train-*-of-*.parquet"))
    val_files = sorted(downloads.glob("val-*-of-*.parquet"))
    if train_files and val_files:
        cache_dir = ROOT / ".hf_cache"
        dataset = load_dataset(
            "parquet",
            data_files={"train": [str(p) for p in train_files], "val": [str(p) for p in val_files]},
            cache_dir=str(cache_dir),
        )
        print(f"Loading local Parquet shards from {downloads}")
    else:
        dataset = load_dataset(REPO_ID, revision=REVISION, cache_dir=str(ROOT / ".hf_cache"))
    seen_hashes: set[str] = set()
    for split in ("train", "val"):
        images_dir = OUTPUT / "images" / split
        labels_dir = OUTPUT / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        removed = 0
        for row in dataset[split]:
            image_name = Path(row["image_name"]).name
            image = row["image"]
            digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
            if digest in seen_hashes:
                removed += 1
                continue
            seen_hashes.add(digest)
            image.save(images_dir / image_name)
            annotations = (row["annotations"] or "").strip()
            label_path = labels_dir / f"{Path(image_name).stem}.txt"
            label_path.write_text(annotations + ("\n" if annotations else ""), encoding="utf-8")
            count += 1
        print(f"{split}: exported {count} images, removed {removed} exact duplicates")

    data_config = {
        "path": str(OUTPUT.relative_to(ROOT)),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["smoke"],
    }
    (OUTPUT / "data.yaml").write_text(yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8")
    source = (
        "Dataset: Pyro-SDIS v2.0\n"
        "Source: https://huggingface.co/datasets/pyronear/pyro-sdis\n"
        "Revision: v2.0\n"
        "License: Apache-2.0\n"
        "Original train/val splits and smoke boxes preserved; exact duplicate pixels removed across both splits. "
        "No test split or segmentation masks are supplied.\n"
    )
    (OUTPUT / "SOURCE.txt").write_text(source, encoding="utf-8")
    print(f"YOLO dataset ready at {OUTPUT}")


if __name__ == "__main__":
    main()
