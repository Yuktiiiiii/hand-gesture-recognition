"""Download LeapGestRecog (Kaggle: gti-upm/leapgestrecog) into data/leapgestrecog.

Tries, in order:
  1. Already present / running on Kaggle with the dataset attached
     (/kaggle/input/leapgestrecog) -> symlink it.
  2. ``kagglehub`` (needs Kaggle credentials: ~/.kaggle/kaggle.json, or the
     KAGGLE_USERNAME / KAGGLE_KEY env vars, or Colab secrets).
  3. The ``kaggle`` CLI.

Manual alternative: download the zip from
https://www.kaggle.com/datasets/gti-upm/leapgestrecog and extract it into
data/leapgestrecog/ — the loader finds the right sub-folder automatically.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET = "gti-upm/leapgestrecog"
TARGET = Path(__file__).resolve().parents[1] / "data" / "leapgestrecog"


def _has_data(p: Path) -> bool:
    return p.exists() and any(p.rglob("*.png"))


def main():
    if _has_data(TARGET):
        print(f"Dataset already present at {TARGET}")
        return
    TARGET.parent.mkdir(parents=True, exist_ok=True)

    kaggle_input = Path("/kaggle/input/leapgestrecog")
    if _has_data(kaggle_input):
        if TARGET.exists() or TARGET.is_symlink():
            TARGET.unlink() if TARGET.is_symlink() else shutil.rmtree(TARGET)
        TARGET.symlink_to(kaggle_input)
        print(f"Linked Kaggle input {kaggle_input} -> {TARGET}")
        return

    try:
        import kagglehub

        path = Path(kagglehub.dataset_download(DATASET))
        if TARGET.exists() and not any(TARGET.iterdir()):
            TARGET.rmdir()
        try:
            TARGET.symlink_to(path, target_is_directory=True)
            print(f"Downloaded with kagglehub to {path} (linked at {TARGET})")
        except OSError:  # e.g. Windows without symlink rights
            shutil.copytree(path, TARGET)
            print(f"Downloaded with kagglehub and copied to {TARGET}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"kagglehub failed ({e}); trying kaggle CLI ...")

    TARGET.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(TARGET)], check=True)
        for z in TARGET.glob("*.zip"):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(TARGET)
            z.unlink()
        print(f"Downloaded with kaggle CLI to {TARGET}")
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"Could not download automatically ({e}).\n"
            "Set up Kaggle credentials (https://www.kaggle.com/docs/api) or download the zip manually from\n"
            "https://www.kaggle.com/datasets/gti-upm/leapgestrecog and extract it into data/leapgestrecog/"
        )


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    main()
