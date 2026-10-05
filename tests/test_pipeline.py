"""Smoke tests: `pytest -q` (runs on CPU in ~1-2 minutes, no download needed)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import build_samples, find_dataset_root, split_samples  # noqa: E402
from src.models import MODEL_NAMES, build_model  # noqa: E402
from src.models.mlp import TorchHOG  # noqa: E402
from src.utils import count_macs  # noqa: E402


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(ROOT / "tests/make_synthetic_dataset.py"), "--out", str(out),
                    "--per-class", "4"], check=True)
    return out


def test_discovery_and_subject_split(synth):
    root = find_dataset_root(synth)
    samples, classes = build_samples(root)
    assert len(classes) == 10 and classes[0] == "palm"
    idx = split_samples(samples, "subject")
    subj = lambda ids: {samples[i].subject for i in ids}  # noqa: E731
    assert subj(idx["test"]) == {"08", "09"} and subj(idx["val"]) == {"06", "07"}
    assert not (subj(idx["train"]) & (subj(idx["val"]) | subj(idx["test"])))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_forward(name):
    m = build_model(name, 10, 64, pretrained=False).eval()
    with torch.no_grad():
        y = m(torch.randn(2, 1, 64, 64))
    assert y.shape == (2, 10)
    assert count_macs(m, 64) > 0


def test_torch_hog_matches_skimage():
    skimage = pytest.importorskip("skimage.feature")
    rng = np.random.default_rng(0)
    img = rng.random((64, 64)).astype(np.float32)
    ref = skimage.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      block_norm="L2-Hys", feature_vector=True)
    ours = TorchHOG()(torch.from_numpy(img)[None, None])[0].numpy()
    assert ours.shape == ref.shape
    assert np.corrcoef(ours, ref)[0, 1] > 0.8


def test_train_end_to_end(synth, tmp_path):
    from src.train import main

    res = main(["--model", "cnn", "--data-dir", str(synth), "--img-size", "64", "--epochs", "2",
                "--num-workers", "0", "--latency-runs", "5", "--cache-dir", str(tmp_path / "cache"),
                "--out-dir", str(tmp_path / "results"), "--ckpt-dir", str(tmp_path / "ckpt")])
    assert 0 <= res["accuracy"] <= 1
    assert (tmp_path / "results/cnn/confusion_matrix_test.png").exists()
    assert (tmp_path / "ckpt/cnn.pt").exists()
