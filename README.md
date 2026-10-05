# Static Hand Gesture Recognition Using Deep Learning
### A Comparative Study of MLP, CNN and Transfer Learning — ICT-4442 Deep Learning Mini Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/hand-gesture-recognition/blob/main/notebooks/run_all_experiments.ipynb)

School of Computer Engineering, Manipal Institute of Technology (MAHE)

| Member | Reg. No. | Model |
|---|---|---|
| Upasana Sahukara | 230911038 | **Model 1** — MLP baseline + HOG hand-crafted-feature ablation |
| Prisha Chadha | 230953116 | **Model 2** — Custom CNN trained from scratch |
| Yukti Bhatia | 230953552 | **Model 3** — Transfer-learning CNN (MobileNetV2 / ResNet18) |

We classify single images of a hand pose into 10 gesture classes and compare three
architecture families under **one identical protocol**, so differences in accuracy and cost
can be attributed to the architecture rather than to the experimental setup. The best model
drives a real-time webcam demo.

---

## Results

Test set = two subjects (08, 09) **never seen during training or model selection**.
Latency = mean per-image CPU inference, batch 1, single thread.

<!-- RESULTS_START -->
_Run `python scripts/run_all.py` (or the Colab notebook). `src/compare.py` fills this table and the plots in automatically._
<!-- RESULTS_END -->

Each run's folder in `results/<run>/` has the confusion matrix, training curves, per-class
classification report, and a JSON file with every metric.

---

## Dataset

**LeapGestRecog** ([Kaggle: gti-upm/leapgestrecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)):
20,000 near-infrared images (640×240) from a Leap Motion sensor, 10 subjects (5 men, 5 women)
× 10 gestures × 200 frames.

Classes: `palm, l, fist, fist_moved, thumb, index, ok, palm_moved, c, down`

```bash
python scripts/download_data.py      # uses kagglehub / kaggle CLI, or links /kaggle/input on Kaggle
```
Or download the zip manually and extract it into `data/leapgestrecog/`. The loader finds the
right folder by itself (the Kaggle archive contains a duplicated nested copy, and we use only one).

## Experimental protocol (identical for all models)

| Aspect | Setting |
|---|---|
| Split | **Subject-wise**: train on subjects 00–05 (12k images), validate on 06–07 (4k), test on 08–09 (4k) |
| Input | Grayscale, resized to 128×128, normalised to [-1, 1]; cached once in `data/cache/` so every run sees the same pixels |
| Augmentation (train only) | Random affine (±15°, ±10% shift, 0.9–1.1 scale), horizontal flip (left/right hand), brightness/contrast jitter ±30% |
| Loss / optimiser | Cross-entropy (label smoothing 0.05), AdamW (wd 1e-4), cosine LR schedule |
| Model selection | Best validation accuracy checkpoint, early stopping (patience 8), max 25 epochs |
| Metrics | Accuracy, macro precision / recall / F1, per-class report, confusion matrix |
| Cost | Parameters, model size, MACs/FLOPs, CPU latency (batch 1, 1 thread), FPS |
| Seed | 42 |

**Why a subject-wise split?** The images are consecutive video frames, so a random split puts
near-duplicate frames of the same person in both train and test, and most models score about 100%.
Holding out whole subjects measures how well a model generalises to **new people**, which is what
matters in deployment. `--split random` is available to show this effect.

## Models

All models take the same `(B, 1, 128, 128)` tensor.

**Model 1 — MLP baseline** (`src/models/mlp.py`)
* `mlp`: flatten → 16384 → 512 → 256 → 10 (BatchNorm, ReLU, Dropout 0.4). It has no spatial
  inductive bias, so it sets the performance floor.
* `hog_mlp` (ablation): the same MLP head on **HOG** features (9 orientations, 8×8 cells,
  2×2 blocks, L2-Hys → 8,100-D). HOG is written in pure PyTorch, so it runs on the GPU inside
  `forward()` and the data pipeline stays shared. It was checked against `skimage.feature.hog` in the tests.

**Model 2 — Custom CNN** (`src/models/custom_cnn.py`)
* 4 stages of `[Conv3×3-BN-ReLU] ×2 → MaxPool → Dropout2d`, widths 32-64-128-256, then
  global average pooling → Dropout → FC 128 → FC 10. About 1.2 M parameters, Kaiming init.
* Ablation `cnn_noaug`: the same network trained without data augmentation.

**Model 3 — Transfer learning** (`src/models/transfer.py`)
* ImageNet-pretrained **MobileNetV2** (main) and **ResNet18** (alternative backbone) with a new
  dropout + linear head. The grayscale input is turned into 3-channel ImageNet-normalised input
  inside the model.
* Two-phase training: 3 epochs on the head only (backbone frozen, BN statistics fixed), then full
  fine-tuning with the backbone LR at 0.1× the head LR.
* Ablation `mobilenet_v2_frozen`: the backbone is never unfrozen (pure feature extraction).

## Quick start

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/hand-gesture-recognition.git
cd hand-gesture-recognition
pip install -r requirements.txt
python scripts/download_data.py

python scripts/run_all.py            # all 7 runs + comparison (GPU recommended)
# or individual runs:
python -m src.train --model mlp
python -m src.train --model hog_mlp
python -m src.train --model cnn
python -m src.train --model mobilenet_v2
python -m src.train --model resnet18
python -m src.compare                # table + plots + README update
```

No GPU? Open the **Colab notebook** (`notebooks/run_all_experiments.ipynb`). It runs everything
on a free T4 in about 30–45 minutes and gives you a zip of `results/` and `checkpoints/` to commit.

Useful flags: `--epochs`, `--batch-size`, `--lr`, `--img-size`, `--split {subject,random}`,
`--no-augment`, `--freeze-epochs`, `--no-pretrained`, `--device {auto,cuda,cpu,mps}`.

### Real-time webcam demo

```bash
python demo/webcam_demo.py --checkpoint checkpoints/mobilenet_v2.pt   # or the best run
```
Put your hand in the green box. The demo shows top-3 probabilities (smoothed over 8 frames),
FPS and model latency. The default `--preprocess ir` imitates the near-infrared training images
(CLAHE + Otsu background suppression). It works best against a plain, dark background.

### Domain-shift test on your own webcam images

```bash
python demo/capture_webcam_set.py --checkpoint checkpoints/mobilenet_v2.pt   # ~50–100 imgs/class
python -m src.evaluate --checkpoint checkpoints/mobilenet_v2.pt --webcam-dir data/webcam
```
This writes `results/<run>/metrics_webcam.json` and a confusion matrix, which measure how well
the model transfers from NIR training images to RGB webcam input.

### Tests

```bash
pytest -q        # synthetic dataset with the same folder layout; CPU, ~30 s
```

## Repository structure

```
├── src/
│   ├── data.py            # discovery, subject split, caching, augmentation, loaders
│   ├── models/
│   │   ├── mlp.py         # Model 1: PixelMLP, HOGMLP (+ GPU HOG)
│   │   ├── custom_cnn.py  # Model 2: GestureCNN
│   │   └── transfer.py    # Model 3: MobileNetV2 / ResNet18
│   ├── train.py           # shared training loop (two-phase for transfer models)
│   ├── evaluate.py        # metrics, confusion matrix, FLOPs, latency, webcam eval
│   ├── compare.py         # comparison table + plots, writes README results
│   └── utils.py
├── scripts/
│   ├── download_data.py
│   └── run_all.py         # every model + ablation in one command
├── demo/
│   ├── webcam_demo.py
│   └── capture_webcam_set.py
├── notebooks/run_all_experiments.ipynb   # Colab / Kaggle, end to end
├── tests/                 # pytest smoke tests + synthetic dataset generator
├── checkpoints/           # best weights per run (created by training)
└── results/               # metrics, plots, comparison (created by training)
```

## References

1. Chaudhary, A. and Raheja, J. L. "Hand Pose Recognition Using Parallel Multi Stream CNN." PMC, 2021.
2. Alqahtani, N. et al. "HGR-ViT: Hand Gesture Recognition with Vision Transformer." *Sensors* 23(12):5555, 2023.
3. Yang, J., Li, W., Zheng, M. "GestureTransformer: A Hybrid CNN-Transformer Model for Hand Gesture Recognition in Smart Educational Environments." *Modern Intelligent Times* 3:1, 2025.
4. Zabihi, S., Mohammadi, A., et al. "Light-Weighted CNN-Attention Based Architecture for Hand Gesture Recognition via ElectroMyography." arXiv:2210.15119, 2022.
5. Dosovitskiy, A. et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." arXiv:2010.11929, 2020.
6. Rautaray, S. S. and Agrawal, A. "Vision Based Hand Gesture Recognition for Human Computer Interaction: A Survey." *Artificial Intelligence Review* 43(1):1–54, 2015.
7. Mantecón, T., del Blanco, C. R., Jaureguizar, F., García, N. "Hand Gesture Recognition Using Infrared Imagery Provided by Leap Motion Controller." ACIVS 2016 (LeapGestRecog dataset).
8. Dalal, N. and Triggs, B. "Histograms of Oriented Gradients for Human Detection." CVPR 2005.
9. Sandler, M. et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018.
10. He, K. et al. "Deep Residual Learning for Image Recognition." CVPR 2016.

## License
MIT. See [LICENSE](LICENSE).
