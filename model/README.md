# Model

This directory contains the model-side research artifacts for **DE-Fake it**:
preprocessing, CNN-LSTM training/evaluation, Grad-CAM + ROI explainability, and
pairwise explanation evaluation.

Google Colab is recommended for the notebooks because model training and
Grad-CAM generation are GPU-heavy.

For paper reproduction, the notebooks are the primary artifacts. Helper scripts
are optional utilities for rebuilding metadata or checking local video files.

---

## Final Model Used by the Application

The deployed backend uses the following configuration:

| Item | Value |
| --- | --- |
| Backbone | EfficientNet-b0 |
| Temporal module | LSTM |
| Final checkpoint | `checkpoint_v35.pt` |
| Input size | 224 × 224 |
| Frame count | 150 preprocessed video frames |
| Binary labels | `FAKE = 0`, `REAL = 1` |
| Method classes | `original`, `Deepfakes`, `FaceShifter`, `FaceSwap`, `NeuralTextures`, `Face2Face`, `others` |

Download the trained checkpoint from the project link below and place it at:

```text
backend/detection/detector/checkpoint_v35.pt
```

For notebook-based evaluation, place the same checkpoint under:

```text
<PROJECT_ROOT>/checkpoints/checkpoint_v35/checkpoint_v35.pt
```

---

## Dataset Sources

The experiments use videos from:

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DeeperForensics](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master)

Expected dataset layout for the notebooks. The repository includes a local
`Dataset/README.md` guide, while actual dataset files are ignored by Git.

```text
<PROJECT_ROOT>/
├── Dataset/
│   ├── FaceForensics++_C23/
│   ├── ff++/
│   │   ├── train/
│   │   │   ├── fake/
│   │   │   │   ├── Deepfakes/
│   │   │   │   ├── Face2Face/
│   │   │   │   ├── FaceShifter/
│   │   │   │   ├── FaceSwap/
│   │   │   │   └── NeuralTextures/
│   │   │   └── real/
│   │   │       └── original/
│   │   └── test/
│   │       ├── fake/
│   │       │   ├── Deepfakes/
│   │       │   ├── Face2Face/
│   │       │   ├── FaceShifter/
│   │       │   ├── FaceSwap/
│   │       │   └── NeuralTextures/
│   │       └── real/
│   │           └── original/
│   ├── celeb-df/
│   ├── DeeperForensics/
│   └── ff++(grad-cam_v2)/
└── checkpoints/
    └── checkpoint_v35/
        └── checkpoint_v35.pt
```

Each notebook has a configuration cell near the top:

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/DE-Fake-it")
DATA_ROOT = PROJECT_ROOT / "Dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
```

Change only `PROJECT_ROOT` to match your own environment.

---

## Quick Reproduction with the Pretrained Model

Use this path when you want to reproduce inference and explanation outputs
without retraining.

1. Download the trained model from the link below.
2. Place it at `<PROJECT_ROOT>/checkpoints/checkpoint_v35/checkpoint_v35.pt`.
3. Download or prepare preprocessed videos under `<PROJECT_ROOT>/Dataset/ff++/test/<label>/<method>/`.
4. Open `model_train_test.ipynb` and set `PROJECT_ROOT` in the first configuration cell.
5. Run the FaceForensics++ test section to load `checkpoint_v35.pt` and evaluate videos.
6. Open `Grad_CAM_and_ROI.ipynb`, set the same `PROJECT_ROOT`, and run one sample folder to generate Grad-CAM and ROI outputs.

This is the recommended smoke test before publishing or reviewing the repository.

---

## Full Reproduction from Raw Videos

Use this path only when rebuilding the processed dataset from raw videos.

1. Put raw FaceForensics++ videos under `Dataset/FaceForensics++_C23/`.
2. Run `preprocessing.ipynb`.
   - The notebook processes one manipulation method at a time.
   - Change `DATA_METHOD` in the configuration cell and repeat for each method, such as `Deepfakes`, `Face2Face`, `FaceShifter`, `FaceSwap`, and `NeuralTextures`.
3. Run `Helpers/validate_video.py` to identify unreadable or short videos.
4. Run `Helpers/cut_frame.py` if you need fixed-length MJPEG videos while preserving the folder structure.
5. Run `Helpers/make_meta_data.py` to generate `Global_metadata.csv`.
6. Run `model_train_test.ipynb` for training and evaluation.

---

## Reproduction Order

Run the model artifacts in this order:

1. `preprocessing.ipynb`
   - Crops faces with MTCNN.
   - Resizes face frames to 224 × 224.
   - Saves preprocessed videos for training and evaluation.

2. `Helpers/make_meta_data.py`
   - Generates `Global_metadata.csv` and full Excel metadata.
   - The compact CSV is used by `model_train_test.ipynb`.

3. `model_train_test.ipynb`
   - Trains and evaluates CNN-LSTM models.
   - The final documented configuration is `EfficientNet-b0` with `checkpoint_v35`.
   - Also includes FaceForensics++, Celeb-DF, and DeeperForensics test sections.

4. `Grad_CAM_and_ROI.ipynb`
   - Generates Grad-CAM videos.
   - Computes facial ROI activation scores.

5. `explanation_pairwise_test.ipynb`
   - Runs pairwise explanation preference evaluation.
   - The paper table uses correctly classified samples only (`N = 60`, 30 real and 30 fake videos).
   - Uses detector modules directly and does not require Django settings.

---

## Metadata Schema

`Global_metadata.csv` is expected in this compact column order:

```text
file_name,label,method
```

For compatibility with the training notebook, the helper writes the compact CSV
without a header by default. If you change `CSV_HEADER = True` in the helper,
also update the notebook CSV loading code.

The full Excel metadata contains:

| Column | Meaning |
| --- | --- |
| `file_name` | Video filename |
| `folder_path` | Path relative to the dataset root |
| `label` | `REAL`, `FAKE`, or `unknown` |
| `split` | `train`, `validation`, `test`, or `unknown` |
| `dataset` | `ff++`, `celeb-df`, `deeperforensics`, `dfdc`, or `unknown` |
| `method` | Deepfake manipulation method |
| `frame` | Number of frames reported by OpenCV |

---

## Helper Scripts

The helper scripts are optional. Pretrained-model reproduction does not require
running them.

Each helper script has a **User configuration** block at the top. The simplest
workflow is to edit those variables and run the script directly.

```bash
python model/Helpers/make_meta_data.py
python model/Helpers/validate_video.py
python model/Helpers/cut_frame.py
```

The configurable values include dataset paths, output paths, and frame count.
`validate_video.py` and `cut_frame.py` search input directories recursively with
`rglob("*.mp4")`. `cut_frame.py` preserves the relative folder structure under
the output directory.

---

## Training Details

- Split: 80/20 train/validation split in the notebook
- Random seed: `42`
- Epochs: `30`
- Sequence length during notebook training: `10`
- Image normalization: ImageNet mean/std
- Loss: binary classification loss + method classification loss
- Class balancing: class weighting for real/fake imbalance
- Candidate backbones: ResNeXt50-32x4d, Xception, EfficientNet-b0
- Final application checkpoint: EfficientNet-b0, `checkpoint_v35`

---

## Outputs

Supported model and explanation outputs:

- Binary prediction: Real/Fake
- Multi-class method prediction
- Grad-CAM heatmap video
- ROI activation table
- LLM-based explanation text
- Pairwise explanation preference summaries

---

## Preprocessed Data

- [FaceForensics++ Fake processed videos](https://drive.google.com/file/d/1KEMw4JPPdlmhFk3QpB2IS0UAbrURudPG/view?usp=drive_link)
- [FaceForensics++ Real processed videos](https://drive.google.com/file/d/1GtuEiGhtL0nIC0Y6d0rBv-QtRfwSaAC6/view?usp=drive_link)
- [Celeb-DF Fake processed videos](https://drive.google.com/file/d/1tgfoaf2LV48ziNFqpXH7nm1USW1hfPUC/view?usp=drive_link)
- [Celeb-DF Real processed videos](https://drive.google.com/file/d/1Rlipbby2MXGWFaoay6th6lSDXMzu5Bg8/view?usp=drive_link)

## Trained Models

- [Download trained models](https://drive.google.com/file/d/12VNleCHv1PB7SUnh0H0QBmObUClwUQy3/view?usp=sharing)

---

## Final Check Before Release

Before publishing the repository, run a smoke test rather than a full retraining:

1. Load `checkpoint_v35.pt`.
2. Run inference on one short preprocessed video.
3. Generate Grad-CAM and ROI outputs for one sample.
4. Run the backend/frontend upload flow with one test video.

Full model retraining is optional because it is expensive, but the checkpoint
load, inference, and explanation path should be verified end to end.
