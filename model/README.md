# Model

This directory contains core deep learning pipeline for **DE-Fake it**:

- Data preprocessing
- CNN-LSTM model training
- Model evaluation
- Grad-CAM visualization and ROI analysis
- LLM prompt generation from ROI statistics


[Google Colab](https://colab.research.google.com/) is recommended for the notebooks because training, evaluation, and
Grad-CAM generation are GPU-heavy.

---


## Pretrained Model

### Model Configuration

| Item | Value |
| --- | --- |
| Backbone | EfficientNet-b0 |
| Temporal module | LSTM |
| Input size | 224 x 224 |
| Binary labels | `FAKE = 0`, `REAL = 1` |
| Method classes | `original`, `Deepfakes`, `FaceShifter`, `FaceSwap`, `NeuralTextures`, `Face2Face`, `others` |
| Checkpoint | `checkpoint_v35_best.pt` |


### Download

Download the pretrained checkpoint from Google Drive: 

**[checkpoint_v35_best.pt](https://drive.google.com/file/d/12VNleCHv1PB7SUnh0H0QBmObUClwUQy3/view?usp=sharing)**

Place the downloaded file at:

```text
<PROJECT_ROOT>/checkpoints/checkpoint_v35/checkpoint_v35_best.pt
```

If you also want to run the web demo, copy the checkpoint to:

```text
backend/detection/detector/checkpoint_v35_best.pt
```

---

## Expected Project Layout

Set the `PROJECT_ROOT` variable in each notebook to the repository root.

```text
<PROJECT_ROOT>/
├── Dataset/
│   ├── ff++/
│   │   ├── train/
│   │   └── test/
│   └── ff++(grad-cam)/
├── checkpoints/
│   └── checkpoint_v35/
│       └── checkpoint_v35_best.pt
└── model/
```

The FaceForensics++ directory is expected to have the following structure:

```text
<PROJECT_ROOT>/Dataset/ff++/test/
├── fake/
│   ├── Deepfakes/
│   ├── Face2Face/
│   ├── FaceShifter/
│   ├── FaceSwap/
│   └── NeuralTextures/
└── real/
    └── original/
```

For instructions on downloading and preparing the datasets used in this project, please refer to **[`../Dataset/README.md`](../Dataset/README.md)**.

---

## Runtime Notes

The notebooks are designed to run on Google Colab.

Some notebooks install the required packages in their first cells (e.g., `facenet-pytorch` and `face_alignment`).

A GPU runtime is recommended.

---


# Workflow

The typical workflow for reproducing our model pipeline is:

```text
Raw videos
      │
      ▼
preprocessing.ipynb
      │
      ▼
Processed Dataset
      │
      ▼
model_train_test.ipynb
      │
      ├── checkpoint_v35_best.pt
      └── (test)_checkpoint_v35_predictions_ff++.xlsx
                     │
                     ▼
gradcam_and_roi.ipynb
                     │
                     ├── ROI images
                     └── video_roi_result.json
                                │
                                ▼
Helpers/make_llm_prompt.py
                                │
                                ▼
LLM Prompt
```


---

# Notebook Guide

| Notebook | Purpose | Input | Output |
| --- | --- | --- | --- |
| `preprocessing.ipynb` | Face detection and preprocessing | Raw videos | Processed dataset |
| `model_train_test.ipynb` | Train, validate, and evaluate the model | Processed dataset | Checkpoints, metrics, prediction file |
| `cross_dataset_evaluation.ipynb` | Evaluate the trained model on external datasets (e.g., Celeb-DF, DeeperForensics) | Pretrained checkpoint + external preprocessed dataset | Prediction file and benchmark metrics |
| `gradcam_and_roi.ipynb` | Generate Grad-CAM visualizations and ROI statistics | Prediction file | ROI images, ROI summaries, JSON |

---

# Quick Start

Using the pretrained model:

1. Download `checkpoint_v35_best.pt`.
2. Prepare the FaceForensics++ dataset.
3. Set `PROJECT_ROOT`, `DATA_ROOT`, and `CHECKPOINT_NAME` in the notebook configuration cells.
4. Run the FaceForensics++ test section of `model_train_test.ipynb`.
5. Run `gradcam_and_roi.ipynb`.
6. Run `Helpers/make_llm_prompt.py`.

Expected key outputs:

- `<PROJECT_ROOT>/checkpoints/checkpoint_v35/(test)_checkpoint_v35_predictions_ff++.xlsx`
- `<PROJECT_ROOT>/Dataset/ff++(grad-cam)/.../*_video_roi_result.json`

Classification metrics such as confusion matrix, ROC-AUC, EER, and pAUC are printed or saved by `model_train_test.ipynb` and `cross_dataset_evaluation.ipynb`.


---

# Helper Scripts

| Script | Purpose |
| --- | --- |
| `make_meta_data.py` | Generate metadata files |
| `validate_video.py` | Validate unreadable or corrupted videos |
| `cut_frame.py` | Create fixed-length clips |
| `make_llm_prompt.py` | Generate LLM prompts from ROI JSON |

Example:

```bash
python model/Helpers/make_meta_data.py
python model/Helpers/validate_video.py
python model/Helpers/cut_frame.py
python model/Helpers/make_llm_prompt.py
```

---

## Notes

- `final_probability` is the final video-level prediction probability used in LLM prompt generation.
- `cam_score` is a frame-level Grad-CAM activation score used only for ROI aggregation.
- The backend uses the same trained checkpoint and LLM prompt generation pipeline.
---
