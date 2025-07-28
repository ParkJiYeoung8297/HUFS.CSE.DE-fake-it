# Model Creation
- You will be able to preprocess the dataset, train a deepfake detection model using CNN-LSTM architecture, and predict on new unseen data using your model.  
- The model includes visual (Grad-CAM) and textual (LLM-based) explanations for interpretability.

### Note: We recommend using [Google Colab](https://colab.research.google.com/) for running the model and notebooks.

---

## Dataset 
Some of the datasets used in this project:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DeeperForensics](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master)

---

## Preprocessing
You can find preprocessing scripts in the following files:

- `validate_video.py`: Remove corrupted videos and filter short videos (less than 150 frames)
- `cut_frame.py`: Cut the first 150 frames of each video and re-encode using MJPEG
- `make_meta_data.py`: Generate metadata (label, method, dataset, split, frame count) into CSV/Excel
- `preprocessing.ipynb`: Provides preprocessing pipeline in notebook format
- Faces are cropped using MTCNN and resized to 224×224 resolution

---

## Model and Train
The model is implemented as a multi-task learning system using CNN and LSTM:

- **CNN Backbone**: ResNeXt50 / Xception / EfficientNet-b0 (transfer learning)
- **Temporal Modeling**: LSTM for sequence analysis
- **Classification Tasks**:
  - Binary classification: Real / Fake
  - Multi-class classification: Deepfake methods (Face2Face, FaceSwap, Deepfakes, NeuralTextures, FaceShifter)

Training Details:
- Load preprocessed videos and `Global_metadata.csv`
- Train/test split (default: 80/20)
- Apply class weighting to balance real/fake data
- Save the best model as `.pt` file

---

## Predict
- Load the saved `.pt` model
- Run predictions on new videos using the same preprocessing
- Supported outputs:
  - Real or Fake (binary)
  - Deepfake method (multi-class)
  - Grad-CAM heatmaps
  - Region-wise (ROI) activation
  - Text explanation via LLM

---

## Helpers
The following utilities are helpful in data and label management:
- Converting and generating global metadata via `make_meta_data.py`
- Filtering out low-quality or short-length videos
- Cutting and saving fixed-length MJPEG sequences
- Grad-CAM and ROI scoring for XAI (`Grad_CAM_and_ROI.ipynb`)

---

## Helpful Links

### Preprocessed Data
- [FaceForensics++ Fake processed videos](https://drive.google.com/file/d/1KEMw4JPPdlmhFk3QpB2IS0UAbrURudPG/view?usp=drive_link)
- [FaceForensics++ Real processed videos](https://drive.google.com/file/d/1GtuEiGhtL0nIC0Y6d0rBv-QtRfwSaAC6/view?usp=drive_link)
- [Celeb-DF Fake processed videos](https://drive.google.com/file/d/1tgfoaf2LV48ziNFqpXH7nm1USW1hfPUC/view?usp=drive_link)
- [Celeb-DF Real processed videos](https://drive.google.com/file/d/1Rlipbby2MXGWFaoay6th6lSDXMzu5Bg8/view?usp=drive_link)

**Note:** Labels for all preprocessed data are provided under `~~~`.

### Trained Models
- [Download trained models](https://drive.google.com/file/d/12VNleCHv1PB7SUnh0H0QBmObUClwUQy3/view?usp=sharing)

---

***If you need any help, feel free to reach out. We'd be happy to assist you!***
