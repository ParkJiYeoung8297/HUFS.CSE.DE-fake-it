# Model Creation
- You will be able to preprocess the dataset, train a deepfake detection model using CNN-LSTM architecture, and predict on new unseen data using your model.  
- The model includes visual (Grad-CAM) and textual (LLM-based) explanations for interpretability.

### Note: We recommend using [Google Colab](https://colab.research.google.com/) for running the model and notebooks.

---

## Dataset 
Some of the datasets used in this project:
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [Deepfake Detection Challenge (DFDC)](https://www.kaggle.com/c/deepfake-detection-challenge/data)
- DeeperForensics 

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
- [Celeb-DF Fake processed videos]()
- [Celeb-DF Real processed videos]()
- [FaceForensics++ Real and Fake]()
- [DFDC Fake]()
- [DFDC Real]()
- [DeeperForensics]()

**Note:** Labels for all preprocessed data are provided under `~~~`.

### Trained Models
- [Download trained models]()

---

***If you need any help, feel free to reach out. We'd be happy to assist you!***
