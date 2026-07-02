# Seeing Through Deepfakes: An Explainable Multi-Task Detection Framework with Deep Learning and Large Language Models

---

## 1. Introduction
This project aims to build an image classification system for video-based Deepfake detection and provide both visual and textual explanations of the model's predictions.

We achieve this by training a CNN-LSTM architecture on the FaceForensics++ (FF++) dataset. Frame-level features are extracted using a pretrained CNN backbone (such as EfficientNet or ResNeXt), and a temporal LSTM layer processes the sequence to classify each video.

In addition to binary classification (Real vs. Fake), the model also performs multi-class classification to identify the specific Deepfake generation technique, such as Face2Face, FaceSwap, Deepfakes, NeuralTextures, or FaceShifter.

To interpret the results, we apply Grad-CAM for visual explanation, compute region-based activation scores using facial alignment, and generate natural language explanations using an LLM (via Ollama).

## 2. Directory Structure
```
HUFS.CSE.DE-fake-it/
├── backend/          # Django backend server
│   └── README.md     # Backend setup, API, logging, and benchmark guide
├── frontend/         # React frontend application
│   └── README.md     # Frontend setup and UI/API guide
├── model/            # Model training and explanation notebooks
│   └── README.md     # Dataset, training, prediction, and XAI guide
├── requirements.txt  # Python dependency file
└── README.md         # Project overview documentation

```
1. backend
  - This is the Django-based backend server. It includes APIs that process uploaded videos and return prediction results using the trained model.

2. frontend
  - This is the React-based user interface. It handles video uploads and displays prediction results to the user.

3. model
  - Contains modules for training the deepfake detection model, performing Grad-CAM interpretation, ROI-based visualization, and generating LLM-based textual explanations.

For implementation-specific instructions, see the README files inside `backend/`, `frontend/`, and `model/`.



## 3. System Architecture

<p align="center">
  <img width="100%" height="100%" alt="image" src="https://github.com/user-attachments/assets/e0991824-4b42-4934-9a9c-b7cbb07d609f" />
</p>


## 4. Demo 
### You can watch the [youtube video](https://youtu.be/ZVpRHxDxAwg) for demo

<p align="left">
  <a href="https://youtu.be/ZVpRHxDxAwg" target="_blank">
    <img width="800" height="450" src="https://github.com/user-attachments/assets/f4037891-c935-411f-a0e9-9f7865603690" />
  </a>
</p>





## 5. Our Results

- Binary Classification (Real vs Fake)

| Model               | Accuracy | Macro F1 | Macro Precision | Macro Recall |
| ------------------- | -------- | -------- | --------------- | ------------ |
| ResNeXt50-32x4d     | 0.94     | 0.93     | 0.94            | 0.91         |
| Xception            | 0.95     | 0.94     | 0.95            | 0.93         |
| **EfficientNet-b0** | **0.95** | **0.94** | **0.93**        | **0.95**     |

- Multi-class Classification (5-way)
  
| Model               | Accuracy | Macro F1 | Macro Precision | Macro Recall |
| ------------------- | -------- | -------- | --------------- | ------------ |
| ResNeXt50-32x4d     | 0.93     | 0.80     | 0.80            | 0.81         |
| Xception            | 0.93     | 0.80     | 0.80            | 0.80         |
| **EfficientNet-b0** | **0.94** | **0.81** | **0.82**        | **0.81**     |

- XAI + ROI Activation

<p align="left">
  <img width="80%" height="80%" alt="image" src="https://github.com/user-attachments/assets/edd9c9b5-e3a1-45b5-925f-ffe29d8f76ca" />
</p>

- LLM Explanation

<p align="left">
  <img width="80%" height="80%" alt="image" src="https://github.com/user-attachments/assets/6a77534f-236b-4864-8b93-6742ddb7fd39" />
</p>

- Pairwise Preference Evaluation of Explanation Settings

We additionally evaluated whether richer explanation settings are preferred by LLM-based judgment. The pairwise test compares binary-only explanations, binary + method explanations, and binary + method + ROI explanations on correctly classified samples only (`N = 60`, 30 real and 30 fake videos).

| Comparison A | Comparison B | Preferred Setting | LLM Judge | Overall | Fake | Real |
| ------------ | ------------ | ----------------- | --------- | ------- | ---- | ---- |
| Binary only | **Binary + Method** | B preferred | Llama3 | 60/60 (100%) | 30/30 (100%) | 30/30 (100%) |
| Binary only | **Binary + Method** | B preferred | GPT-4o | 60/60 (100%) | 30/30 (100%) | 30/30 (100%) |
| Binary + Method | **Binary + Method + ROI** | B preferred | Llama3 | 45/60 (75%) | 27/30 (90%) | 18/30 (60%) |
| Binary + Method | **Binary + Method + ROI** | B preferred | GPT-4o | 49/60 (82%) | 30/30 (100%) | 19/30 (63%) |
| Binary only | **Binary + Method + ROI** | B preferred | Llama3 | 60/60 (100%) | 30/30 (100%) | 30/30 (100%) |
| Binary only | **Binary + Method + ROI** | B preferred | GPT-4o | 60/60 (100%) | 30/30 (100%) | 30/30 (100%) |

The corresponding notebook is available at `model/explanation_pairwise_test.ipynb`.
  
  

## 6. Explainability Pipeline

- Grad-CAM visualization (using the last convolutional layer)
- ROI-based scoring (region separation based on Face Alignment)
- LLM-based explanation using Ollama to generate textual justifications for manipulated regions



## 7. Explanation Evaluation

- `Grad_CAM_and_ROI.ipynb`: Generates Grad-CAM videos, ROI activation scores, and visual explanation outputs.
- `explanation_pairwise_test.ipynb`: Runs pairwise preference tests for explanation settings using LLM-based judgment.

## 8. Preprocessing Tools

- `validate_video.py`: Removes corrupted or too-short videos
- `cut_frame.py`: Extracts the first 150 frames and saves them as MJPEG
- `make_meta_data.py`: Automatically generates labels and metadata
- `Grad_CAM_and_ROI.ipynb`: Generates CAM visualizations and ROI-based visual/textual explanations

## 9. Running the Application

Before running inference, download the trained model and place the checkpoint at:

```text
backend/detection/detector/checkpoint_v35.pt
```

The Django settings file is local-only and is not tracked in Git. Before running the backend, make sure `backend/config/settings.py` exists locally and includes the `detection`, `corsheaders`, and `rest_framework` apps, `MEDIA_URL`, `MEDIA_ROOT`, and CORS access for `http://localhost:3000`.

Run the backend server:

```bash
cd backend
python manage.py runserver
```

Run the frontend development server in a separate terminal:

```bash
cd frontend
npm install
npm start
```

The frontend runs at `http://localhost:3000`, and the backend API runs at `http://localhost:8000`.

## 10. Logging

The backend uses structured console logging for the main processing steps:

- preview video conversion
- upload analysis start and completion
- preprocessing completion
- inference result
- Grad-CAM completion
- LLM explanation completion or failure

Logs are printed in this format:

```text
[INFO] 16:12:37 detection.services.pipeline - Analysis started: sample.mp4
```

The default log level is `INFO`. To see detailed debug logs, run the backend with:

```bash
DEFAKE_LOG_LEVEL=DEBUG python manage.py runserver
```

## 11. Optional Performance Benchmark

Runtime logging is disabled in the default web API flow. To measure processing time for a local video, run:

```bash
cd backend
python manage.py benchmark_detection /path/to/video.mp4
```

Use `--skip-llm` to benchmark only preprocessing, inference, and Grad-CAM:

```bash
python manage.py benchmark_detection /path/to/video.mp4 --skip-llm
```

Benchmark logs are written to `backend/logs/performance.csv` and `backend/logs/performance.md`.

The benchmark command is implemented as a Django custom management command under `backend/detection/management/commands/`. The `management` directory is Django's standard location for app-specific CLI commands that can be run with `python manage.py <command>`.

## 12. Contributors

<table>
  <tr>
    <td align="center" width="150" valign="top">
      <a href="https://github.com/ParkJiYeoung8297">
        <img src="https://github.com/ParkJiYeoung8297.png" width="90" height="90" alt="Jiyeong Park"/>
        <br />
        <sub><b>Jiyeong Park</b></sub>
      </a>
      <br />
      <sub>First Author / Project Lead</sub>
    </td>
    <td align="center" width="150" valign="top">
      <a href="https://github.com/sercanyesilkoy">
        <img src="https://github.com/sercanyesilkoy.png" width="90" height="90" alt="Sercan Yeşilköy"/>
        <br />
        <sub><b>Sercan Yeşilköy</b></sub>
      </a>
      <br />
      <sub>Research Contributor</sub>
    </td>
    <td align="center" width="150" valign="top">
      <a href="https://github.com/dylim-326">
        <img src="https://github.com/dylim-326.png" width="90" height="90" alt="Doyeon Lim"/>
        <br />
        <sub><b>Doyeon Lim</b></sub>
      </a>
      <br />
      <sub>Research Contributor</sub>
    </td>
    <td align="center" width="150" valign="top">
      <a href="https://github.com/huiryeong">
        <img src="https://github.com/huiryeong.png" width="90" height="90" alt="Huiryeong Park"/>
        <br />
        <sub><b>Huiryeong Park</b></sub>
      </a>
      <br />
      <sub>Research Contributor</sub>
    </td>
    <td align="center" width="150" valign="top">
      <a href="https://github.com/kellylee23">
        <img src="https://github.com/kellylee23.png" width="90" height="90" alt="Eunseo Lee"/>
        <br />
        <sub><b>Eunseo Lee</b></sub>
      </a>
      <br />
      <sub>Research Contributor</sub>
    </td>
    <td align="center" width="150" valign="top">
      <a href="https://github.com/Mohsen-Ali-Alawami">
        <img src="https://github.com/Mohsen-Ali-Alawami.png?size=90" width="90" height="90" alt="Prof. Mohsen Ali Alawami"/>
        <br />
        <sub><b>Prof. Mohsen Ali Alawami</b></sub>
      </a>
      <br />
      <sub>Supervisor / Research Contributor</sub>
    </td>
  </tr>
</table>

## 13. Research Supervision

- **Prof. Mohsen Ali Alawami:** Research supervision, methodology refinement, project extension guidance, and paper writing/revisions

## 14. Contact

- Prof. Mohsen Ali Alawami: mohsencomm@hufs.ac.kr
- Jiyeong Park: wldud8297@gmail.com
