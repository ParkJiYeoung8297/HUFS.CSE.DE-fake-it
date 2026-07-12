# Seeing Through Deepfakes: An Explainable Multi-Task Detection Framework with Deep Learning and Large Language Models

![Python](https://img.shields.io/badge/Python-3.11.3-3776AB?logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-5.0.1-092E20?logo=django&logoColor=white)
![React](https://img.shields.io/badge/React-19.1.0-61DAFB?logo=react&logoColor=20232A)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?logo=pytorch&logoColor=white)

We release the code for **Seeing Through Deepfakes**, an explainable deepfake detection framework that connects video classification, visual evidence, facial-region analysis, and natural-language explanation generation.

The project is designed around one question: how can a deepfake detector make its decision easier to inspect? Instead of only returning a real/fake label, the pipeline also produces Grad-CAM visualizations, facial ROI activation scores, and an LLM-generated explanation grounded in the model outputs.

```text
Video input
  -> multi-task CNN-LSTM detection
  -> Grad-CAM visual evidence
  -> facial ROI activation analysis
  -> LLM-generated textual explanation
  -> web-based inspection interface
```

The detection model jointly performs:

- **Binary classification:** real vs. fake
- **Manipulation-type classification:** Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter, or original

## Project Highlights

- Multi-task video-based deepfake detection using a CNN-LSTM architecture
- Comparison of ResNeXt50-32x4d, Xception, and EfficientNet-b0 backbones
- Grad-CAM-based visual explanation for model-sensitive regions
- Facial ROI activation scoring for region-level interpretability
- LLM-based explanation generation from prediction, method, and ROI evidence
- React + Django demo application for reviewing predictions and explanations together

## Directory Structure

```text
HUFS.CSE.DE-fake-it/
├── backend/          # Django inference API and explanation pipeline
├── frontend/         # React demo interface
├── model/            # Research notebooks, training, evaluation, and XAI
├── Dataset/          # Dataset layout and preparation guide
├── requirements.txt  # Shared Python dependencies
└── README.md         # Project-level overview
```

Each component has its own guide. The root README keeps only the project-level overview so that setup and implementation details stay close to the code they describe.

| Guide | Use this for |
| --- | --- |
| [`frontend/README.md`](frontend/README.md) | Running and building the React interface |
| [`backend/README.md`](backend/README.md) | Running the Django server, API behavior, LLM setup, and outputs |
| [`model/README.md`](model/README.md) | Checkpoints, notebooks, training, evaluation, Grad-CAM, ROI, and explanation prompt generation |
| [`Dataset/README.md`](Dataset/README.md) | Dataset sources, expected structure, and preprocessing notes |

## System Architecture

The figure below summarizes the complete inference and explainability pipeline used in the deployed system.

<p align="center">
  <img width="100%" height="100%" alt="System architecture" src="https://github.com/user-attachments/assets/86e4cdc1-e87e-4e19-9eea-5dde0558ca60" />
</p>

## Demo

The demo lets a user upload a video, preview it, run deepfake analysis, and inspect prediction results with Grad-CAM videos, ROI scores, and a text explanation.

Watch the demo video on [YouTube](https://youtu.be/ZVpRHxDxAwg).

<p align="left">
  <a href="https://youtu.be/ZVpRHxDxAwg" target="_blank">
    <img width="800" height="450" alt="Demo video thumbnail" src="https://github.com/user-attachments/assets/f4037891-c935-411f-a0e9-9f7865603690" />
  </a>
</p>



## Quick Start

To run the local demo, prepare the shared Python dependencies, place the pretrained checkpoint where the backend expects it, then start the backend and frontend applications.

```bash
pip install -r requirements.txt
```

Required checkpoint path for the demo:

```text
backend/detection/detector/checkpoint_v35_best.pt
```

Then start the backend and frontend using the [`backend/README.md`](backend/README.md) and [`frontend/README.md`](frontend/README.md) guides.

By default, the frontend runs at `http://localhost:3000` and the backend API runs at `http://localhost:8000`.

## Model And Data

The deployed demo uses the final EfficientNet-b0 + LSTM checkpoint, `checkpoint_v35_best.pt`. The checkpoint is available from [Google Drive](https://drive.google.com/file/d/12VNleCHv1PB7SUnh0H0QBmObUClwUQy3/view).

Experiments use videos from FaceForensics++, Celeb-DF, and DeeperForensics. Dataset files are not tracked in Git because they are large and have separate access conditions. See [`Dataset/README.md`](Dataset/README.md) for local layout and rebuilding notes.

## Reproducing Results

The research workflow is documented in [`model/README.md`](model/README.md). At a high level, reproduction follows this order:

1. Prepare and preprocess datasets
2. Generate metadata
3. Train or evaluate the multi-task detector
4. Run Grad-CAM and ROI analysis
5. Evaluate explanation preferences

The corresponding notebooks and helper scripts are maintained under [`model/`](model/).

## Results

### Binary Classification

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
| --- | --- | --- | --- | --- |
| ResNeXt50-32x4d | 0.94 | 0.93 | 0.94 | 0.91 |
| Xception | 0.95 | 0.94 | 0.95 | 0.93 |
| **EfficientNet-b0** | **0.95** | **0.94** | **0.93** | **0.95** |

### Multi-Class Classification

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
| --- | --- | --- | --- | --- |
| ResNeXt50-32x4d | 0.93 | 0.80 | 0.80 | 0.81 |
| Xception | 0.93 | 0.80 | 0.80 | 0.80 |
| **EfficientNet-b0** | **0.94** | **0.81** | **0.82** | **0.81** |

### Explanation Preference Evaluation

We compare binary-only explanations, binary + method explanations, and binary + method + ROI explanations on correctly classified samples only (`N = 60`, 30 real and 30 fake videos).

| Comparison A | Comparison B | Preferred Setting | LLM Judge | Overall | Fake | Real |
| --- | --- | --- | --- | --- | --- | --- |
| Binary only | **Binary + Method** | B preferred | Llama3 | 60/60 (100%) | 30/30 (100%) | 30/30 (100%) |
| Binary only | **Binary + Method** | B preferred | GPT-4o | 60/60 (100%) | 30/30 (100%) | 30/30 (100%) |
| Binary + Method | **Binary + Method + ROI** | B preferred | Llama3 | 45/60 (75%) | 27/30 (90%) | 18/30 (60%) |
| Binary + Method | **Binary + Method + ROI** | B preferred | GPT-4o | 49/60 (82%) | 30/30 (100%) | 19/30 (63%) |
| Binary only | **Binary + Method + ROI** | B preferred | Llama3 | 60/60 (100%) | 30/30 (100%) | 30/30 (100%) |
| Binary only | **Binary + Method + ROI** | B preferred | GPT-4o | 60/60 (100%) | 30/30 (100%) | 30/30 (100%) |

## Explainability Output

The explanation pipeline combines visual, regional, and textual evidence.

```text
Prediction
  -> Grad-CAM heatmap
  -> facial ROI activation scores
  -> structured ROI summary
  -> LLM-generated explanation
```

### Grad-CAM + ROI

<p align="left">
  <img width="80%" height="80%" alt="image" src="https://github.com/user-attachments/assets/f6f6f88c-2b01-4c57-9e3e-6264c195c55c" />
</p>

### LLM Explanation

REAL Explanation generated by LLM :
<p align="left">
  <img width="80%" height="80%" alt="LLM explanation example (real)" src="https://github.com/user-attachments/assets/712a3cb0-052b-457f-b64c-158aef68aa25" />
</p>

FAKE Explanation generated by LLM :
<p align="left">
  <img width="80%" height="80%" alt="LLM explanation example (fake)" src="https://github.com/user-attachments/assets/21cfb785-51be-41e5-b5af-ec5bf6d67155" />
</p>

## Research Supervision

- **Prof. Mohsen Ali Alawami:** research supervision, methodology refinement, project extension guidance, and paper writing/revisions

## Contributors

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


## Contact

- Prof. Mohsen Ali Alawami: mohsencomm@hufs.ac.kr
- Jiyeong Park: wldud8297@gmail.com
