# Seeing Through Deepfakes: Explainable and Usable Multi-Task Framework with Deep Learning and LLMs

---

## 1. Introduction
This project aims to build an image classification system for video-based Deepfake detection and provide both visual and textual explanations of the model's predictions.

We achieve this by training a CNN-LSTM architecture on the FaceForensics++ (FF++) dataset. Frame-level features are extracted using a pretrained CNN backbone (such as EfficientNet or ResNeXt), and a temporal LSTM layer processes the sequence to classify each video.

In addition to binary classification (Real vs. Fake), the model also performs multi-class classification to identify the specific Deepfake generation technique, such as Face2Face, FaceSwap, Deepfakes, NeuralTextures, or FaceShifter.

To interpret the results, we apply Grad-CAM for visual explanation, compute region-based activation scores using facial alignment, and generate natural language explanations using an LLM (via Ollama).

## 2. Directory Structure
```
HUFS.CSE.DE-fake-it/
├── backend/         # Django backend server
├── frontend/        # React frontend application
├── model/           # Model training and explanation module
├── requirements.txt # Dependency file
└── README.md        # Project overview documentation

```
1. backend/deepfake
  - This is the Django-based backend server. It includes APIs that process uploaded videos and return prediction results using the trained model.

2. frontend/dpfake
  - This is the React-based user interface. It handles video uploads and displays prediction results to the user.

3. model
  - Contains modules for training the deepfake detection model, performing Grad-CAM interpretation, ROI-based visualization, and generating LLM-based textual explanations.



## 3. System Architecture

<p align="center">
  <img width="100%" height="100%" alt="image" src="https://github.com/user-attachments/assets/66172714-2e21-4484-a696-c058a9b3ad72" />
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
  <img width="63%" height="63%" alt="image" src="https://github.com/user-attachments/assets/edd9c9b5-e3a1-45b5-925f-ffe29d8f76ca" />
</p>

- LLM Explanation

<p align="left">
  <img width="63%" height="63%" alt="image" src="https://github.com/user-attachments/assets/6a77534f-236b-4864-8b93-6742ddb7fd39" />
</p>
  
  

## 6. Explainability Pipeline

- Grad-CAM visualization (using the last convolutional layer)
- ROI-based scoring (region separation based on Face Alignment)
- LLM-based explanation using Ollama to generate textual justifications for manipulated regions



## 7. Preprocessing Tools

- `validate_video.py`: Removes corrupted or too-short videos
- `cut_frame.py`: Extracts the first 150 frames and saves them as MJPEG
- `make_meta_data.py`: Automatically generates labels and metadata
- `Grad_CAM_and_ROI.ipynb`: Generates CAM visualizations and ROI-based visual/textual explanations



## 8. Contributors
<!--유저이름만 본인이름으로 변경하면 됨.-->
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/ParkJiYeoung8297">
        <img src="https://github.com/ParkJiYeoung8297.png" width="100px;" alt="username1"/>
        <br />
        <sub><b>Park Ji Yeong</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/username2">
        <img src="https://github.com/username2.png" width="100px;" alt="username2"/>
        <br />
        <sub><b>Sercan Yeşilköy</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/username3">
        <img src="https://github.com/username3.png" width="100px;" alt="username3"/>
        <br />
        <sub><b>Lim Do Yeon</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/username3">
        <img src="https://github.com/username3.png" width="100px;" alt="username3"/>
        <br />
        <sub><b>Park Hui Ryeong</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/kellylee23">
        <img src="https://github.com/kellylee23.png" width="100px;" alt="leeeunseo"/>
        <br />
        <sub><b>Lee Eun Seo</b></sub>
      </a>
    </td>
  </tr>
</table>



## 9. Contact
Contact: wldud8297@gmail.com
