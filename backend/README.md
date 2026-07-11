# Backend

This directory contains the Django backend for **DE-Fake it**. It receives uploaded videos, runs preprocessing and model inference, generates Grad-CAM and ROI-based visual explanations, and returns the analysis result to the React frontend.

## Main Pipeline

```text
video upload
-> face preprocessing
-> EfficientNet-b0 + LSTM inference
-> Grad-CAM generation
-> ROI activation scoring
-> LLM explanation
-> JSON response
```

## Project Structure

```text
backend/
├── config/                  # Django project configuration
├── detection/
│   ├── detector/            # Model, preprocessing, inference, Grad-CAM, ROI logic
│   ├── services/            # Upload analysis pipeline and service wrappers
│   ├── urls.py
│   └── views.py
├── media/                   # Runtime upload/output directory
└── manage.py
```

## Model Checkpoint

The trained model checkpoint is not tracked in Git. Before running inference, [download the trained model](https://drive.google.com/file/d/12VNleCHv1PB7SUnh0H0QBmObUClwUQy3/view) and place it at: 

```text
backend/detection/detector/checkpoint_v35.pt
```

The backend currently uses the `checkpoint_v35` weights with an **EfficientNet-b0 + LSTM** architecture.

## Local Settings

`backend/config/settings.py` is treated as a local settings file and is not tracked in Git. Start from the provided example:

```bash
cp backend/config/settings.example.py backend/config/settings.py
```

Make sure the local settings include:

- `detection`, `corsheaders`, and `rest_framework` in `INSTALLED_APPS`
- `MEDIA_URL = "/media/"`
- `MEDIA_ROOT = os.path.join(BASE_DIR, "media")`
- CORS access for the frontend development server, usually `http://localhost:3000`


## Ollama / LLM Requirement

The backend uses a local Ollama server to generate textual explanations.

Before running the full analysis pipeline, start Ollama and make sure the configured model is available:

```bash
ollama serve
ollama pull llama3
```
By default, the backend sends LLM requests to:
http://localhost:11434/api/generate
If Ollama is not running, the backend does not fail the analysis. It returns the prediction, Grad-CAM videos, ROI table, and a fallback explanation message.


## Run

Install Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Start the Django development server:

```bash
cd backend
python manage.py runserver
```

The backend runs at `http://localhost:8000`.

## Output Files

Analysis outputs are stored under:

```text
backend/media/
├── <uploaded_video>.mp4
└── preprocessed_<video_name>/
    ├── <preprocessed_video>.mp4
    ├── roi_metadata.json
    ├── converted_grad_cam_on_original.mp4
    └── converted_output_box_on_original.mp4
```

Generated outputs include:

- Uploaded source video
- Preprocessed face-cropped video used for inference
- ROI metadata for explainability
- Grad-CAM visualization video
- ROI bounding-box visualization video

Temporary intermediate files may be created during processing and removed automatically.

## Logging

The backend logs the main processing steps to the terminal:

```text
[INFO] 16:12:37 detection.services.pipeline - Analysis started: sample.mp4
```

The default level is `INFO`. To enable debug logs:

```bash
DEFAKE_LOG_LEVEL=DEBUG python manage.py runserver
```

Useful environment variables:

- `DEFAKE_LOG_LEVEL`: backend log level, default `INFO`
- `DEFAKE_LLM_URL`: Ollama generate API URL, default `http://localhost:11434/api/generate`
- `DEFAKE_LLM_MODEL`: Ollama model name, default `llama3`
- `DEFAKE_LLM_TIMEOUT_SEC`: LLM request timeout, default `60`
- `DEFAKE_FRAME_SAMPLE_STRIDE`: frame sampling stride, default `5`
- `DEFAKE_INFERENCE_BATCH_SIZE`: inference batch size, default `16`