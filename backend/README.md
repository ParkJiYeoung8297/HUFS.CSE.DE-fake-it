# Backend

This directory contains the Django backend for **DE-Fake it**. It receives uploaded videos, runs preprocessing and model inference, generates Grad-CAM and ROI-based visual explanations, and returns the analysis result to the React frontend.

## Main Pipeline

```text
video upload
-> face preprocessing
-> EfficientNet-b0 + LSTM inference
-> Grad-CAM generation
-> ROI activation scoring
-> optional LLM explanation
-> JSON response
```

## Project Structure

```text
backend/
├── config/                  # Django project configuration
├── detection/
│   ├── detector/            # Model, preprocessing, inference, Grad-CAM, ROI logic
│   ├── services/            # Upload analysis pipeline and service wrappers
│   ├── utils/               # Result and benchmark helpers
│   ├── management/commands/ # Optional benchmark command
│   ├── urls.py
│   └── views.py
├── media/                   # Runtime upload/output directory
└── manage.py
```

## Model Checkpoint

The trained model checkpoint is not tracked in Git. Before running inference, download the trained model and place it at:

```text
backend/detection/detector/checkpoint_v35.pt
```

The backend currently uses `checkpoint_v35` with the `EfficientNet-b0` model configuration.

## Local Settings

`backend/config/settings.py` is treated as a local settings file and is not tracked in Git. Make sure the local settings include:

- `detection`, `corsheaders`, and `rest_framework` in `INSTALLED_APPS`
- `MEDIA_URL = "/media/"`
- `MEDIA_ROOT = os.path.join(BASE_DIR, "media")`
- CORS access for the frontend development server, usually `http://localhost:3000`

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

## API Endpoints

### `POST /api/show-video/`

Uploads a video and returns a converted preview video URL.

Expected form field:

```text
video
```

Main response fields:

- `message`
- `saved_video_name`
- `video_url`

### `POST /api/upload/`

Uploads a video and runs the full deepfake analysis pipeline.

Expected form field:

```text
video
```

Main response fields:

- `prediction`
- `probability`
- `grad_cam_video_url`
- `output_box_video_url`
- `explanations`
- `table_data`
- `result_file`
- `result_log_file`

If the local LLM server is not available, the backend returns a fallback explanation message instead of failing the whole analysis.

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

## Optional Benchmark

Runtime performance measurement is separated from the web API. To benchmark a local video:

```bash
cd backend
python manage.py benchmark_detection /path/to/video.mp4
```

To skip LLM explanation during benchmarking:

```bash
python manage.py benchmark_detection /path/to/video.mp4 --skip-llm
```

Benchmark outputs are written to:

```text
backend/logs/performance.csv
backend/logs/performance.md
```
