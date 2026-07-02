# Frontend

This directory contains the React interface for **DE-Fake it**. The frontend lets users upload a video, preview the converted video, request deepfake analysis from the Django backend, and view the final prediction with Grad-CAM videos, ROI statistics, and a text explanation.

## Main Features

- Video upload and preview
- Deepfake analysis request through the backend API
- Result page for REAL/FAKE prediction and probability
- Grad-CAM and ROI-box video playback
- ROI table and explanation text display

## Project Structure

```text
frontend/
├── public/                  # Static assets
├── src/
│   ├── components/
│   │   ├── DeepfakeDetector.js
│   │   └── DeepfakeResult.js
│   ├── App.js
│   └── index.js
├── package.json
└── README.md
```

## Run

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm start
```

The frontend runs at `http://localhost:3000`.

## Backend Connection

The frontend expects the backend server to run at:

```text
http://localhost:8000
```

The main API calls are:

- `POST /api/show-video/`: uploads a video and returns a converted preview video URL
- `POST /api/upload/`: uploads a video and returns prediction, Grad-CAM outputs, ROI table data, and explanation text

## Build

To create a production build:

```bash
npm run build
```

The build output is created under `frontend/build/`.
