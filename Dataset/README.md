# Dataset Directory

Place local datasets here when running the model notebooks.

Large dataset files are intentionally ignored by Git. Keep only this guide in
the repository.

Expected structure:

```text
Dataset/
├── FaceForensics++_C23/
├── ff++/
│   ├── train/
│   │   ├── fake/
│   │   │   ├── Deepfakes/
│   │   │   ├── Face2Face/
│   │   │   ├── FaceShifter/
│   │   │   ├── FaceSwap/
│   │   │   └── NeuralTextures/
│   │   └── real/
│   │       └── original/
│   └── test/
│       ├── fake/
│       │   ├── Deepfakes/
│       │   ├── Face2Face/
│       │   ├── FaceShifter/
│       │   ├── FaceSwap/
│       │   └── NeuralTextures/
│       └── real/
│           └── original/
├── celeb-df/
├── DeeperForensics/
└── ff++(grad-cam_v2)/
```

The model notebooks use:

```python
DATA_ROOT = PROJECT_ROOT / "Dataset"
```

