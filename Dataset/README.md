# Dataset

This guide explains how to obtain the original datasets, use the provided
preprocessed data, and reproduce the preprocessing pipeline.

## Preprocessed Data

If you want to skip the preprocessing pipeline, download the preprocessed
videos below.

- [FaceForensics++ Fake processed videos](https://drive.google.com/file/d/1KEMw4JPPdlmhFk3QpB2IS0UAbrURudPG/view?usp=drive_link)
- [FaceForensics++ Real processed videos](https://drive.google.com/file/d/1GtuEiGhtL0nIC0Y6d0rBv-QtRfwSaAC6/view?usp=drive_link)
- [Celeb-DF Fake processed videos](https://drive.google.com/file/d/1tgfoaf2LV48ziNFqpXH7nm1USW1hfPUC/view?usp=drive_link)
- [Celeb-DF Real processed videos](https://drive.google.com/file/d/1Rlipbby2MXGWFaoay6th6lSDXMzu5Bg8/view?usp=drive_link)

## Original Dataset Sources

Our experiments use videos from:

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [DeeperForensics](https://github.com/EndlessSora/DeeperForensics-1.0/tree/master)

Follow the original dataset providers' access rules and licenses before
downloading or redistributing any data.

## Rebuilding Processed Videos

Follow these steps only if you want to regenerate the processed videos from the
original datasets.

1. Place the original datasets under the `Dataset/` directory.
2. Update the configuration in `model/preprocessing.ipynb` to match your local paths.
3. Run `model/preprocessing.ipynb`.
4. Verify that the processed videos have been generated successfully.
5. Reorganize the processed videos into the directory structure required for training.


## Expected Structure

The notebooks assume the following high-level layout:

```text
Dataset/
├── FaceForensics++_C23/  # raw FaceForensics++ videos
├── ff++/                 # processed FaceForensics++ videos
├── celeb-df/             # processed Celeb-DF videos
└── DeeperForensics/      # processed DeeperForensics videos
```


For training and evaluation, processed videos should follow:
```text
Dataset/ff++/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```