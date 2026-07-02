from ..detector.detection_model import run_detection_model


def run_inference(preprocessed_path):
    return run_detection_model(preprocessed_path)
