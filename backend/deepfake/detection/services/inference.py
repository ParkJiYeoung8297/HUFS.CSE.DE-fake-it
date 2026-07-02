import time
from ..detector.detection_model import run_detection_model


def run_inference(preprocessed_path, timings):
    start_time = time.perf_counter()

    result = run_detection_model(preprocessed_path)

    timings['inference'] = time.perf_counter() - start_time

    return result
