import time
from ..detector.preprocessing import process_single_video


def run_preprocessing(save_path, output_path, uploaded_name, timings):
    start_time = time.perf_counter()

    preprocessed_path = process_single_video(
        save_path,
        output_path,
        uploaded_name
    )

    timings['preprocessing'] = time.perf_counter() - start_time

    return preprocessed_path
