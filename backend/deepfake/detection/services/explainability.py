import time

from ..df_model.all_grad_cam import (
    run_gradcam_roi_analysis,
)
from .llm import run_llm


def run_gradcam(output_path, uploaded_name, result, timings):
    start_time = time.perf_counter()

    roi_analyze_result, table_data = run_gradcam_roi_analysis(
        output_path,
        uploaded_name,
        result
    )

    timings['grad_cam'] = time.perf_counter() - start_time

    return roi_analyze_result, table_data


def run_explainability(output_path, uploaded_name, result, timings):
    roi_analyze_result, table_data = run_gradcam(
        output_path,
        uploaded_name,
        result,
        timings
    )
    response_txt = run_llm(roi_analyze_result, timings)

    return response_txt, table_data
