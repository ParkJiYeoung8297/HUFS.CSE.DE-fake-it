import time

from ..df_model.all_grad_cam import (
    all_calculate_roi_scores,
    run_gradcam_roi_analysis,
    run_llm_explanation,
)


def run_gradcam(output_path, uploaded_name, result, timings):
    start_time = time.perf_counter()

    roi_analyze_result, table_data = run_gradcam_roi_analysis(
        output_path,
        uploaded_name,
        result
    )

    timings['grad_cam'] = time.perf_counter() - start_time

    return roi_analyze_result, table_data


def run_llm(roi_analyze_result, timings):
    start_time = time.perf_counter()

    response_txt = run_llm_explanation(roi_analyze_result)

    timings['llm'] = time.perf_counter() - start_time

    return response_txt


def run_explainability(output_path, uploaded_name, result, timings):
    response_txt, table_data, roi_timings = all_calculate_roi_scores(
        output_path,
        uploaded_name,
        result
    )

    timings.update(roi_timings)

    return response_txt, table_data
