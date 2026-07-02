from ..detector.all_grad_cam import (
    run_gradcam_roi_analysis,
)
from .llm import run_llm


def run_gradcam(output_path, uploaded_name, result):
    return run_gradcam_roi_analysis(
        output_path,
        uploaded_name,
        result
    )


def run_explainability(output_path, uploaded_name, result):
    roi_analyze_result, table_data = run_gradcam(
        output_path,
        uploaded_name,
        result,
    )
    response_txt = run_llm(roi_analyze_result)

    return response_txt, table_data
