import time
from ..df_model.all_grad_cam import all_calculate_roi_scores


def run_explainability(output_path, uploaded_name, result, timings):
    start_time = time.perf_counter()

    response_txt, table_data, roi_timings = all_calculate_roi_scores(
        output_path,
        uploaded_name,
        result
    )

    timings['grad_cam'] = time.perf_counter() - start_time
    timings.update(roi_timings)

    return response_txt, table_data