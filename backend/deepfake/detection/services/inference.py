import time
from ..df_model.detection_model import run_detection_model


def run_inference(preprocessed_path, timings):
    start_time = time.perf_counter()

    result = run_detection_model(preprocessed_path)

    timings['inference'] = time.perf_counter() - start_time

    return result

"""

        # 2. ✅ 탐지 모델
        try:
            result = measure_elapsed('inference', timings, run_detection_model, preprocessed_path)
        except Exception as e:
            return JsonResponse({"error": "Detection failed", "detail": str(e)}, status=500)
"""