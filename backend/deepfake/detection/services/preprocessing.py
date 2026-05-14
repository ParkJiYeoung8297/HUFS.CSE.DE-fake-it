import time
from ..df_model.preprocessing import process_single_video


def run_preprocessing(save_path, output_path, uploaded_name, timings):
    start_time = time.perf_counter()

    preprocessed_path = process_single_video(
        save_path,
        output_path,
        uploaded_name
    )

    timings['preprocessing'] = time.perf_counter() - start_time

    return preprocessed_path

"""
        # 1. ✅ 전처리 수행
        try:
            preprocessed_path = measure_elapsed(
                'preprocessing',
                timings,
                process_single_video,
                save_path,
                output_path,
                uploaded_file.name
            )  # 전처리 함수 호출
        except Exception as e:
            return JsonResponse({"error": "Preprocessing failed", "detail": str(e)}, status=500)
"""