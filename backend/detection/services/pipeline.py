import os

from django.conf import settings

from .explainability import run_gradcam
from .inference import run_inference
from .llm import run_llm
from .preprocessing import run_preprocessing
from .uploads import save_uploaded_file
from ..utils.results import save_detection_result


def analyze_uploaded_video(uploaded_file):
    filename, save_path = save_uploaded_file(uploaded_file)
    output_path = os.path.join(settings.MEDIA_ROOT, f"preprocessed_{filename}")

    preprocessed_path = run_preprocessing(save_path, output_path, uploaded_file.name)
    result = run_inference(preprocessed_path)

    response_txt = ""
    table_data = []

    if result["Prediction"] != "Unknown":
        roi_analyze_result, table_data = run_gradcam(
            output_path,
            uploaded_file.name,
            result,
        )
        response_txt = run_llm(roi_analyze_result)

    response_data = {
        "message": "Success",
        "uploaded_video_name": uploaded_file.name,
        "saved_video_name": filename,
        "prediction": result["Prediction"],
        "probability": result["Probability"],
        "grad_cam_video_url": f"/media/preprocessed_{filename}/converted_grad_cam_on_original.mp4",
        "output_box_video_url": f"/media/preprocessed_{filename}/converted_output_box_on_original.mp4",
        "explanations": response_txt,
        "table_data": table_data,
    }
    result_json_path, result_jsonl_path = save_detection_result(
        response_data,
        output_path,
    )
    response_data["result_file"] = result_json_path
    response_data["result_log_file"] = result_jsonl_path

    return response_data
