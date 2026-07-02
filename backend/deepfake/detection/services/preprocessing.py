from ..detector.preprocessing import process_single_video


def run_preprocessing(save_path, output_path, uploaded_name):
    return process_single_video(
        save_path,
        output_path,
        uploaded_name
    )
