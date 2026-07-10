from pathlib import Path

import pandas as pd


# =========================
# Configuration
# =========================
# Edit these values before running the script.

METHOD = "NeuralTextures"
JSON_PATH = Path(f"/path/to/Dataset/ff++(grad-cam)/{METHOD}/{METHOD}_video_roi_result.json")


def build_llm_prompt(summary):
    prompt = (
        f"모델 예측 결과: REAL/FAKE={summary['binary_pred']}, 확률={summary['final_probability']}, "
        f"딥페이크 기법 분류={summary['method_pred']}.\n"
        "참고: original은 위조 흔적이 없는 원본 영상, others는 FaceForensics++의 5가지 기법 외 위조 방식입니다.\n"
        f"아래는 영상 {summary['video_name']}에 대한 Grad-CAM 기반 ROI 활성도 통계입니다. "
        "모든 값은 영상 전체 프레임을 통합한 통계이며, Grad-CAM은 가짜 판단에 영향을 주는 영역을 시각화한 값입니다.\n\n"
        f"분석 대상 얼굴 부위: {summary['facial_region']}. "
        "None은 REAL로 판단되어 Grad-CAM이 그려지지 않은 경우입니다.\n\n"
        "1. First Detection Count: 각 부위가 Grad-CAM에서 1순위로 가장 활성화된 프레임 수입니다.\n"
        f"{summary['first_detection_count']}\n\n"
        "2. Second Detection Count: 각 부위가 2순위로 활성화된 프레임 수입니다.\n"
        f"{summary['second_detection_count']}\n\n"
        "3. First Detection Rate: 전체 프레임 중 각 부위가 1순위로 선택된 비율입니다(%).\n"
        f"{summary['first_detection_rate']}\n\n"
        "4. Second Detection Rate: 전체 프레임 중 각 부위가 2순위로 선택된 비율입니다(%).\n"
        f"{summary['second_detection_rate']}\n\n"
    )

    if summary["binary_pred"] == "FAKE":
        prompt += (
            "5. Raw Detection Probability: 각 프레임별 (Fake일 확률 x 부위별 ROI 평균 활성도)를 계산해 부위별로 합산한 raw 점수입니다.\n"
            f"{summary['raw_detection_probability']}\n\n"
            "6. Normalized Detection Contribution(%): raw 값을 전체 합으로 나눠 정규화한 값이며, "
            "각 부위가 전체 판단에 얼마나 기여했는지 보여줍니다.\n"
            f"{summary['detection_probability']}\n\n"
            "모델 결과가 FAKE인 경우, 위 통계를 바탕으로 어떤 얼굴 부위가 어떻게 작용했는지 중심으로 분석해 주세요. "
            "중요도가 높은 내용을 선별해 독자가 납득할 수 있도록 구체적으로 서술하고, "
            f"영상에 사용된 딥페이크 기법이 {summary['method_pred']}로 예측되었다고 꼭 언급해 주세요.\n"
        )

    elif summary["binary_pred"] == "REAL":
        prompt += (
            "모델 결과가 REAL인 경우, None의 detection_count 값과 얼굴 부위별 낮은 활성도를 중심으로 분석해 주세요. "
            "중요도가 높은 내용을 선별해 독자가 납득할 수 있도록 구체적으로 서술해 주세요.\n"
        )

    prompt += (
        "응답은 분석 내용만 포함하고, 다음 형식처럼 번호를 붙여주세요:\n"
        "1. ...\n2. ...\n3. ...\n"
    )

    return prompt


def build_llm_prompts_from_json(json_path):
    df = pd.read_json(json_path)

    prompts = []
    for _, row in df.iterrows():
        prompts.append(build_llm_prompt(row))

    return prompts


def main():
    prompts = build_llm_prompts_from_json(JSON_PATH)
    for prompt in prompts:
        print(prompt)
        print()


if __name__ == "__main__":
    main()
