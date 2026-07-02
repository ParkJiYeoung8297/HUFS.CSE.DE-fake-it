import json
import os
import requests


LLM_URL = os.environ.get("DEFAKE_LLM_URL", "http://localhost:11434/api/generate")
LLM_MODEL = os.environ.get("DEFAKE_LLM_MODEL", "llama3")
LLM_KEEP_ALIVE = os.environ.get("DEFAKE_LLM_KEEP_ALIVE", "30m")
LLM_TIMEOUT_SEC = int(os.environ.get("DEFAKE_LLM_TIMEOUT_SEC", "60"))
LLM_UNAVAILABLE_MESSAGE = (
    "Textual explanation is unavailable because the local LLM service is not running. "
    "The prediction, probability, Grad-CAM videos, and ROI table were still generated successfully."
)


def build_llm_prompt(summary):
    print(f"summary: {summary}")

    prompt = (
        f"모델 예측 결과: REAL/FAKE={summary['binary_pred']}, 확률={summary['cam_score']}, "
        f"딥페이크 기법 분류={summary['method_pred']}.\n"
        "참고: original은 위조 흔적이 없는 원본 영상, others는 FaceForensics++의 5가지 기법 외 위조 방식입니다.\n"
        f"아래는 영상 {summary['video_name']}에 대한 Grad-CAM 기반 ROI 활성도 통계입니다. "
        "모든 값은 영상 전체 프레임을 통합한 통계이며, Grad-CAM은 가짜 판단에 영향을 주는 영역을 시각화한 값입니다.\n\n"
        f"분석 대상 얼굴 부위: {', '.join(summary['facial_region'])}. "
        "None은 REAL로 판단되어 Grad-CAM이 그려지지 않은 경우입니다.\n\n"
        f"1. First Detection Count: 각 부위가 Grad-CAM에서 1순위로 가장 활성화된 프레임 수입니다.\n"
        f"{summary['first_detection_count']}\n\n"
        f"2. Second Detection Count: 각 부위가 2순위로 활성화된 프레임 수입니다.\n"
        f"{summary['second_detection_count']}\n\n"
        f"3. First Detection Rate: 전체 프레임 중 각 부위가 1순위로 선택된 비율입니다(%).\n"
        f"{summary['first_detection_rate']}\n\n"
        f"4. Second Detection Rate: 전체 프레임 중 각 부위가 2순위로 선택된 비율입니다(%).\n"
        f"{summary['second_detection_rate']}\n\n"
    )

    if summary['binary_pred'] == 'FAKE':
        prompt += (
            f"5. Raw Detection Probability: 각 프레임별 (Fake일 확률 x 부위별 ROI 평균 활성도)를 계산해 부위별로 합산한 raw 점수입니다.\n"
            f"{summary['raw_detection_probability']}\n\n"
            f"6. Normalized Detection Contribution(%): raw 값을 전체 합으로 나눠 정규화한 값이며, "
            f"각 부위가 전체 판단에 얼마나 기여했는지 보여줍니다.\n"
            f"{summary['detection_probability']}\n\n"
            "모델 결과가 FAKE인 경우, 위 통계를 바탕으로 어떤 얼굴 부위가 어떻게 작용했는지 중심으로 분석해 주세요. "
            "중요도가 높은 내용을 선별해 독자가 납득할 수 있도록 구체적으로 서술하고, "
            f"영상에 사용된 딥페이크 기법이 {summary['method_pred']}로 예측되었다고 꼭 언급해 주세요.\n"
        )

    elif summary['binary_pred'] == 'REAL':
        prompt += (
            "모델 결과가 REAL인 경우, None의 detection_count 값과 얼굴 부위별 낮은 활성도를 중심으로 분석해 주세요. "
            "중요도가 높은 내용을 선별해 독자가 납득할 수 있도록 구체적으로 서술해 주세요.\n"
        )

    prompt += (
        "응답은 분석 내용만 포함하고, 다음 형식처럼 번호를 붙여주세요:\n"
        "1. ...\n2. ...\n3. ...\n"
    )

    return prompt



def query_llm_model(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "keep_alive": LLM_KEEP_ALIVE,
    }

    try:
        response = requests.post(
            LLM_URL,
            headers=headers,
            data=json.dumps(data),
            timeout=LLM_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        print(f"LLM request skipped: {exc}")
        return LLM_UNAVAILABLE_MESSAGE
    print("응답 코드 : ", response.status_code)
    if response.status_code == 200:
        print("응답 내용:")
        combined_response = ''

        for line in response.text.splitlines():
            try:
                json_line = json.loads(line)
                combined_response += json_line['response']
            except json.JSONDecodeError:
                continue

        print(combined_response)
        return combined_response

    print("요청 실패:", response.status_code)
    return LLM_UNAVAILABLE_MESSAGE


def warm_up_llm():
    data = {
        "model": LLM_MODEL,
        "prompt": "ready",
        "stream": False,
        "keep_alive": LLM_KEEP_ALIVE,
        "options": {
            "num_predict": 1,
        },
    }

    try:
        requests.post(
            LLM_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(data),
            timeout=LLM_TIMEOUT_SEC,
        )
        print(f"LLM warmed up: {LLM_MODEL}")
    except requests.RequestException as exc:
        print(f"LLM warm-up skipped: {exc}")


def run_llm_explanation(roi_analyze_result):
    prompt = build_llm_prompt(roi_analyze_result)
    print("프롬프트 내용 : ", prompt)
    return query_llm_model(prompt)


def run_llm(roi_analyze_result):
    return run_llm_explanation(roi_analyze_result)
