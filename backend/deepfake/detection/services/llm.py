import json
import time

import requests


def build_llm_prompt(summary):
    print(f"summary: {summary}")
    prompt = (
        f"모델 예측 결과:이 모델은 REAL/FAKE 판단에서 {summary['binary_pred']}로 예측했으며, 확률은 {summary['cam_score']}입니다. 딥페이크 기법 분류 결과: {summary['method_pred']}"
        f"참고 : original은 위조 흔적이 없는 원본 영상, others는 FaceForensics++의 5가지 기법 외의 위조 방식입니다. "
        f"아래는 영상 {summary['video_name']}에 대한 Grad-CAM 기반 ROI 활성도 통계 데이터입니다."
        f"통계 데이터 설명 (모든 값은 영상 전체 프레임을 통합한 통계 기반입니다):\n"
        f" - 가짜에 영향을 주는 부분을 grad-cam으로 시각화하여, 영상 내에 모든 프레임을 통합한 값입니다.\n"
        f" 1. [Facial Regions 분석 대상]: 분석에 사용된 얼굴 부위는 다음과 같습니다: {', '.join(summary['facial_region'])}. ('None'은 REAL로 판단되어 Grad-CAM이 그려지지 않은 경우입니다.)\n\n"
        f" 2. [1순위 활성화 횟수 (First Detection Count)]: 각 부위가 Grad-CAM에서 1순위로 가장 활성화된 프레임 수입니다.\n"
        f" {summary['first_detection_count']}\n\n"
        f" 3. [2순위 활성화 횟수 (Second Detection Count)]: 각 부위가 2순위로 활성화된 프레임 수입니다.\n"
        f" {summary['second_detection_count']}\n\n"
        f" 4. [1순위 활성화 비율 (First Detection Rate)]: 전체 프레임 중 각 부위가 1순위로 선택된 비율입니다 (% 단위).\n"
        f" {summary['first_detection_rate']}\n\n"
        f" 5. [2순위 활성화 비율 (Second Detection Rate)]: 전체 프레임 중 각 부위가 2순위로 선택된 비율입니다 (% 단위).\n"
        f" {summary['second_detection_rate']}\n\n"
    )

    if summary['binary_pred'] == 'FAKE':
        prompt += (
            f" 6. [Raw Detection Probability]: 각 부위에 대해 Grad-CAM의 총 활성 기여도를 raw 점수로 나타낸 값입니다. 각 프레임별에서 (Fake일 확률 × 부위별 ROI 평균 활성도)를 계산하여, 부위별로 모두 더하여 구합니다. \n"
            f" {summary['raw_detection_probability']}\n\n"
            f" 7. [Normalized Detection Contribution (%)]: 위의 raw 값들을 전체 합으로 나눠 정규화한 값입니다. 각 부위가 전체 판단에 얼마나 기여했는지 상대적으로 보여줍니다. \n"
            f" {summary['detection_probability']}\n\n"
        )
        prompt += (
            "모델 결과가 FAKE로 판단된 경우, 모델 예측 결과와 위의 정보를 참고하여 어떤 얼굴 부위가 어떻게 작용했는지를 중심으로 분석하고, "
            "중요도가 높은 분석 내용을 선별하여 독자가 납득할 수 있도록 구체적으로 서술해 주세요.\n"
            "모델 결과가 FAKE로 판단된 경우에는 Grad-CAM 분석이 매우 중요하며, 특정 얼굴 부위들이 어떻게 활성화되었는지를 분석해 주세요.\n"
            f"영상에 사용된 딥페이크 기법은 {summary['method_pred']}로 예측되었다고 꼭 언급해주세요"
            "응답은 분석 내용만 포함하고, 다음 형식처럼 번호를 붙여주세요:\n"
            "1. ...\n2. ...\n3. ...\n"
        )
    elif summary['binary_pred'] == 'REAL':
        prompt += (
            "모델 결과가 REAL로 판단된 경우, None의 detection_count 값과 위의 정보를 참고하여 분석하고, 중요도가 높은 분석 내용을 선별하여 독자가 납득할 수 있도록 구체적으로 서술해 주세요.\n"
            "REAL로 판단된 경우, 얼굴 부위별 감지 횟수가 적고 'None'의 횟수가 대부분이기 때문에 REAL로 판단되었습니다. "
            "응답은 분석 내용만 포함하고, 다음 형식처럼 번호를 붙여주세요:\n"
            "1. ...\n2. ...\n3. ...\n"
        )

    return prompt


def query_llm_model(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3",
        "prompt": prompt
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
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
    return ""


def run_llm_explanation(roi_analyze_result):
    prompt = build_llm_prompt(roi_analyze_result)
    print("프롬프트 내용 : ", prompt)
    return query_llm_model(prompt)


def run_llm(roi_analyze_result, timings):
    start_time = time.perf_counter()
    response_txt = run_llm_explanation(roi_analyze_result)
    timings['llm'] = time.perf_counter() - start_time
    return response_txt
