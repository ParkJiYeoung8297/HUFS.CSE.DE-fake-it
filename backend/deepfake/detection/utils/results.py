import json
import os

from django.conf import settings
from django.utils import timezone


def save_detection_result(result_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    record = {
        "timestamp": timezone.now().isoformat(),
        **result_data,
    }

    per_video_path = os.path.join(output_dir, "analysis_result.json")
    with open(per_video_path, "w", encoding="utf-8") as result_file:
        json.dump(record, result_file, ensure_ascii=False, indent=2, default=str)

    aggregate_path = os.path.join(settings.BASE_DIR, "data.jsonl")
    with open(aggregate_path, "a", encoding="utf-8") as aggregate_file:
        aggregate_file.write(json.dumps(record, ensure_ascii=False, default=str))
        aggregate_file.write("\n")

    return per_video_path, aggregate_path
