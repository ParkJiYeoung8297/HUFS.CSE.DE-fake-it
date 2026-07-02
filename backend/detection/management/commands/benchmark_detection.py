import os
import shutil
import uuid

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from detection.services.explainability import run_gradcam
from detection.services.inference import run_inference
from detection.services.llm import run_llm
from detection.services.preprocessing import run_preprocessing
from detection.utils.performance import PerformanceLogger


class Command(BaseCommand):
    help = "Run the detection pipeline for one video and write performance logs."

    def add_arguments(self, parser):
        parser.add_argument("video_path", help="Path to a local video file.")
        parser.add_argument(
            "--skip-llm",
            action="store_true",
            help="Skip LLM explanation generation while benchmarking.",
        )

    def handle(self, *args, **options):
        source_path = os.path.abspath(options["video_path"])
        if not os.path.isfile(source_path):
            raise CommandError(f"Video file does not exist: {source_path}")

        original_name = os.path.basename(source_path)
        file_size = os.path.getsize(source_path)
        logger = PerformanceLogger(original_name, file_size)

        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        filename = f"{uuid.uuid4().hex}_{original_name}"
        save_path = os.path.join(settings.MEDIA_ROOT, filename)
        output_path = os.path.join(settings.MEDIA_ROOT, f"preprocessed_{filename}")

        response_txt = ""
        table_data = []

        try:
            with logger.step("total"):
                with logger.step("file_save"):
                    shutil.copy2(source_path, save_path)
                    logger.set_video_path(save_path)

                with logger.step("preprocessing"):
                    preprocessed_path = run_preprocessing(
                        save_path,
                        output_path,
                        original_name,
                    )

                with logger.step("inference"):
                    result = run_inference(preprocessed_path)

                if result["Prediction"] != "Unknown":
                    with logger.step("grad_cam"):
                        roi_analyze_result, table_data = run_gradcam(
                            output_path,
                            original_name,
                            result,
                        )

                    if not options["skip_llm"]:
                        with logger.step("llm"):
                            response_txt = run_llm(roi_analyze_result)
        finally:
            logger.save()

        self.stdout.write(self.style.SUCCESS("Benchmark completed."))
        self.stdout.write(f"Prediction: {result['Prediction']}")
        self.stdout.write(f"Probability: {result['Probability']}")
        self.stdout.write(f"Table rows: {len(table_data)}")
        if response_txt:
            self.stdout.write(f"Explanation chars: {len(response_txt)}")
        self.stdout.write(f"Logs: {os.path.join(settings.BASE_DIR, 'logs')}")
