import csv
import os
import resource
import time
import uuid
from contextlib import contextmanager

import cv2
from django.conf import settings


class PerformanceLogger:
    def __init__(self, video_name, file_size=None):
        self.request_id = uuid.uuid4().hex[:8]
        self.video_name = video_name
        self.file_size = file_size
        self.video_path = ""
        self.video_metadata = {
            "duration_sec": "",
            "fps": "",
            "frame_count": "",
            "width": "",
            "height": "",
        }
        self.rows = []

    def set_video_path(self, video_path):
        self.video_path = video_path
        self.video_metadata = self._read_video_metadata(video_path)

    def _read_video_metadata(self, video_path):
        metadata = {
            "duration_sec": "",
            "fps": "",
            "frame_count": "",
            "width": "",
            "height": "",
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return metadata

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()

        metadata["fps"] = round(fps, 3) if fps else 0
        metadata["frame_count"] = int(frame_count) if frame_count else 0
        metadata["width"] = int(width) if width else 0
        metadata["height"] = int(height) if height else 0
        metadata["duration_sec"] = (
            round(frame_count / fps, 3) if fps and frame_count else 0
        )

        return metadata

    def _memory_mb(self):
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if usage > 10_000_000:
            return usage / 1024 / 1024
        return usage / 1024

    def _cpu_seconds(self):
        times = os.times()
        return times.user + times.system + times.children_user + times.children_system

    def _file_size_mb(self):
        if self.file_size is not None:
            return round(self.file_size / 1024 / 1024, 3)

        if self.video_path and os.path.exists(self.video_path):
            return round(os.path.getsize(self.video_path) / 1024 / 1024, 3)

        return ""

    @contextmanager
    def step(self, name):
        wall_start = time.perf_counter()
        cpu_start = self._cpu_seconds()
        memory_start = self._memory_mb()
        status = "success"

        try:
            yield
        except Exception:
            status = "failed"
            raise
        finally:
            wall_sec = time.perf_counter() - wall_start
            cpu_sec = self._cpu_seconds() - cpu_start
            memory_end = self._memory_mb()
            cpu_percent = (cpu_sec / wall_sec) * 100 if wall_sec else 0

            self.rows.append({
                "step": name,
                "wall_sec": round(wall_sec, 3),
                "cpu_sec": round(cpu_sec, 3),
                "cpu_percent": round(cpu_percent, 1),
                "memory_mb": round(memory_end, 1),
                "memory_delta_mb": round(memory_end - memory_start, 1),
                "status": status,
            })

    def _build_log_row(self, row):
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "request_id": self.request_id,
            "video_name": self.video_name,
            "file_size_mb": self._file_size_mb(),
            "duration_sec": self.video_metadata["duration_sec"],
            "fps": self.video_metadata["fps"],
            "frame_count": self.video_metadata["frame_count"],
            "width": self.video_metadata["width"],
            "height": self.video_metadata["height"],
            **row,
        }

    def save(self):
        if not self.rows:
            return

        log_dir = os.path.join(settings.BASE_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)

        csv_path = os.path.join(log_dir, "performance.csv")
        md_path = os.path.join(log_dir, "performance.md")

        fieldnames = [
            "timestamp",
            "request_id",
            "video_name",
            "file_size_mb",
            "duration_sec",
            "fps",
            "frame_count",
            "width",
            "height",
            "step",
            "wall_sec",
            "cpu_sec",
            "cpu_percent",
            "memory_mb",
            "memory_delta_mb",
            "status",
        ]

        log_rows = [self._build_log_row(row) for row in self.rows]
        write_header = not os.path.exists(csv_path)

        with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(log_rows)

        video_info = log_rows[0]
        with open(md_path, "a", encoding="utf-8") as md_file:
            md_file.write(f"\n\n### Request {self.request_id}\n\n")
            md_file.write("| video_name | file_size_mb | duration_sec | fps | frame_count | resolution |\n")
            md_file.write("|---|---:|---:|---:|---:|---|\n")
            md_file.write(
                f"| {video_info['video_name']} | {video_info['file_size_mb']} | "
                f"{video_info['duration_sec']} | {video_info['fps']} | "
                f"{video_info['frame_count']} | {video_info['width']}x{video_info['height']} |\n\n"
            )
            md_file.write("| step | wall_sec | cpu_sec | cpu_percent | memory_mb | memory_delta_mb | status |\n")
            md_file.write("|---|---:|---:|---:|---:|---:|---|\n")
            for row in log_rows:
                md_file.write(
                    f"| {row['step']} | {row['wall_sec']} | {row['cpu_sec']} | "
                    f"{row['cpu_percent']} | {row['memory_mb']} | "
                    f"{row['memory_delta_mb']} | {row['status']} |\n"
                )
