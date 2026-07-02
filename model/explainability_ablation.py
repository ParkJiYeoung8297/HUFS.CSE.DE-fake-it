import argparse
import csv
import os


ROWS = [
    {
        "variant": "CNN + LSTM",
        "grad_cam": "No",
        "roi": "No",
        "llm": "No",
        "explanation_output": "No",
        "purpose": "Detection-only baseline without explanation.",
    },
    {
        "variant": "CNN + LSTM + Grad-CAM",
        "grad_cam": "Yes",
        "roi": "No",
        "llm": "No",
        "explanation_output": "Visual only",
        "purpose": "Visual localization of discriminative regions.",
    },
    {
        "variant": "CNN + LSTM + Grad-CAM + ROI",
        "grad_cam": "Yes",
        "roi": "Yes",
        "llm": "No",
        "explanation_output": "Visual + ROI",
        "purpose": "Region-level quantitative explanation.",
    },
    {
        "variant": "Full model: CNN + LSTM + Grad-CAM + ROI + LLM",
        "grad_cam": "Yes",
        "roi": "Yes",
        "llm": "Yes",
        "explanation_output": "Visual + ROI + Text",
        "purpose": "Human-readable textual explanation based on ROI statistics.",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Write explainability ablation table.")
    parser.add_argument(
        "--output",
        default="model/ablation_results/explainability_ablation.csv",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ROWS[0].keys())
        writer.writeheader()
        writer.writerows(ROWS)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
