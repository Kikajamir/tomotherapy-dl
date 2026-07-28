"""
Local baseline pipeline: raw dataset -> `processed_data.pkl` -> validated
PyTorch Dataset/DataLoader smoke test.

This does NOT train. Training is meant to run on Kaggle (GPU) against the
PKL this script produces -- see `train.py` and the README's "Kaggle"
section. Nothing here touches RTPLAN, the preprocessing algorithm, the
model, the loss, or the optimizer; it only wires together the existing,
unmodified pieces:

    preprocessing.pipeline.run_preprocessing   (discovery + preprocessing)
    preprocessing.validate_dataset             (pkl sanity checks)
    data.dataset.SlidingWindowDataset          (unchanged)

Usage:
    python run_pipeline.py
    python run_pipeline.py --dataset-root "/path/to/DA exit data_new" --output processed_data.pkl
"""

import argparse

import torch
from torch.utils.data import DataLoader

from configs.config import SAVE_FILE, WINDOW_SIZE, STRIDE, BATCH_SIZE
from preprocessing.pipeline import run_preprocessing
from preprocessing.validate_dataset import validate_processed_data, summarize_validation
from data.dataset import SlidingWindowDataset


def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        default=None,
        help="Raw dataset root (defaults to ./'DA exit data_new' next to this script).",
    )
    parser.add_argument(
        "--output",
        default=SAVE_FILE,
        help=f"Output PKL path (default: {SAVE_FILE}).",
    )
    args = parser.parse_args()

    print_header("STEP 1/4 -- Dataset discovery + preprocessing")
    processed_data, skipped, failed = run_preprocessing(
        dataset_root=args.dataset_root, save_path=args.output
    )

    print_header("STEP 2/4 -- Validating processed_data.pkl")
    report = validate_processed_data(processed_data)
    summary = summarize_validation(report)

    print(f"Passed: {summary['passed']}/{summary['total']}")
    if summary["failed"]:
        print("FAILED SAMPLES:")
        for failure in summary["failures"]:
            print(f"  {failure['patient']}: {failure['issues']}")
    else:
        print("No NaN/Inf/shape-mismatch issues found.")

    for entry in report[:3]:
        p, d = entry["planned_stats"], entry["detector_stats"]
        print(
            f"  {entry['patient']} shape={entry['shape']} | "
            f"planned[min={p['min']:.4f} max={p['max']:.4f} mean={p['mean']:.4f} std={p['std']:.4f}] | "
            f"detector[min={d['min']:.4f} max={d['max']:.4f} mean={d['mean']:.4f} std={d['std']:.4f}]"
        )
    if len(report) > 3:
        print(f"  ... ({len(report) - 3} more patients, see full report above)")

    print_header("STEP 3/4 -- Building PyTorch Dataset")
    dataset = SlidingWindowDataset(processed_data, window_size=WINDOW_SIZE, stride=STRIDE)
    print(f"Total sliding-window samples: {len(dataset)}")

    planned, residual = dataset[0]
    print(f"Sample shapes : planned={tuple(planned.shape)}, residual={tuple(residual.shape)}")
    print(f"Sample dtypes : planned={planned.dtype}, residual={residual.dtype}")

    print_header("STEP 4/4 -- DataLoader smoke test")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch_planned, batch_residual = next(iter(loader))
    print(f"Batch shapes  : planned={tuple(batch_planned.shape)}, residual={tuple(batch_residual.shape)}")
    print(f"Batches/epoch : {len(loader)}")

    print_header("Done")
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"Patients used  : {len(processed_data)} "
          f"(skipped {len(skipped)}, failed {len(failed)})")
    print(f"PKL saved to   : {args.output}")
    print()
    print("Next step: upload this PKL as a Kaggle dataset and run "
          "`python train.py --data-path /kaggle/input/<dataset-name>/processed_data.pkl` there.")


if __name__ == "__main__":
    main()
