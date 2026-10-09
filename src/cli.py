import argparse
import sys
import torch

import numpy as np
import SimpleITK as sitk
import preprocess_m as pre

from pathlib import Path
from datetime import datetime
from dataset_m import MedicalDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Niigz Images Loading Tool")
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)

    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-r", "--recursive", action="store_true", default=False)
    parser.add_argument("--skip-preprocess", action="store_true", default=False)

    args = parser.parse_args()

    target_directory = args.input
    if not target_directory or not Path(target_directory).is_dir():
        parser.error(f"Source directory {target_directory} does not exist")

    output_directory = args.output
    if not output_directory or not Path(output_directory).is_dir():
        try:
            Path(output_directory).mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            parser.error("The output needs to be a directory")
        except:
            parser.error("Failed to create the output directory")

    target_directory = Path(target_directory)
    output_directory = Path(output_directory)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.skip_preprocess:
        unique_output_directory = output_directory / "preprocessing_" / run_id
        unique_output_directory.mkdir()

        pattern = "*.nii.gz"
        files = sorted(
            list(
                target_directory.rglob(pattern)
                if args.recursive
                else target_directory.glob(pattern)
            )
        )

        if not files:
            sys.exit("Error: No *.nii.gz files were found in input")

        for target in files:
            pre.preprocess(target, unique_output_directory, args.verbose)

        dataset_dir = unique_output_directory

        logger = pre.getLogger(output_directory=dataset_dir, stdout=args.verbose)
        logger.info("=== BATCH_PROCESSING_COMPLETE ===\n")
    else:
        dataset_dir = target_directory

    dataset = MedicalDataset(dataset_dir)

    if len(dataset) > 0:
        unique_output_directory = output_directory / "slider_" / run_id
        unique_output_directory.mkdir()

        logger = pre.getLogger(
            output_directory=unique_output_directory,
            log_filename="engine.log",
            stdout=args.verbose,
        )
        first_tensor = dataset[0]
        logger.info(f"Loaded tensor shape: {first_tensor.shape}")
        logger.info(f"Tensor dtype: {first_tensor.dtype}")
        logger.info(
            f"Tensor range: {torch.min(first_tensor)} - {torch.max(first_tensor)}"
        )
    else:
        sys.exit("Error: No *.nii.gz files were found in input")
