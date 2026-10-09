import argparse
import sys

import numpy as np
import SimpleITK as sitk

from pathlib import Path
from datetime import datetime
from preprocess_m import preprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Niigz Images Loading Tool')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)

    # Flags
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-r', '--recursive', action='store_true', default=False)
    args = parser.parse_args()

    target_directory = args.input
    if not target_directory or not Path(target_directory).is_dir():
        parser.error("Need to provide the source directory")

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
    unique_output_directory = output_directory / run_id

    pattern = '*.nii.gz'
    files = list(target_directory.rglob(pattern) if args.recursive else target_directory.glob(pattern))

    if not files:
        sys.exit(f"Error: No {pattern} files were found")

    for target in files:
        preprocess(target, unique_output_directory, args.verbose)
