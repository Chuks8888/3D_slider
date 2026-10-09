import time
import logging

import numpy as np
import SimpleITK as sitk
from pathlib import Path

HU_LOWER = -1024.0
HU_UPPER = 3071.0

def getLogger(output_directory: Path, log_filename: str = "preprocessing.log", stdout: bool = False) -> logging.Logger:
    logger = logging.getLogger("sitk_preprocessor")
    logger.setLevel(logging.INFO)

    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = (output_directory / log_filename).resolve()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_file_handler = any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in logger.handlers)
    if not has_file_handler:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stdout:
        has_stream_handler = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers)
        if not has_stream_handler:
            console_header = logging.StreamHandler()
            console_header.setFormatter(formatter)
            logger.addHandler(console_header)

    return logger

def form(data, unit="", decimal=3):
    res = ""
    if data.any():
        res = " x ".join(str(round(element, decimal)) + unit for element in data)

    return res


def preprocess(target: Path, output_directory: Path, verbose: bool = False) -> bool:
    logger = getLogger(output_directory=output_directory, stdout=verbose)

    if not target.is_file():
            logger.warning("Skipping %s: Not a valid file\n", target.name)
            return False
    else:
        logger.info("Processing file:   %s", target.name)

    try:
        image = sitk.ReadImage(target)
    except RuntimeError as e:
        logger.error("Failed to read the file %s: %s\n", target.name, e)
        return False

    if image.GetDimension() != 3:
        logger.warning("Skipping %s: Not a 3D image\n", target.name)
        return False

    size = np.array(image.GetSize())
    if size[2] <= 5:
        logger.warning("Skipping file %s: Insufficient slices along Z-axis (only %d slice/s)\n", target.name, size[2])
        return False

    view = sitk.GetArrayViewFromImage(image)
    spacing = np.array(image.GetSpacing())
    physical_size = size * spacing

    logger.info("Size:              %s", form(size))
    logger.info("Spacing:           %s", form(spacing, "mm"))
    logger.info("Physical Size:     %s", form(physical_size, "mm"))
    logger.info("HU Range:          %.1f to %.1f", view.min(), view.max())

    target_spacing = np.ones_like(physical_size)
    target_size = np.round(physical_size / target_spacing).astype(int)

    logger.info("Target Spacing:    %s", form(target_spacing, "mm"))
    logger.info("Target Size:       %s", form(target_size))

    R = sitk.ResampleImageFilter()
    R.SetInterpolator(sitk.sitkBSpline)
    R.SetSize(target_size.tolist())
    R.SetOutputSpacing(target_spacing.tolist())
    R.SetOutputOrigin(image.GetOrigin())
    R.SetOutputDirection(image.GetDirection())
    R.SetDefaultPixelValue(HU_LOWER)

    t0 = time.perf_counter()

    resampled_float = R.Execute(image)
    resampled_image = sitk.Clamp(
        resampled_float, sitk.sitkFloat32, lowerBound=HU_LOWER, upperBound=HU_UPPER
    )
    final_image = sitk.Cast(resampled_image, sitk.sitkInt16)

    duration = time.perf_counter() - t0

    del image
    del resampled_float

    view = sitk.GetArrayViewFromImage(final_image)
    final_spacing = np.array(final_image.GetSpacing())
    final_size = np.array(final_image.GetSize())

    if not np.allclose(final_spacing.tolist(), target_spacing.tolist()):
        logger.error("Spacing mismatch for %s!\n", target.name)
        return False

    if not np.allclose(final_size.tolist(), target_size.tolist()):
        logger.error("Size mismatch for %s!\n", target.name)
        return False

    logger.info("Resampled Size:    %s", form(final_size))
    logger.info("Resampled Spacing: %s", form(final_spacing, "mm"))
    logger.info("Final HU Range:    %d to %d", view.min(), view.max())
    logger.info("Execution time:    %.2fs", duration)

    output_path = (output_directory / target.name).resolve()
    sitk.WriteImage(final_image, str(output_path))
    logger.info("Saved output to:   %s\n", output_path)

    return True
