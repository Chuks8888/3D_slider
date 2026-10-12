import logging

from pathlib import Path


def getLogger(
    output_directory: Path,
    log_filename: str = "general.log",
    stdout: bool = False,
) -> logging.Logger:
    logger = logging.getLogger("sitk_preprocessor")
    logger.setLevel(logging.INFO)

    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = (output_directory / log_filename).resolve()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_file_handler = any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path
        for h in logger.handlers
    )
    if not has_file_handler:
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if stdout:
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )
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
