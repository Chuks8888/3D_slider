import torch

import numpy as np
import SimpleITK as sitk

from typing import Union
from pathlib import Path
from torch.utils.data import Dataset


class MedicalDataset(Dataset):

    def __init__(
        self,
        target_directory: Union[str, Path],
        hu_lower: float = -1000.0,
        hu_upper: float = 400.0,
    ):
        super().__init__()
        self.target_directory = Path(target_directory)

        if hu_upper - hu_lower == 0.0:
            raise ValueError("Selected HU boundry casues division by zero")

        self.hu_upper = hu_upper
        self.hu_lower = hu_lower

        if not self.target_directory.is_dir():
            raise ValueError(f"Directory does not exist: {self.target_directory}")

        self._validate_preprocessing_log()

        self.files = sorted(list(self.target_directory.glob("*.nii.gz")))

    def _validate_preprocessing_log(self):
        log_path = self.target_directory / "preprocessing.log"

        if not log_path.exists():
            raise RuntimeError(
                f"Missing 'preprocessing.log' in {self.target_directory}. "
                "Did you skip preprocessing on raw data?"
            )

        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            tail = "".join(lines[-50:])

            if "=== BATCH_PROCESSING_COMPLETE ===" not in tail:
                raise RuntimeError(
                    "The preprocessing.log does not contain the completion signature. "
                    "The preprocessing run may have crashed, been interrupted, or the data is raw."
                )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> torch.tensor:
        filepath = self.files[index]

        image = sitk.ReadImage(filepath)
        volume = sitk.GetArrayFromImage(image)

        volume = np.clip(volume, self.hu_lower, self.hu_upper)
        volume = (volume - self.hu_lower) / (self.hu_upper - self.hu_lower)

        # Order of data is 1, 1, Z, Y, X
        tensor = torch.from_numpy(volume)
        tensor = tensor = tensor.float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        return tensor
