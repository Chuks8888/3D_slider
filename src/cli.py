import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path

def form(data, unit="", decimal=3):
    res = ""
    if data.any():
        res = ' x '.join(str(round(element, decimal)) + unit for element in data)

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Niigz Images Loading Tool')
    parser.add_argument('-i', '--input', type=str, required=True)
    args = parser.parse_args()

    target_directory = args.input
    if not target_directory or not Path(target_directory).is_dir():
        parser.error("Need to provide the source directory")
    target_directory = Path(target_directory)

    for target in target_directory.glob('*.nii.gz'):
        image = sitk.ReadImage(target)

        size = np.array(image.GetSize())
        spacing = np.array(image.GetSpacing())
        physical_size = size * spacing

        target_spacing = np.ones_like(physical_size)
        target_size = np.floor(physical_size / target_spacing).astype(int)

        view = sitk.GetArrayFromImage(image)
        del image

        print(f"File: {target.name}")
        print(f"Size of the image:      {form(size)}")
        print(f"Spacing of the image:   {form(spacing, "mm")}")
        print(f"Real Size of the image: {form(physical_size, "mm")}")
        print(f"HU range:               {view.min()} & {view.max()}\n")

        print(f"Target Spacing:         {form(target_spacing, "mm")}")  
        print(f"Target Size:            {form(target_size)}")

        print('\n')
