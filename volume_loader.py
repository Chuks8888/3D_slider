import numpy as np
import SimpleITK as sitk
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Niigz Images Loading Tool')
    parser.add_argument('-s', '--source', type=str, required=True)
    args = parser.parse_args()

    target_directory = args.source
    if not target_directory or not Path(target_directory).is_dir():
        parser.error("Need to provide the source directory")
    target_directory = Path(target_directory)

    for target in target_directory.glob('*.nii.gz'):
        print (f"Opening file: {target}")

        image = sitk.ReadImage(target)
        size = image.GetSize()
        size = ' x '.join(str(element) for element in size)
        print(f"Size of the image: {size}")
        print(f"Spacing of the image: {image.GetSpacing()}")

        array = sitk.GetArrayFromImage(image)
        print(f"Hounsfield values: {array.min()} & {array.max()}")

        print('\n')
