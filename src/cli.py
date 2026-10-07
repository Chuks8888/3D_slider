import argparse
import time
import uuid
from datetime import datetime

import numpy as np
import SimpleITK as sitk
from pathlib import Path

def form(data, unit="", decimal=3):
    res = ""
    if data.any():
        res = ' x '.join(str(round(element, decimal)) + unit for element in data)

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Niigz Images Loading Tool')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-o', '--output', type=str, required=True)
    args = parser.parse_args()

    # Argument Parsing
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

    #HU Bounds
    hu_lower = -1024.0
    hu_upper = 3071.0

    #Filepaths
    target_directory = Path(target_directory)
    output_directory = Path(output_directory)
    
    #Process related variables
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_output_directory = output_directory / run_id

    # Processing Loop
    for target in target_directory.glob('*.nii.gz'):
        if not target:
            continue

        print(f"File: {target.name}")
        image = sitk.ReadImage(target)
        
        if image.GetDimension() != 3:
            print("Skipping: Not a 3D image\n")
            continue

        size = np.array(image.GetSize())
        if size[2] <= 5:
            print("Skipping: Insufficient slices along Z-axis\n")
            continue

        view = sitk.GetArrayViewFromImage(image)
        spacing = np.array(image.GetSpacing())
        physical_size = size * spacing

        print(f"Size of the image:      {form(size)}")
        print(f"Spacing of the image:   {form(spacing, 'mm')}")
        print(f"Real Size of the image: {form(physical_size, 'mm')}")
        print(f"HU range:               {view.min()} & {view.max()}\n")

        target_spacing = np.ones_like(physical_size)
        target_size = np.round(physical_size / target_spacing).astype(int)

        print(f"Target Spacing:         {form(target_spacing, 'mm')}")  
        print(f"Target Size:            {form(target_size)}")

        R = sitk.ResampleImageFilter()
        R.SetInterpolator(sitk.sitkBSpline)

        R.SetSize(target_size.tolist())
        R.SetOutputSpacing(target_spacing.tolist())
        R.SetOutputOrigin(image.GetOrigin())
        R.SetOutputDirection(image.GetDirection())
        R.SetDefaultPixelValue(hu_lower)

        t0 = time.perf_counter()

        resampled_float = R.Execute(image)
        resampled_image = sitk.Clamp(
            resampled_float,
            sitk.sitkFloat32,
            lowerBound=hu_lower,
            upperBound=hu_upper
        )
        final_image = sitk.Cast(resampled_image, sitk.sitkInt16)
        
        duration = time.perf_counter() - t0
        final_spacing = np.array(final_image.GetSpacing())
        final_size = np.array(final_image.GetSize())
        view = sitk.GetArrayViewFromImage(final_image)

        assert np.allclose(final_spacing.tolist(), target_spacing.tolist()), "Spacing mistmatch!"
        assert np.allclose(final_size.tolist(), target_size.tolist()), "Size mistmatch!"

        print(f"\nResampled Size:       {form(final_size)}")
        print(f"Resampled Spacing:      {form(final_spacing, 'mm')}")
        print(f"Fianl HU Range:         {view.min()} to {view.max()}")
        print(f"Execution time:         {duration:.2f}s")

        output_path = unique_output_directory / target.name
        sitk.WriteImage(final_image, str(output_path))
