import torch
import math
import itertools

from typing import List, Tuple


def compute_axis_slices(
    dim_size: int, patch_size: int, stride: int, padding: bool = True
) -> List[Tuple[int, int]]:

    if dim_size < patch_size:
        if not padding:
            raise ValueError(
                f"dim_size ({dim_size}) < patch_size ({patch_size}) requires padding=True"
            )
        return [(0, patch_size)]

    if padding:
        num_steps = math.ceil((dim_size - patch_size) / stride) + 1
        slices = [(i * stride, i * stride + patch_size) for i in range(num_steps)]
    else:
        slices = [
            (i, i + patch_size) for i in range(0, dim_size - patch_size + 1, stride)
        ]
        if slices[-1][0] != dim_size - patch_size:
            slices.append((dim_size - patch_size, dim_size))

    return slices


def compute_sub_cuboids(
    volume_shape: Tuple[int, int, int],
    sub_shape: Tuple[int, int, int],
    overlap: float = 0.5,
    padding: bool = True,
) -> List[Tuple[slice, slice, slice]]:
    d, h, w = volume_shape
    pd, ph, pw = sub_shape

    strides = (int(i * (1.0 - overlap)) for i in sub_shape)

    depth_slices = compute_axis_slices(d, pd, strides[0], padding)
    height_slices = compute_axis_slices(h, ph, strides[1], padding)
    width_slices = compute_axis_slices(w, pw, strides[2], padding)

    cuboids = [
        (slice(z0, z1), slice(h0, h1), slice(w0, w1))
        for (z0, z1), (h0, h1), (w0, w1) in itertools.product(
            depth_slices, height_slices, width_slices
        )
    ]

    return cuboids
