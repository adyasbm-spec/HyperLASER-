
import numpy as np


def pad_width_center(kspace, target_w=384):
    C, H, W = kspace.shape

    if W == target_w:
        return kspace

    if W > target_w:
        start = (W - target_w) // 2
        return kspace[:, :, start:start + target_w]

    diff = target_w - W
    left = diff // 2
    right = diff - left

    return np.pad(
        kspace,
        ((0, 0), (0, 0), (left, right)),
        mode="constant"
    )


def normalize_kspace(kspace, target, eps=1e-8):
    scale = float(np.max(target))

    if not np.isfinite(scale) or scale < eps:
        raise ValueError("Invalid MRI normalization scale")

    kspace = kspace / scale
    target = target / scale

    return kspace, target, scale


def center_crop(image, target_h, target_w):
    H, W = image.shape[-2:]

    start_h = max(0, (H - target_h) // 2)
    start_w = max(0, (W - target_w) // 2)

    return image[
        ...,
        start_h:start_h + target_h,
        start_w:start_w + target_w
    ]
