
import numpy as np


def cartesian_mask(
    shape,
    accel=4,
    acs_frac=0.08,
    seed=None
):
    H, W = shape

    if accel <= 1:
        return np.ones((1, H, W), dtype=np.float32)

    if not 0 < acs_frac < 1:
        raise ValueError("acs_frac must be between 0 and 1")

    rng = np.random.default_rng(seed)

    num_low = max(1, int(round(W * acs_frac)))
    num_total = max(num_low, int(round(W / accel)))
    num_total = min(num_total, W)

    center = W // 2
    start = center - num_low // 2
    end = start + num_low

    selected = np.zeros(W, dtype=bool)
    selected[start:end] = True

    remaining = np.flatnonzero(~selected)
    num_random = num_total - num_low

    if num_random > 0:
        chosen = rng.choice(
            remaining,
            size=num_random,
            replace=False
        )
        selected[chosen] = True

    mask = np.broadcast_to(
        selected[None, None, :],
        (1, H, W)
    ).copy()

    return mask.astype(np.float32)


def apply_mask(kspace, mask):
    return kspace * mask
