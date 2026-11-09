
import os
import glob
import h5py
import numpy as np
import torch

from torch.utils.data import Dataset

from .sampling import cartesian_mask
from .transforms import pad_width_center, normalize_kspace


def to_complex(array):
    array = np.asarray(array)

    if np.iscomplexobj(array):
        return array.astype(np.complex64)

    if array.dtype.fields is not None:
        names = array.dtype.fields.keys()

        if "real" in names and "imag" in names:
            return (
                array["real"] + 1j * array["imag"]
            ).astype(np.complex64)

    if array.ndim > 0 and array.shape[-1] == 2:
        return (
            array[..., 0] + 1j * array[..., 1]
        ).astype(np.complex64)

    return array.astype(np.complex64)


def ifft2c_np(kspace):
    kspace = np.fft.ifftshift(kspace, axes=(-2, -1))
    image = np.fft.ifft2(
        kspace,
        axes=(-2, -1),
        norm="ortho"
    )
    return np.fft.fftshift(image, axes=(-2, -1))


class FastMRIDataset(Dataset):
    def __init__(
        self,
        folder,
        accel_choices=(4, 6, 8),
        acs_frac=0.08,
        target_w=384,
        training=True,
        seed=42
    ):
        self.files = sorted(
            glob.glob(os.path.join(folder, "*.h5"))
        )

        if not self.files:
            raise RuntimeError(
                f"No HDF5 files found in {folder}"
            )

        self.accel_choices = tuple(accel_choices)
        self.acs_frac = acs_frac
        self.target_w = target_w
        self.training = training
        self.seed = seed

        self.index = []

        for fidx, filepath in enumerate(self.files):
            with h5py.File(filepath, "r") as hf:
                shape = hf["kspace"].shape

                if len(shape) == 3:
                    nslices = 1
                elif len(shape) == 4:
                    nslices = shape[0]
                else:
                    raise ValueError(
                        f"Unsupported k-space shape: {shape}"
                    )

            for sidx in range(nslices):
                self.index.append((fidx, sidx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fidx, sidx = self.index[idx]
        filepath = self.files[fidx]

        with h5py.File(filepath, "r") as hf:
            dataset = hf["kspace"]

            if dataset.ndim == 3:
                kspace = dataset[:]
            else:
                kspace = dataset[sidx]

        kspace = to_complex(kspace)

        if kspace.ndim != 3:
            raise ValueError(
                "Expected k-space with shape [C, H, W]"
            )

        kspace = pad_width_center(
            kspace,
            self.target_w
        )

        image = ifft2c_np(kspace)

        target = np.sqrt(
            np.sum(np.abs(image) ** 2, axis=0)
        ).astype(np.float32)

        kspace, target, scale = normalize_kspace(
            kspace,
            target
        )

        if self.training:
            accel = int(
                np.random.choice(self.accel_choices)
            )
            mask_seed = None
        else:
            accel = self.accel_choices[idx % len(self.accel_choices)]
            mask_seed = self.seed + idx

        H, W = kspace.shape[-2:]

        mask = cartesian_mask(
            (H, W),
            accel=accel,
            acs_frac=self.acs_frac,
            seed=mask_seed
        )

        undersampled = kspace * mask

        return {
            "y": torch.from_numpy(
                undersampled.copy()
            ).to(torch.complex64),

            "mask": torch.from_numpy(
                mask.copy()
            ).float(),

            "target": torch.from_numpy(
                target[None, ...].copy()
            ).float(),

            "scale": torch.tensor(scale).float()
        }
