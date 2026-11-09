
import torch


def fft2c(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifft2c(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def extract_acs(kspace, acs_frac=0.08):
    B, C, H, W = kspace.shape

    num_low = max(1, int(round(W * acs_frac)))

    center = W // 2
    start = center - num_low // 2
    end = start + num_low

    acs = torch.zeros_like(kspace)
    acs[:, :, :, start:end] = kspace[:, :, :, start:end]

    return acs


def estimate_sensitivity_maps(
    kspace,
    acs_frac=0.08,
    eps=1e-8
):
    acs = extract_acs(kspace, acs_frac)

    coil_images = ifft2c(acs)

    rss = torch.sqrt(
        torch.sum(
            torch.abs(coil_images) ** 2,
            dim=1,
            keepdim=True
        )
    )

    sensitivity = coil_images / (rss + eps)

    energy = torch.sum(
        torch.abs(sensitivity) ** 2,
        dim=1,
        keepdim=True
    )

    sensitivity = sensitivity / torch.sqrt(
        energy + eps
    )

    return sensitivity
