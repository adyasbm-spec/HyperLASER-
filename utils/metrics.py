
import torch
import torch.nn.functional as F


def compute_psnr(pred, target, data_range=1.0, eps=1e-10):
    mse = torch.mean((pred - target) ** 2)

    psnr = 20 * torch.log10(
        torch.tensor(
            data_range,
            device=pred.device,
            dtype=pred.dtype
        )
    ) - 10 * torch.log10(mse + eps)

    return psnr


def compute_ssim(
    pred,
    target,
    data_range=1.0,
    window_size=7,
    eps=1e-8
):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(
        pred,
        window_size,
        stride=1,
        padding=window_size // 2
    )

    mu_y = F.avg_pool2d(
        target,
        window_size,
        stride=1,
        padding=window_size // 2
    )

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x = F.avg_pool2d(
        pred ** 2,
        window_size,
        stride=1,
        padding=window_size // 2
    ) - mu_x_sq

    sigma_y = F.avg_pool2d(
        target ** 2,
        window_size,
        stride=1,
        padding=window_size // 2
    ) - mu_y_sq

    sigma_xy = F.avg_pool2d(
        pred * target,
        window_size,
        stride=1,
        padding=window_size // 2
    ) - mu_xy

    numerator = (
        (2 * mu_xy + C1) *
        (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x_sq + mu_y_sq + C1) *
        (sigma_x + sigma_y + C2)
    )

    ssim_map = numerator / (denominator + eps)

    return ssim_map.mean()


def compute_nmse(pred, target, eps=1e-10):
    numerator = torch.sum(
        (pred - target) ** 2
    )

    denominator = torch.sum(
        target ** 2
    )

    return numerator / (denominator + eps)


def compute_nrmse(pred, target, eps=1e-10):
    return torch.sqrt(
        compute_nmse(pred, target, eps)
    )
