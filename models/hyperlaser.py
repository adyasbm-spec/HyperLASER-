
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperunet import HyperUNet
from .sensitivity_refiner import SensitivityRefiner


def fft2c(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifft2c(x):
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm="ortho")
    return torch.fft.fftshift(x, dim=(-2, -1))


def sense_forward(x, S, mask):
    return fft2c(S * x) * mask


def sense_adjoint(k, S):
    img = ifft2c(k)

    return torch.sum(
        torch.conj(S) * img,
        dim=1,
        keepdim=True
    )


class HyperLASERRecon(nn.Module):
    def __init__(
        self,
        T=5,
        base_ch=32,
        eps=1e-8
    ):
        super().__init__()

        if T < 1:
            raise ValueError("T must be at least 1.")

        self.T = T
        self.eps = eps

        self.lam = nn.Parameter(
            torch.full((T,), 0.1)
        )

        self.alpha = nn.Parameter(
            torch.full((T,), 0.5)
        )

        self.prior = HyperUNet(
            in_ch=3,
            base_ch=base_ch
        )

        self.sens_refiner = SensitivityRefiner()

    def forward(
        self,
        y,
        mask,
        S,
        num_iter=None
    ):
        if not torch.is_complex(y):
            raise TypeError("Input k-space must be complex-valued.")

        if not torch.is_complex(S):
            raise TypeError("Sensitivity maps must be complex-valued.")

        if y.ndim != 4 or S.ndim != 4:
            raise ValueError("Expected y and S with shape [B,C,H,W].")

        if y.shape != S.shape:
            raise ValueError("K-space and sensitivity map shapes must match.")

        if mask.ndim != 4:
            raise ValueError("Mask must have shape [B,1,H,W].")

        if mask.shape[0] != y.shape[0] or mask.shape[1] != 1:
            raise ValueError("Mask batch and channel dimensions are invalid.")

        if mask.shape[-2:] != y.shape[-2:]:
            raise ValueError("Mask and k-space spatial dimensions must match.")

        Tuse = self.T if num_iter is None else num_iter

        if not isinstance(Tuse, int) or not 1 <= Tuse <= self.T:
            raise ValueError(
                f"num_iter must be an integer between 1 and {self.T}."
            )

        mask = mask.to(device=y.device, dtype=y.real.dtype)
        S = S.to(device=y.device, dtype=y.dtype)

        S = self.sens_refiner(S)

        x = sense_adjoint(y, S)

        coil_energy = torch.sum(
            torch.abs(S) ** 2,
            dim=1,
            keepdim=True
        )

        coil_energy = coil_energy / (
            coil_energy.amax(
                dim=(-2, -1),
                keepdim=True
            ) + self.eps
        )

        for t in range(Tuse):

            lam_t = F.softplus(self.lam[t])
            alpha_t = torch.sigmoid(self.alpha[t])

            pred_k = sense_forward(
                x,
                S,
                mask
            )

            grad = sense_adjoint(
                pred_k - y,
                S
            )

            x_dc = x - lam_t * grad

            mag = torch.abs(x_dc)
            phase = torch.angle(x_dc)

            mag_n = mag / (
                mag.amax(
                    dim=(-2, -1),
                    keepdim=True
                ) + self.eps
            )

            prior_in = torch.cat(
                [
                    mag_n,
                    coil_energy,
                    mask
                ],
                dim=1
            )

            dmag, dphi = self.prior(prior_in)

            mag = mag * (
                1.0 + alpha_t * torch.tanh(dmag)
            )

            phase = phase + torch.tanh(dphi)

            x = torch.polar(mag, phase)

        return x
