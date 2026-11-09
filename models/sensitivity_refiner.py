
import torch
import torch.nn as nn


class SensitivityRefiner(nn.Module):
    def __init__(self, hidden_ch=16, residual_scale=0.1, eps=1e-8):
        super().__init__()

        self.residual_scale = residual_scale
        self.eps = eps

        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, 2, kernel_size=3, padding=1)
        )

    def forward(self, S):
        if not torch.is_complex(S):
            raise TypeError("Sensitivity maps must be complex-valued.")

        if S.ndim != 4:
            raise ValueError("Expected sensitivity maps with shape [B,C,H,W].")

        B, C, H, W = S.shape

        S_real = torch.view_as_real(S)
        S_real = S_real.permute(0, 1, 4, 2, 3)
        S_real = S_real.reshape(B * C, 2, H, W)

        delta = self.net(S_real)

        S_real = S_real + self.residual_scale * delta

        S_real = S_real.reshape(B, C, 2, H, W)
        S_real = S_real.permute(0, 1, 3, 4, 2)

        S_refined = torch.view_as_complex(S_real.contiguous())

        energy = torch.sum(
            torch.abs(S_refined) ** 2,
            dim=1,
            keepdim=True
        )

        S_refined = S_refined / torch.sqrt(energy + self.eps)

        return S_refined
