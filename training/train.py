
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from config import Config
from models.hyperlaser import HyperLASERRecon, fft2c
from data.dataset import FastMRIDataset
from utils.fft import estimate_sensitivity_maps
from utils.metrics import compute_psnr, compute_ssim


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def charbonnier_loss(pred, target, eps=1e-3):
    return torch.mean(
        torch.sqrt((pred - target) ** 2 + eps ** 2)
    )


def kspace_loss(pred, target, S, mask):
    pred_k = fft2c(S * pred)
    return torch.mean(
        torch.abs(mask * (pred_k - target))
    )


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_psnr
):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_psnr": best_psnr
        },
        path
    )


def validate(model, loader, device, config):
    model.eval()

    psnr_values = []
    ssim_values = []

    with torch.no_grad():
        for batch in loader:
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            S = estimate_sensitivity_maps(
                y,
                acs_frac=config.ACS_FRAC
            )

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                output = model(
                    y,
                    mask,
                    S,
                    num_iter=config.NUM_ITERATIONS
                )

            prediction = torch.abs(output).float()

            psnr_values.append(
                compute_psnr(prediction, target).item()
            )

            ssim_values.append(
                compute_ssim(prediction, target).item()
            )

    return (
        float(np.mean(psnr_values)),
        float(np.mean(ssim_values))
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()

    config = Config()

    set_seed(config.SEED)

    device = torch.device(
        args.device or config.DEVICE
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    epochs = args.epochs or config.EPOCHS
    batch_size = args.batch_size or config.BATCH_SIZE
    lr = args.lr or config.LEARNING_RATE

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    train_dataset = FastMRIDataset(
        args.train_dir,
        accel_choices=config.ACCEL_CHOICES,
        acs_frac=config.ACS_FRAC,
        target_w=config.TARGET_WIDTH,
        training=True,
        seed=config.SEED
    )

    val_dataset = FastMRIDataset(
        args.val_dir,
        accel_choices=config.ACCEL_CHOICES,
        acs_frac=config.ACS_FRAC,
        target_w=config.TARGET_WIDTH,
        training=False,
        seed=config.SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    model = HyperLASERRecon(
        T=config.NUM_ITERATIONS,
        base_ch=config.BASE_CHANNELS
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.WEIGHT_DECAY
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    start_epoch = 1
    best_psnr = -float("inf")

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["state_dict"])

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        start_epoch = checkpoint["epoch"] + 1
        best_psnr = checkpoint.get(
            "best_psnr",
            best_psnr
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()

        train_losses = []

        for batch in train_loader:
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)

            S = estimate_sensitivity_maps(
                y,
                acs_frac=config.ACS_FRAC
            )

            with torch.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                output = model(
                    y,
                    mask,
                    S,
                    num_iter=config.NUM_ITERATIONS
                )

                prediction = torch.abs(output)

                loss_img = charbonnier_loss(
                    prediction,
                    target
                )

                loss_k = kspace_loss(
                    output,
                    y,
                    S,
                    mask
                )

                loss_ssim = 1.0 - compute_ssim(
                    prediction.float(),
                    target
                )

                loss = (
                    loss_img
                    + config.KSPACE_WEIGHT * loss_k
                    + config.SSIM_WEIGHT * loss_ssim
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}"
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.GRAD_CLIP
            )

            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        val_psnr, val_ssim = validate(
            model,
            val_loader,
            device,
            config
        )

        train_loss = float(np.mean(train_losses))

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {train_loss:.5f} | "
            f"PSNR: {val_psnr:.4f} | "
            f"SSIM: {val_ssim:.4f}"
        )

        save_checkpoint(
            os.path.join(
                config.CHECKPOINT_DIR,
                "hyperlaser_last.pt"
            ),
            model,
            optimizer,
            epoch,
            best_psnr
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr

            save_checkpoint(
                os.path.join(
                    config.CHECKPOINT_DIR,
                    "hyperlaser_best.pt"
                ),
                model,
                optimizer,
                epoch,
                best_psnr
            )

            print("Best checkpoint saved")


if __name__ == "__main__":
    main()
