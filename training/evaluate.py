
import os
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader

from config import Config
from models.hyperlaser import HyperLASERRecon
from data.dataset import FastMRIDataset
from utils.fft import estimate_sensitivity_maps
from utils.metrics import compute_psnr, compute_ssim, compute_nmse, compute_nrmse
from utils.visualization import save_reconstruction


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="./evaluation_results")
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    config = Config()

    device = torch.device(
        args.device or config.DEVICE
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = FastMRIDataset(
        args.data_dir,
        accel_choices=config.ACCEL_CHOICES,
        acs_frac=config.ACS_FRAC,
        target_w=config.TARGET_WIDTH,
        training=False,
        seed=config.SEED
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    model = HyperLASERRecon(
        T=config.NUM_ITERATIONS,
        base_ch=config.BASE_CHANNELS
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    results = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            S = estimate_sensitivity_maps(
                y,
                acs_frac=config.ACS_FRAC
            )

            output = model(
                y,
                mask,
                S,
                num_iter=config.NUM_ITERATIONS
            )

            prediction = torch.abs(output).float()

            metrics = {
                "PSNR": compute_psnr(
                    prediction, target
                ).item(),

                "SSIM": compute_ssim(
                    prediction, target
                ).item(),

                "NMSE": compute_nmse(
                    prediction, target
                ).item(),

                "NRMSE": compute_nrmse(
                    prediction, target
                ).item()
            }

            results.append(metrics)

            save_reconstruction(
                target[0, 0],
                prediction[0, 0],
                os.path.join(
                    args.output_dir,
                    f"slice_{idx:04d}.png"
                )
            )

            print(
                f"Slice {idx + 1}/{len(dataset)} | "
                f"PSNR: {metrics['PSNR']:.4f} | "
                f"SSIM: {metrics['SSIM']:.4f} | "
                f"NMSE: {metrics['NMSE']:.6f} | "
                f"NRMSE: {metrics['NRMSE']:.6f}"
            )

    if not results:
        raise RuntimeError("No evaluation samples found")

    keys = results[0].keys()

    averages = {
        key: float(np.mean([r[key] for r in results]))
        for key in keys
    }

    with open(
        os.path.join(args.output_dir, "metrics.txt"),
        "w"
    ) as f:
        for key, value in averages.items():
            f.write(f"{key}: {value:.6f}\n")

    print("\nAverage Results")

    for key, value in averages.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
