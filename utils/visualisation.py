
import os
import numpy as np
import matplotlib.pyplot as plt


def save_reconstruction(
    target,
    prediction,
    save_path
):
    os.makedirs(
        os.path.dirname(save_path) or ".",
        exist_ok=True
    )

    target = target.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()

    error = np.abs(prediction - target)

    vmax = max(
        float(target.max()),
        float(prediction.max()),
        1e-8
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        target,
        cmap="gray",
        vmin=0,
        vmax=vmax
    )
    axes[0].set_title("Ground Truth")

    axes[1].imshow(
        prediction,
        cmap="gray",
        vmin=0,
        vmax=vmax
    )
    axes[1].set_title("HyperLASER Reconstruction")

    axes[2].imshow(
        error,
        cmap="hot"
    )
    axes[2].set_title("Absolute Error")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close(fig)
