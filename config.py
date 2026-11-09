
import torch


class Config:

    SEED = 42

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    TARGET_WIDTH = 384

    ACCEL_CHOICES = (4, 6, 8)
    ACS_FRAC = 0.08

    NUM_ITERATIONS = 5
    BASE_CHANNELS = 32

    EPOCHS = 40
    BATCH_SIZE = 1
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-4

    KSPACE_WEIGHT = 0.2
    SSIM_WEIGHT = 0.1

    GRAD_CLIP = 1.0

    NUM_WORKERS = 0

    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR = "./evaluation_results"

    MODEL_NAME = "HyperLASER"
