import os
import glob
import yaml
import torch
import numpy as np
import logging
import warnings

from fly_organelles.run import run, set_weights
from fly_organelles.model import StandardUnet

# Suppress only UserWarning and FutureWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

labels = config["labels"]
yaml_file = config["paths"]["yaml_file"]
log_dir = config["paths"]["log_dir"]
voxel_size = tuple([config["voxel_size"]] * 3)
l_rate = 0.5e-5
batch_size = 14
weights_labels = [1 for _ in labels]
iterations = 1_000_000

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)

# Load datasets
with open(yaml_file, "r") as data_yaml:
    datasets = yaml.safe_load(data_yaml)

# Model setup
model = StandardUnet(len(labels))

# Checkpoint loading
checkpoint_files = glob.glob(os.path.join(os.path.dirname(__file__), "model_checkpoint_*"))
if checkpoint_files:
    def extract_step(fp):
        try:
            return int(os.path.basename(fp).split("_")[-1])
        except Exception:
            return -1
    latest_ckpt = max(checkpoint_files, key=extract_step)
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
else:
    CHECKPOINT_PATH = "/nrs/saalfeld/heinrichl/fly_organelles/run08/model_checkpoint_438000"
    OLD_CHECKPOINT_CHANNELS = ["all_mem", "organelle", "mito", "er", "nuc", "pm", "vs", "ld"]
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True, map_location=device)
    updated_checkpoint = set_weights(model, checkpoint["model_state_dict"], OLD_CHECKPOINT_CHANNELS, labels)
    model.load_state_dict(updated_checkpoint, strict=True)

model = model.to(device)

# Training
run(
    model,
    iterations,
    labels,
    weights_labels,
    datasets,
    voxel_size=voxel_size,
    batch_size=batch_size,
    l_rate=l_rate,
    log_dir=log_dir,
    distance_sigma=6,
)
