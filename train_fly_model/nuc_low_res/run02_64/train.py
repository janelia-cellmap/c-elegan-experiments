#%%
import yaml
import torch
import numpy as np
import logging
from fly_organelles.run import run
from fly_organelles.model import StandardUnet

logger = logging.getLogger(__name__)

log_dir = "/nrs/cellmap/zouinkhim/tensorboard/nuc_64_run01"

CHECKPOINT_PATH = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/run09/model_checkpoint_128000"
labels = ['nuc']
yaml_file = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/datasets_generated_all.yaml"
iterations = 1000000

label_weights = [1]
voxel_size = (64, 64, 64)
l_rate = 0.5e-5
batch_size = 14


if not label_weights:
    label_weights = [
        1.0 / len(labels),
    ] * len(labels)
else:
    if len(label_weights) != len(labels):
        msg = (
            f"If label weights are specified ({type(label_weights)}),"
            f"they need to be of the same length as the list of labels ({len(labels)})"
        )
        raise ValueError(msg)
    normalizer = np.sum(label_weights)
    label_weights = [lw / normalizer for lw in label_weights]
logger.info(
    f"Running training for the following labels:"
    f"{', '.join([f'{lbl} ({lblw:.4f})' for lbl,lblw in zip(labels,label_weights)])}"
)
with open(yaml_file, "r") as data_yaml:
    datasets = yaml.safe_load(data_yaml)
# label_stores, raw_stores, crop_copies = read_data_yaml(data_yaml)

model = StandardUnet(len(labels))

import os
import re

Current_DIR = os.path.dirname(os.path.abspath(__file__))
def get_latest_checkpoint(folder_path: str) -> str:

    max_step = -1
    latest_checkpoint = None
    
    pattern = re.compile(r"^model_checkpoint_(\d+)$")
    
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            step = int(match.group(1))
            if step > max_step:
                max_step = step
                latest_checkpoint = filename
    
    return latest_checkpoint
CH = get_latest_checkpoint(Current_DIR)
if CH: 

    CHECKPOINT_PATH = os.path.join(Current_DIR, CH)
    logger.error(f"Using checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True, map_location=torch.device('cpu'))["model_state_dict"]
else:
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True, map_location=torch.device('cpu'))
    from fly_organelles.run import set_weights
    OLD_CHECKPOINT_CHANNELS = ['mito', 'ld', 'lyso', 'perox', 'yolk', 'nuc']
    checkpoint = set_weights(model, checkpoint["model_state_dict"], OLD_CHECKPOINT_CHANNELS, labels)
model.load_state_dict(checkpoint, strict=True)
model = model.cuda()
run(model,
iterations, 
labels, 
label_weights, 
datasets,
voxel_size = voxel_size,
batch_size = batch_size, 
l_rate=l_rate,
log_dir=log_dir)

