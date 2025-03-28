CHECKPOINT_PATH = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/n/run01/conitnue/model_checkpoint_45000"
#%%

import os
import re

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

CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, get_latest_checkpoint(CHECKPOINT_PATH))

print(CHECKPOINT_PATH)
#%%
from funlib.geometry.coordinate import Coordinate
import numpy as np
input_voxel_size = (8, 8, 8)
read_shape = Coordinate((178, 178, 178)) * Coordinate(input_voxel_size)
write_shape = Coordinate((56, 56, 56)) * Coordinate(input_voxel_size)
output_voxel_size = Coordinate((8, 8, 8))

#%%
import torch
from fly_organelles.model import StandardUnet

#%%
def load_eval_model(num_labels, checkpoint_path):
    model_backbone = StandardUnet(num_labels)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)    
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model_backbone.load_state_dict(checkpoint["model_state_dict"])
    model = torch.nn.Sequential(model_backbone, torch.nn.Sigmoid())
    model.to(device)
    model.eval()
    return model

classes = ['mito', 'ld', 'lyso', 'perox', 'yolk', 'nuc']


output_channels = len(classes)
model = load_eval_model(output_channels, CHECKPOINT_PATH)
block_shape = np.array((56, 56, 56,output_channels))