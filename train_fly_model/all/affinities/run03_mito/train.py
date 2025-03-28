#%%
import yaml
import torch
import numpy as np
import logging
from fly_organelles.run import run
from fly_organelles.model import StandardUnet

from fly_organelles.run import set_weights

logger = logging.getLogger(__name__)

log_dir = "/nrs/cellmap/zouinkhim/tensorboard/aff_run3_8nm_mito"

voxel_size = (8, 8, 8)
l_rate = 0.5e-5
batch_size = 14

affinities_map = [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [6, 0, 0],
                        [0, 6, 0],
                        [0, 0, 6],
    ]



N_AFFINITIES = len(affinities_map)
CHECKPOINT_PATH = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/run09/model_checkpoint_128000"
#%%
OLD_CHECKPOINT_CHANNELS = ['mito', 'ld', 'lyso', 'perox', 'yolk', 'nuc']

labels = ["mito"]

weights_labes = [ f for f in labels  for _ in range(N_AFFINITIES)]

#%%


yaml_file = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/datasets_generated_all.yaml"
iterations = 1000000



with open(yaml_file, "r") as data_yaml:
    datasets = yaml.safe_load(data_yaml)
# label_stores, raw_stores, crop_copies = read_data_yaml(data_yaml)

model = StandardUnet(len(labels)*len(affinities_map))
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True, map_location=torch.device('cpu'))
updated_checkpoint = set_weights(model, checkpoint["model_state_dict"], OLD_CHECKPOINT_CHANNELS, weights_labes)
model.load_state_dict(updated_checkpoint, strict=True)
model = model.cuda()
run(model,
iterations, 
labels, 
None, 
datasets,
voxel_size = voxel_size,
batch_size = batch_size, 
l_rate=l_rate,
log_dir=log_dir,
affinities = True, 
affinities_map = affinities_map,
foreground_factor = 1,
)

