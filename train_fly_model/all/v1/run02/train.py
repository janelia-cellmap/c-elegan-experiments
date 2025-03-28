#%%
import yaml
import torch
import numpy as np
import logging
from fly_organelles.run import run
from fly_organelles.model import StandardUnet

from fly_organelles.run import set_weights

logger = logging.getLogger(__name__)

log_dir = "/nrs/cellmap/zouinkhim/tensorboard/aff_run2"

voxel_size = (8, 8, 8)
l_rate = 1e-4
batch_size = 14

affinities_map = [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [3, 0, 0],
                        [0, 3, 0],
                        [0, 0, 3],
    ]

old_weights = [1,1 ,2,2,2,1]

N_AFFINITIES = len(affinities_map)
CHECKPOINT_PATH = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/run06/model_checkpoint_84000"
#%%
OLD_CHECKPOINT_CHANNELS = ['mito', 'ld', 'lyso', 'perox', 'yolk', 'nuc']

label_weights = [f for f in old_weights for _ in range(N_AFFINITIES)]
labels = [ f for f in OLD_CHECKPOINT_CHANNELS  for _ in range(N_AFFINITIES)]

#%%


yaml_file = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/datasets_generated_all.yaml"
iterations = 1000000





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
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True, map_location=torch.device('cpu'))
updated_checkpoint = set_weights(model, checkpoint["model_state_dict"], OLD_CHECKPOINT_CHANNELS, labels)
model.load_state_dict(updated_checkpoint, strict=True)
model = model.cuda()
run(model,
iterations, 
labels, 
label_weights, 
datasets,
voxel_size = voxel_size,
batch_size = batch_size, 
l_rate=l_rate,
log_dir=log_dir,
affinities = True, 
affinities_map = affinities_map,
)

