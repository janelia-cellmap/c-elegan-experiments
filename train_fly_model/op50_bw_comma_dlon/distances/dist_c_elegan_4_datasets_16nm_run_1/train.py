#%%
import yaml
import torch
import logging
from fly_organelles.run import run
from fly_organelles.model import StandardUnet

from fly_organelles.run import set_weights

logger = logging.getLogger(__name__)

log_dir = "/nrs/cellmap/zouinkhim/tensorboard/dist_c_elegan_4_datasets_16nm_run_1"

CHECKPOINT_PATH = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train_fly_model/all/v2/run03/continue/model_checkpoint_85000"
OLD_CHECKPOINT_CHANNELS = ['mito', 'ld', 'lyso', 'perox', 'yolk', 'nuc']
labels = ['mito']

yaml_file = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train_fly_model/yamls/datasets_generated_4_datasets.yaml"
iterations = 1000000

label_weights = [1]
voxel_size = (16, 16, 16)
l_rate = 0.5e-5
batch_size = 14

with open(yaml_file, "r") as data_yaml:
    datasets = yaml.safe_load(data_yaml)
model = StandardUnet(len(labels))
checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True, map_location=torch.device('cpu'))
updated_checkpoint = set_weights(model, checkpoint["model_state_dict"], OLD_CHECKPOINT_CHANNELS, labels)
model.load_state_dict(updated_checkpoint, strict=True)
run(model,
iterations, 
labels, 
label_weights, 
datasets,
voxel_size = voxel_size,
batch_size = batch_size, 
l_rate=l_rate,
log_dir=log_dir,
distance=True)

