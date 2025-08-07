#%%
from fly_organelles.yaml_utils.yaml_generation import create_yaml_with_crops

input_path = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train_fly_model/yamls/datasets_v2.yaml"
output_path = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train_fly_model/yamls/datasets_generated_4_datasets.yaml"

create_yaml_with_crops(input_path, output_path)
# %%
