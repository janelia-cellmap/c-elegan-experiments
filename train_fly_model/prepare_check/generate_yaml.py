#%%
from fly_organelles.yaml_utils.yaml_generation import create_yaml_with_crops

input_path = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/datasets_all.yaml"
output_path = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/datasets_generated_all_diff_ress.yaml"

create_yaml_with_crops(input_path, output_path)
# %%
