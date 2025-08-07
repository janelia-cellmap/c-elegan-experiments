#%%
import yaml
import time
import numpy as np
import matplotlib.pyplot as plt
import gunpowder as gp
from fly_organelles.train import make_distance_data_pipeline
#%%
labels = ['mito']
# labels = ['mito', 'ld', 'lyso', 'perox', 'yolk', 'nuc']

yaml_file = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train_fly_model/yamls/datasets_generated_4_datasets.yaml"



voxel_size = (16, 16, 16)

with open(yaml_file, "r") as data_yaml:
    datasets = yaml.safe_load(data_yaml)
# label_stores, raw_stores, crop_copies = read_data_yaml(data_yaml)


input_size = gp.Coordinate((178, 178, 178)) * gp.Coordinate(voxel_size)
output_size = gp.Coordinate((56, 56, 56)) * gp.Coordinate(voxel_size)
displacement_sigma = gp.Coordinate((24, 24, 24))
# max_in_request = gp.Coordinate((np.ceil(np.sqrt(sum(input_size**2))),)*len(input_size)) + displacement_sigma * 6
max_out_request = (
    gp.Coordinate((np.ceil(np.sqrt(sum(output_size**2))),) * len(output_size)) + displacement_sigma * 6
)
pad_width_out = output_size / 2.0

pipeline = make_distance_data_pipeline(
    labels,
    datasets,
    pad_width_out,
    voxel_size,
    max_out_request,
    displacement_sigma,
    1
)
#%%
def generate_batches(pipeline, input_size, output_size, voxel_size):
    with gp.build(pipeline) as pp:
        while True:
            request = gp.BatchRequest()
            request.add(gp.ArrayKey("RAW"), input_size, voxel_size=gp.Coordinate(voxel_size))
            request.add(gp.ArrayKey("LABELS"), output_size, voxel_size=gp.Coordinate(voxel_size))
            request.add(gp.ArrayKey("MASK"), output_size, voxel_size=gp.Coordinate(voxel_size))
            batch = pp.request_batch(request)
            yield batch
#%%
batch_generator = generate_batches(pipeline, input_size, output_size, voxel_size)
#%%
start_time = time.time()
batch = next(batch_generator)
end_time = time.time()
print(f"Time taken to generate batch: {end_time - start_time} seconds")
raw = batch[gp.ArrayKey("RAW")].data
print(f"raw shape: {raw.shape} min: {raw.min()} max: {raw.max()}")
gt = batch[gp.ArrayKey("LABELS")].data
print(f"gt shape: {gt.shape} min: {gt.min()} max: {gt.max()}")



raw_example = raw[0, 0, :, :,89]
gt_example = gt[0, 0, :, :,28]
print(gt_example.shape)

# Pad gt_example to fit raw_example, centered
full_gt_example = np.zeros(raw_example.shape)
print(full_gt_example.shape)


full_gt_example[raw_example.shape[0]//2-gt_example.shape[0]//2:raw_example.shape[0]//2+gt_example.shape[0]//2,
                raw_example.shape[1]//2-gt_example.shape[1]//2:raw_example.shape[1]//2+gt_example.shape[1]//2] = gt_example[:]
#%%
plt.imshow(raw_example, cmap="gray")
plt.imshow(full_gt_example, cmap = "gray",alpha=0.5)
plt.show()
# %%
gt_example.shape
# %%
