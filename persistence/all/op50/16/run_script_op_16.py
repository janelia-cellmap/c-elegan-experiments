#%%
data_path = "/nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/em/fibsem-int16/s2"
checkpoint = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/train_fly_model/all/v2/run03/model_checkpoint_48000"
output_path = "/nrs/cellmap/zouinkhim/c-elegan/predictions/v2_jrc_c-elegans-op50-1.zarr/run03_16"
channels = ['mito', 'ld', 'lyso', 'perox', 'yolk', 'nuc']
res = (16,16,16)
run_name = "op_16"
#%%
import cellmap_flow.globals as g
from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from cellmap_flow.post.postprocessors import DefaultPostprocessor
from cellmap_flow.blockwise import CellMapFlowBlockwiseProcessor
g.input_norms = [MinMaxNormalizer(5000,6000)]
g.postprocess = []
from cellmap_flow.utils.data import FlyModelConfig
# %%
model_config = FlyModelConfig(
            chpoint_path=checkpoint,
            channels=channels,
            input_voxel_size=res,
            output_voxel_size=res,
            name=run_name,
        )
process = CellMapFlowBlockwiseProcessor(data_path,model_config, output_path, create=True)
process.run()
# %%
