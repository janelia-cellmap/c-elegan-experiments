#%%
data_path = "/nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/em/fibsem-int16/s3"
checkpoint = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/nuc/run01_32/continue/model_checkpoint_12000"
output_path = "/nrs/cellmap/zouinkhim/c-elegan/predictions/jrc_c-elegans-op50-1_result_tmp.zarr/nuc_32"
channels = ['nuc']
res = (32,32,32)
run_name = "nuc_32"
#%%
import cellmap_flow.globals as g
from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from cellmap_flow.post.postprocessors import DefaultPostprocessor
from cellmap_flow.blockwise import CellMapFlowBlockwiseProcessor
g.input_norms = [MinMaxNormalizer(5000,6000)]
g.postprocess = []
# g.postprocess = [DefaultPostprocessor()]
from cellmap_flow.utils.data import FlyModelConfig
# %%
model_config = FlyModelConfig(
            chpoint_path=checkpoint,
            channels=channels,
            input_voxel_size=res,
            output_voxel_size=res,
            name=run_name,
        )
process = CellMapFlowBlockwiseProcessor(data_path,model_config, output_path)
process.run()
# %%
