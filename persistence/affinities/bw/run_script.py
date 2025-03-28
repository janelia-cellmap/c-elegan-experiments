#%%
data_path = "/nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/em/fibsem-int16/s2"
checkpoint = "/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/all/affinities/new/run04_mito/model_checkpoint_75000"
output_path = "/nrs/cellmap/zouinkhim/c-elegan/predictions/v2__jrc_c-elegans-bw-1.zarr/mito_affinities"
channels = [f'mito_{i}' for i in range(9)]
res = (16,16,16)
run_name = "mito_aff"
#%%
import cellmap_flow.globals as g
from cellmap_flow.norm.input_normalize import MinMaxNormalizer
from cellmap_flow.post.postprocessors import DefaultPostprocessor
from cellmap_flow.blockwise import CellMapFlowBlockwiseProcessor
g.input_norms = [MinMaxNormalizer(0,255)]
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
model_config.model
process = CellMapFlowBlockwiseProcessor(data_path,model_config, output_path)
process.run()
# %%
