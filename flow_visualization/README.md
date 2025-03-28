# Visualisation using CellMap-flow
# P.S. 
- Run flow command in an interactive job
- If inference is bad, change MinMaxInputNormalizer
- when opening cellmap flow you need to specify the right resolution dataset path `.../s2` or `.../s1`


I created a branch to run Fly Model using CellMap-flow. The branch is named `fly`.
```bash
pip install git+https://github.com/janelia-cellmap/cellmap-flow.git@fly
```

After installing CellMap-flow, I ran the following command to generate the flow:

```bash
cellmap_flow_fly all_op50.yaml

```
The command will submit a job for each model. then group the results and result  a neuroglancer link.


Or you can visualize using custom code 
e.g. `model_spec_16.py` 
```bash
cellmap_flow script -s model_spec_16.py -d /nrs/cellmap/data/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1_downscaled.zarr/recon-1/em/fibsem-uint8/s1 -P cellmap
```

# Input Normalization:
### jrc_c-elegans-op50-1:
- Min : 5000
- Max : 6000
- raw path : /nrs/cellmap/data/jrc_c-elegans-op50-1/jrc_c-elegans-op50-1.zarr/recon-1/em/fibsem-int16
### jrc_c-elegans-bw-1:
- Min : 5000
- Max : 6000
- raw path : /nrs/cellmap/data/jrc_c-elegans-bw-1/jrc_c-elegans-bw-1.zarr/recon-1/em/fibsem-int16
### jrc_c-elegans-comma-1:
PS : It is imaged at 6nm, I used a downsampled version (8nm, 16 nm, ...).
- Min : 0
- Max : 255
- raw path : /nrs/cellmap/data/jrc_c-elegans-comma-1/jrc_c-elegans-comma-1_downscaled.zarr/recon-1/em/fibsem-uint8