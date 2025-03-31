For the ckeckpoints check the same folder in `/groups/cellmap/cellmap/zouinkhim/c-elegen/v2`

# Dependencies:
- I forked Larissa repo fly-organelles repo for training : https://github.com/mzouink/fly-organelles
I adapted it to my yamls and i created a new pipeline for affinities.

- The repo is only using [gunpowder](https://github.com/funkey/gunpowder) for data loading and augmentation.

- No Dacapo is used in this project.

# Piplines
currently it does support a two data pipeline:

1 : pull from multiple organelles, binarize them and concat them into one target

2- pull from multiple organelles, create affinities and concat affinites into one target

To train you just need to submit the train.py script in each folder to the cluster.
```bash
bsub -J [RUN_NAME] -P [BILLING] -n 12 -q gpu_h100 -gpu "num=1" -o output.log -e error.log python train.py
```

# The runs: 
1. **op50_bw**: 9 initial runs for the two datasets (**op50** and **bw**) with different parameters and configurations. Resolutions used: 8nm and 16nm. Various learning rates and class weightings were tested.

2. **all**: Added the **comma_1** dataset. Two versions were created; version 2 (v2) is recommended.

3. **affinities**: Trained 4 runs of affinities for mitochondria to obtain instances, as it was not feasible to transition from semantic segmentation of mitochondria to instances directly.

4. **nuc_low_res**: Conducted two test runs at 32nm and 64nm resolutions for nuclei segmentation. These were exploratory runs to evaluate potential improvements in nuclei segmentation but were ultimately not used.





# The best models: 
Overall the the models trained at 16nm resolution performed better than those trained at 8nm.
For the 3 datasets we selected 
- for semantic 6 organelles (mito, ld, lyso, perox, yolk, nuc):
`/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train_fly_model/all/v2/run03/model_checkpoint_48000`
- for affinities for mitochondria to get instances, we used the following checkpoints:
`/groups/cellmap/cellmap/zouinkhim/c-elegen/v2/train/fly_run/all/affinities/run04_mito/model_checkpoint_75000`
#%%