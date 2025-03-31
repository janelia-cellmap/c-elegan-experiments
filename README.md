
# C-elegan experiments:
The goal of the project was to train a segmentation model for for the 3 datasets:
- **jrc_c-elegans-op50-1**
- **jrc_c-elegans-bw-1**
- **jrc_c-elegans-comma-1**

Organelles to segment:
- **mito**
- **ld**
- **lyso**
- **perox**
- **yolk**
- **nuc**

## PS:
- Each folder containes a README file explaining what have been done and how to run the code.
- The checkpoints are  not uploaded in the repo due to their size but can be found in the local folders: `/groups/cellmap/cellmap/zouinkhim/c-elegen/v2`

# Dependencies:
- Cellmap-flow for visualization use **fly branch** for more functionalities. :
https://github.com/janelia-cellmap/cellmap-flow/tree/fly
- fly-organelles repo for training : https://github.com/mzouink/fly-organelles

# Parts of the project:
## 1- train_fly_model:
Contains all the runs. 
Runs: 
1. **op50_bw**: 9 initial runs for the two datasets (**op50** and **bw**) with different parameters and configurations. Resolutions used: 8nm and 16nm. Various learning rates and class weightings were tested.

2. **all**: Added the **comma_1** dataset. Two versions were created; version 2 (v2) is recommended.

3. **affinities**: Trained 4 runs of affinities for mitochondria to obtain instances, as it was not feasible to transition from semantic segmentation of mitochondria to instances directly.

4. **nuc_low_res**: Conducted two test runs at 32nm and 64nm resolutions for nuclei segmentation. These were exploratory runs to evaluate potential improvements in nuclei segmentation but were ultimately not used.

## 2- flow_visualization:
To select the best model i generate a cellmap flow with a variety of models and i give to to annotators to decide if the result is good or not. if not, i train more. and if good, they recommend me which model to select.

## 3- persistence:
When the model is selected and the postprocessing steps are decided. i generate a blockwise processor script that will process the whole volume blockwise using multiple cluster jobs. usually the compting (inference+postprocessing) takes less than 30 min

## 4- validation_scores
Graphs of scores for each dataset cross all the organelles / crops. and the scripts used for it.
