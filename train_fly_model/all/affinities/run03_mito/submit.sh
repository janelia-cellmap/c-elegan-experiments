rm -rf snapshots
rm error.log
rm output.log
bsub -J "aff_run3_16nm_mito" -P cellmap -n 12 -q gpu_h100 -gpu "num=1" -o output.log -e error.log python train.py