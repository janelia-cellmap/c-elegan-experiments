rm err.err
rm out.out
rm -rf daisy_logs
bsub -P cellmap -J comm_aff_pred -n 4 -o out.out -e err.err python run_script.py