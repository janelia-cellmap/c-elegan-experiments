rm err.err
rm out.out
bsub -P cellmap -J op_aff_pred -n 12 -o out.out -e err.err python run_script.py