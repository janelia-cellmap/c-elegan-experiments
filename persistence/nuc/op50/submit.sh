rm err.err
rm out.out
bsub -P cellmap -J nuc_op -n 4 -o out.out -e err.err python run_script.py