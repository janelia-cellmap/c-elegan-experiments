rm err.err
rm out.out
rm -rf daisy_logs
bsub -P cellmap -J nuc_bw -n 4 -o out.out -e err.err python run_script.py