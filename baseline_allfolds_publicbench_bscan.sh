#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

# This script handles exactly ONE dataset per submission (passed as $1),
#   looping over all 10 pre-computed fold-split partitions (0-9) so results
#   can be reported as mean +/- std across folds instead of a single fold-0
#   point estimate -- mirrors MIRAGE's run_bscan_all_tasks_l4.sh.
DATASET=$1          # duke14, glaucoma, oimhs, umn
LR=${2:-"1e-3"}
DATA_ROOT=${3:-"/blue/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/"}
CSV_ROOT=${4:-"/blue/ruogu.fang/tienyuchang/OphFoundation/Public_OCT_split/"}
SCRIPT=${5:-"finetune_visionfm_publicbench_bscan.sh"}

FOLDS=(0 1 2 3 4 5 6 7 8 9)

echo $SLURM_JOBID
echo "=== Dataset: ${DATASET} ==="

#sbatch baseline_allfolds_publicbench_bscan.sh duke14 1e-3
for FOLD in "${FOLDS[@]}"; do
    echo "--- Fold: ${FOLD} ---"
    bash $SCRIPT $DATASET $FOLD $LR $DATA_ROOT $CSV_ROOT
    sleep 1 # Optional: sleep to avoid overwhelming the scheduler
    echo "--- Fold ${FOLD} done ---"
done
echo "=== Dataset ${DATASET} done ==="
