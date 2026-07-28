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

SCRIPT=${1:-"finetune_visionfm_publicbench_bscan.sh"}
FOLD=${2:-0}
LR=${3:-"1e-3"}
DATA_ROOT=${4:-"/orange/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/"}
CSV_ROOT=${5:-"/blue/ruogu.fang/tienyuchang/OphFoundation/Public_OCT_split/"}

module purge
module load conda
conda activate vfm310

echo $SLURM_JOBID

# Same 4 public OCT B-scan datasets as MIRAGE's run_bscan_multirun.sh
#   (OphFoundation's reference benchmark), on the same DATA_ROOT/CSV_ROOT
#   already verified to work with that script -- added here to get the
#   VisionFM comparison requested for Fig 6a. Single fold (default 0); use
#   baseline_allfolds_publicbench_bscan.sh instead to sweep all 10 folds for
#   one dataset at a time (mean +/- std across folds).
DATASETS=(duke14 glaucoma oimhs umn)

#sbatch baseline_multirun_publicbench_bscan.sh finetune_visionfm_publicbench_bscan.sh 0 1e-3
for DATASET in "${DATASETS[@]}"
do
    echo "=== Dataset: ${DATASET} (fold ${FOLD}) ==="
    bash $SCRIPT $DATASET $FOLD $LR $DATA_ROOT $CSV_ROOT
    sleep 1 # Optional: sleep to avoid overwhelming the scheduler
    echo "=== Dataset ${DATASET} done ==="
done
