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

SCRIPT=${1:-"baseline_allfolds_publicbench_bscan.sh"}
LR=${2:-"1e-3"}
DATA_ROOT=${3:-"/blue/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/"}
CSV_ROOT=${4:-"/blue/ruogu.fang/tienyuchang/OphFoundation/Public_OCT_split/"}

echo $SLURM_JOBID

# Same 4 public OCT B-scan datasets as MIRAGE's run_bscan_multirun.sh
#   (OphFoundation's reference benchmark), on the same DATA_ROOT/CSV_ROOT
#   already verified to work with that script -- added here to get the
#   VisionFM comparison requested for Fig 6a. Submits one independent sbatch
#   job per dataset (SCRIPT defaults to baseline_allfolds_publicbench_bscan.sh),
#   which in turn sweeps all 10 folds (0-9) for that dataset within its own
#   job allocation.
DATASETS=(duke14 glaucoma oimhs umn)

#sbatch baseline_multirun_publicbench_bscan.sh baseline_allfolds_publicbench_bscan.sh 1e-3
for DATASET in "${DATASETS[@]}"
do
    echo "=== Submitting Dataset: ${DATASET} (all folds) ==="
    sbatch $SCRIPT $DATASET $LR $DATA_ROOT $CSV_ROOT
    sleep 1 # Optional: sleep to avoid overwhelming the scheduler
done
