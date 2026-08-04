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

SCRIPT=${1:-"finetune_visionfm_publicbench_fundus.sh"}
LR=${2:-"1e-3"}
DATA_ROOT=${3:-"/orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/"}

module purge
module load conda
conda activate vfm

echo $SLURM_JOBID

# Same 7 public fundus-photo datasets as MIRAGE's run_fundus_all_tasks_l4.sh
#   (OphFoundation's reference benchmark), on the same DATA_ROOT already
#   verified to work with that script -- added here to get the VisionFM
#   comparison requested for Fig 6a. Expected class counts, for a sanity
#   check against the "Auto-detected N classes" line each run logs:
#   Glaucoma_fundus:3 IDRiD_data:5 JSIEC:39 MESSIDOR2:5 PAPILA:3 Retina:4 APTOS2019:5
#DATASETS=(Glaucoma_fundus IDRiD_data JSIEC MESSIDOR2 PAPILA Retina APTOS2019)
DATASETS=(IDRiD_data)

#sbatch baseline_multirun_publicbench_fundus.sh finetune_visionfm_publicbench_fundus.sh 1e-3
for DATASET in "${DATASETS[@]}"
do
    echo "=== Dataset: ${DATASET} ==="
    bash $SCRIPT $DATASET $LR $DATA_ROOT
    sleep 1 # Optional: sleep to avoid overwhelming the scheduler
    echo "=== Dataset ${DATASET} done ==="
done
