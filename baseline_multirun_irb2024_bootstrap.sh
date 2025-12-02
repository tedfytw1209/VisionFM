#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8gb
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --output=%x.%j.out
#SBATCH --account=ruogu.fang
#SBATCH --qos=ruogu.fang

date;hostname;pwd

module load conda
conda activate vfm

SCRIPT=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL="vit_base"
FINETUNED_MODEL="VisionFM_OCT"
DATASET=$2
LR=${3:-"1e-3"}
NUM_CLASS=${4:-"2"}
weight_decay="0.05"
Eval_score="auc"
Modality=${5:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${6:-500} # 0, 500, 1000
NUM_K=0

echo $SLURM_JOBID

#sbatch baseline_multirun_irb2024_bootstrap.sh finetune_retfound_UFbenchmark_irb2024v5_bootstrap.sh AMD_all_split 1e-3 2 OCT 500
#sbatch baseline_multirun_irb2024_bootstrap.sh finetune_retfound_UFbenchmark_irb2024v5_bootstrap.sh DR_all_split 1e-3 6 Fundus 500
SUBSET_SEEDS=(1 2 3 4 5 6 7 8 9 10)
for i in "${!SUBSET_SEEDS[@]}"
do
    # Create a job name based on the variables
    SUBSETSEED="${SUBSET_SEEDS[$i]}"
    echo "Running dataset: $DATASET with seed $SUBSETSEED"
    # Submit the job to Slurm
    echo "bash $SCRIPT $DATASET $LR $NUM_CLASS $Modality $SUBSETNUM $SUBSETSEED"
    bash $SCRIPT $DATASET $LR $NUM_CLASS $Modality $SUBSETNUM $SUBSETSEED
    sleep 1 # Optional: sleep to avoid overwhelming the scheduler
done