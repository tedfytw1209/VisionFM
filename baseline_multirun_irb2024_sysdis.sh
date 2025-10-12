#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4gb
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
LR=${2:-"1e-3"}
Num_CLASS=${3:-"2"}
weight_decay="0.05"
Eval_score="auc"
Modality=${4:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${5:-0} # 0, 500, 1000

NUM_K=0

#bash baseline_multirun_irb2024.sh finetune_retfound_UFbenchmark_irb2024v5.sh 1e-3 2 OCT
#bash baseline_multirun_irb2024.sh finetune_retfound_UFbenchmark_irb2024v5.sh 1e-3 2 Fundus
DATASETS=(PD_all_split DKD_all_split Diabetes_all_split) 
CLASSES=(2 2 2)  # Number of classes for each dataset
for i in "${!DATASETS[@]}"
do
    # Create a job name based on the variables
    DATASET="${DATASETS[$i]}"
    NUM_CLASS="${CLASSES[$i]}"
    echo "bash $SCRIPT $DATASET $LR $NUM_CLASS $Modality $SUBSETNUM"
    bash $SCRIPT $DATASET $LR $NUM_CLASS $Modality $SUBSETNUM
    sleep 1 # Optional: sleep to avoid overwhelming the scheduler
done