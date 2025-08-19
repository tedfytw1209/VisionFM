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
conda activate /blue/guoj1/tienyuchang/.conda/envs/vfm
# Go to home directory
#cd $HOME
STUDY=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL=${2:-"RETFound_mae"}
FINETUNED_MODEL=${3:-"RETFound_mae_natureOCT"}
LR=${4:-"5e-4"}
Num_CLASS=${5:-"2"}
weight_decay=${6:-"0.05"}
Eval_score=${7:-"default"}
Modality=${8:-"OCT"} # CFP, OCT, OCT_CFP
ADDCMD=${9:-""}
ADDCMD2=${10:-""}

NUM_K=0
data_type="IRB2024_v4"
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
Pretrain_Fname="/orange/ruogu.fang/tienyuchang/VisionFM_pretrain/VFM_OCT_weights.pth"
Epochs=50
OPTIMIZER="adamw" # "adamw" or "sgd"
BATCH_SIZE=16

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT finetune_visionfm_for_multiclass_classification_UF.py --pretrained_weights $Pretrain_Fname --arch vit_base --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/$STUDY-${data_type}-all-$FINETUNED_MODEL-${Modality}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval-$ADDCMD-$ADDCMD2 --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${data_type}/split/tune5-eval5/${STUDY}.csv --task $STUDY-${data_type}-all-$FINETUNED_MODEL-${Modality}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval-$ADDCMD-$ADDCMD2 --modality $Modality --num_workers 8 --batch_size_per_gpu $BATCH_SIZE --num_labels $Num_CLASS --extra 10
