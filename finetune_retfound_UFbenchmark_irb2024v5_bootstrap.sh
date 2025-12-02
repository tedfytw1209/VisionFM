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

module purge
module load conda
conda activate vfm310
# Go to home directory
#cd $HOME
STUDY=$1 #AMD_all_split 2, Cataract_all_split 2, DR_all_split 6, Glaucoma_all_split 6, DR_binary_all_split 2, Glaucoma_binary_all_split 2
MODEL="vit_base"
FINETUNED_MODEL="VisionFM_OCT"
LR=${2:-"1e-3"}
Num_CLASS=${3:-"2"}
weight_decay="0.05"
Eval_score="auc"
Modality=${4:-"OCT"} # CFP, OCT, OCT_CFP
SUBSETNUM=${5:-0} # 0, 500, 1000
SUBSETSEED=${6:-42} # Seed for subset sampling

NUM_K=0
data_type="IRB2024_v5"
IMG_Path="/orange/ruogu.fang/tienyuchang/IRB2024_imgs_paired/"
Epochs=100
OPTIMIZER="adamw" # "adamw" or "sgd"
BATCH_SIZE=128

MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo $SUBSTUDY
echo $Num_CLASS

# Modify the path to your singularity container 
#sbatch finetune_retfound_UFbenchmark_irb2024v5.sh AMD_all_split 1e-3 2 OCT
python finetune_visionfm_for_multiclass_classification_UF.py --pretrained_weights /orange/ruogu.fang/tienyuchang/VisionFM_pretrain/VFM_${Modality}_weights.pth --arch $MODEL --avgpool_patchtokens 0 --input_size 224 --lr $LR --epochs $Epochs --output_dir ./results/$STUDY-${data_type}-$FINETUNED_MODEL-${Modality}-subtr${SUBSETNUM} --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${data_type}/split/tune5-eval5/${STUDY}.csv --task $STUDY-${data_type}-$FINETUNED_MODEL-subtr${SUBSETNUM} --modality $Modality --num_workers 8 --batch_size_per_gpu $BATCH_SIZE --num_labels $Num_CLASS --extra 10 --img_dir $IMG_Path --new_subset_num $SUBSETNUM --subsetseed $SUBSETSEED --bootstrap_runs

echo "Test"

python inference_visionfm_for_multiclass_classification_UF.py --pretrained_weights ./results/$STUDY-${data_type}-$FINETUNED_MODEL-${Modality}-subtr${SUBSETNUM}/seed_$SUBSETSEED/checkpoint_best_finetune.pth --arch $MODEL --avgpool_patchtokens 0 --input_size 224 --lr $LR --epochs $Epochs --output_dir ./results/$STUDY-${data_type}-$FINETUNED_MODEL-${Modality}-subtr${SUBSETNUM}_test --data_path /orange/ruogu.fang/tienyuchang/OCTRFF_Data/data/UF-cohort/${data_type}/split/tune5-eval5/${STUDY}.csv --task $STUDY-${data_type}-$FINETUNED_MODEL-${Modality}-subtr${SUBSETNUM}_test --modality $Modality --num_workers 8 --batch_size_per_gpu $BATCH_SIZE --num_labels $Num_CLASS --extra 10 --img_dir $IMG_Path --new_subset_num $SUBSETNUM --subsetseed $SUBSETSEED --bootstrap_runs
