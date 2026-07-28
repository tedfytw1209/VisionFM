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

# Single public fundus-photo benchmark dataset: finetune VisionFM (Fundus
#   encoder) then evaluate on the held-out test split, using
#   finetune/inference_visionfm_publicbench_fundus.py -- ImageFolder-based
#   data loading with num_classes auto-inferred from the folder structure,
#   the same convention already verified working for these datasets via
#   MIRAGE's run_cls_tuning_fundus.py on this same DATA_ROOT. No per-dataset
#   class-folder-name hardcoding, unlike RETFoundDataset in
#   finetune_visionfm_for_multiclass_classification_UF.py.
DATASET=$1          # e.g. PAPILA, Glaucoma_fundus, IDRiD_data, JSIEC, MESSIDOR2, Retina, APTOS2019
LR=${2:-"1e-3"}
DATA_ROOT=${3:-"/orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/"}

MODEL="vit_base"
FINETUNED_MODEL="VisionFM_Fundus"
Modality="Fundus"
Epochs=100
OPTIMIZER="adamw"
BATCH_SIZE=64
Eval_score="auc"
data_type="publicbench"

DATA_PATH="${DATA_ROOT%/}/${DATASET}"
WEIGHTS="/orange/ruogu.fang/tienyuchang/VisionFM_pretrain/VFM_${Modality}_weights.pth"
TAG="${DATASET}-${data_type}-${FINETUNED_MODEL}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval"

echo $SLURM_JOBID
echo "Dataset: $DATASET, data_path: $DATA_PATH"

#sbatch finetune_visionfm_publicbench_fundus.sh PAPILA 1e-3
python finetune_visionfm_publicbench_fundus.py --pretrained_weights $WEIGHTS --arch $MODEL --avgpool_patchtokens 0 --input_size 224 --lr $LR --epochs $Epochs --output_dir ./results/$TAG --data_path $DATA_PATH --task $TAG --modality $Modality --num_workers 8 --batch_size_per_gpu $BATCH_SIZE

echo "Test"

python inference_visionfm_publicbench_fundus.py --pretrained_weights ./results/$TAG/checkpoint_best_finetune.pth --arch $MODEL --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/${TAG}_test --data_path $DATA_PATH --task ${TAG}_test --modality $Modality --num_workers 8 --batch_size_per_gpu $BATCH_SIZE
