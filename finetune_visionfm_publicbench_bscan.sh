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
conda activate vfm

# Single public OCT B-scan benchmark dataset + fold: finetune VisionFM (OCT
#   encoder) then evaluate. inference_visionfm_publicbench_bscan.py reports
#   the 'val' split under the name "test" -- these fold CSVs have no true
#   held-out test split, matching MIRAGE's run_cls_tuning_bscan.py behavior
#   for comparability. Data loading is PublicOCTBscanDataset
#   (dataset_public_oct.py), ported from MIRAGE's mutils/dataset_public_oct.py
#   and already verified working against these datasets on the Linux server.
DATASET=$1          # duke14, glaucoma, oimhs, umn
FOLD=${2:-0}        # 0-9
LR=${3:-"1e-3"}
DATA_ROOT=${4:-"/orange/ruogu.fang/tienyuchang/OCTCubeM/assets/ext_oph_datasets/"}
CSV_ROOT=${5:-"/blue/ruogu.fang/tienyuchang/OphFoundation/Public_OCT_split/"}

MODEL="vit_base"
FINETUNED_MODEL="VisionFM_OCT"
Modality="OCT"
Epochs=100
OPTIMIZER="adamw"
BATCH_SIZE=64
Eval_score="auc"
data_type="publicbench_bscan"

WEIGHTS="/orange/ruogu.fang/tienyuchang/VisionFM_pretrain/VFM_${Modality}_weights.pth"
TAG="${DATASET}-fold${FOLD}-${data_type}-${FINETUNED_MODEL}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval"

echo $SLURM_JOBID
echo "Dataset: $DATASET, Fold: $FOLD"

#sbatch finetune_visionfm_publicbench_bscan.sh duke14 0 1e-3
python finetune_visionfm_publicbench_bscan.py --pretrained_weights $WEIGHTS --arch $MODEL --avgpool_patchtokens 0 --input_size 224 --lr $LR --epochs $Epochs --output_dir ./results/$TAG --data_root $DATA_ROOT --csv_root $CSV_ROOT --data_set $DATASET --fold $FOLD --task $TAG --modality $Modality --num_workers 8 --batch_size_per_gpu $BATCH_SIZE

echo "Test"

python inference_visionfm_publicbench_bscan.py --pretrained_weights ./results/$TAG/checkpoint_best_finetune.pth --arch $MODEL --avgpool_patchtokens 0 --input_size 224 --output_dir ./results/${TAG}_test --data_root $DATA_ROOT --csv_root $CSV_ROOT --data_set $DATASET --fold $FOLD --task ${TAG}_test --modality $Modality --num_workers 8 --batch_size_per_gpu $BATCH_SIZE
