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

# Fine-tune VisionFM on one public fundus-photo benchmark. The Python runner
# trains on train/, selects a checkpoint on val/, reloads it, and evaluates
# test/ once. This mirrors RETFound's normal split protocol without using the
# held-out test set for model selection.
DATASET=$1          # e.g. PAPILA, Glaucoma_fundus, IDRiD_data, JSIEC, MESSIDOR2, Retina, APTOS2019
LR=${2:-"1e-3"}
DATA_ROOT=${3:-"/orange/ruogu.fang/tienyuchang/OCTRFF_Data/benchmark/"}

MODEL="vit_base"
FINETUNED_MODEL="VisionFM_Fundus"
Modality="Fundus"
Epochs=100
OPTIMIZER="adamw"
BATCH_SIZE=64
Eval_score="retfound_f1_auc_kappa"
data_type="publicbench"

DATA_PATH="${DATA_ROOT%/}/${DATASET}"
WEIGHTS="/orange/ruogu.fang/tienyuchang/VisionFM_pretrain/VFM_${Modality}_weights.pth"
TAG="${DATASET}-${data_type}-${FINETUNED_MODEL}-bs${BATCH_SIZE}ep${Epochs}lr${LR}opt${OPTIMIZER}-${Eval_score}eval"

echo $SLURM_JOBID
echo "Dataset: $DATASET, data_path: $DATA_PATH"

#sbatch finetune_visionfm_publicbench_fundus.sh PAPILA 1e-3
python finetune_visionfm_publicbench_fundus.py --pretrained_weights $WEIGHTS --arch $MODEL --avgpool_patchtokens 0 --input_size 224 --lr $LR --epochs $Epochs --output_dir ./results/$TAG --data_path $DATA_PATH --task $TAG --modality $Modality --num_workers 8 --batch_size_per_gpu $BATCH_SIZE
