#!/bin/bash
#SBATCH -J RealVLG-Infer
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH -n 7
#SBATCH -o slurm_logs/infer_bbox_%j.out
#SBATCH -e slurm_logs/infer_bbox_%j.err

# VMRD_Cornell_OCID
set -e
mkdir -p slurm_logs

source /share/home/u11124/miniconda3/etc/profile.d/conda.sh
module load gcc/11.4.0
conda activate easygrasp

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 确认 torch + GPU 可用
nvidia-smi
python -c "import torch; print('Torch:', torch.__version__); print('CUDA devices:', torch.cuda.device_count()); print('Visible:', torch.cuda.get_device_name(0))"


set -x



MODEL_PATH=/share/home/u11124/data_10t/llf_data/vlm_models/Qwen2.5-VL-3B-Instruct
# MODEL_PATH=pretrain_models/bbox/qwen3b_cov_grpo_global_step_106/actor/huggingface

python3 inference_scripts/bbox_infer.py \
    --model_path=$MODEL_PATH