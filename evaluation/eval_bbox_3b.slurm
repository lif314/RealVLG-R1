#!/bin/bash
#SBATCH -J RealVLG-Eval-Bbox
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH -n 7
#SBATCH -o slurm_logs/RealVLG_Eval_Bbox_%j.out
#SBATCH -e slurm_logs/RealVLG_Eval_Bbox_%j.err

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

model_path="/share/home/u11124/data_10t/llf_data/vlm_models/Qwen2.5-VL-3B-Instruct"
# model_path="outputs/RealVLG_Benchmark/grpo_RealVLG_GraspNet10P_Bbox_3B/global_step_220/actor/huggingface"
data_root="/share/home/u11124/ssd_datahome/llf_data/GraspNet_VLG"
output_dir="./outputs/evaluation/bbox/qwen3b"
python3 evaluation/eval_bbox.py \
    --model_path=$model_path \
    --data_root=$data_root \
    --output_dir=$output_dir