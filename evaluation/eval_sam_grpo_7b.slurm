#!/bin/bash
#SBATCH -J RealVLG-Eval-SAM
#SBATCH -p L40
#SBATCH -N 1
#SBATCH --gres=gpu:l40:1
#SBATCH -n 7
#SBATCH -o slurm_logs/RealVLG_Eval_SAM_%j.out
#SBATCH -e slurm_logs/RealVLG_Eval_SAM_%j.err

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


data_root="/share/home/u11124/ssd_datahome/llf_data/GraspNet_VLG"
sam_path="./third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
# model_path="/share/home/u11124/data_10t/llf_data/vlm_models/Qwen2.5-VL-3B-Instruct"
# model_path=outputs/RealVLG_Benchmark/done_grpo_RealVLG_GraspNet10P_Bbox_3B/global_step_220/actor/huggingface
model_path=outputs/RealVLG_Benchmark/done_grpo_RealVLG_GraspNet10P_Bbox_7B/global_step_140/actor/huggingface
output_dir="./outputs/evaluation/sam/RealVLG-GRPO-7B"
python3 evaluation/eval_sam.py \
    --model_path=$model_path \
    --sam_model_path=$sam_path \
    --data_root=$data_root \
    --output_dir=$output_dir
