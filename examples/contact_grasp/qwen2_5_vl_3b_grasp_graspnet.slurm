#!/bin/bash
#SBATCH -J easygrasp-vlg-GraspNet-3B
#SBATCH -p A800
#SBATCH -N 1
#SBATCH --gres=gpu:a800:8
#SBATCH -n 56
#SBATCH -o slurm_logs/qwen_gs_3b_%j.out
#SBATCH -e slurm_logs/qwen_gs_3b_%j.err

# GraspNet
set -e
mkdir -p slurm_logs

source /share/home/u11124/miniconda3/etc/profile.d/conda.sh
module load gcc/11.4.0
conda activate easygrasp

export PYTHONUNBUFFERED=1
export RAY_DISABLE_DASHBOARD=1
export RAY_raylet_start_wait_time_s=300

# wandb
export WANDB_API_KEY='3ea4f79a1f7c1b256156316dd61208a9ccd87fed'
export WANDB_MODE="offline"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 确认 torch + GPU 可用
nvidia-smi
python -c "import torch; print('Torch:', torch.__version__); print('CUDA devices:', torch.cuda.device_count()); print('Visible:', torch.cuda.get_device_name(0))"


set -x


# data_len: 26541
# ✅ Train objects: 214127, Test objects: 665
# Generating train split: 214127 examples [00:10, 20236.41 examples/s]
# Generating train split: 665 examples [00:00, 19792.31 examples/s]

project_name="EasyGrasp-VLG-GraspNet-Kinect"
# GAE = "gae"
# GRPO = "grpo"
# REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
# REMAX = "remax"
# RLOO = "rloo"
adv_estimator=grpo
experiment_name=qwen2_5_vl_3b_${project_name}_${adv_estimator}
MODEL_PATH=/share/home/u11124/data_10t/llf_data/vlm_models/Qwen2.5-VL-3B-Instruct
OUTDIR=/share/home/u11124/llf_workspace/EasyR1/outputs/${experiment_name}
DATA_PATH=/ssdfs/datahome/u11124/llf_data

# global_batch_size== verl's ppo_mini_batch_size
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files=${DATA_PATH}/train_gs_kn.parquet \
    data.val_files=${DATA_PATH}/test_gs_kn.parquet \
    data.format_prompt=./examples/format_prompt/contact_grasp_format.jinja \
    worker.reward.reward_function=./examples/reward_function/contact_grasp.py:compute_score \
    worker.reward.reward_type=batch \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.max_try_make_batch=512 \
    trainer.save_freq=2 \
    trainer.val_freq=2 \
    trainer.total_epochs=10 \
    trainer.logger='["console","wandb"]' \
    trainer.save_checkpoint_path=${OUTDIR}