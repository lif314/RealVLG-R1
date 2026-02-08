#!/bin/bash
#SBATCH -J easygrasp-vlg-BBOX-COV-GRPO
#SBATCH -p A800
#SBATCH -N 1
#SBATCH --gres=gpu:a800:4
#SBATCH -n 28
#SBATCH -o slurm_logs/qwen3b_grpo_cov_bbox_%j.out
#SBATCH -e slurm_logs/qwen3b_grpo_cov_bbox_%j.err

# VMRD_Cornell_OCID
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



# Cornell_VLG: 885 samples
# VMRD_VLG: 13876 samples
# OCID_VLG: 11780 samples
# Train size: 25656, Test size: 885

project_name="EasyGrasp-Qwen3B-GRPO-VLG-COV-BBOX"
# GAE = "gae"
# GRPO = "grpo"
# REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
# REMAX = "remax"
# RLOO = "rloo"
adv_estimator=grpo
experiment_name=qwen2_5_vl_3b_${project_name}_${adv_estimator}
MODEL_PATH=/share/home/u11124/data_10t/llf_data/vlm_models/Qwen2.5-VL-3B-Instruct
OUTDIR=/share/home/u11124/llf_workspace/RealVLG-R1/outputs/${experiment_name}
DATA_PATH=/share/home/u11124/data_10t/llf_data/vlg_dataset/RealVLG_Dataset

# global_batch_size== verl's ppo_mini_batch_size
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files=${DATA_PATH}/train_cov_bbox.parquet \
    data.val_files=${DATA_PATH}/test_cov_bbox.parquet \
    data.format_prompt=./examples/format_prompt/bbox_format.jinja \
    worker.reward.reward_function=./examples/reward_function/bbox.py:compute_score \
    worker.reward.reward_type=batch \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.val_freq=2 \
    trainer.total_epochs=20 \
    trainer.max_try_make_batch=128 \
    trainer.logger='["console","wandb"]' \
    trainer.save_checkpoint_path=${OUTDIR}