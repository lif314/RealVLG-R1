#!/bin/bash
#SBATCH -J RealVLG_GraspNet10P_SAM_3B_GSPO
#SBATCH -p A800
#SBATCH -N 1
#SBATCH --gres=gpu:a800:4
#SBATCH -n 28
#SBATCH -o slurm_logs/RealVLG_GraspNet10P_SAM_3B_GSPO_%j.out
#SBATCH -e slurm_logs/RealVLG_GraspNet10P_SAM_3B_GSPO_%j.err

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


project_name="RealVLG_GraspNet10P_SAM_3B_GSPO"
# GAE = "gae"
# GRPO = "grpo"
# REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
# REMAX = "remax"
# RLOO = "rloo"
adv_estimator=gspo
experiment_name=${adv_estimator}_${project_name}
MODEL_PATH=/share/home/u11124/data_10t/llf_data/vlm_models/Qwen2.5-VL-3B-Instruct
OUTDIR=/share/home/u11124/llf_workspace/RealVLG-R1/outputs/RealVLG_Benchmark/${experiment_name}
DATA_PATH=/share/home/u11124/ssd_datahome/llf_data

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${DATA_PATH}/data/train_gs_kn_10p.parquet \
    data.val_files=${DATA_PATH}/data/test_gs_kn.parquet \
    data.format_prompt=./examples/format_prompt/bbox_sam_format.jinja \
    worker.reward.reward_function=./examples/reward_function/bbox_sam_mask_path.py:compute_score \
    worker.reward.reward_type=batch \
    worker.reward.use_sam_model=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.loss_type=gspo_token \
    worker.actor.loss_avg_mode=seq \
    worker.actor.clip_ratio_low=3e-4 \
    worker.actor.clip_ratio_high=4e-4 \
    algorithm.disable_kl=True \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.max_try_make_batch=128 \
    trainer.save_freq=20 \
    trainer.val_freq=20 \
    trainer.total_epochs=10 \
    trainer.logger='["console","wandb"]' \
    trainer.save_checkpoint_path=${OUTDIR}