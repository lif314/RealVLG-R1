#!/bin/bash
#SBATCH -J RealVLG-Model-Merge
#SBATCH -N 1
#SBATCH -n 7
#SBATCH -o slurm_logs/model_merge_%j.out
#SBATCH -e slurm_logs/model_merge_%j.err

set -e
mkdir -p slurm_logs

source /share/home/u11124/miniconda3/etc/profile.d/conda.sh
module load gcc/11.4.0
conda activate easygrasp

# local_dir=outputs/RealVLG_Benchmark/done_grpo_RealVLG_GraspNet10P_Bbox_7B/global_step_140/actor
# local_dir=outputs/RealVLG_Benchmark/done_grpo_RealVLG_GraspNet10P_Contact_7B/global_step_140/actor
# local_dir=outputs/RealVLG_Benchmark/done_grpo_RealVLG_GraspNet10P_Grasp_3B/global_step_240/actor
# local_dir=outputs/RealVLG_Benchmark/done_grpo_RealVLG_GraspNet10P_Grasp_7B/global_step_120/actor
# local_dir=outputs/RealVLG_Benchmark/done_gspo_RealVLG_GraspNet10P_Bbox_3B_GSPO/global_step_160/actor
# local_dir=outputs/RealVLG_Benchmark/done_gspo_RealVLG_GraspNet10P_Bbox_7B_GSPO/global_step_180/actor
# local_dir=outputs/RealVLG_Benchmark/grpo_RealVLG_GraspNet10P_Grasp_3B_GSPO/global_step_120/actor
# local_dir=outputs/RealVLG_Benchmark/done_gspo_RealVLG_GraspNet10P_Contact_7B_GSPO/global_step_140/actor
local_dir=outputs/RealVLG_Benchmark/grpo_RealVLG_GraspNet10P_Grasp_7B_GSPO/global_step_200/actor
python3 scripts/model_merger.py \
         --local_dir $local_dir