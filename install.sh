#!/bin/bash
#SBATCH -J install
#SBATCH -p A800
#SBATCH -N 1
#SBATCH --gres=gpu:a800:1
#SBATCH -n 7
#SBATCH -o slurm_logs/install_%j.out
#SBATCH -e slurm_logs/install_%j.err

set -e

source /share/home/u11124/miniconda3/etc/profile.d/conda.sh
module load cuda/12.4
module load gcc/11.4.0
conda activate realvlgr1

pip install setuptools==78.1.1

if ! python -c "import torch; assert torch.__version__ == '2.6.0'" 2>/dev/null; then
  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
fi

pip install --no-cache-dir vllm==0.8.4 # torch 2.6.0

pip install ray[default]==2.48.0
pip install opentelemetry-api==1.26.0
pip install opentelemetry-sdk==1.26.0

pip install shapely

pip install flashinfer-python

pip install --no-cache-dir flash-attn/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation

pip install -r requirements.txt
pip install --no-deps -e .

# install sam2 if need to use sam2 for segmentation
cd third_party/sam2
pip install -e .