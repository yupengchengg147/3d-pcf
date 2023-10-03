#!/bin/bash
#SBATCH --job-name="yub2"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4,VRAM:36G
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=/storage/slurm/logs/slurm-%j.out
#SBATCH --error=/storage/slurm/logs/slurm-%j.out


nvidia-smi:
pwd
source /storage/user/ype/miniconda3/bin/activate mon
module load cuda/11.6.0
cd /usr/stud/ype/3d-forecasting
srun python train_mo_witht_mn.py
# --ntasks-per-node=1
#  --nodes=1
# --gres=gpu:4,VRAM:36G