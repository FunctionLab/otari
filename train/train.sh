#!/bin/bash
#SBATCH -t 0-08:0
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%j.out
#SBATCH --mem=100G

python train.py --config_path configs.yml
