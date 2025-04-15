#!/bin/bash
#SBATCH -t 0-08:0
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%j.out
#SBATCH --mem=100G

python variant_effect_prediction.py \
    --variant_path /path/to/variant_file.tsv \
    --model_path /path/to/model_file.pt \
    --output_path /path/to/output_file.tsv \
    --annotate
