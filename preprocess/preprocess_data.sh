#!/bin/bash
#SBATCH -t 1-00:0
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%j.out
#SBATCH --mem=100G

abundance_data_dir="${1:-}"  # path to abundance data
dataset="${2:-'espresso'}"   # dataset name (default 'espresso')
outdir="${3:-}"              # path to output dir

python preprocess_data.py \
    --abundance_data $abundance_data_dir \
    --dataset $dataset \
    --output $outdir
