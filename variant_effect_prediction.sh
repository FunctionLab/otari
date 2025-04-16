#!/bin/bash
#SBATCH -t 0-08:0
#SBATCH -p gpu
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%j.out
#SBATCH --mem=100G

vcf_filepath="${1:-}"  # path to vcf file
model_path="${2:-}"    # model weights path
outdir="${3:-}"        # path to output dir

python variant_effect_prediction.py \
    --variant_path $vcf_filepath \
    --model_path $model_path \
    --output_path $outdir \
    --annotate
