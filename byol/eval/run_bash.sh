#!/bin/bash

#SBATCH --job-name=finetune
#SBATCH --constraint=A100
#SBATCH --time=10-23
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --output=/share/nas2/dmohan/byol_finetuning/byol/byol/logs/out-slurm_%j.out


pwd;

nvidia-smi 

source /share/nas2/dmohan/byol_finetuning/venv/bin/activate
echo ">>>eval"
python $1