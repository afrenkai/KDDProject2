#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=120g
#SBATCH -J "Model tune"
#SBATCH -p short
#SBATCH -t 05:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sppradhan/tune_models_%j.txt
module load cuda
module load python/3.10.13
source ~/KDDProject2/kdd/bin/activate
cd Tasks/Task\ 5/
python tune_models.py