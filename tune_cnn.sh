#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=100g
#SBATCH -J "CNN tune"
#SBATCH -p academic
#SBATCH -t 16:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/afrenk/KDDProject2/log.txt
module load cuda
module load python/3.10.13
source ~/KDDProject2/kdd/bin/activate
pip install -r requirements.txt
cd Tasks/Task\ 5/
python tune_cnn.py
