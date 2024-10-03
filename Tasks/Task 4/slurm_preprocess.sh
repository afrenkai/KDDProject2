#!/bin/bash

#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=100g
#SBATCH -J "Preprocess"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:2

module load cuda
module load python/3.10.13
source ~/KDDProject2/kdd/bin/activate
pip install -r requirements.txt
cd Tasks/Task\ 4/
python preprocessing.py