#!/bin/bash

#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=100g
#SBATCH -J "SVM test"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=/home/$USER/KDDProject2/train_svm-%j.txt
module load cuda
module load python/3.10.13
source ~/KDDProject2/kdd/bin/activate
pip install -r requirements.txt
cd Tasks/Task\ 5/
python preprocessing.py