#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=100g
#SBATCH -J "CNN test"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sppradhan/train_cnn_%j.txt
module load cuda
module load python/3.10.13
source ~/KDDProject2/kdd/bin/activate
cd Tasks/Task\ 5/
python base_model_cnn.py