#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=100g
#SBATCH -J "CNN test"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/sppradhan/KDDProject2/train_cnn_%j.txt
module load cuda11.7
module load python/3.10.13
source ~/KDDProject2/kdd/bin/activate
pip install -r requirements.txt
cd Tasks/Task\ 5/
python base_model_cnn.py