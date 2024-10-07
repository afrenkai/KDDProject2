#!/bin/bash
#SBATCH -N 1 


#SBATCH -n 2 


#SBATCH --mem=8g 


#SBATCH -J "Hello World Job" 


#SBATCH -p short 


#SBATCH -t 12:00:00 



echo "Hello World" 



