#!/bin/bash 

#SBATCH -p t4v2,rtx6000,t4v1,p100 
#SBATCH -c 2 
#SBATCH --mem=32G 
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/%j.log
#SBATCH --qos=normal
#SBATCH --ntasks=1 
#SBATCH --open-mode=append


. /etc/profile.d/lmod.sh  
module use $HOME/env_scripts
module load transformers4 
hostname

python ../py_scripts/multitask_nlp_train.py $1 $2 roberta roberta-base 
