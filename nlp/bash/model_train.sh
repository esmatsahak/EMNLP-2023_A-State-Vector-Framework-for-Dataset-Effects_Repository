#!/bin/bash 

#SBATCH -p t4v2,rtx6000,t4v1,p100 
#SBATCH -c 2 
#SBATCH --mem=32G 
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/%x.log
#SBATCH --qos=normal
#SBATCH --ntasks=1 
#SBATCH --open-mode=append

. /etc/profile.d/lmod.sh  
module use $HOME/multitask_nlp_project/env_scripts
module load nlp_module
hostname
python ../py_scripts/multitask_nlp_train.py $id $seed $encoder $model 

