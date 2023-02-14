#!/usr/bin/env bash

envname=("walker_walk" "cheetah_run" "humanoid_walk")
types=("medium_replay" "medium_expert" "medium" "expert")
seeds=(0 1 2 3 4 5)

for seed in ${seeds[@]}
do
  for env in ${envname[@]}
  do
    for typ in ${types[@]}
    do
        echo "#!/bin/bash" >> temprun.sh
        echo "#SBATCH --job-name=bpr-vd4rl" >> temprun.sh
        echo "#SBATCH --cpus-per-task=4"  >> temprun.sh   # ask for 4 CPUs
        echo "#SBATCH --gres=gpu:2" >> temprun.sh         # ask for 2 GPU
        echo "#SBATCH --mem=24G" >> temprun.sh            # ask for 32 GB RAM
        echo "#SBATCH --time=24:00:00" >> temprun.sh
        echo "module load miniconda/3" >> temprun.sh
        echo "conda activate drqv2" >> temprun.sh
        echo "python drqbc/main.py  task_name=offline_${env}_${typ} offline_dir=/my/offline/dataset/dir/offline_data/main/${env}/${typ}/84px nstep=3 seed=${seed}">> temprun.sh

        eval "sbatch temprun.sh"
        rm temprun.sh
    done
  done
done
