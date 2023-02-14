#!/usr/bin/env bash

envname=("halfcheetah")
types=("medium-expert")
seeds=(0 1 2)
#methods=("None")

for seed in ${seeds[@]}
do
  for env in ${envname[@]}
  do
    for typ in ${types[@]}
    do
            echo "#!/bin/bash" >> temprun.sh
            echo "#SBATCH --job-name=CQL_BPR" >> temprun.sh
            echo "#SBATCH --cpus-per-task=4"  >> temprun.sh   # ask for 4 CPUs
            #echo "#SBATCH --mem-per-cpu=2G"  >> temprun.sh
            echo "#SBATCH --gres=gpu:1" >> temprun.sh         # ask for 2 GPU
            echo "#SBATCH --mem=24G" >> temprun.sh            # ask for 32 GB RAM
            echo "#SBATCH --time=12:00:00" >> temprun.sh
            echo "#SBATCH --account=rrg-bengioy-ad" >> temprun.sh
            echo "#SBATCH --output=\"/scratch/rislam4/slurm_output/CQL/slurm-%j.out\"" >> temprun.sh
            echo "source ~/vd4rl/bin/activate" >> temprun.sh
            echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/rislam4/.mujoco/mujoco200/bin" >> temprun.sh
            #sleep 2
            echo "MUJOCO_GL="egl" python examples/cql_mujoco_new.py --policy_lr=1e-4 --lagrange_thresh=-1.0  --min_q_weight=5.0 --min_q_version=3 --gpu=1 --env=${env}-${typ}-v2 --seed ${seed}" >> temprun.sh

            eval "sbatch temprun.sh"
			rm temprun.sh
    done
  done
done