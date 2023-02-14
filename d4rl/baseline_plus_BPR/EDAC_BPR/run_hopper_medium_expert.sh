
CUDA_VISIBLE_DEVICES=1 nohup python -u -m scripts.sac --env_name hopper-medium-expert-v2 --num_qs 50 --eta 1 --seed 0 > 0.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u -m scripts.sac --env_name hopper-medium-expert-v2 --num_qs 50 --eta 1 --seed 1 > 1.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u -m scripts.sac --env_name hopper-medium-expert-v2 --num_qs 50 --eta 1 --seed 2 > 2.log &
CUDA_VISIBLE_DEVICES=1 nohup python -u -m scripts.sac --env_name hopper-medium-expert-v2 --num_qs 50 --eta 1 --seed 3 > 3.log &
