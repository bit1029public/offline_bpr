env='walker2d-medium-expert-v2'
seeds=(0 1 2 3 4)
opt_pre=True
name=pretrain
for seed in ${seeds[@]}
do
	CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --env $env --name $name --seed $seed --opt_predictor ${opt_pre} > logs/${env}_${name}_${opt_pre}_${seed}.log &
done
