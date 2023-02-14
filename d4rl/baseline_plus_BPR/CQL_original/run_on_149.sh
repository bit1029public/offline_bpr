envs=("halfcheetah")
tasks=("medium-expert")
seeds=(0 1 2)
for seed in ${seeds[@]}
do
for env in ${envs[@]}
do
for task in ${tasks[@]}
do
	nohup python -u examples/cql_mujoco_new.py --policy_lr=1e-4 --seed=${seed} --lagrange_thresh=-1.0  --min_q_weight=5.0 --min_q_version=3 --gpu=1 --env=${env}-${task}-v2 > ${env}_${task}_${seed}.log &
done
done
done
