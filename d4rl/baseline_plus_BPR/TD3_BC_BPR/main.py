import numpy as np
import torch
import gym
import argparse
import os
import d4rl

import utils
from torch.utils.tensorboard import SummaryWriter


# Runs policy for X episodes and returns D4RL score
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, mean, std, it, writer, seed_offset=100, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            state = (np.array(state).reshape(1, -1) - mean)/std
            action = policy.select_action(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes
    d4rl_score = eval_env.get_normalized_score(avg_reward) * 100

    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}, D4RL score: {d4rl_score:.3f}")
    print("---------------------------------------")
    writer.add_scalar("Reward", d4rl_score, it)
    return d4rl_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment
    # Policy name
    parser.add_argument("--policy", default="TD3_BC")
    # OpenAI gym environment name
    parser.add_argument("--env", default="halfcheetah-medium-expert-v2")
    # OpenAI gym environment name
    parser.add_argument("--name", default="with_repr")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=5e3, type=int)
    # Max time steps to run environment
    parser.add_argument("--max_timesteps", default=1.1e6, type=int)
    # Save model and optimizer parameters
    parser.add_argument("--save_model", action="store_true")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_model", default="")
    # Model load file name, "" doesn't load, "default" uses file_name
    parser.add_argument("--load_emb_model", default="")
    # eigen value
    parser.add_argument("--save_eigen", action="store_true")
    # TD3
    # Std of Gaussian exploration noise
    parser.add_argument("--expl_noise", default=0.1)
    # Batch size for both actor and critic
    parser.add_argument("--batch_size", default=256, type=int)
    # Discount factor
    parser.add_argument("--discount", default=0.99)
    # Target network update rate
    parser.add_argument("--tau", default=0.005)
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.2)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)
    # TD3 + BC
    parser.add_argument("--alpha", default=2.5)
    parser.add_argument("--normalize", default=True)
    parser.add_argument("--opt_predictor", default=True)

    args = parser.parse_args()

    assert args.name in ["no_repr", "with_repr", "pretrain"]
    if args.name == "no_repr":
        import TD3_BC
    elif args.name == "with_repr":
        import TD3_BC_with_repr as TD3_BC
    elif args.name == 'pretrain':
        import TD3_BC_pretrain as TD3_BC
    else:
        import TD3_BC

    file_name = f"{args.policy}_{args.seed}_{args.name}"
    file_path = f"{args.policy}/{args.name}/{args.opt_predictor}/{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Name: {args.name}, Optimize predictor: {args.opt_predictor}")
    print("---------------------------------------")
    writer = SummaryWriter('cmp_runs/'+'finetune/'+args.env+'/'+file_name)

    if not os.path.exists(f"./results/{args.env}"):
        os.makedirs(f"./results/{args.env}")

    if not os.path.exists(f"./results/{args.env}/{file_path}"):
        os.makedirs(f"./results/{args.env}/{file_path}")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)

    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    if args.name == "no_repr":
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": args.discount,
            "tau": args.tau,
            "save_eigen": args.save_eigen,
            # TD3
            "policy_noise": args.policy_noise * max_action,
            "noise_clip": args.noise_clip * max_action,
            "policy_freq": args.policy_freq,
            # TD3 + BC
            "alpha": args.alpha,
        }
    else:
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": args.discount,
            "tau": args.tau,
            "save_eigen": args.save_eigen,
            # TD3
            "policy_noise": args.policy_noise * max_action,
            "noise_clip": args.noise_clip * max_action,
            "policy_freq": args.policy_freq,
            # TD3 + BC
            "alpha": args.alpha,
            "opt_predictor": args.opt_predictor,
        }

    # Initialize policy
    policy = TD3_BC.TD3_BC(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{args.env}/{policy_file}")
        print("load model")
    if args.load_emb_model != "":

        policy.load_enc(args.load_emb_model)
        print("load model")

    # if args.env == "halfcheetah-medium-expert-v2":
    # 	import pickle as pkl
    # 	use_data = pkl.load(open('halfcheetah-medium-expert-v3','rb'))
    # else:
    use_data = d4rl.qlearning_dataset(env)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(use_data)
    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0, 1

    evaluations = []

    fine_tune_t = 0
    if args.name == "pretrain":
        start_ts = 1e5
        print('Pretraining begins.......')
    else:
        start_ts = 0

    for t in range(int(args.max_timesteps)):
        policy.train(replay_buffer, args.batch_size)
        # Evaluate episode
        if t > start_ts:
            fine_tune_t += 1
        if t > start_ts and ((t + 1) % args.eval_freq == 0):
            print(f"Time steps: {t+1}")
            evaluations.append(eval_policy(
                policy, args.env, args.seed, mean, std, fine_tune_t, writer))
            np.save(f"./results/{args.env}/{file_path}/ret.npy", evaluations)
            if args.save_model:
                policy.save(f"./models/{args.env}/{file_name}")
        if args.save_eigen:
            np.save(f"./results/{args.env}/{file_name}_effective_eigen",
                    np.array(policy.effective_eigen))
