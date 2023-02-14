import numpy as np
import torch

import gtimer as gt
import os
import random

from experiment_utils.prepare_data import load_hdf5

from lifelong_rl.core.logging.logging import logger
from lifelong_rl.core.logging.logging_setup import setup_logger
from lifelong_rl.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from lifelong_rl.data_management.replay_buffers.mujoco_replay_buffer import MujocoReplayBuffer
from lifelong_rl.envs.env_processor import make_env
import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.envs.env_utils import get_dim
from lifelong_rl.samplers.data_collector.path_collector import MdpPathCollector, LatentPathCollector
from lifelong_rl.samplers.data_collector.step_collector import MdpStepCollector, RFCollector, \
    GoalConditionedReplayStepCollector


def experiment(
        # Variant
        variant,

        # Experiment config
        experiment_config,

        # Misc arguments
        exp_postfix='',
        use_gpu=True,
        log_to_tensorboard=False,
        
        # data config
        data_args=None,
):
    print('base_experiment begin')
    """
    Reset timers
    (Useful if running multiple seeds from same command)
    """
    gt.reset()
    gt.start()
    """
    Setup logging
    """
    seed = variant['seed']
    log_dir = os.path.join(
        '{}{}'.format(variant['algorithm'], exp_postfix),
        '{}_{}'.format(variant['env_name'], seed))

    print('setup logger')
    setup_logger(log_dir=log_dir,
                 variant=variant,
                 log_to_tensorboard=log_to_tensorboard)
    print('logger set!')
    output_csv = logger.get_tabular_output()
    """
    Set GPU mode for pytorch (+ possible other things later)
    """
    ptu.set_gpu_mode(use_gpu)
    """
    Set experiment seeds
    """
    # fix random seed for experiment
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    """
    Environment setup
    """
    expl_env, env_infos = make_env(variant['env_name'],
                                    **variant.get('env_kwargs', {}))

    obs_dim = get_dim(expl_env.observation_space)
    action_dim = get_dim(expl_env.action_space)

    if env_infos['mujoco']:
        replay_buffer = MujocoReplayBuffer(variant['replay_buffer_size'],
                                           expl_env)
    else:
        replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'],
                                        expl_env)

    eval_env, _ = make_env(variant['env_name'],
                           **variant.get('env_kwargs', {}))
    """
    Import offline data from d4rl
    """
    load_hdf5(expl_env, replay_buffer, data_args)
    obs_dim = replay_buffer.obs_dim()
    """
    Experiment-specific configuration
    """
    config = experiment_config['get_config'](
        variant,
        expl_env=expl_env,
        eval_env=eval_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
    )

    if 'algorithm_kwargs' not in config:
        config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())
    if 'offline_kwargs' not in config:
        config['offline_kwargs'] = variant.get('offline_kwargs', dict())
    """
    Path collectors for sampling from environment
    """

    collector_type = variant.get('collector_type', 'step')

    if collector_type == 'gcr':
        eval_path_collector = GoalConditionedReplayStepCollector(
            eval_env,
            config['evaluation_policy'],
            replay_buffer,
            variant['resample_goal_every'],
        )
    else:
        eval_path_collector = MdpPathCollector(
            eval_env,
            config['evaluation_policy'],
        )
    """
    Finish timer
    """
    gt.stamp('initialization', unique=False)
    """
    Offline RL pretraining
    """
    logger.set_tabular_output(
        os.path.join(logger.log_dir, 'offline_progress.csv'))

    offline_algorithm = experiment_config['get_offline_algorithm'](
        config,
        eval_path_collector=eval_path_collector,
    )
    offline_algorithm.to(ptu.device)
    offline_algorithm.train()

    logger.set_tabular_output(output_csv)
