from lifelong_rl.models.networks import ParallelizedEnsembleFlattenMLP, Mlp
from lifelong_rl.policies.base.base import MakeDeterministic
from lifelong_rl.policies.models.tanh_gaussian_policy import TanhGaussianPolicy
from lifelong_rl.trainers.q_learning.sac import SACTrainer
import lifelong_rl.util.pythonplusplus as ppp
import os
import torch
import lifelong_rl.torch.pytorch_util as ptu
from torch.nn import functional as F


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
):
    """
    Policy construction
    """

    num_qs = variant['trainer_kwargs']['num_qs']
    M = variant['policy_kwargs']['layer_size']
    num_q_layers = variant['policy_kwargs']['num_q_layers']
    num_p_layers = variant['policy_kwargs']['num_p_layers']
    embedding_size = variant["trainer_kwargs"]["embedding_size"]

    qfs, target_qfs = ppp.group_init(
        2,
        ParallelizedEnsembleFlattenMLP,
        ensemble_size=num_qs,
        hidden_sizes=[M] * num_q_layers,
        input_size=embedding_size + action_dim,
        output_size=1,
        layer_norm=None,
    )

    policy = TanhGaussianPolicy(
        obs_dim=embedding_size,
        action_dim=action_dim,
        hidden_sizes=[M] * num_p_layers,
        layer_norm=None,
    )
    embedder = Mlp([M], embedding_size, obs_dim)
    # fake_policy = TanhGaussianPolicy(
    #     obs_dim=embedding_size,
    #     action_dim=action_dim,
    #     hidden_sizes=[M] * num_p_layers,
    #     layer_norm=None,
    # )
    predictor = Mlp([M]*3, action_dim, embedding_size, output_activation=F.tanh)
    
    
    
    trainer = SACTrainer(
        env=eval_env,
        embedder=embedder,
        policy=policy,
        predictor=predictor,
        qfs=qfs,
        target_qfs=target_qfs,
        replay_buffer=replay_buffer,
        **variant['trainer_kwargs'],
    )
    """
    Create config dict
    """

    config = dict()
    config.update(
        dict(
            trainer=trainer,
            exploration_policy=policy,
            evaluation_policy=MakeDeterministic(policy),
            evaluation_embedder=embedder,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            replay_buffer=replay_buffer,
            qfs=qfs,
        ))
    config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())

    return config
