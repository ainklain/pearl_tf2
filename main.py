
from rlkit_tf2.envs import ENVS
from rlkit_tf2.envs.wrappers import NormalizedBoxEnv
from rlkit_tf2.tf2.sac.policies import TanhGaussianPolicy
from rlkit_tf2.tf2.networks import FlattenMLP, MLPEncoder   # , RecurrentEncoder
from rlkit_tf2.tf2.sac.sac import PEARLSoftActorCritic
from rlkit_tf2.tf2.sac.agent import PEARLAgent
from rlkit_tf2.launchers.launcher_util import setup_logger
from configs.default import default_config


import os
import pathlib
import numpy as np
import json
import tensorflow as tf



def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


def main():
    env_name = 'point-robot'
    variant = default_config

    with open("./configs/{}.json".format(env_name)) as f:
        exp_params = json.load(f)
    variant = deep_update_dict(exp_params, variant)

    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    # encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    encoder_model = MLPEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=obs_dim + action_dim + reward_dim,
        output_size=context_encoder,
    )
    qf1 = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    target_vf = FlattenMLP(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf, target_vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    def example():
        train_tasks = list(tasks[:variant['n_train_tasks']])
        meta_batch = 64
        indices = np.random.choice(train_tasks, meta_batch)

    # # optionally load pre-trained weights
    # if variant['path_to_weights'] is not None:
    #     path = variant['path_to_weights']
    #     context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
    #     qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
    #     qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
    #     vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
    #     # TODO hacky, revisit after model refactor
    #     algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
    #     policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))
    #
    # # optional GPU mode
    # ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    # if ptu.gpu_enabled():
    #     algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()
