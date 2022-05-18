import gym
from gym import spaces
import numpy as np
import importlib
from core import environment
import os
import torch
from rtfm import featurizer as X
from torch.nn import functional as F
from argparse import Namespace

flags = Namespace({'env': 'groups_simple_stationary-v0', 'mode': 'train_DQN_writer', 
        'xpid': 'groups_simple_stationary:paper_txt2pi:yeswiki:default', 
        'prefix': 'default', 'model': 'paper_txt2pi', 
        'resume': 'checkpoints/groups_simple_stationary:paper_txt2pi:yeswiki:default/model.tar', 
        'resume_scheduler': False, 'resume_strategy': 'all', 'shuffle_wiki': False, 
        'wiki': 'yes', 'height': 6, 'width': 6, 'partial_observability': False, 
        'max_placement': 1, 'max_name': 8, 'max_inv': 8, 'max_wiki': 80, 
        'max_task': 40, 'time_penalty': -0.02, 'pretrained_emb': False, 'demb': 30, 
        'drnn': 100, 'drnn_small': 10, 'drep': 400, 'disable_checkpoint': False, 
        'savedir': 'checkpoints', 'num_actors': 30, 'total_frames': 100000000, 
        'batch_size': 24, 'unroll_length': 80, 'queue_timeout': 1, 
        'num_threads': 4, 'disable_cuda': False, 'entropy_cost': 0.0006, 
        'baseline_cost': 0.5, 'discounting': 0.99, 'reward_clipping': 'abs_one', 
        'learning_rate': 0.0005, 'alpha': 0.99, 'momentum': 0, 'epsilon': 0.01, 
        'random_agent': False, 'num_buffers': 60})

class metaEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    global Net
    Net = importlib.import_module('model.{}'.format(flags.model)).Model

    def __init__(self, height, width, max_placement, demb, drollout, dwiki, dtask):
        super(metaEnv, self).__init__()
        gym_env = Net.create_env(flags)
        self.env = environment.Environment(gym_env)
        self.model = Net.make(flags, gym_env)
        self.model.eval()
        checkpointpath = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                                'model.tar')))
        checkpoint = torch.load(checkpointpath, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.action_space = spaces.Tuple((
        spaces.Discrete(6),
        spaces.Discrete(3),
        spaces.Discrete(3)))
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=999,
                                shape=(height, width, max_placement, demb, drollout), 
                                dtype=np.uint8),
                                spaces.Box(low=0, high=999, #wiki
                                shape=(dwiki, drollout), 
                                dtype=np.uint8),
                                spaces.Box(low=0, high=999, #task
                                shape=(dtask, drollout), 
                                dtype=np.uint8)))

    def step(self, action):
        info = None
        eps_done = False
        # steps = 0
        eps_observation = self.env.initial()
        rollout_ls = []
        while not eps_done:
            agent_outputs = self.model(observation)
            eps_observation = self.env.step(agent_outputs['action'])
            rollout_ls.append(eps_observation['name'])
            policy = F.softmax(agent_outputs['policy_logits'], dim=-1)
            log_policy = F.log_softmax(agent_outputs['policy_logits'], dim=-1)
            e = -torch.sum(policy * log_policy, dim=-1)
            # steps += 1
            done = observation['done'].item()
            if observation['done'].item():
                pass
        reward = eps_observation['episode_return'].item()
        wiki = eps_observation['wiki'].item()
        task = eps_observation['task'].item()
        rollout = torch.stack(rollout_ls)
        observation = (rollout, wiki, task)
        if action[0] == -1: #delete idx
            done = True
        return observation, reward, done, info
    def reset(self):
        observation = self.env.initial()
        return observation  # reward, done, info can't be included
    def render(self, mode='human'):
        ...
    def close (self):
        ...