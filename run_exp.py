# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Must be run with OMP_NUM_THREADS=1
#
'''
For debugging the env using random actions run:
OMP_NUM_THREADS=1 python torchbeast.py --env MiniGrid-MultiRoom-N2-S4-v0 --num_actors 1 --num_threads 1 --random_agent --mode test
'''

import argparse
import logging
import os
import sys
from termios import FF1
import tqdm
import importlib

os.environ['OMP_NUM_THREADS'] = '1'

import threading
import time
import timeit
import traceback
import pprint
import typing
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch import multiprocessing as mp

import gym
import random
import exp_utils
from rtfm import tasks

from core import environment
from core import file_writer
from core import prof
from core import vtrace

import pandas as pd
import wandb

from metaWriter import UCBBandit, DQNWriter
from rtfm.tasks import RockPaperScissors
import pickle

project_name = "num_objects"

Net = None

# rollout_file = open("rollout.txt", "w")
# pickle_dataset = open("pickle_dataset", "wb")
# debug = open("debug.txt", 'w')
bandit_report = open("bandit_report.txt", 'w')

from types import MethodType


logging.basicConfig(
    format=('[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
            '%(message)s'),
    level=0)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]

LOG_COUNT = 0
def compute_baseline_loss(advantages):
    # Take the mean over batch, sum over time.
    return 0.5 * torch.sum(torch.mean(advantages ** 2, dim=1))


def compute_entropy_loss(logits):
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    return -torch.sum(torch.mean(entropy_per_timestep, dim=1))


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction='none')
    cross_entropy = cross_entropy.view_as(advantages)
    advantages.requires_grad = False
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return torch.sum(torch.mean(policy_gradient_loss_per_timestep, dim=1))


def act(i: int, free_queue: mp.SimpleQueue, full_queue: mp.SimpleQueue, model: torch.nn.Module, buffers: Buffers, flags):
    try:
        logging.info('Actor %i started.', i)
        timings = prof.Timings()  # Keep track of how fast things are.

        gym_env = Net.create_env(flags)
        seed = i ^ int.from_bytes(os.urandom(4), byteorder='little')
        gym_env.seed(seed)
        env = environment.Environment(gym_env)
        env_output = env.initial()
        agent_output = model(env_output)
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]

            # Do new rollout
            for t in range(flags.unroll_length):
                timings.reset()

                with torch.no_grad():
                    agent_output = model(env_output)

                timings.time('model')

                env_output = env.step(agent_output['action'])
                s1 = 'env output \n' + str(env_output) + '\n'
                # f.write(s1)

                timings.time('step')

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]

                timings.time('write')
            full_queue.put(index)

        if i == 0:
            logging.info('Actor %i: %s', i, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


def get_batch(free_queue: mp.SimpleQueue,
              full_queue: mp.SimpleQueue,
              buffers: Buffers,
              flags,
              timings,
              lock=threading.Lock()) -> typing.Dict[str, torch.Tensor]:
    with lock:
        timings.time('lock')
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time('dequeue')
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    timings.time('batch')
    for m in indices:
        free_queue.put(m)
    timings.time('enqueue')
    batch = {
        k: t.to(device=flags.device, non_blocking=True)
        for k, t in batch.items()
    }
    timings.time('device')
    return batch


def learn(actor_model,
          model,
          batch,
          optimizer,
          scheduler,
          flags,
          lock=threading.Lock()):
    """Performs a learning (optimization) step."""
    with lock:
        learner_outputs = model(batch)

        # Use last baseline value (from the value function) to bootstrap.
        bootstrap_value = learner_outputs['baseline'][-1]

        # At this point, the environment outputs at time step `t` are the inputs
        # that lead to the learner_outputs at time step `t`. After the following
        # shifting, the actions in actor_batch and learner_outputs at time
        # step `t` is what leads to the environment outputs at time step `t`.
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {
            key: tensor[:-1]
            for key, tensor in learner_outputs.items()
        }

        rewards = batch['reward']
        if flags.reward_clipping == 'abs_one':
            clipped_rewards = torch.clamp(rewards, -1, 1)
        elif flags.reward_clipping == 'soft_asymmetric':
            squeezed = torch.tanh(rewards / 5.0)
            # Negative rewards are given less weight than positive rewards.
            clipped_rewards = torch.where(rewards < 0, 0.3 * squeezed,
                                          squeezed) * 5.0
        elif flags.reward_clipping == 'none':
            clipped_rewards = rewards

        discounts = (~batch['done']).float() * flags.discounting

        # This could be in C++. In TF, this is actually slower on the GPU.
        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch['policy_logits'],
            target_policy_logits=learner_outputs['policy_logits'],
            actions=batch['action'],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs['baseline'],
            bootstrap_value=bootstrap_value)

        # Compute loss as a weighted sum of the baseline loss, the policy
        # gradient loss and an entropy regularization term.
        pg_loss = compute_policy_gradient_loss(learner_outputs['policy_logits'],
                                               batch['action'],
                                               vtrace_returns.pg_advantages)
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs['baseline'])
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs['policy_logits'])
        aux_loss = learner_outputs['aux_loss'][0]

        total_loss = pg_loss + baseline_loss + entropy_loss + aux_loss

        episode_returns = batch['episode_return'][batch['done']]
        episode_lens = batch['episode_step'][batch['done']]
        won = batch['reward'][batch['done']] > 0.8
        stats = {
            'mean_win_rate': torch.mean(won.float()).item(),
            'mean_episode_len': torch.mean(episode_lens.float()).item(),
            'mean_episode_return': torch.mean(episode_returns).item(),
            'total_loss': total_loss.item(),
            'pg_loss': pg_loss.item(),
            'baseline_loss': baseline_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'aux_loss': aux_loss.item(),
        }
        #wandb watch
        # wandb.watch(model, total_loss, log='all', log_freq=10)

        optimizer.zero_grad()
        model.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optimizer.step()
        scheduler.step()

        # Interestingly, this doesn't require moving off cuda first?
        actor_model.load_state_dict(model.state_dict())
        return stats


def create_buffers(observation_shapes, num_actions, flags) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        aux_loss=dict(size=(T + 1, ), dtype=torch.float32),
    )
    for k, shape in observation_shapes.items():
        specs[k] = dict(size=(T + 1, *shape), dtype=torch.long)
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    if flags.xpid is None:
        flags.xpid = 'torchbeast-%s' % time.strftime('%Y%m%d-%H%M%S')
    plogger = file_writer.FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
        symlink_latest=False,
    )
    # checkpointpath = os.path.expandvars(
    #     os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
    #                                      'RPS2.tar')))
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                         'model.tar')))
    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info('Using CUDA.')
        flags.device = torch.device('cuda')
    else:
        logging.info('Not using CUDA.')
        flags.device = torch.device('cpu')

    env = Net.create_env(flags)
    model = Net.make(flags, env)
    buffers = create_buffers(env.observation_space, len(env.action_space), flags)
    metadat = {}
    for k,v in buffers.items():
        metadat[k] = len(v)

    model.share_memory()

    actor_processes = []
    ctx = mp.get_context('fork')
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, buffers, flags))
        actor.start()
        actor_processes.append(actor)

    learner_model = Net.make(flags, env).to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),#loss,
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if flags.resume:
        save = torch.load(flags.resume, map_location='cpu')
        learner_model.load_state_dict(save['model_state_dict'])
        optimizer.load_state_dict(save['optimizer_state_dict'])
        if flags.resume_scheduler:
            scheduler.load_state_dict(save['scheduler_state_dict'])
        # tune only the embedding layer
        if flags.resume_strategy == 'emb':
            keep = []
            for group in optimizer.param_groups:
                if group['params'][0].size() == (len(learner_model.vocab), flags.demb):
                    keep.append(group)
            optimizer.param_groups = keep

    logger = logging.getLogger('logfile')
    stat_keys = [
        'total_loss',
        'mean_episode_return',
        'pg_loss',
        'baseline_loss',
        'entropy_loss',
        'aux_loss',
        'mean_win_rate',
        'mean_episode_len',
    ]
    logger.info('# Step\t%s', '\t'.join(stat_keys))

    frames, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        batch_count = 0
        while frames < flags.total_frames:
            timings.reset()
            batch_count += 1
            batch = get_batch(free_queue, full_queue, buffers, flags, timings)

            stats = learn(model, learner_model, batch, optimizer, scheduler,
                          flags)
            timings.time('learn')
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
        if i == 0:
            logging.info('Batch and learn: %s', timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name='batch-and-learn-%d' % i, args=(i,))
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info('Saving checkpoint to %s', checkpointpath)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'flags': vars(flags),
        }, checkpointpath)

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5)
            wandb.log(stats)
            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)
            if stats.get('episode_returns', None):
                mean_return = 'Return per episode: %.1f. ' % stats[
                    'mean_episode_return']
            else:
                mean_return = ''
            total_loss = stats.get('total_loss', float('inf'))
            logging.info('After %i frames: loss %f @ %.1f fps. %sStats:\n%s',
                        frames, total_loss, fps, mean_return,
                        pprint.pformat(stats))
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info('Learning finished after %d frames.', frames)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()

# ------ helper functions for meta writer -------
def add_sym(orders, ls):
    new_orders = []
    for order in orders:
        for c in ls:
            new_order = order + [c]
            new_orders.append(new_order)
    return new_orders

def get_all_orders(orders, n, ls):
    while len(orders[0]) < n:
        orders = add_sym(orders, ls)
    orders_dict = {}
    for i in range(len(orders)):
        orders_dict[i] = orders[i]
    return orders_dict

def orders_to_str(orders_dict):
    new_dict = {}
    for key, value in orders_dict.items():
        s = ''
        for token in value:
            s += token + ' '
        s = s[:-1]
        if s[-1] == '.':
            s = s[:-2] + '.'
        new_dict[key] = s
    return new_dict
# ------------------------------------------------

def train_meta_writer(flags, difficulty = 'character_level', num_eps = 100):
    num_eps = int(num_eps)
    counter = 0
    rolling_win_rate = []
    rolling_wins = 0
    rolling_window = 10
    rolling_win_counter = 0

    from rtfm import featurizer as X
    gym_env = RockPaperScissors(featurizer=X.Concat([X.Text(),X.ValidMoves(), X.RelativePosition()]), max_placement=1)
    if flags.mode == 'test_render':
        gym_env.featurizer = X.Concat([gym_env.featurizer, X.Terminal()])
    env = environment.Environment(gym_env)

    if difficulty == 'character_level':
        env.gym_env.get_wiki = MethodType(lambda x : 'c beats b. c beats a. b beats c.', env.gym_env)
        meta_action_map = {0: 'a', 1: 'b', 2: 'c'}
    elif difficulty == 'statement_level':
        env.gym_env.get_wiki = MethodType(lambda x : 'c beats a. b beats c.', env.gym_env)
        all_orders = get_all_orders([['a'], ['b'], ['c'], ['beats'],['.']], 4, ['a', 'b', 'c', 'beats', '.'])
        meta_action_map = orders_to_str(all_orders)

    meta_writer = UCBBandit(len(meta_action_map), bonus_multiplier=1/np.sqrt(2))

    first_line = '------------- Record of written wikis ------------- \n'
    bandit_report.write(first_line)

    if not flags.random_agent:
        model = Net.make(flags, gym_env)
        model.eval()
        if flags.xpid is None:
            checkpointpath = './results_latest/model.tar'
        else:
            checkpointpath = os.path.expandvars(
                os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                                 'model.tar')))
        checkpoint = torch.load(checkpointpath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    observation = env.initial()
    returns = []
    won = []
    entropy = []
    ep_len = []
    meta_action = None     #for UCBBandit
    while len(won) < num_eps:
        done = False
        steps = 0
        while not done:
            if flags.random_agent:
                action = torch.zeros(1, 1, dtype=torch.int32)
                action[0][0] = random.randint(0, len(gym_env.action_space) - 1)
                observation = env.step(action)
            else:
                agent_outputs = model(observation)
                observation = env.step(agent_outputs['action'])
                policy = F.softmax(agent_outputs['policy_logits'], dim=-1)
                log_policy = F.log_softmax(agent_outputs['policy_logits'], dim=-1)
                e = -torch.sum(policy * log_policy, dim=-1)
                entropy.append(e.mean(0).item())

            steps += 1
            done = observation['done'].item()
            if observation['done'].item():
                # BANDIT WRITER CODE
                meta_action = meta_writer.step(meta_action, observation['episode_return'].item())
                if difficulty == 'character_level':
                    wiki_str = meta_action_map[meta_action] + ' beats b.' + ' c beats a. b beats c.'
                elif difficulty == 'statement_level':
                    wiki_str = meta_action_map[meta_action] + ' c beats a. b beats c.'
                bandit_report.write(wiki_str + '\n')
                env.gym_env.get_wiki = MethodType(lambda x : wiki_str, env.gym_env)

                counter += 1
                if counter % 10 == 0:
                    print('eps: ', counter)

                rolling_win_counter += 1
                if observation['reward'][0][0].item() > 0.5:
                    if rolling_win_counter < 10:
                        rolling_wins += 1
                    else:
                        rolling_win_rate.append(rolling_wins / rolling_window)
                        rolling_wins = 0
                        rolling_win_counter = 0
                returns.append(observation['episode_return'].item())
                won.append(observation['reward'][0][0].item() > 0.5)
                ep_len.append(steps)
            if flags.mode == 'test_render':
                sleep_seconds = os.environ.get('DELAY', '0.3')
                time.sleep(float(sleep_seconds))

                if observation['done'].item():
                    print('Done: {}'.format('You won!!' if won[-1] else 'You lost!!'))
                    print('Episode steps: {}'.format(observation['episode_step']))
                    print('Episode return: {}'.format(observation['episode_return']))
                    done_seconds = os.environ.get('DONE', None)
                    if done_seconds is None:
                        print('Press Enter to continue')
                        input()
                    else:
                        time.sleep(float(done_seconds))

    env.close()
    logging.info('Average returns over %i episodes: %.2f. Win rate: %.2f. Entropy: %.2f. Len: %.2f', num_eps, sum(returns)/len(returns), sum(won)/len(returns), sum(entropy)/max(1, len(entropy)), sum(ep_len)/len(ep_len))
    second_line = '------------- Rolling win rate ------------- \n'
    bandit_report.write(str(second_line))
    bandit_report.write(str(rolling_win_rate))
    print(meta_writer.Q.argmax())

def collect_rollouts(flags, num_eps: int = 100000): #num_eps originall 2
    def dot_split(sent):
        ls0 = sent.split()
        ls1 = []
        for elem in ls0:
            if '.' in elem:
                ls1.append(elem[:-1])
                ls1.append('.')
            else:
                ls1.append(elem)
        return ls1

    from rtfm import featurizer as X
    gym_env = Net.create_env(flags)
    # gym_env = RockPaperScissors(featurizer=X.Concat([X.Text(),X.ValidMoves(), X.RelativePosition()]), max_placement=1)
    if flags.mode == 'test_render':
        gym_env.featurizer = X.Concat([gym_env.featurizer, X.Terminal()])
    env = environment.Environment(gym_env)
    textFeaturizer = X.Text(gym_env.featurizer)
    # env.gym_env.get_wiki = MethodType(lambda x : 'a beats c. c beats b. b beats a.', env.gym_env)
    print(env.gym_env.vocab._index2word)

    if not flags.random_agent:
        model = Net.make(flags, gym_env)
        model.eval()
        if flags.xpid is None:
            checkpointpath = './results_latest/model.tar'
        else:
            checkpointpath = os.path.expandvars(
                os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                                 'model.tar')))
        checkpoint = torch.load(checkpointpath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    observation = env.initial()
    returns = []
    won = []
    entropy = []
    ep_len = []
    dataset = []
    while len(won) < num_eps:
        if len(won) % 100 == 0:
            print('episode number: ', len(won))
        done = False
        steps = 0
        relevant_statement = env.gym_env.get_relevant_statements()
        debug.write(str(env.gym_env.vocab) + '\n')
        # relstat_emb = textFeaturizer.lookup_sentence(relevant_statement, env.gym_env.vocab, max_len=env.gym_env.max_wiki, eos='pad', pad='pad')
        # debug.write(str(relstat_emb) + '\n')
        entry = {}
        entry['relstat_emb'] = env.gym_env.vocab.word2index(dot_split(relevant_statement))
        rollout = []
        get_task_flag = True
        while not done:
            if flags.random_agent:
                action = torch.zeros(1, 1, dtype=torch.int32)
                # action[0][0] = random.randint(0, gym_env.action_space.n - 1)
                action[0][0] = random.randint(0, len(gym_env.action_space) - 1)
                observation = env.step(action)
            else:
                agent_outputs = model(observation)
                observation = env.step(agent_outputs['action'])
                if get_task_flag:
                    entry['task_emb'] = observation['task']
                    debug.write(str(observation['task']) + '\n')
                    get_task_flag = False
                rollout.append(observation['name'])
                flatwiki = torch.flatten(observation['wiki']).tolist()
                rollout_file.write(str(observation['name'].size()) + '\n')
                policy = F.softmax(agent_outputs['policy_logits'], dim=-1)
                log_policy = F.log_softmax(agent_outputs['policy_logits'], dim=-1)
                e = -torch.sum(policy * log_policy, dim=-1)
                entropy.append(e.mean(0).item())

            steps += 1
            done = observation['done'].item()
            if observation['done'].item():
                entry['rollout'] = torch.stack(rollout)
                dataset.append(entry)
                returns.append(observation['episode_return'].item())
                won.append(observation['reward'][0][0].item() > 0.5)
                ep_len.append(steps)

    env.close()
    logging.info('Average returns over %i episodes: %.2f. Win rate: %.2f. Entropy: %.2f. Len: %.2f', num_eps, sum(returns)/len(returns), sum(won)/len(returns), sum(entropy)/max(1, len(entropy)), sum(ep_len)/len(ep_len))
    pickle.dump(dataset,pickle_dataset)
    pickle_dataset.close()

def test(flags, num_eps: int = 10): #num_eps originall 2
    from rtfm import featurizer as X
    gym_env = Net.create_env(flags)
    # gym_env = RockPaperScissors(featurizer=X.Concat([X.Text(),X.ValidMoves(), X.RelativePosition()]), max_placement=1)
    if flags.mode == 'test_render':
        gym_env.featurizer = X.Concat([gym_env.featurizer, X.Terminal()])
    env = environment.Environment(gym_env)
    # env.gym_env.get_wiki = MethodType(lambda x : 'a beats c. c beats b. b beats a.', env.gym_env)

    if not flags.random_agent:
        model = Net.make(flags, gym_env)
        model.eval()
        if flags.xpid is None:
            checkpointpath = './results_latest/model.tar'
        else:
            checkpointpath = os.path.expandvars(
                os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                                 'model.tar')))
        checkpoint = torch.load(checkpointpath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    observation = env.initial()
    returns = []
    won = []
    entropy = []
    ep_len = []
    while len(won) < num_eps:
        done = False
        steps = 0
        print('wiki: ', env.gym_env.get_wiki())
        while not done:
            if flags.random_agent:
                action = torch.zeros(1, 1, dtype=torch.int32)
                # action[0][0] = random.randint(0, gym_env.action_space.n - 1)
                action[0][0] = random.randint(0, len(gym_env.action_space) - 1)
                observation = env.step(action)
            else:
                agent_outputs = model(observation)
                observation = env.step(agent_outputs['action'])
                f1.write(str(observation) + '\n')
                policy = F.softmax(agent_outputs['policy_logits'], dim=-1)
                log_policy = F.log_softmax(agent_outputs['policy_logits'], dim=-1)
                e = -torch.sum(policy * log_policy, dim=-1)
                entropy.append(e.mean(0).item())

            steps += 1
            done = observation['done'].item()
            if observation['done'].item():
                returns.append(observation['episode_return'].item())
                won.append(observation['reward'][0][0].item() > 0.5)
                ep_len.append(steps)

    env.close()
    logging.info('Average returns over %i episodes: %.2f. Win rate: %.2f. Entropy: %.2f. Len: %.2f', num_eps, sum(returns)/len(returns), sum(won)/len(returns), sum(entropy)/max(1, len(entropy)), sum(ep_len)/len(ep_len))

def train_simple(flags, num_eps: int = 10):
    from rtfm import featurizer as X
    gym_env = Net.create_env(flags)
    # gym_env = RockPaperScissors(featurizer=X.Concat([X.Text(),X.ValidMoves(), X.RelativePosition()]), max_placement=1)
    if flags.mode == 'test_render':
        gym_env.featurizer = X.Concat([gym_env.featurizer, X.Terminal()])
    env = environment.Environment(gym_env)
    # env.gym_env.get_wiki = MethodType(lambda x : 'a beats c. c beats b. b beats a.', env.gym_env)

    if not flags.random_agent:
        model = Net.make(flags, gym_env)
        # model.eval()
        if flags.xpid is None:
            checkpointpath = './results_latest/model.tar'
        else:
            checkpointpath = os.path.expandvars(
                os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid,
                                                 'model.tar')))
        checkpoint = torch.load(checkpointpath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

    observation = env.initial()
    returns = []
    won = []
    entropy = []
    ep_len = []
    while len(won) < num_eps:
        done = False
        steps = 0
        while not done:
            if flags.random_agent:
                action = torch.zeros(1, 1, dtype=torch.int32)
                # action[0][0] = random.randint(0, gym_env.action_space.n - 1)
                action[0][0] = random.randint(0, len(gym_env.action_space) - 1)
                observation = env.step(action)
            else:
                agent_outputs = model(observation)
                observation = env.step(agent_outputs['action'])
                policy = F.softmax(agent_outputs['policy_logits'], dim=-1)
                log_policy = F.log_softmax(agent_outputs['policy_logits'], dim=-1)
                e = -torch.sum(policy * log_policy, dim=-1)
                entropy.append(e.mean(0).item())

            steps += 1
            done = observation['done'].item()
            if observation['done'].item():
                returns.append(observation['episode_return'].item())
                won.append(observation['reward'][0][0].item() > 0.5)
                ep_len.append(steps)

    env.close()
    logging.info('Average returns over %i episodes: %.2f. Win rate: %.2f. Entropy: %.2f. Len: %.2f', num_eps, sum(returns)/len(returns), sum(won)/len(returns), sum(entropy)/max(1, len(entropy)), sum(ep_len)/len(ep_len))

def main(flags):
    flags.num_buffers = 2 * flags.num_actors
    flags.model = 'paper_txt2pi'

    config_dict = dict(vars(flags))
    print('flags: ', config_dict)
    
    global Net
    Net = importlib.import_module('model.{}'.format(flags.model)).Model
    if flags.mode == 'train':
        with wandb.init(project="GroupsSimpleStationary", config=config_dict, 
        name='fromcheckpoint_singlegame_1lie1', dir='/scratch0/NOT_BACKED_UP/sml/christan/rtfm'):
            train(flags)
    elif flags.mode == 'test':
        test(flags)
    elif flags.mode == 'train_writer':
        flags.env = 'rock_paper_scissors-v0'
        flags.demb = 10
        flags.drnn_small = 10
        flags.drnn = 100
        flags.drep = 300
        flags.num_actors = 20
        flags.batch_size = 24
        flags.learning_rate = 0.0007
        flags.height = 10
        flags.width = 10
        flags.xpid = 'RPSdefault_butsize10'
        if flags.difficulty == 'character_level':
            train_meta_writer(flags, num_eps = flags.num_eps)
        elif flags.difficulty == 'statement_level':
            train_meta_writer(flags, difficulty = 'statement_level', num_eps = flags.num_eps)
    elif flags.mode == 'collect_rollouts':
        collect_rollouts(flags)


if __name__ == '__main__':
    parser = exp_utils.get_parser()
    flags = parser.parse_args()

    # flags.xpid = flags.xpid or exp_utils.compose_name(flags.model, flags.wiki, flags.env, flags.prefix)

    #run the experiment and time it
    start = time.time()
    main(flags)
    end = time.time()
    execution_time = end - start

    print('execution_time in seconds: ', execution_time)
