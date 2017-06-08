"""Session using baselines code to build an agent."""

import itertools

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from baselines import deepq, logger
from baselines.common import tf_util
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.deepq.simple import ActWrapper


def model(inpt, num_actions, scope, reuse=False):
    """Pass a source through a model of neural layers.

    Args:
        name (str): of the tensorflow scope to build under
        source (`Tensor`): input tensor
        num_actions (int): available output actions
        reuse (bool): If true, reuse existing variable scopes
    """
    with tf.variable_scope(scope, reuse=reuse):
        return fully_connected(
            fully_connected(inpt, num_outputs=64, activation_fn=tf.nn.tanh),
                num_outputs=num_actions, activation_fn=None)


def train(model_file, game="CartPole-v1"):
    """Train at a game."""
    with tf_util.make_session(8):
        env = gym.make(game)

        def make_placeholder(name):
            """Make a placeholder input."""
            return tf_util.BatchInput(env.observation_space.shape, name=name)

        act_params = {
            'make_obs_ph': make_placeholder,
            'q_func': model,
            'num_actions': env.action_space.n
        }
        act, train, update_target, debug = deepq.build_train(
            **act_params,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4)
        )
        act = ActWrapper(act, act_params)

        replay_buffer = ReplayBuffer(50000)

        exploration = LinearSchedule(
            schedule_timesteps=100000,
            initial_p=1.0,
            final_p=0.02
        )

        tf_util.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = env.reset()
        for t in itertools.count():
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action)
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0)
            if not len(episode_rewards) % 100:
                env.render()

            if t > 1000:
                obses_t, actions, rewards, obses_tp1, dones = (
                    replay_buffer.sample(32)
                )
                train(
                    obses_t, actions, rewards, obses_tp1, dones,
                    np.ones_like(rewards)
                )
            if not t % 1000:
                update_target()
            if not t % 3000:
                if model_file:
                    tf_util.save_state(model_file)
                yield act

            if done and len(episode_rewards) % 10 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular(
                    "mean episode reward",
                    round(np.mean(episode_rewards[-101:-1]), 1)
                )
                logger.record_tabular(
                    "% time spent exploring",
                    int(100 * exploration.value(t))
                )
                logger.dump_tabular()
