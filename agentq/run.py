"""Run, agentq, run!

Module for running agentq through a session.
"""
import sys
sys.path.insert(0, '/home/aaron/src/github.com/AAorris/agentq')

import gym
import tensorflow as tf

from agentq import actor, utils


def cartpole():
    env = gym.make('CartPole-v1')
    state = env.reset()

    observations = tf.placeholder(tf.float32, [None, env.spec.max_episode_steps])
    model = actor.thin_nn_model

    agentq = actor.create_actor(observations, model, env.action_space.n)

    for episode in range(10):
        rewards = []
        for transitions, prospects in utils.run_episode(env, agentq):
            discounted_reward, advantage = prospects
            rewards.append(discounted_reward)
        print('Avg Reward: {}'.format(sum(rewards) / float(len(rewards))))

cartpole()
