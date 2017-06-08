"""Openai Gym utilities."""

from random import random
from argparse import Namespace
from collections import namedtuple, deque
from itertools import imap, islice, count, repeat, starmap

import numpy as np

from .transition import Transition
from .discount import run_review


def run_gym(env, agent):
    """Run a gym episode."""
    state = env.reset()
    done = False
    while not done:
        action = agent.action(state)
        newstate, reward, done, _ = env.step(action)
        yield Transition(state, action, newstate, reward)
        state = newstate


def run_episode(env, agent):
    transitions = run_gym(env, agent)
    prospects = run_review(agent, transitions)
    return transitions, prospects
