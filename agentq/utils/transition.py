"""A transition is an element that holds a before and after state."""

from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'newstate', 'reward'])
