"""Structure for an actor object using tensorflow.

Meant to be returned by a 'create actor' function.

Attributes:
    output_actions (Tensor): holding a batch of actions
    update_epsilon_expr (Expression): update the chance of actions being random
    new_epsilon (Tensor): the constant float value to update epsilon to
"""

from collections import namedtuple

Actor = namedtuple('Actor', [
    'output_actions', 'update_epsilon_expr', 'new_epsilon'
])
