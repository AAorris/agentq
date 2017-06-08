"""Action selection agent API."""

from collections import namedtuple

from tensorflow import argmax, cond, float32, shape, name_scope, placeholder
from tensorflow import get_variable as variable
from tensorflow import constant_initializer as const
from tensorflow.contrib.layers import fully_connected

from .models.random_action import sometimes_random_actions
from .utils.actor import Actor


def thin_nn_model(observations, num_actions, scope):
    """Create a thin neural network model."""
    with name_scope(scope):
        layer = observations
        for hidden_neurons in [100, 25, num_actions]:
            layer = fully_connected(layer, hidden_neurons)
        return layer


def create_actor(observations, create_model, num_actions, random_model=None):
    """Create an actor.

    Args:
        observations (Tensor): from the environment
        model (Tensor): action scores (we will apply argmax here)
        num_actions (int): the number of actions in the model
        random_model (Optional[Tensor]): If not specified, is uniformly random

    Returns:
        dict: {new_epsilon, update_epsilon_expr, output_actions}
            new_epsilon (Tensor): holding a single float probability
            update_epsilon_expr (expression): to run in a tensorflow session
            output_actions (Tensor): the generated batch of outputs

    Example:
        actor = create_actor(states, my_model, env.action_space.n)
    """
    batch_size = shape(observations)[0]

    # epsilon drives how often random actions are selected.
    epsilon = variable("epsilon", (), initializer=const(0))

    # Action values from the model, with the highest value's index is chosen.
    values = create_model(observations, num_actions, scope="q_model")
    value_max = argmax(values, axis=1)

    # Apply chance of a random action.
    output_actions = sometimes_random_actions(
        batch_size, num_actions, nonrand=value_max,
        random_probability=epsilon
    )

    # Allow updates to epsilon (the chance of a random action)
    new_epsilon = placeholder(float32, (), name="new_epsilon")
    update_epsilon_expr = epsilon.assign(
        cond(new_epsilon >= 0, lambda: new_epsilon, lambda: epsilon)
    )

    return Actor(output_actions, update_epsilon_expr, new_epsilon)
