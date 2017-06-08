"""Models for stochastic actions."""

# Only supports tensorflow
# But would like to take a generic "engine" argument if possible.

from tensorflow import int64, float32, random_uniform, stack, where


def random_actions(batch_size, num_actions):
    """Simple uniform random action between 0 and num_actions.

    Args:
        batch_size (int): number of predictions to make
        num_actions (int): number of total actions available

    Returns:
        int64: chosen action
    """
    return random_uniform(
        stack([batch_size]), minval=0, maxval=num_actions, dtype=int64
    )


def sometimes_random_actions(
        batch_size, num_actions, nonrand, random_probability, rand=None):
    """Return actions that are random... sometimes.

    Used to introduce random actions into your model, where the chance of
    random actions may decrease or increase over time.

    Args:
        batch_size (int): number of actions to generate
        num_actions (int): possible actions to choose from
        nonrand (Tensor): to evaluate when random_chance is not satisfied
        random_probability (Tensor): chance (0-1) of choosing a random action
        rand (Tensor): used to evaluate a random action. Default: uniform

    Returns:
        int64: either random or nonrandom action
    """
    if not rand:
        rand = random_actions(batch_size, num_actions)  # can specify others

    random_rolls = random_uniform(
        stack([batch_size]), minval=0, maxval=1, dtype=float32
    )

    is_random = random_rolls < random_probability  # 1 is always random

    actions = where(is_random, rand, nonrand)
