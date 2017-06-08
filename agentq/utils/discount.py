"""Discount is a concept where reward is summed up.

When discounting rewards, a level of "impatience" causes farther off
rewards to have less value than short term rewards.
"""


def discount(rewards, impatience):
    """Sum the rewards, where far off rewards are devalued by impatience.

    For example, if rewards were [1, 1, 1, 1] and impatience was 0.5,
    returns: sum([1, 0.5, 0.25, 0.125])

    Args:
        rewards (list): list of rewards to accumulate
        impatience (float): factor to discount the rewards at
    """
    total, factor = 0, 1.0
    for r in rewards:
        total += r * factor
        factor *= impatience
    return total


def run_review(agent, transitions):
    """Run a review of a gym episode with the agent."""
    if not isinstance(transitions, list):
        transitions = list(transitions)

    _future_rewards = deque([item.reward for item in transitions])
    for i, current in enumerate(transitions):
        _future_rewards.popleft()
        _current_value = agent.value(current.state)
        discounted_reward = discount(_future_rewards, 0.97)
        advantage = discounted_reward - _current_value
        yield discounted_reward, advantage
