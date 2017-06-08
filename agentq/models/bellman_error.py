"""Bellman's equation implementation for error in learning.

The equation is an optimzation problem. What we optimize should be
'an objective function' that is minimizing or maximizing something.

The equation requires state. We feed in observations of state.

We will define an optimal plan by defining a 'policy function'
that decides on actions based on a given state.

But how is the plan optimal? For this, we must define a
'value function', that will provide feedback on how effective the
policy was.


[V] The Value of a given state is

    [max] (with action 'a' a being in the set of good actions)

        [F(x, a)] The payoff of an action
        plus

        [B] The discount factor, or 'impatience'
        times

        [V] The value of

            [T(x, a)] The estimated resulting state

Or succinctly as ``V(x) = max(F(x, a) + BV(T(x, a)))``

When optimized, this equation will give us the optimal actions to take
from then onwards, regardless of the previous history.

But to optimize the equation, we will need to find that policy.
In our case, we will be using 'backward induction' to solve the equation.
This does the reverse.

    Given the history of an attempt:
        We know the value of the previous step, and we know its action.
        We can define the policy as choosing those actions from those states
        And calculate the error(between our internal model and the results).

        I assume that the error is introduced using noise and exploration.
        By exploring, we may find improvements to our model. When we see
        greater value in the history of our attempt, we can optimize to
        minimize the loss between our model and the run, thereby getting
        closer to the optimal action.

"""

from collections import namedtuple





def backwards_induction(measurements, actions, payoffs, impatience,
):
