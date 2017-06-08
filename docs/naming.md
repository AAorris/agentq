# Variable naming guide

#### `{}_ph` : Placeholder

Represents a placeholder tensor given by tensorflow.

#### epsilon : Chance of a random action

Epsilon represents the probability of a random action, where 1.0
means the actions are always random.

#### gamma : Discount rate

When predicting future rewards, we slowly devalue them as we are less sure that
they will actually happen. A high gamma means less trust for the future.
