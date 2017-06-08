"""Huber loss implementation - https://en.wikipedia.org/wiki/Huber_loss."""


def huber_loss(x, delta=1.0, math):
    """Calculate huber loss.

    Commonly used are squared loss and absolute loss.
    Each has their ups and downs.

    Huber loss takes the best of both worlds by using:

        squared loss on 'small' values and
        absolute loss on 'large' values.

    The switch is a simple if statement, where squared loss will be used if

        abs(x) < delta.

    Rather than plain squared or absolute loss, the following are calculated:

        small: 0.5 * x ^ 2
        large: delta * (|x| - 0.5 * delta)

    The result:

        loss is quadratic close to 0
        loss is linear closer to |infinity|

    This system is less sensitive to outliers with large losses.

    Args:
        x (number): to process
        delta (float): what is considered 'small'
        math (Object): math implementation containing: {where, abs, square}

    Returns:
        float: loss
    """
    return math.where(
        math.abs(x) < delta,
        0.5 * math.square(x),
        delta * (math.abs(x) - 0.5 * delta)
    )
