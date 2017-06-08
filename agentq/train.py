"""Q Learning training model."""


def create_trainer(model, target_model, num_actions, optimizer, gamma):
    """Create a trainer for a q learning network.

    Args:
        model (Tensor): to train
        target_model (Tensor): to compare against and send updates to.
        num_actions (int): number of possible actions
        optimizer (Optimizer): tensorflow optimizer like AdamOptimizer
        gamma (float): rate at which to disregard future rewards

    Returns:
        dict: {
            train (callable): optimizes error in Bellman's equation. (?)
            update_target (callable): copies model parameters to target network
        }
    """
