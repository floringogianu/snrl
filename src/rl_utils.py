""" Utility functions used across the library.
"""
from typing import NamedTuple
import torch


__all__ = ["DQNLoss", "C51Loss", "get_estimator_device", "to_device"]


class EpsilonGreedyOutput(NamedTuple):
    """ The output of the epsilon greedy policy. """

    action: int
    q_value: float
    full: object


class DQNLoss(NamedTuple):
    r""" Object returned by :attr:`get_dqn_loss`. """

    loss: torch.Tensor
    qsa: torch.Tensor
    qsa_targets: torch.Tensor
    q_values: torch.Tensor
    q_targets: torch.Tensor


class C51Loss(NamedTuple):
    r""" Object returned by :attr:`get_categorical_loss`. """
    loss: torch.Tensor
    support: torch.Tensor
    qsa_probs: torch.Tensor
    target_qsa_probs: torch.Tensor
    qs_probs: torch.Tensor
    target_qs_probs: torch.Tensor


def get_estimator_device(estimator):
    r""" Returns the estimator's device.
    """
    params = estimator.parameters()
    if isinstance(params, list):
        return next(params[0]["params"]).device
    return next(params).device


def to_device(data, device):
    r""" Moves the data on the specified device irrespective of
    data being a tensor or a tensor in a container.

    Usefull in several situations:
        1. move a `batch: [states, rewards, ...]`
        2. move a `state: [image, instructions]`
        3. move a batch of mixed types.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(el, device) for el in data]
    return data.to(device)
