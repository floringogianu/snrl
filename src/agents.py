r""" Module containing the C51 Policy Improvement class.
"""
import collections
from copy import deepcopy

import torch
from numpy import random
from torch.nn.utils import clip_grad_norm_

from src.rl_utils import (
    C51Loss,
    DQNLoss,
    EpsilonGreedyOutput,
    get_estimator_device,
    to_device,
)
from src.schedules import constant_schedule, get_schedule


def get_target_distribution(
    next_states, rewards, mask, gamma, target_estimator, support
):
    r""" Computes the target distribution TZ(x_,a).

    The size of `next_states` can be smaller than that of the actual
    batch size to save computation.
    """
    bsz = rewards.shape[0]
    bsz_ = next_states.shape[0]
    bin_no = support.shape[0]
    v_min, v_max = support[0].item(), support[-1].item()
    delta_z = (v_max - v_min) / (bin_no - 1)

    probs = target_estimator(next_states, probs=True)
    qs = torch.mul(probs, support.expand_as(probs))
    argmax_a = qs.sum(2).max(1)[1].unsqueeze(1).unsqueeze(1)
    action_mask = argmax_a.expand(bsz_, 1, bin_no)
    _qa_probs = probs.gather(1, action_mask).squeeze()

    # Next-states batch can be smaller so we scatter qa_probs in
    # a tensor the size of the full batch with each row summing to 1
    qa_probs = torch.eye(bsz, bin_no, device=_qa_probs.device)
    qa_probs.masked_scatter_(mask.expand_as(qa_probs), _qa_probs)

    # Mask gamma and reshape it torgether with rewards to fit p(x,a).
    rewards = rewards.expand_as(qa_probs)
    gamma = (mask.float() * gamma).expand_as(qa_probs)

    # Compute projection of the application of the Bellman operator.
    bellman_op = rewards + gamma * support.unsqueeze(0).expand_as(rewards)
    bellman_op = torch.clamp(bellman_op, v_min, v_max)

    # Compute categorical indices for distributing the probability
    m = torch.zeros(bsz, bin_no, device=qa_probs.device)
    b = (bellman_op - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()

    # Fix disappearing probability mass when l = b = u (b is int)
    l[(u > 0) * (l == u)] -= 1
    u[(l < (bin_no - 1)) * (l == u)] += 1

    # Distribute probability
    """
    for i in range(bsz):
        for j in range(self.bin_no):
            uidx = u[i][j]
            lidx = l[i][j]
            m[i][lidx] = m[i][lidx] + qa_probs[i][j] * (uidx - b[i][j])
            m[i][uidx] = m[i][uidx] + qa_probs[i][j] * (b[i][j] - lidx)
    for i in range(bsz):
        m[i].index_add_(0, l[i], qa_probs[i] * (u[i].float() - b[i]))
        m[i].index_add_(0, u[i], qa_probs[i] * (b[i] - l[i].float()))
    """
    # Optimized by https://github.com/tudor-berariu
    offset = (
        torch.linspace(0, ((bsz - 1) * bin_no), bsz, device=qa_probs.device)
        .long()
        .unsqueeze(1)
        .expand(bsz, bin_no)
    )

    m.view(-1).index_add_(
        0, (l + offset).view(-1), (qa_probs * (u.float() - b)).view(-1)
    )
    m.view(-1).index_add_(
        0, (u + offset).view(-1), (qa_probs * (b - l.float())).view(-1)
    )
    return m, probs


def get_categorical_loss(batch, estimator, gamma, support, target_estimator=None):
    r""" Computes the C51 KL loss phi(TZ(x_,a)) || Z(x,a).

        Largely analogue to the DQN loss routine.

        Args:
            batch (list): The (states, actions, rewards, next_states, done_mask)
                batch.
            estimator (nn.Module): The *online* estimator.
            gamma (float): Discount factor γ.
            support (torch.tensor): A vector of size `bin_no` that is the
                support of the categorical distribution.
            target_estimator (nn.Module, optional): Defaults to None. The target
                estimator. If None the target is computed using the online
                estimator.

        Returns:
            DQNLoss: A simple namespace containing the loss and its byproducts.
    """
    assert target_estimator is not None, "Online target not implemented yet."

    states, actions, rewards, next_states, mask = batch
    bsz = states.shape[0]
    bin_no = support.shape[0]

    # Compute probability distribution of Q(s, a)
    log_qs_probs = estimator(states, log_probs=True)
    action_mask = actions.view(bsz, 1, 1).expand(bsz, 1, bin_no)
    log_qsa_probs = log_qs_probs.gather(1, action_mask).squeeze()

    # Compute probability distribution of Q(s_, a)
    with torch.no_grad():
        target_qsa_probs, target_qs_probs = get_target_distribution(
            next_states, rewards, mask, gamma, target_estimator, support
        )

    # Compute the cross-entropy of phi(TZ(x_,a)) || Z(x,a) for each
    # transition in the batch
    loss = -torch.sum(target_qsa_probs * log_qsa_probs, 1)

    return C51Loss(
        loss=loss,
        support=support,
        qsa_probs=log_qsa_probs,
        target_qsa_probs=target_qsa_probs,
        qs_probs=log_qs_probs,
        target_qs_probs=target_qs_probs,
    )


class PolicyImprovement:
    r""" C51 DQN policy improvement.
    """

    def __init__(self, estimator, action_space, epsilon=0.1, **kwargs):
        self.estimator = estimator
        self.action_no = action_space

        self.epsilon_args = epsilon
        if isinstance(epsilon, float):
            self.epsilon = constant_schedule(epsilon)
        elif isinstance(epsilon, (tuple, list)):
            if not isinstance(epsilon[0], str):
                self.epsilon_args = epsilon = ["linear", *epsilon]
            self.epsilon = get_schedule(*epsilon)
        elif isinstance(epsilon, collections.abc.Iterator):
            self.epsilon = epsilon

    def act(self, state):
        """ Epsilone Greedy Policy Improvement step."""
        epsilon = next(self.epsilon)
        if epsilon < random.uniform():
            qvals = self.estimator(state)
            q_val, argmax_a = qvals.max(1)
            return EpsilonGreedyOutput(
                action=argmax_a.item(), q_value=q_val, full=qvals
            )
        return EpsilonGreedyOutput(
            action=random.randint(self.action_no), q_value=None, full=None
        )

    def __call__(self, state):
        return self.act(state)

    def __str__(self):
        if isinstance(self.epsilon_args, list):
            epsilon = ",".join([str(i) for i in self.epsilon_args])
        else:
            epsilon = self.epsilon
        name = self.__class__.__name__
        name += f"(ε={epsilon})"
        return name

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f"{name} @ {obj_id}"


class C51PolicyEvaluation:
    r""" Categorical C51 DQN.

    For more information see `A distributional perspective on RL
    <https://arxiv.org/pdf/1707.06887.pdf>`_.

    """

    def __init__(
        self,
        estimator,
        optimizer,
        gamma=0.99,
        target_estimator=True,
        clip_grad_norm=False,
        div_grad_by_rho=False,
        **kwargs,
    ):
        self.device = get_estimator_device(estimator)
        self.estimator = estimator
        self.optimizer = optimizer
        self.gamma = gamma
        self.support = support = estimator.support.clone()
        self.v_min, self.v_max = support[0].item(), support[-1].item()
        self.bin_no = self.support.shape[0]
        if target_estimator in (True, None):
            self.target_estimator = deepcopy(estimator)
        else:
            self.target_estimator = target_estimator
        self.clip_grad_norm = clip_grad_norm
        self.div_grad_by_rho = div_grad_by_rho

    def __call__(self, batch, cb=None, update=True, backward=True):
        batch = to_device(batch, self.device)

        loss = get_categorical_loss(
            batch,
            self.estimator,
            self.gamma,
            self.support,
            target_estimator=self.target_estimator,
        )

        if backward:
            if cb:
                loss = cb(loss)
            else:
                loss = loss.loss.mean()

            loss.backward()

        if update:
            self.update_estimator()

        return loss

    def update_estimator(self):
        r""" Do the estimator optimization step. Usefull when computing
        gradients across several steps/batches and optimizing using the
        accumulated gradients.
        """
        if self.clip_grad_norm:
            clip_grad_norm_(self.estimator.parameters(), 10.0)
        elif self.div_grad_by_rho:
            self.estimator.div_grad_by_rho()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_target_estimator(self):
        r""" Update the target net with the parameters in the online model."""
        self.target_estimator.load_state_dict(self.estimator.state_dict())

    def get_estimator_state(self):
        r""" Return a reference to the estimator. """
        return self.estimator.state_dict()

    def __str__(self):
        lr = self.optimizer.param_groups[0]["lr"]
        name = self.__class__.__name__
        name += f"(\u03B3={self.gamma}, \u03B1={lr}, "
        name += f"Ω=[{self.v_min, self.v_max}], bins={self.bin_no})"
        return name

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f"{name} @ {obj_id}"


""" Deep Q-Learning policy improvement.
"""


def get_ddqn_targets(qsa_target, q_targets, mask, estimator, next_states):
    """ Compute the DDQN argmax_a Q(s', a')
    """
    with torch.no_grad():
        next_q_values = estimator(next_states)
        argmax_actions = next_q_values.max(1, keepdim=True)[1]
        qsa_target[mask] = q_targets.gather(1, argmax_actions)
    return qsa_target


def get_dqn_loss(  # pylint: disable=bad-continuation
    batch,
    estimator,
    gamma,
    target_estimator=None,
    is_double=False,
    loss_fn=torch.nn.MSELoss(reduction="mean"),
):
    r""" Computes the DQN loss or its Double-DQN variant.

    Args:
        batch (list): The (states, actions, rewards, next_states, done_mask)
            batch.
        estimator (nn.Module): The *online* estimator.
        gamma (float): Discount factor γ.
        target_estimator (nn.Module, optional): Defaults to None. The target
            estimator. If None the target is computed using the online
            estimator.
        is_double (bool, optional): Defaults to False. If True it computes
            the Double-DQN loss using the `target_estimator`.
        loss_fn (torch.nn.Loss): Defaults to torch.nn.MSELoss. Custom loss
            function, eg.: torch.nn.SmoothL1Loss.

    Returns:
        DQNLoss: A simple namespace containing the loss and its byproducts.
    """

    states, actions, rewards, next_states, mask = batch

    # Compute Q(s, a)
    q_values = estimator(states)
    qsa = q_values.gather(1, actions)
    mask = mask.squeeze(1)

    if next_states.nelement() != 0:
        # Compute Q(s_, a).
        if target_estimator:
            with torch.no_grad():
                q_targets = target_estimator(next_states)
        else:
            with torch.no_grad():
                q_targets = estimator(next_states)
    else:
        q_targets = None

    # Bootstrap for non-terminal states
    qsa_targets = torch.zeros_like(qsa)

    if q_targets is not None:
        if is_double:
            qsa_targets = get_ddqn_targets(
                qsa_targets, q_targets, mask, estimator, next_states
            )
        else:
            qsa_targets[mask] = q_targets.max(1, keepdim=True)[0]

    # Compute temporal difference error
    qsa_targets = (qsa_targets * gamma) + rewards
    loss = loss_fn(qsa, qsa_targets)

    return DQNLoss(
        loss=loss,
        qsa=qsa,
        qsa_targets=qsa_targets,
        q_values=q_values,
        q_targets=q_targets,
    )


class DQNPolicyEvaluation:
    r""" Object doing the Deep Q-Learning Policy Evaluation step.

    As other objects in this library we override :attr:`__call__`. During a
    call as the one in the example below, several things happen:

        1. Put the batch on the same device as the estimator,
        2. Compute DQN the loss,
        3. Calls the callback if available (eg.: when doing prioritized
           experience replay),
        4. Computes gradients and updates the estimator.

    Example:

    .. code-block:: python

        # construction
        policy_improvement = DQNPolicyImprovement(
            estimator,
            optim.Adam(estimator.parameters(), lr=0.25),
            gamma,
        )

        # usage
        for step in range(train_steps):
            # sample the env and push transitions in experience replay
            batch = experience_replay.sample()
            policy_improvement(batch, cb=None)

            if step % target_update_freq == 0:
                policy_improvement.update_target_estimator()

    Args:
        estimator (nn.Module): Q-Values estimator.
        optimizer (nn.Optim): PyTorch optimizer.
        gamma (float): Discount factor.
        target_estimator (nn.Module, optional): Defaults to None. This
            assumes we always want a target network, since it is a DQN
            update. Therefore if `None`, it will clone `estimator`. However
            if `False` the update rule will use the online network for
            computing targets.
        is_double (bool, optional): Defaults to `False`. Whether to use
            Double-DQN or not.
    """

    # pylint: disable=too-many-arguments, bad-continuation
    def __init__(
        self,
        estimator,
        optimizer,
        gamma=0.99,
        target_estimator=None,
        is_double=False,
        loss_fn="SmoothL1Loss",
        div_grad_by_rho=False,
        **kwargs,
    ):
        # pylint: enable=bad-continuation
        self.estimator = estimator
        self.target_estimator = target_estimator
        if target_estimator in (True, None):
            self.target_estimator = deepcopy(estimator)
        else:
            self.target_estimator = target_estimator
        self.optimizer = optimizer
        self.gamma = gamma
        self.is_double = is_double
        self.loss_fn = getattr(torch.nn, loss_fn)(reduction="none")
        self.div_grad_by_rho = div_grad_by_rho
        self.device = get_estimator_device(estimator)
        self.optimizer.zero_grad()

    def __call__(self, batch, cb=None, update=True, backward=True):
        r""" Performs a policy improvement step. Several things happen:
            1. Put the batch on the device the estimator is on,
            2. Computes DQN the loss,
            3. Calls the callback if available,
            4. Computes gradients and updates the estimator.

        Args:
            batch (list): A (s, a, r, s_, mask, (meta, optional)) list. States
                and States_ can also be lists of tensors for composed states
                (eg. frames + nlp_instructions).
            cb (function, optional): Defaults to None. A function performing
                some other operations with/on the `dqn_loss`. Examples
                include weighting the loss and updating priorities in
                prioritized experience replay.
        """

        batch = to_device(batch, self.device)

        dqn_loss = get_dqn_loss(
            batch,
            self.estimator,
            self.gamma,
            target_estimator=self.target_estimator,
            is_double=self.is_double,
            loss_fn=self.loss_fn,
        )

        if backward:
            if cb:
                loss = cb(dqn_loss)
            else:
                loss = dqn_loss.loss.mean()

            loss.backward()

        if update:
            self.update_estimator()

        return loss

    def update_estimator(self):
        r""" Do the estimator optimization step. Usefull when computing
        gradients across several steps/batches and optimizing using the
        accumulated gradients.
        """
        if self.div_grad_by_rho:
            self.estimator.div_grad_by_rho()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_target_estimator(self):
        r""" Update the target net with the parameters in the online model."""
        self.target_estimator.load_state_dict(self.estimator.state_dict())

    def get_estimator_state(self):
        r""" Return a reference to the estimator. """
        return self.estimator.state_dict()

    def __str__(self):
        lr = self.optimizer.param_groups[0]["lr"]
        name = self.__class__.__name__
        if self.is_double:
            name = f"Double{name}"
        return name + f"(\u03B3={self.gamma}, \u03B1={lr}, loss={self.loss_fn})"

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f"{name} @ {obj_id}"


AGENTS = {
    "C51": {
        "policy_improvement": PolicyImprovement,
        "policy_evaluation": C51PolicyEvaluation,
    },
    "DQN": {
        "policy_improvement": PolicyImprovement,
        "policy_evaluation": DQNPolicyEvaluation,
    },
}


def main():
    from torch import nn

    action_no = 3
    bsz = 6
    bsz_ = 4
    support = (-1, 1, 7)

    class Net(nn.Module):
        def __init__(self, action_no, bin_no):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(3 * 7 * 7, 24)
            self.fc2 = nn.Linear(24, action_no * bin_no)
            self.action_no, self.bin_no = action_no, bin_no

        def forward(self, x, **kwargs):
            y = self.fc2(torch.relu(self.fc1(x)))
            y = y.view(x.shape[0], self.action_no, self.bin_no)
            return torch.softmax(y, dim=2)

    x = torch.rand(bsz_, 3 * 7 * 7)
    rewards = torch.zeros(bsz, 1)
    mask = torch.ones(bsz, 1, dtype=torch.bool)
    rewards[2, 0] = 0.33
    mask[2, 0] = False
    mask[4, 0] = False

    net = Net(action_no, support[2])
    y = net(x)

    print(
        "target p(Q(s_,a)):\n",
        get_target_distribution(x, rewards, mask, 0.92, net, torch.linspace(*support)),
    )


if __name__ == "__main__":
    main()
