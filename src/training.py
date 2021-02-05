""" Functions that will be common to all.
"""
from functools import reduce

import torch
from torch import optim as O

import rlog
import src.estimators as E
from src.agents import AGENTS
from src.replay import ExperienceReplay
from src.rl_routines import Episode
from src.wrappers import get_env


def train_one_epoch(
    env,
    agent,
    epoch_step_cnt,
    update_freq,
    target_update_freq,
    opt,
    logger,
    total_steps=0,
    last_state=None,
):
    """ Policy iteration for a given number of steps. """

    replay, policy, policy_evaluation = agent
    policy_evaluation.estimator.train()
    policy_evaluation.target_estimator.train()

    while True:
        # do policy improvement steps for the length of an episode
        # if _state is not None then the environment resumes from where
        # this function returned.
        for transition in Episode(env, policy, _state=last_state):

            _state, _pi, reward, state, done = transition
            total_steps += 1

            # push one transition to experience replay
            replay.push((_state, _pi.action, reward, state, done))

            # learn if a minimum no of transitions have been pushed in Replay
            if replay.is_ready:
                if total_steps % update_freq == 0:
                    # sample from replay and do a policy evaluation step
                    batch = replay.sample()

                    # divide the learning rate by the spectral radius
                    if hasattr(opt.optim, "div_by_rho") and opt.optim.div_by_rho:
                        base_lr = opt.optim.args["lr"]
                        rhos = policy.estimator.get_spectral_norms()
                        for group in policy_evaluation.optimizer.param_groups:
                            if group["rho_idx"] is not None:
                                group["lr"] = base_lr / max(1, rhos[group["rho_idx"]])

                    # compute the loss and optimize
                    loss = policy_evaluation(batch).detach().item()
                    logger.put(trn_loss=loss, lrn_steps=batch[0].shape[0])

                if total_steps % target_update_freq == 0:
                    policy_evaluation.update_target_estimator()

            # some more stats
            logger.put(trn_reward=reward, trn_done=done, trn_steps=1)
            if (policy.estimator.spectral is not None) and (total_steps % 1000 == 0):
                logger.put(**policy.estimator.get_spectral_norms())
            if total_steps % 50_000 == 0:
                msg = "[{0:6d}] R/ep={trn_R_ep:2.2f}, tps={trn_tps:2.2f}"
                logger.info(msg.format(total_steps, **logger.summarize()))

            # exit if done
            if total_steps % epoch_step_cnt == 0:
                return total_steps, _state

        # This is important! It tells Episode(...) not to attempt to resume
        # an episode intrerupted when this function exited last time.
        last_state = None


def validate(policy, env, steps, logger):
    """ Validation routine """
    policy.estimator.eval()

    done_eval, step_cnt = False, 0
    with torch.no_grad():
        while not done_eval:
            for _, _, reward, _, done in Episode(env, policy):
                logger.put(reward=reward, done=done, val_frames=1)
                step_cnt += 1
                if step_cnt >= steps:
                    done_eval = True
                    break
    env.close()


def get_estimator(opt, env):
    estimator_args = opt.estimator.args
    if opt.estimator.name == "MLP":
        estimator_args["layers"] = [
            reduce(lambda x, y: x * y, env.observation_space.shape),
            *estimator_args["layers"],
        ]
    if opt.estimator.name in ("MinAtarNet", "RandomMinAtarNet"):
        estimator_args["input_ch"] = env.observation_space.shape[-1]
    estimator = getattr(E, opt.estimator.name)(opt.action_cnt, **estimator_args)
    estimator.to(opt.device)

    if (opt.agent.name == "DQN") and ("support" in opt.estimator.args):
        raise ValueError("DQN estimator should not have a support.")
    if (opt.agent.name == "C51") and ("support" not in opt.estimator.args):
        raise ValueError("C51 requires an estimator with support.")

    return estimator


def get_optimizer(opt, estimator):
    # Create custom param groups
    if hasattr(opt.optim, "div_by_rho") and opt.optim.div_by_rho:
        assert (
            opt.estimator.args["spectral"] is not None
        ), "When dividing by rho you should hook at least a layer."
        assert all(
            [s[-1] == "L" for s in str(opt.estimator.args["spectral"]).split(",")]
        ), "Spectral norm layers should not be active when dividing the optim step."

        param_groups = [
            {"params": p, "name": n, "lr": opt.optim.args["lr"], "rho_idx": None}
            for n, p in estimator.named_parameters()
        ]
        param_groups_ = [g for g in param_groups if "weight" in g["name"]]
        for k in estimator.get_spectral_norms().keys():
            param_groups_[int(k)]["rho_idx"] = k
    else:
        param_groups = estimator.parameters()

    optimizer = getattr(O, opt.optim.name)(param_groups, **opt.optim.args)

    if hasattr(opt.optim, "div_by_rho") and opt.optim.div_by_rho:
        rlog.info("Checking the groups are alright, alright, alright...")
        for group in optimizer.param_groups:
            rlog.info("{:<36} rho_idx={}".format(group["name"], group["rho_idx"]))

    return optimizer


def experiment_factory(opt):
    """ Configures an environment and an agent """
    env = get_env(opt)
    opt.action_cnt = env.action_space.n

    # we use the warmup steps in the epsilon-greedy schedule to set the warmup
    # in the replay. If epsilon-greedy is not a schedule the warmup will be
    # given either replay.args.warmup_steps or by batch-size
    if isinstance(opt.agent.args["epsilon"], list):
        opt.replay["warmup_steps"] = opt.agent.args["epsilon"][-1]
    replay = ExperienceReplay(**opt.replay)
    assert (
        replay.warmup_steps == opt.agent.args["epsilon"][-1]
    ), "warmup steps and warmup epsilon should be equal."

    estimator = get_estimator(opt, env)
    optimizer = get_optimizer(opt, estimator)
    policy_evaluation = AGENTS[opt.agent.name]["policy_evaluation"](
        estimator, optimizer, **opt.agent.args
    )
    policy_improvement = AGENTS[opt.agent.name]["policy_improvement"](
        estimator, opt.action_cnt, **opt.agent.args
    )
    return env, (replay, policy_improvement, policy_evaluation)
