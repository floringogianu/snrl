from functools import reduce

import bsuite
import gym
import torch
from bsuite.utils import gym_wrapper
from liftoff import parse_opts

import rlog
import src.io_utils as ioutil
from src.c51 import C51PolicyEvaluation, C51PolicyImprovement
from src.estimators import MLP
from src.replay import ExperienceReplay
from src.rl_routines import Episode
from src.wrappers import TorchWrapper


def experiment_factory(opt, only_env=False):
    env = gym_wrapper.GymFromDMEnv(bsuite.load_from_id(opt.env.name))
    env = TorchWrapper(env, opt.device)
    if only_env:
        return env

    replay = ExperienceReplay(**opt.replay)
    layers = [
        reduce(lambda x, y: x * y, env.observation_space.shape),  # input
        *opt.estimator["layers"],  # hidden
        env.action_space.n,  # output
    ]
    estimator = MLP(layers, spectral=opt.spectral, **opt.estimator)
    estimator.to(opt.device)

    optimizer = getattr(torch.optim, opt.optim.name)(
        estimator.parameters(), **opt.optim.kwargs
    )
    policy_improvement = C51PolicyImprovement(
        estimator, opt.epsilon, env.action_space.n
    )
    policy_evaluation = C51PolicyEvaluation(estimator, optimizer, opt.gamma)
    rlog.info(replay)
    rlog.info(estimator)
    return env, (replay, policy_improvement, policy_evaluation)


def game_settings_(opt):
    env = {
        "name": opt.game,
        "settings": bsuite.sweep.SETTINGS[opt.game],
        "episodes": bsuite.sweep.EPISODES[opt.game],
        "id": int(opt.game.split("/")[1]),
    }
    opt.env = ioutil.dict_to_namespace(env)
    if opt.valid_freq is None:
        opt.valid_freq = int(opt.env.episodes / 20)
    return opt


def train_one_ep(env, agent, steps, update_freq, target_update_freq):
    """ Train the agent for one episode. """
    replay, policy, policy_evaluation = agent
    policy_evaluation.estimator.train()
    policy_evaluation.target_estimator.train()

    for transition in Episode(env, policy):
        _state, _pi, reward, state, done = transition
        steps += 1

        # push one transition to experience replay
        replay.push((_state, _pi.action, reward, state, done))

        if replay.is_ready:
            if steps % update_freq == 0:
                # sample from replay and do a policy evaluation step
                batch = replay.sample()
                loss = policy_evaluation(batch).detach().item()
                rlog.put(trn_loss=loss, lrn_steps=batch[0].shape[0])

            if steps % target_update_freq == 0:
                policy_evaluation.update_target_estimator()

        rlog.put(trn_reward=reward, trn_done=done)
    return steps


def validate(env, agent, valid_ep_cnt):
    """ Validate the agent. """
    _, policy, _ = agent
    policy.estimator.eval()
    policy = type(policy)(policy.estimator, 0.0, policy.action_no)

    ep_rw = []
    with torch.no_grad():
        for _ in range(valid_ep_cnt):
            episode = Episode(env, policy)
            for _, _, reward, _, done in episode:
                rlog.put(reward=reward, done=done, val_frames=1)
                continue
            ep_rw.append(episode.Rt)

    env.close()
    return ep_rw


def run(opt):
    """ Entry Point. """

    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_R_ep", metargs=["trn_reward", "trn_done"]),
        rlog.AvgMetric("trn_loss", metargs=["trn_loss", 1]),
        rlog.FPSMetric("lrn_tps", metargs=["lrn_steps"]),
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )

    opt = game_settings_(opt)
    env, agent = experiment_factory(opt)

    rlog.info(ioutil.config_to_string(opt))
    ioutil.save_config(opt, opt.out_dir)

    steps = 0
    for ep in range(1, opt.env.episodes + 1):
        steps = train_one_ep(env, agent, steps, opt.update_freq, opt.target_update_freq)

        if ep % opt.valid_freq == 0:
            rlog.traceAndLog(ep)
            validate(env, agent, opt.valid_episodes)
            rlog.traceAndLog(ep)


def main():
    """ Liftoff
    """
    run(parse_opts())


if __name__ == "__main__":
    main()
