""" Offline script for checking the  effective rank of models.
"""
from copy import deepcopy
from pathlib import Path

import rlog
import torch
from liftoff import parse_opts

import src.io_utils as ioutil
from src.agents import AGENTS
from src.rl_routines import Episode
from src.training import get_estimator
from src.wrappers import get_env


def get_activation(activations):
    def hook(_, __, output):
        activations.append(output.detach().cpu())

    return hook


def compute_rank(phi, delta=0.01):
    """ Approximate rank as the first k values that capture more than (1-delta)%
        variance of all singular values, i.e.: sum_i^k s_i / sum_i^d s_i.
        https://arxiv.org/pdf/2010.14498.pdf
        https://arxiv.org/pdf/1909.12255.pdf
    """
    _, cols = phi.shape
    sigmas = torch.svd(phi.t() @ phi).S.sqrt()
    a, b = 0, sigmas.sum()
    for k in range(cols):
        a += sigmas[k]
        if a / b >= (1 - delta):
            return k + 1


def check_effective_features_rank(policy, env, steps):
    """ Computes the effective rank of 𝚽. """
    policy.estimator.eval()

    activations = []
    policy.estimator.head[1].register_forward_hook(get_activation(activations))

    done_eval, step_cnt = False, 0
    with torch.no_grad():
        while not done_eval:
            for _, _, reward, _, done in Episode(env, policy):
                rlog.put(reward=reward, done=done, val_frames=1)
                step_cnt += 1
                if step_cnt % 32 == 0:
                    rlog.put(rank=compute_rank(torch.cat(activations)))
                    activations.clear()
                if step_cnt >= steps:
                    done_eval = True
                    break
    env.close()


def load_policy(env, ckpt_path, opt):
    opt.action_cnt = env.action_space.n
    estimator = get_estimator(opt, env)
    agent_args = opt.agent.args
    agent_args["epsilon"] = opt.val_epsilon
    policy = AGENTS[opt.agent.name]["policy_improvement"](
        estimator, opt.action_cnt, **agent_args
    )
    idx = int(ckpt_path.stem.split("_")[1])
    rlog.info(f"Loading {ckpt_path.stem}")
    ckpt = ioutil.load_checkpoint(
        ckpt_path.parent, idx=idx, verbose=False, device=torch.device(opt.device)
    )

    if opt.estimator.args["spectral"] is not None:
        ioutil.special_conv_uv_buffer_fix(policy.estimator, ckpt["estimator_state"])
    policy.estimator.load_state_dict(ckpt["estimator_state"])
    return policy, idx


def run(opt):
    """ Entry point """
    if "sRank" not in opt.experiment:
        opt.experiment += "--sRank"

    rlog.init(opt.experiment, path=opt.out_dir, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("avg_rank", metargs=["rank", 1]),
        # rlog.ValueMetric("rank", metargs=["rank"]),
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.SumMetric("val_ep_cnt", metargs=["done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )

    opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(opt.out_dir)
    ckpt_paths = sorted(root.glob("**/checkpoint*"))

    rlog.info("Begin empirical estimation of feature matrix rank.")
    rlog.info("Runing experiment on {}".format(opt.device))
    rlog.info("Found {:3d} checkpoints.".format(len(ckpt_paths)))


    # Sample only every other third checkpoint
    if "MinAtar" in opt.game:
        ckpt_paths = ckpt_paths[0::3]
        rlog.warning("IMPORTANT! Sampling only every other third checkpoint.")
    else:
        ckpt_paths = ckpt_paths[0::5]
        rlog.warning("IMPORTANT! Sampling only every other fifth checkpoint.")

    sampled_steps = min(opt.valid_step_cnt, opt.train_step_cnt)
    rlog.info("Sampling {:6d} steps from the environment".format(sampled_steps))

    for ckpt_path in ckpt_paths:

        env = get_env(opt, mode="testing")
        policy, step = load_policy(env, ckpt_path, deepcopy(opt))
        check_effective_features_rank(policy, env, sampled_steps)

        rlog.traceAndLog(step=step)


def main():
    """ Liftoff
    """
    run(parse_opts())


if __name__ == "__main__":
    main()
