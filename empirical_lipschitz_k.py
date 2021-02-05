""" Offline script for various stuff such as empirically checking the
    Lipschitz constant or re-evalutating the models.
"""
from copy import deepcopy
from pathlib import Path

import rlog
import torch
from liftoff import parse_opts

import src.io_utils as ioutil
from src.training import get_estimator
from src.wrappers import get_env
from src.agents import AGENTS


def get_grad_eigenvalues(x, pi):
    """ Computes the eigenvalues of the Jaccobian """
    y = pi.full[0, pi.action]
    (Jyx,) = torch.autograd.grad(y, x)
    # return list(Jyx.squeeze().svd().S.detach().cpu().numpy())
    return torch.norm(Jyx.flatten()).item()


def check_lipschitz_constant(policy, env, steps):
    """ Validation routine """
    policy.estimator.eval()

    obs, done = env.reset(), False

    for _ in range(1, steps + 1):

        obs = obs.float().requires_grad_()
        policy.estimator.zero_grad()

        pi = policy.act(obs)
        Jyx_norm = get_grad_eigenvalues(obs, pi)

        obs, reward, done, _ = env.step(pi.action)
        rlog.put(Jyx_norm=Jyx_norm, reward=reward, done=done, val_frames=1)

        if done:
            obs, done = env.reset(), False


def load_policy(env, ckpt_path, opt):
    opt.action_cnt = env.action_space.n
    estimator = get_estimator(opt, env)
    agent_args = opt.agent.args
    agent_args["epsilon"] = 0.0  # purely max
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


# results/experiment/variation/0
def run(opt):
    """ Entry point of the experiment """

    # no need to run this for all the seeds
    if opt.run_id not in [0, 1, 2]:
        return

    # this is a bit of a hack, it would be nice to change it
    # when launching the experiment. It generally only affects the logger.
    if "JyxNorm" not in opt.experiment:
        opt.experiment += "--JyxNorm"

    rlog.init(opt.experiment, path=opt.out_dir, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("Jyx_norm_avg", metargs=["Jyx_norm", 1]),
        rlog.MaxMetric("Jyx_norm_max", metargs=["Jyx_norm"]),
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.SumMetric("val_ep_cnt", metargs=["done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )

    opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(opt.out_dir)
    ckpt_paths = sorted(root.glob("**/checkpoint*"))

    rlog.info("Begin empirical estimation of norm(Jyx).")
    rlog.info("Runing experiment on {}.".format(opt.device))
    rlog.info("Found {:3d} checkpoints.".format(len(ckpt_paths)))

    # Sample only every other third checkpoint
    if (Path(opt.out_dir) / "max_ckpt").exists():
        ckpt_paths = [
            p
            for p in ckpt_paths
            if int(p.stem.split("_")[1])
            == int((Path(opt.out_dir) / "max_ckpt").read_text())
        ]
        rlog.info("IMPORTANT! Found max_ckpt @{}.".format(ckpt_paths[0]))
    else:
        if "MinAtar" in opt.game:
            ckpt_paths = ckpt_paths[0::3]
            rlog.warning("IMPORTANT! Sampling only every other third checkpoint.")
        else:
            ckpt_paths = ckpt_paths[0::5]
            rlog.warning("IMPORTANT! Sampling only every other fifth checkpoint.")

    for ckpt_path in ckpt_paths:
        env = get_env(opt, mode="testing")
        policy, step = load_policy(env, ckpt_path, deepcopy(opt))

        check_lipschitz_constant(policy, env, opt.valid_step_cnt)
        rlog.traceAndLog(step=step)


def main():
    """ Liftoff
    """
    run(parse_opts())


if __name__ == "__main__":
    main()
