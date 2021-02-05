""" Entry point. """
import gc
from pathlib import Path

from liftoff import parse_opts

from termcolor import colored as clr
import rlog
import src.io_utils as ioutil
from src.agents import AGENTS
from src.wrappers import get_env

from src.training import experiment_factory, train_one_epoch, validate


def run(opt):
    """ Entry point of the program. """

    if __debug__:
        print(
            clr(
                "Code might have assertions. Use -O in liftoff when running stuff.",
                color="red",
                attrs=["bold"],
            )
        )

    ioutil.create_paths(opt)

    rlog.init(opt.experiment, path=opt.out_dir, tensorboard=True, relative_time=True)
    rlog.addMetrics(
        rlog.AvgMetric("trn_R_ep", metargs=["trn_reward", "trn_done"]),
        rlog.SumMetric("trn_ep_cnt", metargs=["trn_done"]),
        rlog.AvgMetric("trn_loss", metargs=["trn_loss", 1]),
        rlog.FPSMetric("trn_tps", metargs=["trn_steps"]),
        rlog.FPSMetric("lrn_tps", metargs=["lrn_steps"]),
        rlog.AvgMetric("val_R_ep", metargs=["reward", "done"]),
        rlog.SumMetric("val_ep_cnt", metargs=["done"]),
        rlog.AvgMetric("val_avg_step", metargs=[1, "done"]),
        rlog.FPSMetric("val_fps", metargs=["val_frames"]),
    )

    # Initialize the objects we will use during training.
    env, (replay, policy_improvement, policy_evaluation) = experiment_factory(opt)

    guts = [
        env,
        replay,
        policy_evaluation.estimator,
        policy_evaluation.optimizer,
        policy_improvement,
        policy_evaluation,
    ]
    rlog.info(("\n\n{}" * len(guts)).format(*guts))

    if opt.estimator.args.get("spectral", None) is not None:
        for k in policy_evaluation.estimator.get_spectral_norms().keys():
            # k = f"min{str(k)[1:]}"
            rlog.addMetrics(rlog.ValueMetric(k, metargs=[k]))

    # if we loaded a checkpoint
    if Path(opt.out_dir).joinpath("replay.gz").is_file():

        # sometimes the experiment is intrerupted while saving the replay
        # buffer and it gets corrupted. Therefore we attempt restoring
        # from the previous checkpoint and replay.
        try:
            idx = replay.load(Path(opt.out_dir) / "replay.gz")
            ckpt = ioutil.load_checkpoint(opt.out_dir, idx=idx)
            rlog.info(f"Loaded most recent replay (step {idx}).")
        except:
            gc.collect()
            rlog.info("Last replay gzip is faulty.")
            idx = replay.load(Path(opt.out_dir) / "prev_replay.gz")
            ckpt = ioutil.load_checkpoint(opt.out_dir, idx=idx)
            rlog.info(f"Loading a previous snapshot (step {idx}).")

        # load state dicts

        # load state dicts
        ioutil.special_conv_uv_buffer_fix(
            policy_evaluation.estimator, ckpt["estimator_state"]
        )
        policy_evaluation.estimator.load_state_dict(ckpt["estimator_state"])
        ioutil.special_conv_uv_buffer_fix(
            policy_evaluation.target_estimator, ckpt["target_estimator_state"]
        )
        policy_evaluation.target_estimator.load_state_dict(
            ckpt["target_estimator_state"]
        )
        policy_evaluation.optimizer.load_state_dict(ckpt["optim_state"])

        last_epsilon = None
        for _ in range(ckpt["step"]):
            last_epsilon = next(policy_improvement.epsilon)
        rlog.info(f"Last epsilon: {last_epsilon}.")
        # some counters
        last_epoch = ckpt["step"] // opt.train_step_cnt
        rlog.info(f"Resuming from epoch {last_epoch}.")
        start_epoch = last_epoch + 1
        steps = ckpt["step"]
    else:
        steps = 0
        start_epoch = 1
        # add some hardware and git info, log and save
        opt = ioutil.add_platform_info(opt)

    rlog.info("\n" + ioutil.config_to_string(opt))
    ioutil.save_config(opt, opt.out_dir)

    # Start training

    last_state = None  # used by train_one_epoch to know how to resume episode.
    for epoch in range(start_epoch, opt.epoch_cnt + 1):

        # train for 250,000 steps
        steps, last_state = train_one_epoch(
            env,
            (replay, policy_improvement, policy_evaluation),
            opt.train_step_cnt,
            opt.update_freq,
            opt.target_update_freq,
            opt,
            rlog.getRootLogger(),
            total_steps=steps,
            last_state=last_state,
        )
        rlog.traceAndLog(epoch * opt.train_step_cnt)

        # validate for 125,000 steps
        validate(
            AGENTS[opt.agent.name]["policy_improvement"](
                policy_improvement.estimator, opt.action_cnt, epsilon=opt.val_epsilon
            ),
            get_env(opt, mode="testing"),
            opt.valid_step_cnt,
            rlog.getRootLogger(),
        )
        rlog.traceAndLog(epoch * opt.train_step_cnt)

        # save the checkpoint
        if opt.agent.save:
            ioutil.checkpoint_agent(
                opt.out_dir,
                steps,
                estimator=policy_evaluation.estimator,
                target_estimator=policy_evaluation.target_estimator,
                optim=policy_evaluation.optimizer,
                cfg=opt,
                replay=replay,
                save_replay=(epoch % 8 == 0 or epoch == opt.epoch_cnt),
            )


def main():
    """ Liftoff
    """
    run(parse_opts())


if __name__ == "__main__":
    main()
