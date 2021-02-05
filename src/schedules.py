""" Various exploration schedules. Probably can be used for learning rates
as well.

* constant_schedule(value)
    constant_schedule(.1)   =>   .1, .1, .1, .1, .1, ...

* linear_schedule(start, end, steps_no, warmup)
    linear_schedule(.5, .1, 5)  =>  .5, .4, .3, .2, .1, .1, .1, .1, ...
    linear_schedule(.5, .1, 5, 3)  =>  .5, .5, .5, .5, .4, .3, .2, .1, ...

* log_schedule(start, end, steps_no)
    log_schedule(1, 0.001, 3)   =>   1., .1, .01, .001, .001, .001, ...
"""

import itertools


def float_range(start, end, step):
    x = start
    if step > 0:
        while x < end:
            yield x
            x += step
    else:
        while x > end:
            yield x
            x += step


def constant_schedule(epsilon=0.05):
    return itertools.repeat(epsilon)


def linear_schedule(start, end, steps_no, warmup_steps=0):
    start, end, steps_no = float(start), float(end), float(steps_no)
    steps_no -= warmup_steps

    step = (end - start) / (steps_no - 1.0)
    if warmup_steps:
        schedules = [
            itertools.repeat(start, times=warmup_steps),
            float_range(start, end, step),
            itertools.repeat(end),
        ]
    else:
        schedules = [float_range(start, end, step), itertools.repeat(end)]
    return itertools.chain(*schedules)


def log_schedule(start, end, steps_no, warmup_steps=0):
    from math import log, exp

    steps_no -= warmup_steps
    log_start, log_end = log(start), log(end)
    log_step = (log_end - log_start) / (steps_no - 1)
    log_range = float_range(log_start, log_end, log_step)
    if warmup_steps:
        schedules = [
            itertools.repeat(start, times=warmup_steps),
            map(exp, log_range),
            itertools.repeat(end),
        ]
    else:
        schedules = [map(exp, log_range), itertools.repeat(end)]
    return itertools.chain(*schedules)


SCHEDULES = {"linear": linear_schedule, "log": log_schedule}


def get_schedule(name="linear", start=1, end=0.01, steps=0, warmup_steps=0):
    r""" Returns a constant, linear or logarithmic scheduler.

        name (str, optional): Defaults to "linear". Schedule type.
        start (int, optional): Defaults to 1. Start value.
        end (float, optional): Defaults to 0.01. End value.
        steps (int, optional): Defaults to 0. No of steps during which the
            value is degraded towards its `end` value.

        warmup_steps (int, optional): Defaults to 0. No of steps during which
            the schedule remains constant at the start value.

    Returns:
        iterator: A schedule for a given value.
    """

    if name == "constant":
        return constant_schedule(start)
    return SCHEDULES[name](start, end, steps, warmup_steps)


def get_random_schedule(args, probs):
    assert len(args) == len(probs)
    import numpy as np

    return get_schedule(*args[np.random.choice(len(args), p=probs)])


if __name__ == "__main__":

    const = get_schedule("constant", start=0.1)
    print("Constant(0.1):")
    for _ in range(10):
        print(f" {next(const):.2f}")
    print("\n")

    linear = get_schedule("linear", start=0.5, end=0.1, steps=5)
    print("Linear Schedule(.5, .1, 5):")
    for _ in range(10):
        print(f" {next(linear):.2f}")
    print("\n")

    linear = get_schedule(
        "linear", start=0.5, end=0.1, steps=10, warmup_steps=3
    )
    print("Linear Schedule(.5, .1, 10, warmup_steps=3):")
    for _ in range(10):
        print(f" {next(linear):.2f}")
    print("\n")

    logarithmic = get_schedule("log", start=1, end=0.001, steps=4)
    print("Logarithmic Schedule(1, .001, 4):")
    for _ in range(10):
        print(f" {next(logarithmic):.3f}")
    print("\n")

    logarithmic = get_schedule(
        "log", start=1, end=0.001, steps=8, warmup_steps=4
    )
    print("Logarithmic Schedule(1, .001, 8, warmup_steps=4):")
    for _ in range(10):
        print(f" {next(logarithmic):.3f}")
    print("\n")
