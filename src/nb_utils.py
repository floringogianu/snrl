import os
import pickle
from pathlib import Path
from collections import defaultdict
from functools import reduce
import yaml
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator


IGNORE = [".__start", ".__end", ".__journal", ".__crash", ".__leaf"]


def get_file_paths(results_path, include_crashed=False):
    """Returns a list of lists, each containing
    the results file_paths of a trial."""
    file_paths = defaultdict(list)
    for root, _, files in os.walk(results_path):
        if ".__start" in files and ("__crash" not in files or include_crashed):
            for file in files:
                if file not in IGNORE:
                    file_paths[Path(root)].append(file)
    return file_paths


def load_cfg(yaml_path):
    with open(yaml_path, "r") as stream:
        return yaml.safe_load(stream)


def load_pkl(pkl_path):
    with open(pkl_path, "rb") as stream:
        try:
            return pickle.load(stream)
        except Exception as err:
            print(f"Pickle is corrupted: {pkl_path}")
            raise err


def get_config_val(cfg, key):
    """For composed keys."""
    keys = key.split(".")
    return reduce(lambda c, k: c.get(k, {}), keys, cfg)


def _merge_pkls(root, paths):
    merged = {}
    paths = sorted(paths, key=lambda x: int(x.split("_")[0]))

    for path in paths:
        try:
            pkl = load_pkl(root / path)
        except (pickle.UnpicklingError, EOFError):
            print(f"WARN: corrupted {root / path}")
            raise
        pkl = {m: v for m, v in pkl.items() if m != "text"}
        if pkl:
            for metric, events in pkl.items():
                if metric not in merged:
                    merged[metric] = {event["step"]: event["value"] for event in events}
                else:
                    merged[metric].update(
                        {event["step"]: event["value"] for event in events}
                    )
    return merged


def _get_from_pkl(root_path, files, index_key, metrics):
    # check for duplicates indicating resumed experiments
    pkl_fragments = [f for f in files if "pkl" in f]

    if len(pkl_fragments) == 0:
        return None
    if len(pkl_fragments) > 1:
        # concatenate data of resumed experiments
        try:
            pkl = _merge_pkls(root_path, pkl_fragments)
        except:
            return None

        # the columns of the dataframe: index_key] + pkl_metrics
        data_dict = {index_key: list(pkl[metrics[0]].keys())}  # the time index
        data_dict.update({m: list(pkl[m].values()) for m in metrics})
    else:

        pkl = load_pkl(root_path / pkl_fragments[0])
        if metrics[0] not in pkl:
            print(f"WARN: Probably to early for {root_path}.")
            return None

        data_dict = {index_key: [ev[index_key] for ev in pkl[metrics[0]]]}
        data_dict.update({m: [ev["value"] for ev in pkl[m]] for m in metrics})

    return data_dict


def _get_from_tb(root_path, index_key, metrics):
    ea = event_accumulator.EventAccumulator(
        str(root_path), size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    root_logger = ea.scalars.Keys()[0].split("/")[0]

    # {"steps": [0, 1, 2, ..., N]}
    data_dict = {
        index_key: [ev.step for ev in ea.Scalars(f"{root_logger}/{metrics[0]}")]
    }
    data_dict.update(
        {k: [ev.value for ev in ea.Scalars(f"{root_logger}/{k}")] for k in metrics}
    )
    return data_dict


def get_data(
    trials,
    cfg_keys,
    pkl_metrics=None,
    tb_metrics=None,
    index_key="step",
    cb=None,
    cfg_fname="cfg.yml",
):
    print(f"Processing {len(trials)} trials.")
    dataframes = []
    for root_path, files in trials.items():
        cfg = load_cfg(root_path / cfg_fname)

        if pkl_metrics is not None:
            data_dict = _get_from_pkl(root_path, files, index_key, pkl_metrics)
        elif tb_metrics is not None:
            try:
                data_dict = _get_from_tb(root_path, index_key, tb_metrics)
            except:
                data_dict = None

        if data_dict is None:
            continue

        data = pd.DataFrame(data_dict)

        # add config values (mostly hyperparams)
        for k in cfg_keys:
            try:
                val = get_config_val(cfg, k)
                val = ",".join([str(v) for v in val]) if isinstance(val, list) else val
                data[k] = val
            except Exception as err:
                print(err)
                print(k, " -> ", val)
                print(f"WARN, no {k} key in cfg {root_path}")

        # additional stuff
        if cb:
            for col, val in cb(root_path, cfg, data_dict):
                data[col] = val

        data = data.sort_values(by=["step"])
        dataframes.append(data)
    return pd.concat(dataframes, ignore_index=True)


def custom_experiment_name(root_path, cfg, pkl):
    # print(root_path, cfg["experiment"])
    exp_hash = Path(root_path).parts[-2].split("_")[2]
    return "experiment", f"{cfg['experiment']}_{exp_hash}"


def fix_hue_(df, hue):
    # fix the hue
    hue_key = hue
    if df[hue].dtype not in (str, object):
        print(df[hue].dtype)
        df[f"{hue}_"] = [f"${str(x)}$" for x in df[hue]]
        hue_key = f"{hue}:"
    return hue_key


# Quick plotting
# pylint: disable=bad-continuation
def plot(
    data,
    x="step",
    y="R/ep",
    hue=None,
    style=None,
    window=10,
    width=9,
    height=5,
    title=None,
    ylim=None,
    ci="sd",
    legend="brief",
):
    # pylint: enable=bad-continuation
    df = data.copy()
    if window:
        new_col = f"avg_{y}"
        group = [c for c in df.columns if c not in ["ep_cnt", "step", x, y]]
        df[new_col] = (
            df.groupby(group, as_index=False)[y]
            .rolling(window=window)
            .mean()
            .reset_index(0, drop=True)
        )
        print(f"Done rolling average of {y}, grouped by: ", group)

    y = f"avg_{y}" if window else y
    hue_key = fix_hue_(df, hue)
    hue_order = sorted(list(df[hue_key].unique()))

    with matplotlib.rc_context({"figure.figsize": (width, height)}):
        ax = sns.lineplot(
            x=x,
            y=y,
            hue=hue_key,
            style=style,
            data=df,
            ci=ci,
            hue_order=hue_order,
            legend=legend,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(title)


def smooth(data, y, x="step", window=10):
    df = data.copy()
    new_col = f"avg_{y}"
    group = [c for c in df.columns if c not in ["ep_cnt", "step", x, y]]
    df[new_col] = (
        df.groupby(group, as_index=False)[y]
        .rolling(window=window)
        .mean()
        .reset_index(level=group)[y]
    )
    print(f"Done rolling average of {y}, grouped by: ", group)
    return df


def plot_grid(
    df,
    y="val_R_ep",
    hue=None,
    size=None,  # this is the "size" arg in lineplot, not some figsize
    style=None,
    col_wrap=5,
    height=6,
    aspect=1.618,
    legend_out=True,
    legend_kw=None,
    baseline=None,
    title=None,
    custom_colors=None,
    ci=95,
    window=10,
    sharex=True,
):
    """ Plot on a grid using FaceGrid """

    # smooth the data
    if window is not None:
        df_ = smooth(df, y, window=window)
        y = f"avg_{y}"
    else:
        df_ = df.copy()

    # sort by name and fix game name to match dopamine convention
    sorted_games = sorted(df_.game.unique())
    sorted_games_ = [n.replace("_", "") for n in sorted_games]

    # fix the hue order
    hue_order = None
    if hue is not None:
        hue_order = sorted(list(df_[hue].unique()))

    # have a reference line
    hue2palette = None
    if custom_colors is not None:
        palette = sns.color_palette()
        palette = [c for c in palette if c not in custom_colors.values()]
        hue2palette = {
            hue: palette[i % len(palette)]
            for i, hue in enumerate(hue_order)
            if hue not in custom_colors
        }
        hue2palette.update(custom_colors)

    print(hue2palette)

    # win
    g = sns.FacetGrid(
        df_,
        col="game",
        col_wrap=col_wrap,
        sharex=sharex,
        sharey=False,
        height=height,
        aspect=aspect,
        col_order=sorted_games,
        legend_out=legend_out,
    )

    if baseline is None:
        baseline = pd.read_csv("./dopamine_average_max_scores.csv")

    # use the dopamine baselines to create the horizontal lines
    xmax = df.step.max()
    algo2color = {"C51": "b", "RAINBOW": "c", "IQN": "r", "DQN": "y"}

    for ax, game in zip(g.axes, sorted_games_):
        # we sort the algos
        base_game = baseline.loc[baseline.Game == game].sort_values(
            by="Value", ascending=False
        )
        base_algos = base_game.Agent.values
        base_vals = base_game.Value.values

        for i, (algo, val) in enumerate(zip(base_algos, base_vals)):
            ax.axhline(y=val, ls="--", lw=2.0, c=algo2color[algo])
            if i % 2 == 0:
                ax.text(1_000_000, val, algo, fontsize=16)
            else:
                ax.text(
                    xmax, val, algo, fontsize=16, horizontalalignment="right",
                )

    # map the lineplots to the grid
    g.map_dataframe(
        sns.lineplot,
        "step",
        y,
        hue,
        size=size,
        style=style,
        hue_order=hue_order,
        palette=hue2palette,
        ci=ci,
    )
    if legend_kw is not None:
        g.add_legend(**legend_kw)
    if title is not None:
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(title, fontsize=24)
    return g


def _plot_on_ax(df, x, y, hue, ax, style, ci, ylim):
    hue_key = fix_hue_(df, hue)
    hue_order = sorted(list(df[hue_key].unique()))

    print(x, y, hue_key, style, ci, ylim)
    sns.lineplot(
        data=df, x=x, y=y, hue=hue_key, style=style, hue_order=hue_order, ax=ax, ci=ci,
    )
    if ylim is not None:
        ax.set_ylim(*ylim)


# pylint: disable=bad-continuation
def _plot_grid(
    data,
    x="ep_cnt",
    y="R/ep",
    rows=None,
    cols=None,
    hue=None,
    style=None,
    window=10,
    width=8,
    height=5,
    ylim=None,
    ci="sd",
):
    """ From scratch implementation of grid plots. """

    # pylint: enable=bad-continuation
    df = data.copy()
    base_wh, base_hw = width, height

    # smooth the curves
    if window is not None:
        df = smooth(df, y, window=window)
        y = f"avg_{y}"

    # make subplots
    row_vals = sorted(df[rows].unique()) if rows else []
    col_vals = sorted(df[cols].unique()) if cols else []
    nrows, ncols = max(len(row_vals), 1), max(len(col_vals), 1)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(base_wh * ncols, base_hw * nrows)
    )
    fig.subplots_adjust(hspace=0.3)

    cnt = 0
    if rows:
        for rval in row_vals:
            if cols:
                for ci, cval in enumerate(col_vals):
                    dff = df.loc[(df[cols] == cval) & (df[rows] == rval)]
                    ax = axes.flatten()[cnt]
                    _plot_on_ax(dff, x, y, hue, ax, style, ci, ylim)
                    ax.set(title=f"{cols}={cval}, {rows}={rval}")
                    cnt += 1
            else:
                dff = df.loc[(df[rows] == rval)]
                ax = axes.flatten()[cnt]
                _plot_on_ax(dff, x, y, hue, ax, style, ci, ylim)
                ax.set(title=f"{rows}={rval}")
                cnt += 1
    else:
        for cval in col_vals:
            dff = df.loc[(df[cols] == cval)]
            ax = axes.flatten()[cnt]
            _plot_on_ax(dff, x, y, hue, ax, style, ci, ylim)
            ax.set(title=f"{cols}={cval}")
            cnt += 1
