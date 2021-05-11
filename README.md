# snrl

Code-base for the paper _Spectral Normalisation for Deep Reinforcement Learning: An Optimisation Perspective_. Code developed together with [@tudor-berariu](https://github.com/tudor-berariu).

## Installation

For installing `OpenAI Gym` you will probably need some dependencies:

```sh
apt install apt install -y libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb ffmpeg curl patchelf libglfw3 libglfw3-dev cmake zlib1g zlib1g-dev swig
```

MinAtar requires manual installation, check the [instructions](https://github.com/kenjyoung/MinAtar#quick-start).

#### Conda environment

Use `conda install -f environment.yml` to create a conda env with all the required packages to run experiments. This will install PyTorch and some other dependencies.

## Running experiments

The project uses [liftoff](https://github.com/tudor-berariu/liftoff) to run experiments. Running a single experiment with one of the available config files is done with the command:

```sh
liftoff online.py configs/dqn_minatar.yaml
```

Running multiple experiments at once is possible when using `liftoff`. In the `./config` folder there are multiple experiments already configured. For example `./configs/minatar_dqn_mega_redux` can be used to generate all the various combinations of learning rates, epsilon values and normalizations. The following command will simulate the generation of in excess of `12,000` experiments (`~6000` experiments `x2` seeds):

```sh
liftoff-prepare configs/minatar_dqn_mega_redux --runs-no 2
```

To actually generate the config files on disk simply append `--do` to the above command:

```sh
liftoff-prepare configs/minatar_dqn_mega_redux --runs-no 2 --do
```

You can now use liftoff to start a heap of experiments to be run. For example the following command will randomly pick `8` experiments from the config files generated at the previous step and launch them. Once once of the experiments finishes, another one is launched and so on untill all of the configured experiments will be executed.

```sh
liftoff online.py ./results/TIMESTAMP_minatar_dqn_mega_redux/ --procs-no 8
```

#### Running on GPU

Depending on the configuration file, some experiments will run on GPU. In this case the `--per-gpu` argument control how many experiment runs can be executed on a single GPU. The command below instructs liftoff to use the first two GPUs, execute at most `4` experiment runs per GPU for a total of `8` concurrent experiment runs.

```sh
liftoff online.py ./results/TIMESTAMP_minatar_dqn_mega_redux/ --gpus 0 1 --per-gpu 4 --procs-no 8
```


#### Monitoring

You can use `liftoff status` or `liftoff status --all` for monitoring currently running experiments or for getting a summary of all the past experiments.