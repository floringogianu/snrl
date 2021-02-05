""" An object more closer to the Arcade Learning Environment that the one
provided by OpenAI Gym.
"""
import random
from collections import deque

import atari_py
import gym
import numpy as np
import torch
from gym.spaces import Box, Discrete
from minatar import Environment

try:
    import cv2
except ModuleNotFoundError as err:
    print(
        "OpenCV is required when using the ALE env wrapper. ",
        "Try `conda install -c menpo opencv`.",
        err,
    )

__all__ = ["ALE"]


class ALE:
    """ A wrapper over atari_py, the Arcade Learning Environment python
    bindings that provides: frame concatentation of `history_len`, sticky
    actions probability, end game after first life in training mode, clip
    rewards during training.
    
    All credits for this wrapper go to
    [@Kaixhin](https://github.com/Kaixhin/Rainbow/blob/master/env.py)
    
    Returns:
        env: An ALE object with settings close to the original DQN paper.
    """

    # pylint: disable=too-many-arguments, bad-continuation
    def __init__(
        self,
        game,
        seed,
        device,
        training=True,
        clip_rewards_val=1,
        history_length=4,
        random_starts=True,
        sticky_action_p=0,
        max_episode_length=108e3,
    ):
        # pylint: enable=bad-continuation
        self.game_name = game
        self.device = device

        # Manage stochasticity
        self.sticky_action_p = sticky_action_p
        self.random_starts = random_starts
        if (sticky_action_p != 0) and random_starts:
            raise ValueError("Can't use both random_starts and sticky actions.")
        if (sticky_action_p == 0) and not random_starts:
            print("ALE: WARNING, the game is deterministic!")

        # Configure ALE
        self.ale = atari_py.ALEInterface()
        self.ale.setInt("random_seed", seed)
        self.ale.setInt("max_num_frames_per_episode", max_episode_length)
        self.ale.setFloat("repeat_action_probability", sticky_action_p)
        self.ale.setInt("frame_skip", 0)
        self.ale.setBool("color_averaging", False)
        self.ale.loadROM(
            atari_py.get_game_path(game)
        )  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.action_space = Discrete(len(self.actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        # Used to check if resetting only from loss of life
        self.life_termination = False
        self.window = history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=history_length)
        self.training = training  # Consistent with model training mode
        self.clip_val = clip_rewards_val

    def _get_state(self):
        state = cv2.resize(
            self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_AREA,
        )
        return torch.tensor(state, dtype=torch.uint8, device=self.device)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(
                torch.zeros(84, 84, device=self.device, dtype=torch.uint8)
            )

    def _random_noops(self):
        # Perform up to 30 random no-ops before starting
        for _ in range(random.randrange(30)):
            self.ale.act(0)  # Assumes raw action 0 is always no-op
            if self.ale.game_over():
                self.ale.reset_game()

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Do the random ops
            if self.random_starts:
                self._random_noops()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device, dtype=torch.uint8)
        reward, done = 0, False
        for t in range(4):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                # Only set flag when not truly done
                self.life_termination = not done
                done = True
            self.lives = lives
        # clip the reward
        if self.clip_val and self.training:
            reward = max(min(reward, self.clip_val), -self.clip_val)
        # Return state, reward, done
        state = torch.stack(list(self.state_buffer), 0).unsqueeze(0).byte()
        return state, reward, done, {}

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def render(self):
        frame = self.ale.getScreenRGB()[:, :, ::-1]
        frame = cv2.resize(frame, (320, 420), interpolation=cv2.INTER_LANCZOS4)
        cv2.imshow("screen", frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def __str__(self):
        mode = "train" if self.training else "eval"
        if self.random_starts:
            stochasticity = "no_op_30"
        elif self.sticky_action_p != 0:
            stochasticity = f"{self.sticky_action_p:.2f}_sticky_action"
        else:
            stochasticity = "deterministic"

        return "ALE(game={}, mode={}, stochasticity={}, hist_len={}, repeat_act=4)".format(
            self.game_name, mode, stochasticity, self.window
        )


class MinAtar(gym.Env):
    """ Minatar does not have a Gym-like API so we make one.
    """

    def __init__(self, game, **kwargs):
        self._env = env = Environment(game, **kwargs)
        self._action_space = Discrete(env.num_actions())
        self._observation_space = Box(
            low=False, high=True, shape=env.state_shape(), dtype=np.bool
        )

    def reset(self):
        self._env.reset()
        return self._env.state()

    def step(self, action):
        reward, done = self._env.act(action)
        observation_ = self._env.state()
        return observation_, reward, done, {}

    def render(self):
        raise NotImplementedError("Better fails, right?")

    def close(self):
        pass

    @property
    def action_space(self):
        """ Return the action space """
        return self._action_space

    @property
    def observation_space(self):
        """ Return the observation space """
        return self._observation_space

    @property
    def sticky_action_prob(self):
        return self._env.sticky_action_prob

    @sticky_action_prob.setter
    def sticky_action_prob(self, value):
        self._env.sticky_action_prob = value
