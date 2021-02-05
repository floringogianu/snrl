""" This files implements a storage efficient Experience Replay.
"""

import io
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from gzip import GzipFile
from pathlib import Path

import numpy as np
import torch
from numpy import random

from src.rl_utils import to_device


def _collate(batch, batch_size, histlen):
    device = batch[0][0].device
    frame_size = batch[0][0].shape[2:]
    states = torch.cat(batch[0][::-1], 0).view(batch_size, histlen, *frame_size)
    actions = torch.tensor(  # pylint: disable=E1102
        batch[1][::-1], device=device, dtype=torch.long
    ).unsqueeze_(1)
    rewards = torch.tensor(  # pylint: disable=E1102
        batch[2][::-1], device=device, dtype=torch.float
    ).unsqueeze_(1)
    if all(batch[4]):
        # if all next_states are terminal
        next_states = torch.empty(0, device=device)
    else:
        # concatenates only non-terminal next_states
        next_states = torch.cat(batch[3][::-1], 0).view(-1, histlen, *frame_size)
    mask = ~torch.tensor(  # pylint: disable=E1102
        batch[4][::-1], device=device, dtype=torch.bool
    ).unsqueeze_(1)

    # if we train with full RGB information (three channels instead of one)
    if states.ndimension() == 5:
        bsz, hist, nch, height, width = states.size()
        states = states.view(bsz, hist * nch, height, width)
        bsz, hist, nch, height, width = next_states.size()
        next_states = next_states.view(bsz, hist * nch, height, width)

    return [states, actions, rewards, next_states, mask]


class ExperienceReplay:
    r""" Experience Replay Buffer which stores states in order and samples
    concatenated states of a given history length.

    Args:
        capacity (int, optional): Defaults to 100_000. ER size.
        batch_size (int, optional): Defaults to 32.
        hist_len (int, optional): Defaults to 4. Size of the state.
        async_memory (bool, optional): Defaults to True. If enabled it will
            try to take advantage of the time it takes to do a policy
            improvement step and sample asyncronously the next batch.

        bootstrap_args (list, optional): Defaults to None.
    """

    def __init__(  # pylint: disable=bad-continuation
        self,
        capacity: int = 100_000,
        batch_size: int = 32,
        hist_len: int = 4,
        **kwargs,
    ) -> None:

        self.memory = []
        self.capacity = capacity
        self.batch_size = batch_size
        self.histlen = hist_len
        self.device = kwargs.get("device", torch.device("cpu"))
        self.warmup_steps = kwargs.get("warmup_steps", batch_size)

        self.position = 0
        self._size = 0
        self.__last_state = None
        self._collate = kwargs.get("collate") or _collate

    def push(self, transition: list) -> int:
        """ Push a transition to the experience replay buffer.

        Args:
            transition (list): A list containing [s, a, r, s_, d]

        Returns:
            int: current insertion position
        """

        with torch.no_grad():
            state = transition[0][:, -1:].clone().to(self.device)
        to_store = [state, transition[1], transition[2], bool(transition[4])]
        if len(self.memory) < self.capacity:
            self.memory.append(to_store)
        else:
            self.memory[self.position] = to_store
        self.__last_state = transition[3][:, -1:].to(self.device)
        pos = self.position
        self.position += 1
        self._size = max(self._size, self.position)
        self.position = self.position % self.capacity
        return pos

    def sample(self, gods_idxs=None) -> list:
        """ Sample a batch from the experience replay buffer.

        Args:
            gods_idxs (list, optional): A list of indices to sample from.
                Usefull for custom samplers, such as Prioritized ER.
                Defaults to None.

        Returns:
            list: A list of batched states, actions, rewards, next_states, done.
        """
        batch = [], [], [], [], []
        memory = self.memory
        nmemory = len(self.memory)

        if gods_idxs is None:
            gods_idxs = random.randint(0, nmemory, (self.batch_size,))

        for idx in gods_idxs[::-1]:
            transition = memory[idx]
            batch[0].append(transition[0])
            batch[1].append(transition[1])
            batch[2].append(transition[2])
            is_final = bool(transition[3])
            batch[4].append(is_final)
            if not is_final:
                if idx == self.position - 1:
                    batch[3].append(self.__last_state)
                else:
                    batch[3].append(self.memory[(idx + 1) % self.capacity][0])

            last_screen = transition[0]
            found_done = False
            bidx = idx
            for _ in range(self.histlen - 1):
                if not is_final:
                    batch[3].append(batch[0][-1])
                if not found_done:
                    bidx = (bidx - 1) % self.capacity
                    if bidx < self._size:
                        new_transition = memory[bidx]
                        if new_transition[3]:
                            found_done = True
                        else:
                            last_screen = new_transition[0]
                    else:
                        found_done = True
                batch[0].append(last_screen)

        return self._collate(batch, self.batch_size, self.histlen)

    def push_and_sample(self, transition: list) -> list:
        """ Pushes a transition and returns a sampled batch.

        Args:
            transition (list): A list containing [s, a, r, s_, d]

        Returns:
            list: A list of batched states, actions, rewards, next_states, done.
        """

        if isinstance(transition[0], list):
            for trans in transition:
                self.push(trans)
        else:
            self.push(transition)
        return self.sample()

    @property
    def is_ready(self):
        return self._size > self.warmup_steps

    def load(self, path):
        print(f"Loading from {path}.")
        with open(path, "rb") as f:
            with GzipFile(fileobj=f) as inflated:
                data = io.BytesIO(inflated.read())
                replay_data = torch.load(data)
                self._set(replay_data)
                return replay_data["idx"]

    def save(self, path, idx, save_all=False):
        """ Snapshots the ER buffer.

            Saves into `replay.gz` and then copies it to `prev_replay.gz` if
            the save was successful.

            If save_all it will just store on disk all the results.
        """
        root = Path(path)
        fname = f"replay_{idx:08d}.gz" if save_all else "replay.gz"
        fpath = root / fname
        with open(fpath, "wb") as f:
            with GzipFile(fileobj=f) as outfile:
                torch.save(
                    {
                        "memory": self.memory,
                        "last_state": self.__last_state,
                        "position": self.position,
                        "size": self._size,
                        "capacity": self.capacity,
                        "idx": idx,
                    },
                    outfile,
                )
        return fpath

    def _set(self, checkpoint):
        self.memory = checkpoint["memory"]
        self.position = checkpoint["position"]
        self._size = checkpoint["size"]
        self.__last_state = checkpoint["last_state"]
        if self.capacity != checkpoint["capacity"]:
            print(
                "Warning! Checkpoint's capacity {} different ER's {}.".format(
                    checkpoint["capacity"], self.capacity
                )
            )
        self.capacity = checkpoint["capacity"]

    def __len__(self):
        return self._size

    def __str__(self):
        return "{}(N={}, size={}, batch={}, hlen={}, warmup={})".format(
            self.__class__.__name__,
            self.capacity,
            self._size,
            self.batch_size,
            self.histlen,
            self.warmup_steps,
        )

    def __repr__(self):
        obj_id = hex(id(self))
        name = self.__str__()
        return f"{name} @ {obj_id}"


class AsyncExperienceReplay:
    def __init__(self, er):
        self.is_async = True
        self.__er = er
        self.__executor = ThreadPoolExecutor(max_workers=1)
        self.__sample_result = None
        self.__push_result = None
        # delegation
        self.save = self.__er.save
        self.load = self.__er.load

    def sample(self, gods_idxs=None):
        if self.__push_result is not None:
            self.__push_result.result()
            self.__push_result = None

        if self.__sample_result is None:
            batch = self.__er.sample(gods_idxs=gods_idxs)
        else:
            batch = self.__sample_result.result()

        self.__sample_result = self.__executor.submit(
            self.__er.sample, gods_idxs=gods_idxs
        )
        return batch

    def push(self, transition):
        if self.__push_result is not None:
            self.__push_result.result()
            self.__push_result = None

        if self.__sample_result is not None:
            self.__sample_result.result()

        self.__push_result = self.__executor.submit(self.__er.push, transition)

    def push_and_sample(self, transition):
        if self.__push_result is not None:
            self.__push_result.result()
            self.__push_result = None

        if self.__sample_result is not None:
            batch = self.__sample_result.result()
        else:
            batch = self.__er.sample()

        self.__sample_result = self.__executor.submit(
            self.__er.push_and_sample, transition
        )
        return batch

    def clear_ahead_results(self):
        """ Waits for any asynchronous push and cancels any sample request.
        """
        if self.__sample_result is not None:
            self.__sample_result.cancel()
            self.__sample_result = None
        if self.__push_result is not None:
            self.__push_result.result()
            self.__push_result = None


# Batch/Offline Experience Replay


def load_file(path):
    """ Loads one gzipped numpy object. """
    with open(path, "rb") as f:
        with GzipFile(fileobj=f) as inflated:
            return np.load(inflated, allow_pickle=False)


def load_checkpoint(root, idx):
    """ Loads all the checkpoint files for a given checkpoint index. """
    print("Loading checkpoint {}.".format(idx))
    fpaths = []
    fpaths = [
        root / "$store$_{}_ckpt.{}.gz".format(el, idx)
        for el in ["observation", "action", "reward", "terminal"]
    ]
    fpaths += [
        root / "{}_ckpt.{}.gz".format(el, idx) for el in ["add_count", "invalid_range"]
    ]
    objects = [load_file(fpath) for fpath in fpaths]
    return objects[:4], objects[4:]


def get_valid_checkpoints(root):
    """ Return a list of valid checkpoint indices, the criteria being that
        for each checkpoint index all the checkpiont files need to be
        present
    """
    # count the number of files for each checkpoint index found
    ckpt_idxs = [item.stem.split(".")[-1] for item in root.iterdir()]
    ckpt_cnt = Counter(ckpt_idxs)  # { idx: no_of_files_with_this_idx }
    valid_ckpts = [int(idx) for idx in ckpt_cnt if ckpt_cnt[idx] in [6, 7]]
    valid_ckpts = [idx for idx in valid_ckpts if idx not in [50]]  # tainted
    return valid_ckpts


class OfflineExperienceReplay:
    """ Wrapper over Experience Replay that facilitates loading Dopamine
        checkpoints.
    """

    def __init__(self, root, batch_size=32, device=None, is_async=False):
        self.__root = Path(root)
        self.__batch_size = batch_size
        self.__device = device
        self.__valid_ckpts = get_valid_checkpoints(root)
        self.__replay_buffers = []  # where we store the ER buffers
        self.__is_async = is_async

    def sample(self, idxs=None):
        er = random.choice(self.__replay_buffers)
        return er.sample(gods_idxs=idxs)

    def reload_N(self, N=1, ckpt_idxs=None):
        """ Loads N replay buffers. """
        # clear the memory
        del self.__replay_buffers[:]

        if ckpt_idxs is None:
            # sample N checkpoints
            ckpt_idxs = random.choice(self.__valid_ckpts, N, replace=False)

        # and load them in parallel
        with ThreadPoolExecutor(max_workers=N) as tpe:
            replay_futures = [tpe.submit(self._load_one, idx) for idx in ckpt_idxs]
        self.__replay_buffers = [f.result() for f in replay_futures]
        print(f"Loaded {N} Replay buffers.")

        # self.__replay_buffers = [self._load_one(idx) for idx in ckpt_idxs]

    def _dopamine2wintermute(self, idx, device=None):
        """ Load a Dopamine checkpoint. """
        ckpt, _ = load_checkpoint(self.__root, idx)
        ckpt = [torch.from_numpy(el) for el in ckpt]
        ckpt[0].unsqueeze_(1).unsqueeze_(1)  # add dimension to observations

        if device is not None:
            ckpt = to_device(ckpt, device)

        # convert to wintermute format
        memory = list(zip(*ckpt))
        return {
            "memory": memory,
            "last_state": memory[-1][0],
            "size": len(memory),
            "position": 0,  # assumes at capacity
            "capacity": len(memory),
        }

    def _load_one(self, idx, dopamine=True):
        """ Loads one replay buffer.
        """
        replay_buffer = ExperienceReplay(
            capacity=int(1e6), batch_size=self.__batch_size
        )

        if dopamine:
            ckpt = self._dopamine2wintermute(idx, device=self.__device)
            replay_buffer._set(ckpt)
        else:
            fpath = self.__root / f"replay_{idx:08d}.gz"
            replay_buffer.load(fpath)

        print("Checkpoint {} loaded.".format(idx))

        if self.__is_async:
            return AsyncExperienceReplay(replay_buffer)
        return replay_buffer
