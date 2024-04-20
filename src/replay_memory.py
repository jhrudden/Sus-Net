import torch
from collections import namedtuple
from src.utils import EnhancedOrderedDict
from typing import Union, List

# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple(
    "Batch", ("states", "actions", "rewards", "next_states", "imposters", "dones")
)


class FastReplayBuffer:
    def __init__(
        self,
        max_size: int,
        trajectory_size: int,
        state_size: int,
        n_agents: int,
        n_imposters: int,
    ):
        assert max_size > 0, "Replay buffer size must be positive"
        assert trajectory_size > 0, "Trajectory size must be positive"
        assert state_size > 0, "State size must be positive"
        assert n_agents > 0, "Number of agents must be positive"

        self.max_size = max_size
        self.trajectory_size = trajectory_size
        self.state_size = state_size

        self.trajectory_dict = EnhancedOrderedDict(max_size)

        # initalizing the timestep buffer
        self.states = torch.empty((max_size, state_size))
        self.actions = torch.empty((max_size, n_agents), dtype=torch.long)
        self.rewards = torch.empty((max_size, n_agents))
        self.next_states = torch.empty((max_size, state_size))
        self.dones = torch.empty((max_size, 1), dtype=torch.bool)
        self.starts = torch.empty((max_size, 1), dtype=torch.bool)
        self.imposters = torch.empty((max_size, n_imposters), dtype=torch.int16)

        # initializing current index and buffer size
        self.idx = 0
        self.size = 0

        # initializing the trajectory length tracker
        self.trajectory_lengths = torch.zeros(max_size)

    def get_last_trajectory(self):
        pass

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        imposters,
        is_start: bool = False,
    ):
        """Add a transition to the buffer.

        :param state: 1-D np.ndarray of state-features
        :param action: Integer action
        :param reward: Float reward
        :param next_state: 1-D np.ndarray of state-features
        :param done: Boolean value indicating the end of an episode
        """
        # check if we are overwriting a trajectory
        # if we are, then pop trajectory_size - 1 elemnts from the smart_buffer or until we reach the start of an episode
        if self.trajectory_dict.has(self.idx):
            self.starts[self.idx] = False
            for i in range(self.trajectory_size):
                new_idx = (self.idx + i) % self.max_size
                if self.starts[new_idx]:
                    break
                self.trajectory_dict.pop()

        self.states[self.idx] = torch.tensor(state)
        self.actions[self.idx] = torch.tensor(action)
        self.rewards[self.idx] = torch.tensor(reward)
        self.next_states[self.idx] = torch.tensor(next_state)
        self.dones[self.idx] = torch.tensor(done)
        self.starts[self.idx] = torch.tensor(is_start)
        self.imposters[self.idx] = torch.tensor(imposters)

        self.trajectory_dict.insert(self.idx)

        # Circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size
        # Update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        Parameters
            - batch_size (int): Number of transitions to sample
        """
        assert self.size > 0, "Replay buffer is empty, can't sample"

        sample_idx = torch.tensor(
            self.trajectory_dict.sample(n_samples=batch_size), dtype=torch.int
        )

        return self._get_sequence(sample_idx)

    def get_last_trajectory(self):
        """
        Get the last trajectory from the buffer
        """
        assert self.size > 0, "Replay buffer is empty, can't sample"
        sample_idx = self.trajectory_dict.get_last()
        return self._get_sequence(sample_idx)

    def _get_sequence(self, sample_idx: Union[int, torch.Tensor]):
        """
        Fetches a single or multiple sequences from the buffer
        """
        if isinstance(sample_idx, int):
            sample_idx = torch.tensor([sample_idx], dtype=torch.int)

        batch_size = sample_idx.size(0)

        seq = torch.ones((batch_size, self.trajectory_size), dtype=torch.int) * -1

        for i in range(self.trajectory_size):
            new_idx = (sample_idx - i) % self.max_size
            neg = seq[:, i] == -1

            seq[neg, i] = new_idx[neg].squeeze()
            starts = self.starts[new_idx].squeeze()

            fill_condition = (starts & neg & (i < self.trajectory_size - 1)).squeeze()
            if fill_condition.sum() > 0:
                seq[fill_condition, i:] = new_idx[fill_condition].repeat(
                    1, self.trajectory_size - i
                )

            if not torch.any(neg):
                break

        seq = torch.flip(seq, [1])

        return Batch(
            states=self.states[seq],
            actions=self.actions[seq],
            rewards=self.rewards[seq],
            next_states=self.next_states[seq],
            imposters=self.imposters[
                seq[:, 0]
            ],  # imposters don't change over the trajectory
            dones=self.dones[seq],
        )

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env: Gymnasium environment
        :param num_steps: Number of steps to populate the replay memory
        """

        step = 0
        episode_id = 0
        while step < num_steps:
            episode_id += 1
            s, _ = env.reset()
            state = env.flatten_state(s)
            done = False
            truncation = False
            start = True
            while not done and not truncation:
                imposters = env.imposter_idxs
                action = env.sample_actions()
                n_s, reward, done, truncation, _ = env.step(action)
                next_state = env.flatten_state(n_s)
                self.add(
                    state, action, reward, next_state, done, imposters, is_start=start
                )
                state = next_state
                step += 1
                start = False
                if step >= num_steps:
                    break
