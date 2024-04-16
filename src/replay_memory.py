import torch
from collections import namedtuple

# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple("Batch", ("states", "actions", "rewards", "next_states", "dones"))


class TrajectoryReplayBuffer:

    def __init__(self, max_size, trajectory_size, state_size):

        self.max_size = max_size
        self.trajectory_size = trajectory_size
        self.state_size = state_size

        # initalizing the timestep buffer
        self.states = torch.empty((max_size, state_size))
        self.actions = torch.empty((max_size, 1), dtype=torch.long)
        self.rewards = torch.empty((max_size, 1))
        self.next_states = torch.empty((max_size, state_size))
        self.dones = torch.empty((max_size, 1), dtype=torch.bool)
        self.timesteps = torch.empty((max_size, 1), dtype=torch.long)

        # initializing current index and buffer size
        self.idx = 0
        self.size = 0

        # initializing the trajectory length tracker
        self.trajectory_lengths = torch.zeros(max_size)

    def add(self, state, action, reward, next_state, done, timestep):
        """Add a transition to the buffer.

        :param state: 1-D np.ndarray of state-features
        :param action: Integer action
        :param reward: Float reward
        :param next_state: 1-D np.ndarray of state-features
        :param done: Boolean value indicating the end of an episode
        """

        self.states[self.idx] = torch.tensor(state)
        self.actions[self.idx] = torch.tensor(action)
        self.rewards[self.idx] = torch.tensor(reward)
        self.next_states[self.idx] = torch.tensor(next_state)
        self.dones[self.idx] = torch.tensor(done)
        self.timesteps[self.idx] = timestep

        self.trajectory_lengths[self.idx] = 0
        self.trajectory_lengths[
            self.idx - min(timestep, self.trajectory_size - 1) : self.idx + 1
        ] += 1

        # Circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size
        # Update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        :param batch_size: Number of transitions to sample
        :rtype: Batch
        """

        valid_indexes = torch.where(self.trajectory_lengths == self.trajectory_size)[0]
        sample_indices = torch.multinomial(
            valid_indexes.float(), batch_size, replacement=False
        )

        # Calculate the full set of indices to extract
        idx = torch.arange(self.trajectory_size) + sample_indices[:, None]

        return Batch(
            states=self.states[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_states=self.next_states[idx],
            dones=self.dones[idx],
        )

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env: Gymnasium environment
        :param num_steps: Number of steps to populate the replay memory
        """

        # TODO

        pass
