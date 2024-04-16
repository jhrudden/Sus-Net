import torch
import numpy as np
from collections import namedtuple

# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple("Batch", ("states", "actions", "rewards", "next_states", "dones"))


class ReplayMemory:
    def __init__(self, max_size, state_size, sequence_size):

        self.max_size = max_size
        self.sequence_size = sequence_size
        self.state_size = state_size

        self.sample_idx_to_buffer_idx = {}

        self.start_idxs = {}

        # Preallocating all the required memory, for speed concerns
        self.states = torch.empty((max_size, state_size))
        self.actions = torch.empty((max_size, 1), dtype=torch.long)
        self.rewards = torch.empty((max_size, 1))
        self.next_states = torch.empty((max_size, state_size))
        self.dones = torch.empty((max_size, 1), dtype=torch.bool)
        self.timesteps = torch.empty((max_size, 1), dtype=torch.long)
        self.is_start = torch.empty((max_size, 1), dtype=torch.bool)
        self.episode_idx = torch.empty((max_size, 1), dtype=torch.long)

        # Pointer to the current location in the circular buffer
        self.idx = 0
        # Indicates number of transitions currently stored in the buffer
        self.size = 0
        self.ep_idx = 0

    def add(self, state, action, reward, next_state, done, is_start, timestep):
        """Add a transition to the buffer.

        :param state: 1-D np.ndarray of state-features
        :param action: Integer action
        :param reward: Float reward
        :param next_state: 1-D np.ndarray of state-features
        :param done: Boolean value indicating the end of an episode
        """

        # if is_start:
        #     for _ in range(self.sequence_size):
        #         self.add(# dummy )
        offset_idx = (self.idx + self.sequence_size - 1) % self.max_size

        if is_start:
            self.idx = offset_idx

        # YOUR CODE HERE: Store the input values into the appropriate
        # attributes, using the current buffer position `self.idx`
        self.states[self.idx] = torch.tensor(state)
        self.actions[self.idx] = torch.tensor(action)
        self.rewards[self.idx] = torch.tensor(reward)
        self.next_states[self.idx] = torch.tensor(next_state)
        self.dones[self.idx] = torch.tensor(done)
        self.is_start[self.idx] = is_start
        self.timesteps[self.idx] = timestep

        self.sample_idx_to_buffer_idx[self.idx] = self.idx
        del_idx = (self.idx + self.sequence_size - 1) % self.max_size
        if del_idx in self.sample_idx_to_buffer_idx:
            del self.sample_idx_to_buffer_idx[del_idx]

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

        # YOUR CODE HERE: Randomly sample an appropriate number of
        # transitions *without replacement*. If the buffer contains less than
        # `batch_size` transitions, return all of them. The return type must
        # be a `Batch`.

        sample_indices = np.random.choice(
            np.arange(self.size), size=batch_size, replace=False
        )
        batch = Batch(
            states=self.states[sample_indices],
            actions=self.actions[sample_indices],
            rewards=self.rewards[sample_indices],
            next_states=self.next_states[sample_indices],
            dones=self.dones[sample_indices],
        )

        return batch

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env: Gymnasium environment
        :param num_steps: Number of steps to populate the replay memory
        """

        # YOUR CODE HERE: Run a random policy for `num_steps` time-steps and
        # populate the replay memory with the resulting transitions.
        # Hint: Use the self.add() method.

        step_idx = 0
        done = trunc = True

        while True:
            step_idx += 1

            if done or trunc:
                S, _ = env.reset()

            A = env.action_space.sample()
            S_prime, R, done, trunc, _ = env.step(A)
            self.add(state=S, action=A, reward=R, next_state=S_prime, done=done)

            S = S_prime

            if step_idx >= num_steps:
                break

    def get_del_idx_offset(self, idx):
        i = idx + self.sequence_size - 1
        if i >= self.max_size:
            return i - self.max_size
