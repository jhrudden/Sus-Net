import numpy as np
import torch
from collections import namedtuple

# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple(
    "Batch", ("states", "actions", "rewards", "next_states", "imposters", "dones")
)


class ReplayBuffer:
    def __init__(
        self,
        max_size: int,
        state_size: int,
        trajectory_size: int,
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
        self.n_agents = n_agents
        self.n_imposters = n_imposters

        # initializing the timestep buffer
        self.states = torch.empty(
            (self.max_size, self.trajectory_size, self.state_size)
        )
        self.actions = torch.empty((self.max_size, self.n_agents), dtype=torch.long)
        self.rewards = torch.empty((self.max_size, self.n_agents))
        self.next_states = torch.empty(
            (self.max_size, self.trajectory_size, self.state_size)
        )
        self.dones = torch.empty((self.max_size, 1), dtype=torch.bool)
        self.imposters = torch.empty(
            (self.max_size, self.n_imposters), dtype=torch.int16
        )

        # initializing current index and buffer size
        self.idx = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done, imposters):
        """
        Add a transition to the buffer.

        Parameters
            - state (np.ndarray): Current state
            - action (np.ndarray): Action taken
            - reward (float): Reward received
            - next_state (np.ndarray): Next state
            - done (bool): Whether the episode ended
            - imposters (np.ndarray): List of imposter indices
        """
        self.states[self.idx] = torch.tensor(state)
        self.actions[self.idx] = torch.tensor(action)
        self.rewards[self.idx] = torch.tensor(reward)
        self.next_states[self.idx] = torch.tensor(next_state)
        self.dones[self.idx] = torch.tensor(done)
        self.imposters[self.idx] = torch.tensor(imposters)

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

        sample_idx = torch.randint(0, self.size, (batch_size,))

        return Batch(
            states=self.states[sample_idx],
            actions=self.actions[sample_idx],
            rewards=self.rewards[sample_idx],
            imposters=self.imposters[sample_idx],
            next_states=self.next_states[sample_idx],
            dones=self.dones[sample_idx],
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
            state_sequence = np.zeros((self.trajectory_size, self.state_size))
            s, _ = env.reset()
            state = env.flatten_state(s)
            # fill the sequence with the current state for the first `trajectory_size` steps
            for i in range(self.trajectory_size):
                state_sequence[i] = state

            done = False
            truncation = False
            while not done and not truncation:
                imposters = env.imposter_idxs
                action = env.sample_actions()
                n_s, reward, done, truncation, _ = env.step(action)
                next_state = env.flatten_state(n_s)
                next_sequence = np.roll(
                    state_sequence.copy(), -1, axis=0
                )  # shift the sequence by one step back (copying the array to avoid reference issues)
                next_sequence[-1] = (
                    next_state.copy()
                )  # replace the last state in the sequence with the new state
                self.add(
                    state=state_sequence,
                    action=action,
                    reward=reward,
                    next_state=next_sequence,
                    done=done,
                    imposters=imposters,
                )
                state = next_state
                state_sequence = next_sequence
                step += 1

                if done:
                    break

                if step >= num_steps:
                    break
