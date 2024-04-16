from src.env import Action
from typing import List
import numpy as np
from collections import defaultdict


class AgentPolicy:

    def __init__(
        self,
        available_actions: List[Action],
        epsilon: float = 0,
        consistent: bool = False,
    ):

        self.available_actions = available_actions
        self.n_actions = len(available_actions)
        self.epsilon = epsilon
        self.consistent = consistent
        self.Q = defaultdict(lambda: np.zeros(self.n_actions))

    def get_action(self, state):

        if np.random.random() >= self.epsilon:
            return np.random.choice(self.available_actions)

        if self.consistent:
            return self.available_actions[np.argmax(self.Q[state])]

        # randomly breaking ties
        max_action_value = np.max(self.Q[state])
        optimal_action_idxs = np.where(self.Q[state] == max_action_value)
        return self.available_actions[np.random.choice(optimal_action_idxs)]
