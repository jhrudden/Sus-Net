from typing import List, Tuple
from src.env import FourRoomEnv
from abc import ABC, abstractmethod
import numpy as np
import torch


Q1_mask = np.zeros((9, 9))
Q1_mask[:5, :5] = 1.0
Q2_mask = np.zeros((9, 9))
Q2_mask[:5, 5:] = 1.0
Q3_mask = np.zeros((9, 9))
Q3_mask[5:, 5:] = 1.0
Q4_mask = np.zeros((9, 9))
Q4_mask[5:, :5] = 1.0

ROOM_MASKS = [Q1_mask, Q2_mask, Q3_mask, Q4_mask]

# TODO: Sequence 
class SequenceFeaturizer():
    def __init__(self, env: FourRoomEnv):
        self.env = env
        self.state_size = env

    def featurize(self, state_sequence: torch.tensor, imposter_locations: torch.tensor) -> torch.tensor:
        assert state_sequence.shape[0] == imposter_locations.shape[0], "State sequence and imposter locations must have the same length"
        assert state_sequence.size(1) == self.env.flattened_state_size, "State sequence must have the same size as the flattened state size"
        return torch.stack([
            self._featurize_state(self.env.unflatten_state(s_i.numpy()), imposter_locations[idx].numpy())
            for idx, s_i in  enumerate(torch.unbind(state_sequence, dim=0))
        ])
    
    def _featurize_state(self, state: Tuple, imposter_positions) -> np.array:
        print(state, imposter_positions)
        return torch.tensor([])

class SpatialFeaturizer(ABC):
    """Returns a 3D Numpy array for the specific feacture."""

    def __init__(
        self,
        game_width,
        game_height,
    ):
        self.game_width = game_width
        self.game_height = game_height

    @abstractmethod
    def extract_features(agent_state: Tuple) -> np.array:
        raise NotImplementedError("Need to implement extract features method.")

    def get_blank_features(self, num_channels):
        return np.zeros((num_channels, self.game_width, self.game_height))


class SelfPositionFeaturizer(SpatialFeaturizer):
    """
    1 channel: 1 in positon of agent and 0 everywhere else
    """

    def extract_features(self, agent_state: Tuple):
        features = self.get_blank_features(num_channels=1)
        x, y = agent_state.agent_position
        features[x, y] = 1
        return features


class AgentsAtPositionFeaturizer(SpatialFeaturizer):
    """
    1 channel: Number of agents located in a particular position.
    """

    def extract_features(self, agent_state: Tuple):
        features = self.get_blank_features(num_channels=1)
        for alive, (x, y) in zip(
            agent_state.other_agents_alive, agent_state.other_agent_positions
        ):
            features[x, y] += alive

        return features


class AgentPositionsFeaturizer(SelfPositionFeaturizer):
    """
    n_agent_channels:
    Makes a channel per agent. Each channel has 1 in the location of the agent, 0 everywhere else.
    Exepcted to help with learning which agent is which over time.
    """

    def extract_features(self, agent_state: Tuple):
        n_agents = 1 + len(agent_state.other_agent_positions)

        features = self.get_blank_features(num_channels=n_agents)

        # position of the current agent
        features[0] = super().extract_features()

        # positions of other agents
        for idx, (alive, (x, y)) in enumerate(
            zip(agent_state.other_agents_alive, agent_state.other_agent_positions)
        ):
            features[idx, x, y] += alive

        return features


class JobFeaturizer(SpatialFeaturizer):
    """
    2 channels:
        - positions of incomplete jobs.
        - positions of done jobs
    """

    def extract_features(self, agent_state: Tuple):

        features = self.get_blank_features(num_channels=2)

        for job_position, job_done in zip(
            agent_state.job_positions, agent_state.completed_jobs
        ):
            x, y = job_position
            features[int(job_done), x, y] = 1

        return features


class CombineSpatialFeaturizer(SpatialFeaturizer):
    """
    Combines multiple Spatial featurizers into a single 3D array.
    """

    def __init__(self, featurizers: List[SpatialFeaturizer]):
        self.featurizers = featurizers

    def extract_features(self, agent_state: Tuple):
        features = [f.extract_features(agent_state) for f in self.featurizers]
        return np.concatenate(features, axis=0)


class PartiallyObservableFeaturizer(CombineSpatialFeaturizer):
    """
    Zeros everything that is not visible to the agent.
    """

    def __init__(self, featurizers: List[SpatialFeaturizer], add_obs_mask_feature=True):
        super().__init__(featurizers=featurizers)
        self.add_obs_mask_feature = add_obs_mask_feature

    def extract_features(self, agent_state: Tuple):
        features = super().extract_features(agent_state=agent_state)

        # agent position
        x, y = agent_state.agent_position

        # determine observability mask
        obs_mask = self.get_blank_features(num_channels=1)
        for room_mask in ROOM_MASKS:
            if room_mask[x, y] > 0:
                obs_mask += room_mask

        obs_mask = np.minimum(obs_mask, 1)  # 1 if cell is observable, 0 if not

        # zero out all that is not in the room
        features = features * obs_mask

        # add channel to indicate what we can/cannot observe? the observation mask?
        if self.add_obs_mask_feature:
            return np.concatenate([features, obs_mask], axis=0)

        return features
