from typing import List, Tuple
import numpy as np
import torch

from src.env import FourRoomEnv
from abc import ABC, abstractmethod


Q1_mask = np.zeros((9, 9))
Q1_mask[:5, :5] = 1.0
Q2_mask = np.zeros((9, 9))
Q2_mask[:5, 5:] = 1.0
Q3_mask = np.zeros((9, 9))
Q3_mask[5:, 5:] = 1.0
Q4_mask = np.zeros((9, 9))
Q4_mask[5:, :5] = 1.0

ROOM_MASKS = [Q1_mask, Q2_mask, Q3_mask, Q4_mask]

class SequenceStateFeaturizer():
    def __init__(self, env: FourRoomEnv, state: Tuple, imposter_locations: List[Tuple]):
        self.env = env
        self.state_size = env.flattened_state_size
        self.states = [env.unflatten_state(s) for s in torch.unbind(state, dim=0)]
        self.imposter_locations = imposter_locations

        self.sp_f = AgentPositionsFeaturizer(env.n_cols, env.n_rows)

        self.spatial = self._featurize_state()
    
    """goals of FeaturizedState:
        1. holds all logic about spatial and non-spatial features
           - will hold a global spatial featurized state
           - will expose get_agent_state(agent_idx: int) method to return:
                 - agent spatial featurized version (permuted so the agent is top channel)
                 - agent non-spatial featurized version
                 - all state will be for sequence of trajectory length
    """
    def _featurize_state(self) -> torch.tensor:
        spatial_features = torch.stack([
            self.sp_f.extract_features(state)
            for state in self.states
        ])
        # Needs to also featurize non-spatial features
        return spatial_features
    
    def get_agent_state(self, agent_idx: int) -> Tuple[torch.tensor, torch.tensor]:
        raise NotImplementedError("Need to implement get agent state method.")

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
        return torch.zeros((num_channels, self.game_width, self.game_height), dtype=torch.float32)


class PositionFeaturizer(SpatialFeaturizer):
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


class AgentPositionsFeaturizer(PositionFeaturizer):
    """
    n_agent_channels:
    Makes a channel per agent. Each channel has 1 in the location of the agent, 0 everywhere else.
    Exepcted to help with learning which agent is which over time.
    """

    def extract_features(self, agent_state: Tuple):
        positions = agent_state[0]
        n_agents, _ = positions.shape
        features = self.get_blank_features(num_channels=n_agents)
        
        for i, (x, y) in enumerate(positions):
            features[i, x, y] = 1

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
