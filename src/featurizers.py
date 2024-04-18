from typing import Generator, List, Tuple
import numpy as np
import torch

from src.env import FourRoomEnv, StateFields
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


class SequenceStateFeaturizer:
    def __init__(
        self,
        env: FourRoomEnv,
        state_sequence: Tuple,
        imposter_locations: List[Tuple],
    ):
        self.env = env
        self.state_size = env.flattened_state_size
        self.states = [
            env.unflatten_state(s) for s in torch.unbind(state_sequence, dim=0)
        ]
        self.sequence_len = len(self.states)
        self.imposter_locations = imposter_locations

        self.sp_f = CombineFeaturizer(
            featurizers=[AgentPositionsFeaturizer(env=env), JobFeaturizer(env=env)]
        )
        self.local_non_sp_f = CombineFeaturizer([AliveAgentFeaturizer(env=env)])
        self.global_non_sp_f = CombineFeaturizer([JobStatusFeaturizer(env=env)])

        self.spatial, self.agent_non_spacial, self.global_non_spatial = (
            self._featurize_state()
        )

    """goals of FeaturizedState:
        1. holds all logic about spatial and non-spatial features
           - will hold a global spatial featurized state
           - will expose get_agent_state(agent_idx: int) method to return:
                 - agent spatial featurized version (permuted so the agent is top channel)
                 - agent non-spatial featurized version
                 - all state will be for sequence of trajectory length
    """

    def _featurize_state(self) -> torch.tensor:
        """
        Featurizes the state of the environment based on order of input states.

        THIS METHOD DOES NOT:
        - do any agent-specific featurization

        Returns:
            torch.tensor: 3D tensor of spatial features.
            TODO: 1D tensor of non-spatial features.
        """
        spatial_features = torch.stack(
            [self.sp_f.extract_features(state) for state in self.states]
        )
        agent_non_spacial_features = torch.stack(
            [self.local_non_sp_f.extract_features(state) for state in self.states]
        ).view(self.sequence_len, -1, self.env.n_agents)

        global_non_spacial_features = torch.stack(
            [self.global_non_sp_f.extract_features(state) for state in self.states]
        )

        return spatial_features, agent_non_spacial_features, global_non_spacial_features

    def generator(self) -> Generator[Tuple[torch.tensor, torch.tensor], None, None]:
        """
        Generator that yields the featurized state from each agent's perspective.

        Yields:
            Tuple[torch.tensor, torch.tensor]: Tuple of spatial and non-spatial features.
            # TODO: only returns spatial features for now.
        """

        spatial_rep = self.spatial.clone()
        agent_non_spacial_rep = self.agent_non_spacial.clone()
        global_non_spacial_rep = self.global_non_spatial.clone()

        print(spatial_rep.shape)
        print(agent_non_spacial_rep.shape)
        print(global_non_spacial_rep.shape)

        n_channels = torch.arange(spatial_rep.shape[1])
        agent_non_spacial_dim = torch.arange(agent_non_spacial_rep.shape[2])

        for agent_idx in range(self.env.n_agents):
            n_channels[0] = agent_idx
            agent_non_spacial_dim[0] = agent_idx
            if agent_idx > 0:
                n_channels[agent_idx] = agent_idx - 1
                agent_non_spacial_dim[agent_idx] = agent_idx - 1

            non_spatial = torch.cat(
                [
                    agent_non_spacial_rep[:, :, agent_non_spacial_dim]
                    .clone()
                    .view(self.sequence_len, -1),
                    global_non_spacial_rep,
                ],
                dim=1,
            )

            print(non_spatial)

            yield (spatial_rep[:, n_channels, :, :].clone(), non_spatial)


class SpatialFeaturizer(ABC):
    """Returns a 3D Numpy array for the specific feacture."""

    def __init__(
        self,
        env,
    ):
        self.env = env

    @abstractmethod
    def extract_features(agent_state: Tuple) -> np.array:
        raise NotImplementedError("Need to implement extract features method.")

    def get_blank_features(self, num_channels):
        return torch.zeros(
            (num_channels, self.env.n_cols, self.env.n_rows), dtype=torch.float32
        )


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

    def extract_features(self, agent_state: Tuple, alive_only: bool = True):
        print(agent_state)
        positions = agent_state[self.env.state_fields[StateFields.AGENT_POSITIONS]]
        alive = agent_state[self.env.state_fields[StateFields.ALIVE_AGENTS]]
        n_agents, _ = positions.shape
        features = self.get_blank_features(num_channels=n_agents)

        for i, (x, y) in enumerate(positions):
            if alive_only and alive[i]:
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

        job_positions = agent_state[self.env.state_fields[StateFields.JOB_POSITIONS]]
        job_statuses = agent_state[self.env.state_fields[StateFields.JOB_STATUS]]

        for job_position, job_done in zip(job_positions, job_statuses):
            x, y = job_position
            features[int(job_done), x, y] = 1

        return features


class CombineFeaturizer(SpatialFeaturizer):
    """
    Combines multiple Spatial featurizers into a single 3D array.
    """

    def __init__(self, featurizers: List[SpatialFeaturizer]):
        self.featurizers = featurizers

    def extract_features(self, agent_state: Tuple):
        return torch.cat(
            [f.extract_features(agent_state) for f in self.featurizers], axis=0
        )


class PartiallyObservableFeaturizer(CombineFeaturizer):
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


class NonSpatialFeaturizer(ABC):
    """Returns a 1D Numpy array for the specific feacture."""

    def __init__(
        self,
        env,
    ):
        self.env = env

    @abstractmethod
    def extract_features(agent_state: Tuple) -> np.array:
        raise NotImplementedError("Need to implement extract features method.")


class AliveAgentFeaturizer(NonSpatialFeaturizer):

    def extract_features(self, agent_state: Tuple) -> np.array:
        return torch.tensor(
            agent_state[self.env.state_fields[StateFields.ALIVE_AGENTS]],
            dtype=torch.float32,
        )


class JobStatusFeaturizer(NonSpatialFeaturizer):

    def extract_features(self, agent_state: Tuple) -> np.array:
        return torch.tensor(
            agent_state[self.env.state_fields[StateFields.JOB_STATUS]],
            dtype=torch.float32,
        )
