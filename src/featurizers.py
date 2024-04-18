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

class StateSequenceFeaturizer(ABC):
    """
    Featurizer that takes a sequence of states and imposter locations and featurizes them. 

    Exposes a generator that yields the featurized state from each agent's perspective.

    Parameters:
        env (FourRoomEnv): The environment.
        sequence_len (int): The length of the sequence.
    """
    def __init__(self, env: FourRoomEnv, sequence_len: int):
        self.env = env
        self.state_size = env.flattened_state_size
        self.sequence_len = sequence_len

    @abstractmethod
    def fit(self, state_sequence: Tuple, imposter_locations: List[Tuple]) -> None:
        """
        Featurizes the state sequence and imposter locations. Stores the featurized states, this impacts state returned by generator.

        Parameters:
            state_sequence (Tuple): The state sequence.
            imposter_locations (List[Tuple]): The imposter locations.
        """
        raise NotImplementedError("Need to implement fit method.")

    @abstractmethod
    def generator(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Generator that yields the featurized state from each agent's perspective.

        Yields:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of spatial and non-spatial features.
        """
        raise NotImplementedError("Need to implement generator method.")


class PerspectiveFeaturizer(StateSequenceFeaturizer):
    """
    Featurizer that takes a sequence of states and imposter locations and featurizes them from each agent's perspective.

    How does this differ from GlobalFeaturizer? 
    PerspectiveFeaturizer featurizes takes a more agent-centric view of the environment.
    All state representations are from the perspective of the agent. 
    - Spatial features, agent in question is always at the top, and other agents are ordered based on their index in increasing order.
    - Non-spatial are also ordered based on the agent in question. TODO: Dima, say more
    """
    def __init__(self, env: FourRoomEnv, sequence_len: int):
        super().__init__(env, sequence_len)

        self.sp_f = CombineFeaturizer(
            featurizers=[AgentPositionsFeaturizer(env=env), JobFeaturizer(env=env)]
        )
        self.agent_non_sp_f = CombineFeaturizer(
            [
                StateFieldFeaturizer(env=env, state_field=StateFields.ALIVE_AGENTS),
                StateFieldFeaturizer(env=env, state_field=StateFields.TAG_COUNTS),
            ]
        )
        self.global_non_sp_f = CombineFeaturizer(
            [
                StateFieldFeaturizer(env=env, state_field=StateFields.JOB_STATUS),
            ]
        )

    def fit(self, state_sequence: Tuple, imposter_locations: List[Tuple]):
        assert len(state_sequence) == self.sequence_len, f"Sequence length mismatch. Expected: {self.sequence_len} states. Got: {len(state_sequence)} states."
        self.states = [
            self.env.unflatten_state(s) for s in torch.unbind(state_sequence, dim=0)
        ]
        self.sequence_len = len(self.states)
        self.imposter_locations = imposter_locations

        self.spatial, self.agent_non_spacial, self.global_non_spatial = (
            self._featurize_state()
        )

    def _featurize_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_features = torch.stack(
            [self.sp_f.extract_features(state) for state in self.states]
        )
        agent_non_spacial_features = torch.stack(
            [self.agent_non_sp_f.extract_features(state) for state in self.states]
        ).view(self.sequence_len, -1, self.env.n_agents)

        global_non_spacial_features = torch.stack(
            [self.global_non_sp_f.extract_features(state) for state in self.states]
        )

        return spatial_features, agent_non_spacial_features, global_non_spacial_features

    def generator(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        spatial_rep = self.spatial.clone()
        agent_non_spacial_rep = self.agent_non_spacial.clone()
        global_non_spacial_rep = self.global_non_spatial.clone()

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

            yield (spatial_rep[:, n_channels, :, :].clone(), non_spatial)


class GlobalFeaturizer(StateSequenceFeaturizer):
    """
    Featurizer that takes a sequence of states and imposter locations and featurizes them from a global perspective.

    How does this differ from PerspectiveFeaturizer?
    GlobalFeaturizer does not shift ordering of channels based on the agent in question. Instead we just simply append one-hot encoding of the agent index to the non-spatial features.
    """
    def __init__(self, env: FourRoomEnv, sequence_len: int):
        super().__init__(env, sequence_len)

        self.spatial_features = CombineFeaturizer(
            featurizers=[
                AgentPositionsFeaturizer(env=env),
                JobFeaturizer(env=env),
            ]
        )

        self.non_spatial_features = CombineFeaturizer(
            featurizers=[
                StateFieldFeaturizer(env=env, state_field=StateFields.ALIVE_AGENTS),
                StateFieldFeaturizer(env=env, state_field=StateFields.TAG_COUNTS),
                StateFieldFeaturizer(env=env, state_field=StateFields.JOB_STATUS),
            ]
        )

    def fit(self, state_sequence: Tuple, imposter_locations: List[Tuple]) -> None:
        assert len(state_sequence) == self.sequence_len, f"Sequence length mismatch. Expected: {self.sequence_len} states. Got: {len(state_sequence)} states."
        self.states = [
            self.env.unflatten_state(s) for s in torch.unbind(state_sequence, dim=0)
        ]
        self.sequence_len = len(self.states)
        self.imposter_locations = imposter_locations

        self.spatial, self.non_spatial = self._featurize_state()

    def _featurize_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_features = torch.stack(
            [self.spatial_features.extract_features(state) for state in self.states]
        )
        non_spatial_features = torch.stack(
            [self.non_spatial_features.extract_features(state) for state in self.states]
        )

        return spatial_features, non_spatial_features

    def generator(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        for agent_idx in range(self.env.n_agents):
            agent_idx_tensor = torch.zeros(self.sequence_len, self.env.n_agents)
            agent_idx_tensor[:, agent_idx] = 1

            yield (self.spatial.clone(), torch.cat([self.non_spatial.clone(), agent_idx_tensor], dim=1))


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


class StateFieldFeaturizer(NonSpatialFeaturizer):

    def __init__(
        self,
        env,
        state_field,
    ):
        super().__init__(env=env)
        self.state_field = state_field

    def extract_features(self, agent_state: Tuple) -> torch.Tensor:
        return torch.tensor(
            agent_state[self.env.state_fields[self.state_field]],
            dtype=torch.float32,
        )
