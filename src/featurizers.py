from enum import StrEnum, auto
from typing import Generator, List, Tuple
import numpy as np
import torch

from src.env import FourRoomEnv, StateFields
from abc import ABC, abstractmethod

"""
TODO:
1. Make it batchable
2. Add dead rewards
"""

Q1_mask = np.zeros((9, 9))
Q1_mask[:5, :5] = 1.0
Q2_mask = np.zeros((9, 9))
Q2_mask[:5, 5:] = 1.0
Q3_mask = np.zeros((9, 9))
Q3_mask[5:, 5:] = 1.0
Q4_mask = np.zeros((9, 9))
Q4_mask[5:, :5] = 1.0

ROOM_MASKS = [Q1_mask, Q2_mask, Q3_mask, Q4_mask]

class FeaturizerType(StrEnum):
    PERPSECTIVE = auto()
    GLOBAL = auto()

    @staticmethod
    def build(featurizer_type: str, env: FourRoomEnv):
        assert featurizer_type in [f.value for f in FeaturizerType], f"Invalid featurizer type: {featurizer_type}"
        if featurizer_type == FeaturizerType.PERPSECTIVE:
            return PerspectiveFeaturizer(env=env)
        elif featurizer_type == FeaturizerType.GLOBAL:
            return GlobalFeaturizer(env=env)


class StateSequenceFeaturizer(ABC):
    """
    Featurizer that takes a sequence of states and imposter locations and featurizes them.

    Exposes a generator that yields the featurized state from each agent's perspective.

    Parameters:
        env (FourRoomEnv): The environment.
        sequence_len (int): The length of the sequence.
    """

    def __init__(self, env: FourRoomEnv):
        self.env = env
        self.state_size = env.flattened_state_size

    @property
    def featurized_shape(self):
        raise NotImplementedError("Need to implement featurized_shape property.")

    @abstractmethod
    def fit(self, state_sequence: torch.Tensor) -> None:
        """
        Featurizes the state sequence and imposter locations. Stores the featurized states, this impacts state returned by generator.

        Parameters:
            state_sequence (torch.Tensor): A sequence of states.
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
    - Non-spatial are also ordered based on the agent in question.
    """

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

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

    @property
    def featurized_shape(self):
        non_spatial_shape = torch.sum(
            torch.stack(
                [self.agent_non_sp_f.shape, self.global_non_sp_f.shape], axis=0
            ),
            axis=0,
        )
        return self.sp_f.shape, non_spatial_shape

    def fit(self, state_sequence: torch.Tensor) -> None:
        assert (
            state_sequence.dim() == 3
        ), f"Expected 3D tensor. Got: {state_sequence.dim()}"

        self.B, self.T, S = state_sequence.size()

        for seq_idx in range(self.T):
            batch_states = [
                self.env.unflatten_state(s) for s in state_sequence[:, seq_idx, :]
            ]
            spatial, agent_non_spatial, global_non_spatial = self._featurize_state(
                batch_states
            )

            if seq_idx == 0:
                # Start with a list
                spatial_list = [spatial]
                agent_non_spatial_list = [agent_non_spatial]
                global_non_spatial_list = [global_non_spatial]
            else:
                # Append new tensor to the list
                spatial_list.append(spatial)
                agent_non_spatial_list.append(agent_non_spatial)
                global_non_spatial_list.append(global_non_spatial)

            self.spatial = torch.stack(spatial_list)
            self.agent_non_spatial = torch.stack(agent_non_spatial_list)
            self.global_non_spatial = torch.stack(global_non_spatial_list)

        self.spatial = self.spatial.transpose(0, 1)
        self.agent_non_spatial = self.agent_non_spatial.transpose(0, 1)
        self.global_non_spatial = self.global_non_spatial.transpose(0, 1)

    def _featurize_state(
        self, states: List[Tuple]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial_features = torch.stack(
            [self.sp_f.extract_features(state) for state in states]
        )
        agent_non_spatial_features = torch.stack(
            [self.agent_non_sp_f.extract_features(state) for state in states]
        ).view(self.B, -1, self.env.n_agents)

        global_non_spatial_features = torch.stack(
            [self.global_non_sp_f.extract_features(state) for state in states]
        )

        return spatial_features, agent_non_spatial_features, global_non_spatial_features

    def generator(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        spatial_rep = self.spatial.detach().clone()
        agent_non_spatial_rep = self.agent_non_spatial.detach().clone()
        global_non_spatial_rep = self.global_non_spatial.detach().clone()

        C = spatial_rep.size(2)  # number of channels size (B, T, C, H, W)

        agents = torch.arange(self.env.n_agents)

        channel_order = torch.arange(C)

        for agent_idx in range(self.env.n_agents):
            channel_order[0] = agent_idx
            agents[0] = agent_idx
            if agent_idx > 0:
                channel_order[agent_idx] = agent_idx - 1
                agents[agent_idx] = agent_idx - 1

            non_spatial = torch.cat(
                [
                    agent_non_spatial_rep[:, :, :, agents]
                    .detach()
                    .clone()
                    .view(self.B, self.T, -1),
                    global_non_spatial_rep,
                ],
                dim=2,
            )

            yield (spatial_rep[:, :, channel_order, :, :].detach().clone(), non_spatial)


class GlobalFeaturizer(StateSequenceFeaturizer):
    """
    Featurizer that takes a sequence of states and imposter locations and featurizes them from a global perspective.

    How does this differ from PerspectiveFeaturizer?
    GlobalFeaturizer does not shift ordering of channels based on the agent in question. Instead we just simply append one-hot encoding of the agent index to the non-spatial features.
    """

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

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

    @property
    def featurized_shape(self):
        non_sp_shape = self.non_spatial_features.shape
        non_sp_shape[0] += self.env.n_agents
        return self.spatial_features.shape, non_sp_shape

    def fit(self, state_sequence: torch.Tensor) -> None:
        assert (
            state_sequence.dim() == 3
        ), f"Expected 3D tensor. Got: {state_sequence.dim()}"

        self.B, self.T, S = state_sequence.size()

        for seq_idx in range(self.T):
            batch_states = [
                self.env.unflatten_state(s) for s in state_sequence[:, seq_idx, :]
            ]
            spatial, non_spatial = self._featurize_state(batch_states)

            if seq_idx == 0:
                spatial_list = [spatial]  # Start with a list
                non_spatial_list = [non_spatial]
            else:
                spatial_list.append(spatial)  # Append new tensor to the list
                non_spatial_list.append(non_spatial)

            self.spatial = torch.stack(spatial_list)  # Convert list to tensor stack
            self.non_spatial = torch.stack(non_spatial_list)

        self.non_spatial = self.non_spatial.transpose(0, 1)
        self.spatial = self.spatial.transpose(0, 1)

    def _featurize_state(self, batch_state) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_features = torch.stack(
            [self.spatial_features.extract_features(state) for state in batch_state]
        )
        non_spatial_features = torch.stack(
            [self.non_spatial_features.extract_features(state) for state in batch_state]
        )

        return spatial_features, non_spatial_features

    def generator(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        for agent_idx in range(self.env.n_agents):
            agent_idx_tensor = torch.zeros(self.B, self.T, self.env.n_agents)
            agent_idx_tensor[:, :, agent_idx] = 1

            yield (
                self.spatial.detach().clone(),
                torch.cat([self.non_spatial.detach().clone(), agent_idx_tensor], dim=2),
            )


class Featurizer(ABC):
    """Extracts features from the environment state."""

    def __init__(self, env: FourRoomEnv):
        self.env = env

    @abstractmethod
    def extract_features(self, state: Tuple) -> torch.Tensor:
        """Extracts features from the environment state."""
        raise NotImplementedError("Need to implement extract_features method.")

    @property
    def shape(self) -> Tuple:
        """Returns the shape of the features."""
        raise NotImplementedError("Need to implement shape property.")


class SpatialFeaturizer(Featurizer):
    """Returns a 3D Numpy array for the specific feacture."""

    def __init__(
        self,
        env,
    ):
        super().__init__(env=env)

    def get_blank_features(self, num_channels):
        return torch.zeros(
            (num_channels, self.env.n_cols, self.env.n_rows), dtype=torch.float32
        )

    @property
    def shape(self):
        return torch.tensor([1, self.env.n_cols, self.env.n_rows], dtype=torch.int)


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

    @property
    def shape(self):
        return torch.tensor(
            [self.env.n_agents, self.env.n_cols, self.env.n_rows], dtype=torch.int
        )


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

    @property
    def shape(self):
        return torch.tensor([2, self.env.n_cols, self.env.n_rows], dtype=torch.int)


class CombineFeaturizer(Featurizer):
    """
    Combines featurizers into a single tensor.
    """

    def __init__(self, featurizers: List[Featurizer]):
        shapes = [f.shape[1:] for f in featurizers]
        assert all(
            torch.all(shape.eq(shapes[0])) for shape in shapes
        ), "All featurizers must have the same shape (ignoring the number of first dimension)."
        self.featurizers = featurizers

    def extract_features(self, agent_state: Tuple):
        return torch.cat(
            [f.extract_features(agent_state) for f in self.featurizers], axis=0
        )

    @property
    def shape(self):
        assert len(self.featurizers) > 0, "No featurizers provided."
        shapes = torch.stack([f.shape for f in self.featurizers], axis=0)
        last_dims = shapes[0, 1:]
        return torch.tensor([shapes[:, 0].sum().item(), *last_dims], dtype=torch.int)


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

    @property
    def shape(self):
        raise NotImplementedError("Need to implement shape property.")


class StateFieldFeaturizer(Featurizer):

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

    @property
    def shape(self):
        return self.env.compute_state_dims(self.state_field)
