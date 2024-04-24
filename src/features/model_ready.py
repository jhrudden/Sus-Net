from enum import StrEnum, auto
from abc import ABC, abstractmethod
from typing import List, Tuple
import torch

from src.features.component import (
    AgentPositionsFeaturizer,
    CompositeFeaturizer,
    DistanceToImposterFeaturizer,
    JobFeaturizer,
    StateFieldFeaturizer,
    OneHotAgentPositionFeaturizer,
)
from src.environment.base import FourRoomEnv, StateFields


class FeaturizerType(StrEnum):
    PERPSECTIVE = auto()
    GLOBAL = auto()
    FLAT = auto()

    @staticmethod
    def build(featurizer_type: str, env: FourRoomEnv):
        assert featurizer_type in [
            f.value for f in FeaturizerType
        ], f"Invalid featurizer type: {featurizer_type}"
        if featurizer_type == FeaturizerType.PERPSECTIVE:
            return PerspectiveFeaturizer(env=env)
        elif featurizer_type == FeaturizerType.GLOBAL:
            return GlobalFeaturizer(env=env)
        elif featurizer_type == FeaturizerType.FLAT:
            return FlatFeaturizer(env=env)


class SequenceStateFeaturizer(ABC):
    """
    Featurizer that takes a sequence of states and imposter locations and featurizes them.

    Exposes a method to generate featurized states from each agent's perspective.

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
        Featurizes the state sequence and imposter locations. Stores the featurized states, this impacts state returned by generate_featurized_states.

        Parameters:
            state_sequence (torch.Tensor): A sequence of states.
        """
        raise NotImplementedError("Need to implement fit method.")

    @abstractmethod
    def generate_featurized_states(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns the featurized state from each agent's perspective.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of spatial and non-spatial features.
        """
        raise NotImplementedError(
            "Need to implement generate_featurized_states method."
        )


class PerspectiveFeaturizer(SequenceStateFeaturizer):
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

        self.sp_f = CompositeFeaturizer(
            featurizers=[AgentPositionsFeaturizer(env=env), JobFeaturizer(env=env)]
        )
        self.agent_non_sp_f = CompositeFeaturizer(
            [
                StateFieldFeaturizer(env=env, state_field=StateFields.ALIVE_AGENTS),
                *(
                    [StateFieldFeaturizer(env=env, state_field=StateFields.TAG_COUNTS)]
                    if "Tagging" in env.__class__.__name__
                    else []
                ),
            ]
        )
        self.global_non_sp_f = CompositeFeaturizer(
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

    def generate_featurized_states(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        spatial_rep = self.spatial.detach().clone()
        agent_non_spatial_rep = self.agent_non_spatial.detach().clone()
        global_non_spatial_rep = self.global_non_spatial.detach().clone()

        featurized = []

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

            featurized.append(
                (
                    spatial_rep[:, :, channel_order, :, :]
                    .detach()
                    .clone()
                    .requires_grad_(True),
                    non_spatial.requires_grad_(True),
                )
            )

        return featurized


class GlobalFeaturizer(SequenceStateFeaturizer):
    """
    Featurizer that takes a sequence of states and imposter locations and featurizes them from a global perspective.

    How does this differ from PerspectiveFeaturizer?
    GlobalFeaturizer does not shift ordering of channels based on the agent in question. Instead we just simply append one-hot encoding of the agent index to the non-spatial features.
    """

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

        self.spatial_features = CompositeFeaturizer(
            featurizers=[
                AgentPositionsFeaturizer(env=env),
                JobFeaturizer(env=env),
            ]
        )

        self.non_spatial_features = CompositeFeaturizer(
            featurizers=[
                StateFieldFeaturizer(env=env, state_field=StateFields.ALIVE_AGENTS),
                *(
                    [StateFieldFeaturizer(env=env, state_field=StateFields.TAG_COUNTS)]
                    if "Tagging" in env.__class__.__name__
                    else []
                ),
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

    def generate_featurized_states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        featurized = []
        for agent_idx in range(self.env.n_agents):
            agent_idx_tensor = torch.zeros(self.B, self.T, self.env.n_agents)
            agent_idx_tensor[:, :, agent_idx] = 1

            featurized.append(
                (
                    self.spatial.detach().clone().requires_grad_(True),
                    torch.cat(
                        [self.non_spatial.detach().clone(), agent_idx_tensor], dim=2
                    ).requires_grad_(True),
                )
            )

        return featurized


class FlatFeaturizer(SequenceStateFeaturizer):
    """
    Quite simple, just return the flattened state.
    """

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)
        self.featurizer = CompositeFeaturizer(
            [
                OneHotAgentPositionFeaturizer(env=env),
                DistanceToImposterFeaturizer(env=env),
            ]
        )

    @property
    def featurized_shape(self):
        return (
            self.env.flattened_state_size,
            self.env.flattened_state_size,
        )  # current returning zeros for spatial features (this is a hack, need to fix this)

    def fit(self, state_sequence: torch.Tensor) -> None:
        assert (
            state_sequence.dim() == 3
        ), f"Expected 3D tensor. Got: {state_sequence.dim()}"

        self.B, self.T, _ = state_sequence.size()

        self.flattened_state = state_sequence.view(self.B, self.T, -1).to(torch.float32)

        for seq_idx in range(self.T):
            batch_states = [
                self.env.unflatten_state(s) for s in state_sequence[:, seq_idx, :]
            ]

            featurized_batch_timestep_states = []
            for batch_idx, batch_state in enumerate(batch_states):

                featurized_timestep_batch_state = torch.cat(
                    [
                        self.flattened_state[batch_idx, seq_idx],
                        self.featurizer.extract_features(batch_state),
                    ]
                )
                featurized_batch_timestep_states.append(featurized_timestep_batch_state)

            featurized_batch_timestep_states = torch.stack(
                featurized_batch_timestep_states
            )

            if seq_idx == 0:
                feature_sequence_list = [featurized_batch_timestep_states]
            else:
                feature_sequence_list.append(featurized_batch_timestep_states)

        self.featurized_state = torch.stack(feature_sequence_list)
        self.featurized_state = self.featurized_state.transpose(0, 1)

    def generate_featurized_states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        featurized = []
        for agent_idx in range(self.env.n_agents):
             
            featurized.append(
                (
                    torch.zeros(self.B, self.T, 1).requires_grad_(True),
                    self.featurized_state.detach().clone().requires_grad_(True),
                )
            )

        return featurized
