from typing import List
from env import AgentState, AgentStateWithTagging
from abc import ABC, abstractmethod
import numpy as np


class SpacialFeaturizer(ABC):
    """Returns a 3D Numpy array for the specific feacture."""

    def __init__(
        self,
        game_width,
        game_height,
    ):
        self.game_width = game_width
        self.game_height = game_height

    @abstractmethod
    def extract_features(agent_state: AgentState) -> np.array:
        raise NotImplementedError("Need to implement extract featurs method.")

    def get_blank_featurs(self, num_channels):
        return np.zeros((num_channels, self.game_width, self.game_height))


class SelfPositionFeaturizer(SpacialFeaturizer):
    """
    1 channel: 1 in positon of agent and 0 everywhere else
    """

    def extract_features(self, agent_state: AgentState):
        featurs = self.get_blank_featurs(num_channels=1)
        x, y = agent_state.agent_position
        featurs[x, y] = 1
        return featurs


class AgentsAtPositionFeaturizer(SpacialFeaturizer):
    """
    1 channel: Number of agents located in a particular position.
    """

    def extract_features(self, agent_state: AgentState):
        featurs = self.get_blank_featurs(num_channels=1)
        for alive, (x, y) in zip(
            agent_state.other_agents_alive, agent_state.other_agent_positions
        ):
            featurs[x, y] += alive

        return featurs


class AgentPositionsFeaturizer(SelfPositionFeaturizer):
    """
    n_agent_channels:
    Makes a channel per agent. Each channel has 1 in the location of the agent, 0 everywhere else.
    Exepcted to help with learning which agent is which over time.
    """

    def extract_features(self, agent_state: AgentState):
        n_agents = 1 + len(agent_state.other_agent_positions)

        featurs = self.get_blank_featurs(num_channels=n_agents)

        # position of the current agent
        featurs[0] = super().extract_features()

        # positions of other agents
        for idx, (alive, (x, y)) in enumerate(
            zip(agent_state.other_agents_alive, agent_state.other_agent_positions)
        ):
            featurs[idx, x, y] += alive

        return featurs


class JobFeaturizer(SpacialFeaturizer):
    """
    2 channels:
        - positions of incomplete jobs.
        - positions of done jobs
    """

    def extract_features(self, agent_state: AgentState):

        featurs = self.get_blank_featurs(num_channels=2)

        for job_position, job_done in zip(
            agent_state.job_positions, agent_state.completed_jobs
        ):
            x, y = job_position
            featurs[int(job_done), x, y] = 1

        return featurs


class CombineSpacialFeaturizer(SpacialFeaturizer):
    """
    Combines multiple spacial featurizers into a single 3D array.
    """

    def __init__(self, featurizers: List[SpacialFeaturizer]):
        self.featurizers = featurizers

    def extract_features(self, agent_state: AgentState):
        features = [f.extract_features(agent_state) for f in self.featurizers]
        return np.concatenate(features, axis=0)


class PartiallyObservableFeaturizer(CombineSpacialFeaturizer):
    """
    Zeros everything that is not visible to the agent.
    """

    def extract_features(self, agent_state: AgentState):
        features = super().extract_features(agent_state=agent_state)

        # determine room in which the player is in

        # zero out all that is not in the room

        # add channel to indicate what we can/cannot observe?

        return features
