from env import AgentState, AgentStateWithTagging
from abc import ABC, abstractmethod
import numpy as np


class SpacialFeaturizer(ABC):

    @abstractmethod
    def __init__(
        self,
        game_width,
        game_height,
    ):
        self.game_width = game_width
        self.game_height = game_height

    @abstractmethod
    def extract_feature(agent_state: AgentState) -> np.array:
        raise NotImplementedError("Need to implement extract feature method.")

    def get_blank_feature(self):
        return np.zeros((self.game_width, self.game_height))


class SelfPositionFeaturizer(SpacialFeaturizer):
    """2D channel with 1 in positon of agent and 0 everywhere else"""

    def extract_feature(self, agent_state: AgentState):
        feature = self.get_blank_feature()
        x, y = agent_state.agent_position
        feature[x, y] = 1
        return feature


class OthersPositionFeaturizer(SpacialFeaturizer):
    """Number of agents located in a particular position."""

    def extract_feature(self, agent_state: AgentState):
        feature = self.get_blank_feature()
        for alive, (x, y) in zip(
            agent_state.other_agents_alive, agent_state.other_agent_positions
        ):
            feature[x, y] += alive

        return feature


class AllPositionFeaturizer(SelfPositionFeaturizer):
    """
    Makes a channel per agent. Each channel has 1 in the location of the agent, 0 everywhere else.
    Exepcted to help with learning which agent is which over time.
    """

    def extract_feature(self, agent_state: AgentState):
        n_agents = 1 + len(agent_state.other_agent_positions)

        feature = np.zeros((n_agents, self.game_width, self.game_height))

        # position of the current agent
        feature[0] = super().extract_feature()

        # positions of other agents
        for idx, (alive, (x, y)) in enumerate(
            zip(agent_state.other_agents_alive, agent_state.other_agent_positions)
        ):
            feature[idx, x, y] += alive

        return feature


class JobFeaturizer(SpacialFeaturizer):
    """
    Makes 2 channels:
        - positions of incomplete jobs.
        - positions of done jobs
    """

    def extract_feature(self, agent_state: AgentState):

        feature = np.zeros((2, self.game_width, self.game_height))

        for job_position, job_done in zip(
            agent_state.job_positions, agent_state.completed_jobs
        ):
            x, y = job_position
            feature[int(job_done), x, y] = 1

        return feature
