from env import AgentState, AgentStateWithTagging
from abc import ABC, abstractmethod
import numpy as np


class SpacialFeatureExtractor(ABC):

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
        return np.zeros(self.game_width, self.game_height)


class SelfPositionFeatureExtractor(SpacialFeatureExtractor):

    def extract_feature(self, agent_state: AgentState):
        feature = self.get_blank_feature()
        x, y = agent_state.agent_position
        feature[x, y] = 1
        return feature


class OthersPositionFeatureExtractor(SpacialFeatureExtractor):

    def extract_feature(self, agent_state: AgentState):
        feature = self.get_blank_feature()
        for alive, (x, y) in zip(
            agent_state.other_agents_alive, agent_state.other_agent_positions
        ):
            feature[x, y] += alive

        return feature
