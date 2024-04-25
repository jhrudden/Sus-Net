from typing import List, Tuple
import numpy as np
import torch

from src.environment import FourRoomEnv, StateFields
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


class ComponentFeaturizer(ABC):
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


class BaseSpatialFeaturizer(ComponentFeaturizer):
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


class PositionFeaturizer(BaseSpatialFeaturizer):
    """
    1 channel: 1 in positon of agent and 0 everywhere else
    """

    def extract_features(self, agent_state: Tuple):
        features = self.get_blank_features(num_channels=1)
        x, y = agent_state.agent_position
        features[x, y] = 1
        return features


class AgentsAtPositionFeaturizer(BaseSpatialFeaturizer):
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


class JobFeaturizer(BaseSpatialFeaturizer):
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


class CompositeFeaturizer(ComponentFeaturizer):
    """
    Combines featurizers into a single tensor.
    """

    def __init__(self, featurizers: List[ComponentFeaturizer]):
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
    
    def __repr__(self) -> str:
        return str([f.__class__.__name__ for f in self.featurizers])


class PartiallyObservableFeaturizer(CompositeFeaturizer):
    """
    Zeros everything that is not visible to the agent.
    """

    def __init__(
        self, featurizers: List[BaseSpatialFeaturizer], add_obs_mask_feature=True
    ):
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


class StateFieldFeaturizer(ComponentFeaturizer):

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


class OneHotAgentPositionFeaturizer(ComponentFeaturizer):

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

    def extract_features(self, state: Tuple) -> torch.Tensor:

        agent_positions = state[self.env.state_fields[StateFields.AGENT_POSITIONS]]
        one_hot_positions = torch.zeros(
            self.env.n_agents, self.env.n_cols + self.env.n_rows
        )

        for agent_idx, pos in enumerate(agent_positions):
            alive = state[self.env.state_fields[StateFields.ALIVE_AGENTS]][agent_idx]

            if alive:
                one_hot_positions[agent_idx, pos[0]] = 1
                one_hot_positions[agent_idx, self.env.n_cols + pos[1]] = 1

        return one_hot_positions.view(-1)

    @property
    def shape(self) -> torch.tensor:

        return torch.tensor(
            [self.env.n_agents * (self.env.n_cols + self.env.n_rows)], dtype=torch.int
        )


class DistanceToImposterFeaturizer(ComponentFeaturizer):

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

    def extract_features(self, state: Tuple) -> torch.Tensor:

        agent_positions = state[self.env.state_fields[StateFields.AGENT_POSITIONS]]
        imposter_diastances = torch.zeros((self.env.n_agents - 1) * 2)

        alive_agents = state[self.env.state_fields[StateFields.ALIVE_AGENTS]]

        # NOTE: ONLY IF IMPOSTER IN 0th IDX
        imposter_x, imposter_y = agent_positions[0]

        idx = 0

        for agent_idx, pos in enumerate(agent_positions):
            if agent_idx != 0 and alive_agents[agent_idx]:
                imposter_diastances[idx] = imposter_x - pos[0]
                imposter_diastances[idx + 1] = imposter_y - pos[1]
                idx += 2

        return imposter_diastances.view(-1)

    @property
    def shape(self) -> torch.tensor:

        return torch.tensor([(self.env.n_agents - 1) * 2], dtype=torch.int)


class WallsFeaturizer(ComponentFeaturizer):

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

    def extract_features(self, state: Tuple) -> torch.Tensor:

        agent_positions = state[self.env.state_fields[StateFields.AGENT_POSITIONS]]
        imposter_x, imposter_y = agent_positions[0]

        grid = torch.zeros((self.env.n_cols+2, self.env.n_rows+2))
        grid[1:-1,1:-1] = torch.tensor(self.env.grid)

        surrounding = grid[imposter_x:imposter_x+3, imposter_y:imposter_y+3].clone().view(-1)

        return surrounding

    @property
    def shape(self) -> torch.tensor:
        return torch.tensor([9], dtype=torch.int)


class ImposterVSCrewRoomLocaionFeaturizer(ComponentFeaturizer):

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

    def extract_features(self, state: Tuple) -> torch.Tensor:

        # first 4 bits for imposter room, 4 last for all crew
        room_features = torch.zeros(8)

        alive_agents = state[self.env.state_fields[StateFields.ALIVE_AGENTS]]

        for agent_idx, pos in enumerate(state[self.env.state_fields[StateFields.AGENT_POSITIONS]]):

            if not alive_agents[agent_idx]:
                continue
            
            rooms = torch.zeros(4)
            for room_idx, room_mask in enumerate(ROOM_MASKS):
                rooms[room_idx] = room_mask[pos[0], pos[1]]

            if agent_idx == 0:
                room_features[:4] += rooms
            else:
                room_features[4:] += rooms
                
        return room_features


    @property
    def shape(self) -> torch.tensor:
        return torch.tensor([8], dtype=torch.int)

    


class ImposterScentFeaturizer(ComponentFeaturizer):

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

    def extract_features(self, state: Tuple) -> torch.Tensor:

        agent_positions = state[self.env.state_fields[StateFields.AGENT_POSITIONS]]

        # scent for each of the 4 directions
        # left, right, top, bottom
        imposter_scent = torch.zeros(4)

        alive_agents = state[self.env.state_fields[StateFields.ALIVE_AGENTS]]

        # NOTE: ONLY IF IMPOSTER IN 0th IDX
        imposter_x, imposter_y = agent_positions[0]

        for agent_idx, pos in enumerate(agent_positions):
            if agent_idx != 0 and alive_agents[agent_idx]:
                dx = pos[0] - imposter_x
                dy = pos[1] - imposter_y

                x_scent = (self.env.n_cols - dx) / self.env.n_cols
                y_scent = (self.env.n_rows - dy) / self.env.n_rows

                if x_scent > 0:
                    imposter_scent[0] += x_scent
                else:
                    imposter_scent[1] += x_scent

                if y_scent > 0:
                    imposter_scent[2] += y_scent
                else:
                    imposter_scent[3] += y_scent

        return imposter_scent.view(-1)

    @property
    def shape(self) -> torch.tensor:

        return torch.tensor(4, dtype=torch.int)



class CoordinateAgentPositionsFeaturizer(ComponentFeaturizer):

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

    def extract_features(self, state: Tuple) -> torch.Tensor:

        agent_positions = state[self.env.state_fields[StateFields.AGENT_POSITIONS]]
        
        coordinates = torch.zeros(self.env.n_agents * 2)

        for agent_idx, pos in enumerate(agent_positions):
            coordinates[2*agent_idx] = pos[0]
            coordinates[2*agent_idx+1] = pos[1]

        return coordinates

    @property
    def shape(self) -> torch.tensor:
        return torch.tensor([self.env.n_agents * 2], dtype=torch.int)


class AliveCrewFeaturizer(ComponentFeaturizer):

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

    def extract_features(self, state: Tuple) -> torch.Tensor:

        alive_agents = state[self.env.state_fields[StateFields.ALIVE_AGENTS]]
        
        alive_features = torch.zeros(self.env.n_agents-1)

        for agent_idx, alive in enumerate(alive_agents):
            if agent_idx > 0 and alive:
                alive_features[agent_idx-1] = 1

        return alive_features

    @property
    def shape(self) -> torch.tensor:
        return torch.tensor([self.env.n_agents-1], dtype=torch.int)


class L1CrewFeaturizer(ComponentFeaturizer):

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

    def extract_features(self, state: Tuple) -> torch.Tensor:
        
        
        alive_agents = state[self.env.state_fields[StateFields.ALIVE_AGENTS]]
        agent_positions = state[self.env.state_fields[StateFields.AGENT_POSITIONS]]

        # NOTE: ONLY IF IMPOSTER IN 0th IDX
        imposter_x, imposter_y = agent_positions[0]

        l1_crew = torch.ones(self.env.n_crew) * -1

        for agent_idx, pos in enumerate(agent_positions):
            if agent_idx > 0 and alive_agents[agent_idx]:
                l1_crew[agent_idx-1] = abs(imposter_x - pos[0]) + abs(imposter_y - pos[1])

        return l1_crew

    @property
    def shape(self) -> torch.tensor:
        return torch.tensor([self.env.n_crew], dtype=torch.int)


class ClosestAliveCrewFeaturizer(ComponentFeaturizer):

    def __init__(self, env: FourRoomEnv):
        super().__init__(env)

    def extract_features(self, state: Tuple) -> torch.Tensor:
        
        
        alive_agents = state[self.env.state_fields[StateFields.ALIVE_AGENTS]]
        agent_positions = state[self.env.state_fields[StateFields.AGENT_POSITIONS]]

        # NOTE: ONLY IF IMPOSTER IN 0th IDX
        imposter_x, imposter_y = agent_positions[0]

        l1_crew = torch.ones(self.env.n_crew) * self.env.n_cols + self.env.n_rows
        closest = torch.zeros(self.env.n_crew)

        for agent_idx, pos in enumerate(agent_positions):
            if agent_idx > 0 and alive_agents[agent_idx]:
                l1_crew[agent_idx-1] = abs(imposter_x - pos[0]) + abs(imposter_y - pos[1])

        closest[torch.argmin(l1_crew)] = 1

        return closest

    @property
    def shape(self) -> torch.tensor:
        return torch.tensor([self.env.n_crew], dtype=torch.int)

