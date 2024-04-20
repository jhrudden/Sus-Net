from enum import Enum
import json
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from gymnasium import Env, spaces
import torch
import logging
from enum import StrEnum, auto

"""
TODO:
- Jobs require multiple time steps to complete
   - any action besides Fix or Sabotage will not have an immediate effect (but resets the timer)
   - Fixing a job will take 3 time steps to complete (BASE)
   - Sabotaging or fixing a job doesn't do anything unless job is in a good state

- How do we handle time step rewards?

- Crew win when finish tasks!

- Should jobs and killing cause team rewards?
"""

class SusMetrics(StrEnum):
    IMP_KILLED_CREW = auto()
    IMP_VOTED_OUT = auto()
    CREW_VOTED_OUT = auto()
    SABOTAGED_JOBS = auto()
    COMPLETED_JOBS = auto()
    TOTAL_STALEMATES = auto()
    TOTAL_TIME_STEPS = auto()
    IMPOSTER_WON = auto()
    CREW_WON = auto()

    @classmethod
    def can_increment(cls, metric: str):
        return metric in [
            SusMetrics.IMP_KILLED_CREW,
            SusMetrics.IMP_VOTED_OUT,
            SusMetrics.CREW_VOTED_OUT,
            SusMetrics.SABOTAGED_JOBS,
            SusMetrics.COMPLETED_JOBS,
            SusMetrics.TOTAL_STALEMATES,
            SusMetrics.TOTAL_TIME_STEPS,
        ]

class MetricHandler:
    def __init__(self):
        self.metrics = {metric: 0 for metric in SusMetrics}
    
    def increment(self, event, amount=1) -> None:
        """
        Increment the metric by the specified amount

        Args:
        - event (Metrics): The metric to increment
        """
        if SusMetrics.can_increment(event):
            self.metrics[event] += amount
        else:
            raise ValueError(f"Invalid metric: {event}")
        
    def update(self, event: SusMetrics, value: Any) -> None:
        if event not in SusMetrics:
            raise ValueError(f"Invalid metric: {event}")
        self.metrics[event] = value

    def reset(self) -> None:
        for key in self.metrics.keys():
            self.metrics[key] = 0

    def get_metrics(self) -> Dict[SusMetrics, int]:
        return {metric: self.metrics[metric] for metric in SusMetrics}
    
    def __repr__(self):
        return json.dumps(self.metrics, indent=4)


# TODO: move this logging file etc.
def configure_logging(name="SUSSY_ENV", debug=False):
    logger = logging.getLogger(name)
    logger.propagate = False

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

    logger.setLevel(logging.DEBUG if debug else logging.WARNING)

    return logger


class AgentTypes(Enum):
    AGENT = 0
    CREW_MEMBER = 1
    IMPOSTER = 2


class StateFields(Enum):
    AGENT_POSITIONS = 0
    JOB_POSITIONS = 1
    JOB_STATUS = 2
    ALIVE_AGENTS = 3
    USED_TAGS = 4
    TAG_COUNTS = 5
    TAG_RESET_COUNT = 6


class Action(Enum):

    # Move Actions
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    # Job Actions
    KILL = 5
    FIX = 6
    SABOTAGE = 7

    @property
    def is_move_action(self):
        return self in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY)

    @property
    def is_job_action(self):
        return self in (Action.KILL, Action.FIX, Action.SABOTAGE)


def move(action, position):
    if action == Action.UP:
        return np.array([position[0], position[1] + 1])
    elif action == Action.DOWN:
        return np.array([position[0], position[1] - 1])
    elif action == Action.LEFT:
        return np.array([position[0] - 1, position[1]])
    elif action == Action.RIGHT:
        return np.array([position[0] + 1, position[1]])
    else:
        return position


CREW_ACTIONS = [
    Action.STAY,
    Action.UP,
    Action.DOWN,
    Action.RIGHT,
    Action.LEFT,
    Action.FIX,
]

IMPOSTER_ACTIONS = [
    Action.STAY,
    Action.UP,
    Action.DOWN,
    Action.RIGHT,
    Action.LEFT,
    Action.SABOTAGE,
    Action.KILL,
]

class FourRoomEnv(Env):
    def __init__(
        self,
        n_imposters: int,
        n_crew: int,
        n_jobs: int,
        is_action_order_random=True,
        random_state: Optional[int] = None,
        kill_reward: int = -3,
        job_reward=1,
        time_step_reward: int = 0,
        game_end_reward: int = 10,
        debug: bool = False,
    ):
        super().__init__()

        self._validate_init_args(n_imposters, n_crew, n_jobs)

        if random_state is not None:
            np.random.seed(random_state)

        self.logger = configure_logging(debug=debug)

        self.state_fields = {
            field: idx
            for idx, field in enumerate(
                [
                    StateFields.AGENT_POSITIONS,
                    StateFields.JOB_POSITIONS,
                    StateFields.JOB_STATUS,
                    StateFields.ALIVE_AGENTS,
                ]
            )
        }
        self.metrics = MetricHandler()

        self.is_action_order_random = is_action_order_random
        self.n_imposters = n_imposters
        self.n_crew = n_crew
        self.n_agents = n_imposters + n_crew
        self.n_jobs = n_jobs
        self.kill_reward = kill_reward
        self.job_reward = job_reward
        self.time_step_reward = time_step_reward
        self.game_end_reward = game_end_reward

        self.job_positions = None
        self.agent_positions = None
        self.alive_agents = None
        self.completed_jobs = None

        # used to shuffle the order in which get_agent_state builds states
        # if imposters are always first, eventually alg will learn to vote out first players
        self.agent_state_order_list = None
        self.agent_state_order_dict = None

        # NOTE: This is the 2D grid of 4 rooms that we saw in the previous examples however, no goal and start states are defined
        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        self.walls = np.array(
            [
                [0, 4],
                [2, 4],
                [3, 4],
                [4, 4],
                [5, 4],
                [6, 4],
                [8, 4],
                [4, 0],
                [4, 2],
                [4, 3],
                [4, 5],
                [4, 6],
                [4, 8],
            ]
        )

        self.grid = np.ones((9, 9), dtype=bool)
        self.grid[self.walls[:, 0], self.walls[:, 1]] = 0
        self.valid_positions = np.argwhere(self.grid)

        self.imposter_actions = IMPOSTER_ACTIONS
        self.crew_actions = CREW_ACTIONS
        self.n_imposter_actions = len(IMPOSTER_ACTIONS)
        self.n_crew_actions = len(CREW_ACTIONS)

        self.n_rows = 9
        self.n_cols = 9

        self.action_space = spaces.Discrete(len(Action))

        self.observation_space = spaces.Tuple(
            (
                spaces.Box(
                    low=0, high=self.n_rows, shape=(self.n_agents, 2), dtype=int
                ),  # Agent positions
                spaces.Box(
                    low=0, high=self.n_rows, shape=(self.n_jobs, 2), dtype=int
                ),  # Job positions
                spaces.MultiBinary(self.n_jobs),  # Completed jobs
                spaces.MultiBinary(self.n_agents),  # Alive agents
            )
        )

    @property
    def flattened_state_size(self):
        return spaces.flatten_space(self.observation_space).shape[0]

    def flatten_state(self, state):
        return spaces.flatten(self.observation_space, state)

    def unflatten_state(self, state):
        # if tensor, convert to numpy array
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        return spaces.unflatten(self.observation_space, state)

    def _validate_init_args(self, n_imposters, n_crew, n_jobs):
        assert n_imposters > 0, f"Must have at least one imposter. Got {n_imposters}."
        assert n_crew > 0, f"Must have at least one crew member. Got {n_crew}."
        assert n_jobs > 0, f"Must have at least one job. Got {n_jobs}."
        assert (
            n_imposters < n_crew
        ), f"Must be more crew members than imposters. Got {n_imposters} imposters and {n_crew} crew members."

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Tuple, Dict]:
        """
        Reset the environment to the initial state

        Specifically, this method sets:
        - The agent positions to random locations in the environment
        - The job positions to random locations in the environment
        - The completed jobs to all zeros
        - The alive agents to all ones

        Args:
        - seed (int): An optional seed to use for the random number generator.
        Returns:
        Tuple: A tuple containing the initial state and an empty dictionary
        """
        if seed is not None:
            np.random.seed(seed)

        # reset metrics
        self.metrics.reset()

        # randomize imposter positions
        self.imposter_idxs = np.random.choice(
            range(self.n_agents), size=self.n_imposters, replace=False
        )
        self.imposter_mask = np.zeros(self.n_agents, dtype=bool)
        self.imposter_mask[self.imposter_idxs] = True
        self.crew_mask = ~self.imposter_mask
        self.crew_idxs = np.where(self.crew_mask)[0]

        # Select agent and job positions randomly from the valid positions

        # random agent positions
        agent_cells = np.random.choice(
            len(self.valid_positions), size=self.n_agents, replace=True
        )
        self.agent_positions = self.valid_positions[agent_cells]

        # random job positions
        # NOTE: any two jobs can't be at the same position
        job_cells = np.random.choice(
            len(self.valid_positions), size=self.n_jobs, replace=False
        )

        self.job_positions = self.valid_positions[job_cells]

        self.alive_agents = np.ones(self.n_agents, dtype=bool)
        self.completed_jobs = np.zeros(self.n_jobs, dtype=bool)

        # Agent Action Map: keeps tracks of actions available to each agent
        # when agent_step is called, this list is indexed to get the action
        self.agent_action_map = {}
        for agent_idx in range(self.n_agents):
            self.agent_action_map[agent_idx] = self.crew_actions.copy()

        # overwrite actions for imposters
        for imposter_idx in self.imposter_idxs:
            self.agent_action_map[imposter_idx] = self.imposter_actions.copy()

        return (
            (
                self.agent_positions,
                self.job_positions,
                self.completed_jobs,
                self.alive_agents,
            ),
            self.metrics.get_metrics(),
        )

    def sample_actions(self):
        actions = np.zeros(self.n_agents, dtype=int)
        for agent_idx in self.agent_action_map:
            actions[agent_idx] = np.random.choice(len(self.agent_action_map[agent_idx]))
        return actions

    def step(self, agent_actions):
        """
        Executes a step in the environment by applying the actions of all agents and updating the environment's state accordingly.

        This function processes each agent's action in a specified or random order, updates the agents' positions,
        handles interactions such as kills, fixes, and sabotages, and determines whether the game reaches a
        terminal state based on the conditions of alive agents or completed objectives.

        Parameters:
        - agent_actions (list or dict): A collection of actions for each agent to take during this step.
            The actions should be indexed by the agent's index in the list.

        Returns:
        - tuple containing:
            - A tuple of (agent_positions, job_positions, completed_jobs, alive_agents) reflecting the new state of the environment.
            - agent_rewards (numpy.ndarray): An array of rewards received by each agent during this step.
            - done (bool): A flag indicating whether the game has reached a terminal state.
            - truncated (bool): A flag indicating whether the episode was truncated (not applicable in this context, but included for API consistency).
            - info (dict): An empty dictionary that could be used for debugging or logging additional information in the future.

        Side Effects:
        - Updates `self.agent_positions` based on the actions that involve movement.
        - Modifies `self.completed_jobs` and `self.alive_agents` based on actions that involve fixing, sabotaging, or killing.
        - Alters `self.agent_rewards` to reflect the rewards accumulated by each agent during this step.
        """
        assert (
            len(agent_actions) == self.n_agents
        ), f"Expected {self.n_agents} actions, got {len(agent_actions)}"
        assert all(
            action < self.action_space.n for action in agent_actions
        ), f"Invalid action(s) {agent_actions}"

        truncated = False
        done = False
        self.metrics.increment(SusMetrics.TOTAL_TIME_STEPS, 1)

        # initialize the agent reward array before computing all agent rewards
        self.agent_rewards = np.ones(self.n_agents) * self.time_step_reward

        # getting the order in which agent actions will be performed
        agent_action_order = list(range(self.n_agents))
        if self.is_action_order_random:
            np.random.shuffle(agent_action_order)

        # perform action for each agent
        for agent_idx in agent_action_order:

            self._agent_step(
                agent_idx=agent_idx, agent_action=Action(agent_actions[agent_idx])
            )

        team_win, team_reward = self.check_win_condition()
        done = done or team_win

        self.agent_rewards = self._merge_rewards(self.agent_rewards, team_reward)

        return (
            (
                self.agent_positions,
                self.job_positions,
                self.completed_jobs,
                self.alive_agents,
            ),
            self.agent_rewards,
            done,
            truncated,
            self.metrics.get_metrics(),
        )

    def check_win_condition(self):
        """
        Checks if the game has reached a terminal state and returns the reward for each agent.

        This method checks the win conditions of the game to determine if the game has reached a terminal state.
        The game ends when all imposters are killed or when the number of imposters is greater than or equal to the number of crew members.
        The team reward is calculated based on the win condition.

        Returns:
        - tuple containing:
            - A boolean indicating whether the game has reached a terminal state.
            - Team reward
        """

        # NOTE: We might need to know the role of agents for this...
        # check for no imposters (crew members won)
        # TODO: Check if all jobs are completed
        done = False
        reward = 0
        if np.sum(self.alive_agents[self.imposter_mask]) == 0:
            self.logger.debug("CREW won!")
            self.metrics.update(SusMetrics.CREW_WON, 1)
            done = True
            reward = self.game_end_reward

        # check more or = imposters than crew (imposters won)
        elif (
            self.alive_agents.sum()
            - self.alive_agents[self.imposter_mask].sum()  # crew memebrs
            <= self.alive_agents[self.imposter_mask].sum()  # imposters
        ):
            self.logger.debug("IMPOSTERS won!")
            self.metrics.update(SusMetrics.IMPOSTER_WON, 1)
            done = True
            reward = -1 * self.game_end_reward

        if done:
            self.logger.debug(
                f"""
GAME OVER!
    Alive Crew: {np.argwhere(self.alive_agents & ~self.imposter_mask).flatten()}
    Alive Imposters: {np.argwhere(self.alive_agents & self.imposter_mask).flatten()}
    Completed Jobs: {list(map(tuple, self.job_positions[self.completed_jobs]))}
Metrics:
{str(self.metrics)}
            """
            )

        return done, reward

    def _agent_step(self, agent_idx, agent_action) -> None:
        """
        Processes a single step for an agent by executing the specified action within the environment.

        This method handles the movement, killing, fixing, and sabotaging actions of an agent.
        Movement is allowed based on the room map, killing removes another agent at the same position,
        fixing completes a job at the agent's position, and sabotaging undoes a completed job at the agent's position.
        Rewards or penalties are assigned to agents based on their actions.

        Parameters:
        - agent_idx (int): Index of the agent performing the action.
        - agent_action (Action): The action to be performed by the agent.
            This is an instance of an Action enumeration that includes MOVE, KILL, FIX, and SABOTAGE actions.
        """

        if self.alive_agents[agent_idx] == 0:  # agent is dead
            return

        # get agent position
        pos = self.agent_positions[agent_idx]

        # moving the agent position
        if agent_action.is_move_action:
            new_pos = move(agent_action, pos)
            if self._is_valid_position(new_pos):
                self.agent_positions[agent_idx] = new_pos

        # agent attempts kill action
        elif agent_action == Action.KILL:

            # who else is at this position
            agents_at_pos = self._get_agents_at_pos(pos, crew_only=True)

            if agents_at_pos:
                # choosing random victim
                victim_idx = np.random.choice(agents_at_pos)
                assert (
                    victim_idx not in self.imposter_idxs
                ), "Imposter cannot be killed. Only voted out!"

                self.logger.debug(
                    f"""
                Agent {victim_idx} ({self.agent_positions[victim_idx]}) got killed by {agent_idx} ({self.agent_positions[agent_idx]})!!!
                """
                )

                self.metrics.increment(SusMetrics.IMP_KILLED_CREW, 1)

                # updating alive list
                self.alive_agents[victim_idx] = 0

                # setting rewards
                self.agent_rewards[victim_idx] = self.kill_reward
                self.agent_rewards[agent_idx] = self.kill_reward

        # agent attempts to fix
        elif agent_action == Action.FIX:
            job_idx = self._get_job_at_pos(pos)
            if job_idx is not None and not self.completed_jobs[job_idx]:
                self.completed_jobs[job_idx] = 1
                self.metrics.increment(SusMetrics.COMPLETED_JOBS, 1)
                self.agent_rewards[agent_idx] = self.job_reward
                self.logger.debug(f"Agent {agent_idx} fixed a job at {pos}!")

        # agent attempts to sabotage
        elif agent_action == Action.SABOTAGE:
            job_idx = self._get_job_at_pos(pos)
            if job_idx is not None and self.completed_jobs[job_idx]:
                self.completed_jobs[job_idx] = 0
                self.metrics.increment(SusMetrics.SABOTAGED_JOBS, 1)
                self.agent_rewards[agent_idx] = -1 * self.job_reward
                self.logger.debug(f"Imposter {agent_idx} sabotaged a job at {pos}!")

    def _get_agents_at_pos(self, pos, crew_only=True) -> List[int]:
        if crew_only:
            alive = np.argwhere(self.alive_agents & ~self.imposter_mask).flatten()
        else:
            alive = np.argwhere(self.alive_agents).flatten()

        agents_at_pos = alive[np.all(self.agent_positions[alive] == pos, axis=1)]
        return agents_at_pos.tolist()

    def _get_job_at_pos(self, pos) -> int | None:
        idx = np.argwhere(np.all(self.job_positions == pos, axis=1))
        return idx[0][0] if idx.size > 0 else None

    def _is_valid_position(self, pos):
        assert self.n_cols == self.n_rows  # this function assumes a square grid
        valid = np.all(pos >= 0) and np.all(pos < self.n_cols)
        return valid and self.grid[pos[1], pos[0]]

    def _merge_rewards(self, agent_rewards, team_reward):
        """
        Merges the rewards for each agent with the team reward.
        """
        agent_rewards += team_reward
        # negate imposters rewards
        agent_rewards[: self.n_imposters] *= -1
        return agent_rewards


class FourRoomEnvWithTagging(FourRoomEnv):
    def __init__(
        self, *args, tag_reset_interval: int = 50, vote_reward: int = 3, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.state_fields = {
            field: idx
            for idx, field in enumerate(
                [
                    StateFields.AGENT_POSITIONS,
                    StateFields.JOB_POSITIONS,
                    StateFields.JOB_STATUS,
                    StateFields.ALIVE_AGENTS,
                    StateFields.USED_TAGS,
                    StateFields.TAG_COUNTS,
                    StateFields.TAG_RESET_COUNT,
                ]
            )
        }
        self.tag_counts = np.zeros(self.n_agents, dtype=int)
        self.used_tag_actions = np.zeros(self.n_agents, dtype=bool)
        self.tag_reset_timer = 0
        self.tag_reset_interval = tag_reset_interval
        self.vote_reward = vote_reward

        self.n_imposter_actions = self.n_imposter_actions + self.n_agents - 1
        self.n_crew_actions = self.n_crew_actions + self.n_agents - 1

        self.action_space = spaces.Discrete(
            len(Action) + self.n_agents
        )  # Add tagging action (1 for each agent)

        self.observation_space = spaces.Tuple(
            (
                spaces.Box(
                    low=0, high=self.n_rows, shape=(self.n_agents, 2), dtype=int
                ),  # Agent positions
                spaces.Box(
                    low=0, high=self.n_rows, shape=(self.n_jobs, 2), dtype=int
                ),  # Job positions
                spaces.MultiBinary(self.n_jobs),  # Completed jobs
                spaces.MultiBinary(self.n_agents),  # Alive agents
                spaces.MultiBinary(self.n_agents),  # Who has used their tag
                spaces.Box(
                    low=0, high=self.n_agents, shape=(self.n_agents,), dtype=int
                ),  # Tag counts
                spaces.Box(
                    low=1, high=self.tag_reset_interval, shape=(1,), dtype=int
                ),  # Time left for tag reset
            )
        )

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Tuple, Dict]:
        state, _ = super().reset(seed, **kwargs)
        self.tag_counts = np.zeros(self.n_agents, dtype=int)
        self.used_tag_actions = np.zeros(self.n_agents, dtype=bool)
        self.tag_reset_timer = 0

        # updating agent_action_map to include tagging actions
        for agent_idx in range(self.n_agents):
            tag_actions = np.arange(self.n_agents)[
                np.arange(self.n_agents) != agent_idx
            ]
            self.agent_action_map[agent_idx] = np.hstack(
                [self.agent_action_map[agent_idx], tag_actions]
            )

        self.logger.debug(
            f"""
New Game Started!
-----------------
    Agent Positions: {list(map(tuple, self.agent_positions))}
    Imposters: {np.argwhere(self.imposter_mask).flatten()}
    Crew Members: {np.argwhere(~self.imposter_mask).flatten()}
    Alive Agents: {self.alive_agents}
    Job Positions: {list(map(tuple, self.job_positions))}
    Completed Jobs: {self.completed_jobs}
    Tag Counts: {self.tag_counts}
    Used Tag Actions: {self.used_tag_actions}
    Time Left for Tag Reset: {self.tag_reset_interval - self.tag_reset_timer}
-----------------
        """
        )

        state = (
            *state,
            self.used_tag_actions,
            self.tag_counts,
            self.tag_reset_interval - self.tag_reset_timer,
        )

        return state, {}

    def _agent_tag(self, agent_idx, agent_tagged):
        """Can only tag someone if your tag is unused and if the tagged agent is alive."""
        if (
            self.used_tag_actions[agent_idx] == 0
            and self.alive_agents[agent_tagged] > 0
        ):
            self.tag_counts[agent_tagged] += 1
            self.used_tag_actions[agent_idx] = 1

            self.logger.debug(
                f"""Agent {agent_idx} ({self.agent_positions[agent_idx]}) tagged Agent {agent_tagged} ({self.agent_positions[agent_tagged]})! {agent_tagged}'s new tag count: {self.tag_counts[agent_tagged]}"""
            )
        else:
            self.logger.debug(
                f"""Agent {agent_idx} tried to tag Agent {agent_tagged} but failed!"""
            )

    def step(self, agent_actions):
        """
        Executes a step in the environment by applying the actions of all agents and updating the environment's state accordingly.

        This function processes each agent's action in a specified or random order, updates the agents' positions,
        handles interactions such as kills, fixes, tagging, and sabotages, and determines whether the game reaches a
        terminal state based on the conditions of alive agents or completed objectives.

        Parameters:
        - agent_actions (list or dict): A collection of actions for each agent to take during this step.
            The actions should be indexed by the agent's index in the list.

        Returns:
        - tuple containing:
            - A tuple of (agent_position, job_positions, completed_jobs, alive_agents, used_tag_actions, agent_tag_count, time_till_vote_reset) reflecting the new state of the environment.
            - agent_rewards (numpy.ndarray): An array of rewards received by each agent during this step.
            - done (bool): A flag indicating whether the game has reached a terminal state.
            - truncated (bool): A flag indicating whether the episode was truncated (not applicable in this context, but included for API consistency).
            - info (dict): An empty dictionary that could be used for debugging or logging additional information in the future.

        Side Effects:
        - Updates `self.agent_positions` based on the actions that involve movement. Also updates `self.tag_reset_timer` at the start of the step.
        - Modifies `self.completed_jobs`, `self.tag_counts`, `self.used_tag_action` and `self.alive_agents` based on actions that involve fixing, sabotaging, tagging or killing.
        - Alters `self.agent_rewards` to reflect the rewards accumulated by each agent during this step.
        """
        assert (
            len(agent_actions) == self.n_agents
        ), f"Expected {self.n_agents} actions, got {len(agent_actions)}"
        assert all(
            action < self.action_space.n for action in agent_actions
        ), f"Invalid action(s) {agent_actions}"

        self.metrics.increment(SusMetrics.TOTAL_TIME_STEPS, 1)

        # kill agents who have been tagged too many times (tag count > half the number of agents)

        truncated = False
        done = False
        reset_tag_counts = False
        team_reward = 0

        # initialize the agent reward array before computing all agent rewards
        self.agent_rewards = np.ones(self.n_agents) * self.time_step_reward

        # getting the order in which agent actions will be performed
        agent_action_order = list(range(self.n_agents))
        if self.is_action_order_random:
            np.random.shuffle(agent_action_order)

        # perform action for each agent
        for agent_idx in agent_action_order:

            agent_action = self.agent_action_map[agent_idx][agent_actions[agent_idx]]

            if isinstance(agent_action, int):  # this is a tag action
                self._agent_tag(agent_idx=agent_idx, agent_tagged=agent_action)

            else:
                self._agent_step(agent_idx=agent_idx, agent_action=agent_action)

        self.tag_counts *= self.alive_agents  # reset tag counts for dead agents

        self.tag_reset_timer += 1

        if self.tag_reset_timer >= self.tag_reset_interval:
            highest_vote_idx = np.argmax(self.tag_counts)
            highest_vote = self.tag_counts[highest_vote_idx]

            quorum = (self.alive_agents.sum() + 1) // 2

            if highest_vote >= quorum:
                self.alive_agents[highest_vote_idx] = 0 # kick out the agent with the highest vote
                is_imposter = self.imposter_mask[highest_vote_idx]
                # Reward Crew if Imposter is voted out, else reward Imposters (team reward penalizes the team that lost a member)
                team_reward += self.vote_reward * (-1 if is_imposter else 1)

                if is_imposter:
                    self.metrics.increment(SusMetrics.IMP_VOTED_OUT, 1)
                else:
                    self.metrics.increment(SusMetrics.CREW_VOTED_OUT, 1)

                self.logger.debug(
                    f"""Agent {highest_vote_idx} got voted OUT! Tag Count / Alive Agents: {highest_vote} / {self.alive_agents.sum()}"""
                )

            self._reset_tagging_state()

        team_win, win_team_reward = self.check_win_condition()
        team_reward += win_team_reward
        done = done or team_win

        self.agent_rewards = self._merge_rewards(self.agent_rewards, team_reward)

        return (
            (
                self.agent_positions,
                self.job_positions,
                self.completed_jobs,
                self.alive_agents,
                self.used_tag_actions,  # Who has used their tag
                self.tag_counts,  # Tag counts
                self.tag_reset_interval
                - self.tag_reset_timer,  # Time left for tag reset
            ),
            self.agent_rewards,
            done,
            truncated,
            self.metrics.get_metrics(),
        )

    def _reset_tagging_state(self):
        self.tag_counts = np.zeros(self.n_agents, dtype=int)
        self.used_tag_actions = np.zeros(self.n_agents, dtype=bool)
        self.tag_reset_timer = 0
        self.logger.debug("Tagging state reset!")
