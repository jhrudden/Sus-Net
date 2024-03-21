from enum import Enum
from collections import defaultdict
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gymnasium import Env, spaces

"""
TODO:
- Action ordering choice:
    - Current implementation, but always random order each timestep
    - Step takes agent by agent, and returns next state + reward, where next state gets passed to next agent
    - All actions are taken at the same time, so always reference initial state for each agent instead of updated state based on order
- should self.time_step_reward depend on the agents job? (1 for imposters, -1 for crew members)

- Jobs require multiple time steps to complete
   - any action besides Fix or Sabotage will not have an immediate effect (but resets the timer)
   - Fixing a job will take 3 time steps to complete (BASE)
   - Sabotaging or fixing a job doesn't do anything unless job is in a good state

- Instead of voting, add tag count to state
    - Every x timesteps, tag counts are reset
    - An agent can tag 1 other person, and the tagged person get's their count increased by 1
    - At end of x timesteps, if tag count of an agent > half the # agents, the agent dies
    - Tag count is reset when an agent dies or when the timer resets


- Who am I? Which agent am I? 
    - Need to figure out who the agent is in the state
    - Q looks like Q(s_{agent}, s_{whole}, a_{agent}) 

"""


class Action(Enum):

    # Move Actions
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

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
        return (position[0], position[1] + 1)
    elif action == Action.DOWN:
        return (position[0], position[1] - 1)
    elif action == Action.LEFT:
        return (position[0] - 1, position[1])
    elif action == Action.RIGHT:
        return (position[0] + 1, position[1])
    else:
        return position


def reverse_action(action):
    if action == Action.UP:
        return Action.DOWN
    elif action == Action.DOWN:
        return Action.UP
    elif action == Action.LEFT:
        return Action.RIGHT
    elif action == Action.RIGHT:
        return Action.LEFT
    elif action == Action.STAY:
        return Action.STAY


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
    ):
        super().__init__()

        self._validate_init_args(
            n_imposters, n_crew, n_jobs
        )

        if random_state is not None:
            np.random.seed(random_state)

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


        # NOTE: This is the 2D grid of 4 rooms that we saw in the previous examples however, no goal and start states are defined
        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        self.walls = [
            (0, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 9),
            (5, 10),
            (6, 4),
            (7, 4),
            (9, 4),
            (10, 4),
        ]

        self.n_rows = 11
        self.n_cols = 11

        self.valid_positions = [
            (x, y)
            for x in range(11)
            for y in range(11)
            if (x, y) not in self.walls
        ]

        self.action_space = spaces.Discrete(len(Action))

        self.observation_space = spaces.Tuple(
            (
                spaces.MultiDiscrete([self.n_rows, self.n_cols] * self.n_agents),  # Agent positions
                spaces.MultiDiscrete([self.n_rows, self.n_cols] * self.n_jobs),  # Job positions
                spaces.MultiBinary(self.n_jobs),  # Completed jobs
                spaces.MultiBinary(self.n_agents),  # Alive agents
            )
        )

        self.reset()
    
    def _validate_init_args(self, n_imposters, n_crew, n_jobs):
        assert n_imposters > 0, f"Must have at least one imposter. Got {n_imposters}."
        assert n_crew > 0, f"Must have at least one crew member. Got {n_crew}."
        assert n_jobs > 0, f"Must have at least one job. Got {n_jobs}."
        assert n_imposters < n_crew, f"Must be more crew members than imposters. Got {n_imposters} imposters and {n_crew} crew members."


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
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

        # Select agent and job positions randomly from the valid positions

        # random agent positions
        agent_cells = np.random.choice(len(self.valid_positions), size=self.n_agents, replace=True)
        self.agent_positions = [self.valid_positions[cell] for cell in agent_cells]

        # random job positions
        # NOTE: any two jobs can't be at the same position
        job_cells = np.random.choice(len(self.valid_positions), size=self.n_jobs, replace=False)
        self.job_positions = [self.valid_positions[cell] for cell in job_cells]


        self.alive_agents = np.ones(self.n_agents)
        self.completed_jobs = np.zeros(self.n_jobs)

        return (
            (
                self.agent_positions,
                self.job_positions,
                self.completed_jobs,
                self.alive_agents,
            ),
            {},
        )

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
            - A tuple of (agent_posions, jtiob_positions, completed_jobs, alive_agents) reflecting the new state of the environment.
            - agent_rewards (numpy.ndarray): An array of rewards received by each agent during this step.
            - done (bool): A flag indicating whether the game has reached a terminal state.
            - truncated (bool): A flag indicating whether the episode was truncated (not applicable in this context, but included for API consistency).
            - info (dict): An empty dictionary that could be used for debugging or logging additional information in the future.

        Side Effects:
        - Updates `self.agent_positions` based on the actions that involve movement.
        - Modifies `self.completed_jobs` and `self.alive_agents` based on actions that involve fixing, sabotaging, or killing.
        - Alters `self.agent_rewards` to reflect the rewards accumulated by each agent during this step.
        """

        truncated = False
        done = False

        # initialize the agent reward array before computing all agent rewards
        self.agent_rewards = np.ones(self.n_agents) * self.time_step_reward
        team_reward = 0

        # getting the order in which agent actions will be performed
        agent_action_order = list(range(self.n_agents))
        if self.is_action_order_random:
            np.random.shuffle(agent_action_order)

        # perform action for each agent
        for agent_idx in agent_action_order:
            print(f"Agent {agent_idx} is performing action {agent_actions[agent_idx]}")
            self._agent_step(agent_idx=agent_idx, agent_action=agent_actions[agent_idx])

        # TODO: Should we shuffle imposters and crew members and store their locations in the env implicitly?
        # check for no imposters (crew members won)
        if np.sum(self.alive_agents[: self.n_imposters]) == 0:
            done = True
            team_reward = self.game_end_reward

        # TODO: Same as above note
        # check more or = imposters than crew (imposters won)
        if np.sum(self.alive_agents[: self.n_imposters]) >= np.sum(
            self.alive_agents[self.n_imposters :]
        ):
            team_reward = -1 * self.game_end_reward
            done = True

        return (
            (
                self.agent_positions,
                self.job_positions,
                self.completed_jobs,
                self.alive_agents,
            ),
            (self.agent_rewards, team_reward),
            done,
            truncated,
            {},
        )

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

        print(f"{agent_idx}: {agent_action}")

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
            agents_at_pos = self._get_agents_at_pos(pos)
            agents_at_pos.remove(agent_idx)

            if agents_at_pos:
                # choosing random victim
                victim_idx = np.random.choice(agents_at_pos)

                # updating alive list
                self.alive_agents[victim_idx] = 0

                # setting rewards
                self.agent_rewards[victim_idx] = self.kill_reward
                self.agent_rewards[agent_idx] = self.kill_reward

        # agent attempts to fix
        elif agent_action == Action.FIX:
            job_idx = self._get_job_idx_at_pos(pos)
            if job_idx is not None and not self.completed_jobs[job_idx]:
                self.completed_jobs[job_idx] = 1
            self.agent_rewards[agent_idx] = self.job_reward

        # agent attempts to sabotage
        elif agent_action == Action.SABOTAGE:
            job_idx = self._get_job_idx_at_pos(pos)
            if job_idx is not None and self.completed_jobs[job_idx]:
                self.completed_jobs[job_idx] = 0
            self.agent_rewards[agent_idx] = -1 * self.job_reward

    def _get_agents_at_pos(self, pos) -> List[int]:
        # only returns living agents
        return [
            i
            for i, val in enumerate(self.agent_positions)
            if val == pos and self.alive_agents[i]
        ]

    def _get_job_idx_at_pos(self, pos) -> int | None:
        for job_idx, job_pos in enumerate(self.job_positions):
            if job_pos == pos:
                return job_idx
        return None
    
    def _is_valid_position(self, pos):
        return pos not in self.walls and 0 <= pos[0] < self.n_cols and 0 <= pos[1] < self.n_rows
    
    def render(self):
        """
        Use matplotlib to render the current state of the environment.

        This method draws the current state of the environment using a 2D grid of cells.
        The walls are drawn in black, the agents are drawn in blue, the jobs are drawn in green,
        and the completed jobs are drawn in red.
        """

        # this should be a 2D grid of cells with the following colors:
        # - black for walls
        # - blue for crew
        # - red for imposters
        # - green for yellow
        # - grey for completed jobs
        # - white for empty cells
        grid = np.zeros((self.n_rows, self.n_cols, 3))

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                grid[i, j] = [1, 1, 1]

        # draw walls
        for wall in self.walls:
            grid[wall[1], wall[0]] = [0, 0, 0]
        
        # draw jobs
        for job in self.job_positions:
            grid[job[1], job[0]] = [0, 1, 0]
        
        # draw completed jobs
        for job_idx, completed in enumerate(self.completed_jobs):
            job = self.job_positions[job_idx]
            grid[job[1], job[0]] = [1, 0, 0] if completed else [0, 1, 0]
            
        # draw agents
        for agent_idx, pos in enumerate(self.agent_positions):
            if self.alive_agents[agent_idx]:
                grid[pos[1], pos[0]] = [0, 0, 1]

        # flip the grid to match the coordinate system
        grid = np.flip(grid, 0)

        plt.imshow(grid)
        plt.show()
