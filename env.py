from enum import Enum
from collections import defaultdict
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gymnasium import Env, spaces


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
        return (position[0], position[1] - 1)
    elif action == Action.DOWN:
        return (position[0], position[1] + 1)
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


class MazeEnv(Env):
    def __init__(
        self,
        n_rooms: int,
        n_agents: int,
        # n_crew: int,
        # n_imposters: int,
        n_jobs: int,
        is_action_order_random=False,
        random_state: Optional[int] = None,
        kill_reward: int = 3,
        fix_reward: int = 1,
        sabotage_reward: int = 1,
        time_step_reward: int = 0,
    ):

        super().__init__()
        assert n_rooms > 0, "Number of rooms must be greater than 0"
        if random_state is not None:
            np.random.seed(random_state)

        self.action_space = spaces.Discrete(len(Action))

        self.observation_space = spaces.Tuple(
            (
                spaces.MultiDiscrete([n_rooms] * n_agents),  # Agent positions
                spaces.MultiDiscrete([n_rooms] * n_jobs),  # Job positions
                spaces.MultiBinary(n_jobs),  # Completed jobs
                spaces.MultiBinary(n_agents),  # Alive agents
            )
        )
        self.is_action_order_random = is_action_order_random
        self.n_rooms = n_rooms
        self.n_agents = n_agents
        self.kill_reward = kill_reward
        self.fix_reward = fix_reward
        self.sabotage_reward = sabotage_reward
        self.time_step_reward = time_step_reward

        self.job_positions = None
        self.agent_positions = None
        self.alive_agents = None
        self.completed_jobs = None

        self._generate_rooms()

        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state

        Returns:
        Tuple: A tuple containing the initial state and an empty dictionary
        """

        rooms_list = np.arange(self.n_rooms)
        agent_pos_indices = np.random.choice(
            rooms_list, size=self.n_agents, replace=True
        )
        self.agent_positions = [self.rooms[i] for i in agent_pos_indices]
        job_pos_indices = np.random.choice(rooms_list, size=self.n_agents, replace=True)
        self.alive_agents = np.ones(self.n_agents)
        self.job_positions = [self.rooms[i] for i in job_pos_indices]
        self.completed_jobs = np.zeros(self.n_agents)
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

        # getting the order in which agent actions will be performed
        agent_action_order = list(range(self.n_agents))
        if self.is_action_order_random:
            np.random.shuffle(agent_action_order)

        # perform action for each agent
        for agent_idx in agent_action_order:
            self._agent_step(agent_idx=agent_idx, agent_action=agent_actions[agent_idx])

        # check for terminal states
        if np.sum(self.alive_agents) == 0:  # no living remain
            done = True

        # TODO: any other terminal states?
        # NEED TO ADD internal list of imposters and crew members
        # to determine when game ends

        # TODO: update team rewards? Maybe the function should return both, individual
        # and team rewards and then our algorithms will handle distributing to
        # crew members and imposters?

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
            if agent_action in self.room_map[pos]:
                self.agent_positions[agent_idx] = self.room_map[pos][agent_action]

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
                self.agent_rewards[victim_idx] = self.kill_reward * -1
                self.agent_rewards[agent_idx] = self.kill_reward

        # agent attempts to fix
        elif agent_action == Action.FIX:
            job_idx = self._get_job_idx_at_pos(pos)
            if job_idx is not None and not self.completed_jobs[job_idx]:
                self.completed_jobs[job_idx] = 1
            self.agent_rewards[agent_idx] = self.fix_reward

        # agent attempts to sabotage
        elif agent_action == Action.SABOTAGE:
            job_idx = self._get_job_idx_at_pos(pos)
            if job_idx is not None and self.completed_jobs[job_idx]:
                self.completed_jobs[job_idx] = 0
            self.agent_rewards[agent_idx] = self.sabotage_reward

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

    def _generate_rooms(self):
        rooms = [(0, 0)]
        room_map = defaultdict(dict)
        while len(rooms) < self.n_rooms:
            random_action = np.random.choice(list(Action))
            random_room = rooms[np.random.choice(len(rooms))]
            while random_action in room_map[random_room]:
                random_room = rooms[np.random.choice(len(rooms))]
                random_action = np.random.choice(list(Action))
            new_room = move(random_action, random_room)
            if new_room not in rooms:
                rooms.append(new_room)
            room_map[random_room][random_action] = new_room
            room_map[new_room][reverse_action(random_action)] = random_room

        self.rooms = rooms  # List[Tuple[int , int ]]
        self.room_map = room_map  # what actions you can take from the room

        print(room_map)

    def render(self):
        fig, ax = plt.subplots()
        room_size = 0.8  # Size of the room, adjust as needed for spacing
        door_width = 0.2  # Width of the doors

        # Set limits based on the room positions
        all_x = [x for x, y in self.rooms]
        all_y = [y for x, y in self.rooms]
        ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
        ax.set_ylim(min(all_y) - 1, max(all_y) + 1)

        # Draw each room and its connections
        for room in self.rooms:
            room_center = (room[0], room[1])
            # Draw room as a square
            ax.add_patch(
                patches.Rectangle(
                    (room_center[0] - room_size / 2, room_center[1] - room_size / 2),
                    room_size,
                    room_size,
                    fill=None,
                    edgecolor="black",
                )
            )

            # For each connected room, draw a doorway
            if room in self.room_map:
                for action, next_room in self.room_map[room].items():
                    if action == Action.UP:
                        # Door on the bottom edge (visually appears at the bottom because we're not changing the x-coordinate)

                        ax.add_patch(
                            patches.Rectangle(
                                (
                                    room_center[0] - door_width / 2,
                                    room_center[1] - room_size / 2 - door_width,
                                ),
                                door_width,
                                door_width,
                                color="black",
                            )
                        )
                    elif action == Action.DOWN:
                        # Door on the top edge (visually appears at the top because we're not changing the x-coordinate)
                        ax.add_patch(
                            patches.Rectangle(
                                (
                                    room_center[0] - door_width / 2,
                                    room_center[1] + room_size / 2,
                                ),
                                door_width,
                                door_width,
                                color="black",
                            )
                        )
                    elif action == Action.LEFT:
                        # Door on the right edge
                        ax.add_patch(
                            patches.Rectangle(
                                (
                                    room_center[0] - room_size / 2 - door_width,
                                    room_center[1] - door_width / 2,
                                ),
                                door_width,
                                door_width,
                                color="black",
                            )
                        )
                    elif action == Action.RIGHT:
                        # Door on the left edge
                        ax.add_patch(
                            patches.Rectangle(
                                (
                                    room_center[0] + room_size / 2,
                                    room_center[1] - door_width / 2,
                                ),
                                door_width,
                                door_width,
                                color="black",
                            )
                        )

        # Draw agents and jobs
        for j, job in enumerate(self.job_positions):
            if self.completed_jobs[j] == 0:
                ax.add_patch(patches.Circle(job, 0.3, color="yellow"))
            else:
                ax.add_patch(patches.Circle(job, 0.3, color="red"))

        for i, agent in enumerate(self.agent_positions):
            image_pos = (agent[0] - 0.5, agent[1] - 0.5)
            if self.alive_agents[i] == 0:
                ax.add_patch(patches.Circle(agent, 0.3, color="black", alpha=0.3))
            else:
                ax.add_patch(
                    patches.Rectangle(image_pos, 1, 1, color="gray", alpha=0.3)
                )

        ax.set_aspect("equal")
        plt.gca().invert_yaxis()  # Invert y-axis to match the coordinate system used in your environment
        plt.axis("off")  # Optionally hide the axis
        plt.title(
            f"Maze with {self.n_rooms} rooms (random seed: {np.random.get_state()[1][0]})"
        )
        plt.show()
