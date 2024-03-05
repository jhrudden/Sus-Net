from enum import Enum
from collections import defaultdict
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from gymnasium import Env, spaces


class MoveAction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def move(action, position):
    if action == MoveAction.UP:
        return (position[0], position[1] - 1)
    elif action == MoveAction.DOWN:
        return (position[0], position[1] + 1)
    elif action == MoveAction.LEFT:
        return (position[0] - 1, position[1])
    elif action == MoveAction.RIGHT:
        return (position[0] + 1, position[1])


def reverse_action(action):
    if action == MoveAction.UP:
        return MoveAction.DOWN
    elif action == MoveAction.DOWN:
        return MoveAction.UP
    elif action == MoveAction.LEFT:
        return MoveAction.RIGHT
    elif action == MoveAction.RIGHT:
        return MoveAction.LEFT


class MazeEnv(Env):
    def __init__(self, n_rooms: int, n_agents: int, random_state: Optional[int] = None):
        super().__init__()
        assert n_rooms > 0, "Number of rooms must be greater than 0"
        if random_state is not None:
            np.random.seed(random_state)

        self.action_space = spaces.Discrete(
            4
        )  # Need to change this to include KILL + FIX + SABOTAGE actions
        self.observation_space = spaces.Tuple(
            (
                spaces.MultiDiscrete([n_rooms, n_rooms]),  # Agent positions
                spaces.MultiDiscrete([n_rooms, n_rooms]),  # Job positions
                spaces.MultiBinary(n_agents),  # Completed jobs
                spaces.MultiBinary(n_agents),  # Alive agents
            )
        )

        self.n_rooms = n_rooms
        self.n_agents = n_agents
        self._generate_rooms()
        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state

        Returns:
        Tuple: A tuple containing the initial state and an empty dictionary
        """
        self.t = 0
        rooms_list = np.arange(self.n_rooms)
        agent_pos_indices = np.random.choice(
            rooms_list, size=self.n_agents, replace=True
        )
        self.agent_positions = [self.rooms[i] for i in agent_pos_indices]
        job_pos_indices = np.random.choice(rooms_list, size=self.n_agents, replace=True)
        self.alive_agents = np.ones(self.n_agents)
        self.job_positions = [self.rooms[i] for i in job_pos_indices]
        self.completed_jobs = np.zeros(self.n_agents)
        self.turn = 0
        # TODO: Should turn be a part of the state?
        return (
            (
                self.agent_positions,
                self.job_positions,
                self.completed_jobs,
                self.alive_agents,
            ),
            {},
        )

    # TODO: Should we be turned based or just take a list of actions?
    # TODO: Rewards should be indvidualized for each agent unless terminal state is reached
    def step(self, action):
        """
        Take a step in the environment.
        """
        truncated = False
        done = False
        if np.sum(self.alive_agents) == 0:
            # all agents are dead
            reward = -1
            done = True

        elif np.sum(self.completed_jobs) == self.n_agents:
            # all jobs are completed
            reward = 1
            done = True

        elif self.alive_agents[self.turn] == 0:
            # agent is dead
            reward = -1
        else:
            # agent is alive
            # TODO: Need to handle non-move actions
            reward = 0
            pos = self.agent_positions[self.turn]
            if action in self.room_map[pos]:
                self.agent_positions[self.turn] = self.room_map[pos][action]
            else:
                # do nothing
                pass

        # increment the turn
        self.turn = (self.turn + 1) % self.n_agents

        # TODO: Need to check if t is incremented by a wrapper or if we need to do it here
        self.t += 1

        return (
            (
                self.agent_positions,
                self.job_positions,
                self.completed_jobs,
                self.alive_agents,
            ),
            reward,
            done,
            truncated,
            {},
        )

    def _generate_rooms(self):
        rooms = [(0, 0)]
        room_map = defaultdict(dict)
        while len(rooms) < self.n_rooms:
            random_action = np.random.choice(list(MoveAction))
            random_room = rooms[np.random.choice(len(rooms))]
            while random_action in room_map[random_room]:
                random_room = rooms[np.random.choice(len(rooms))]
                random_action = np.random.choice(list(MoveAction))
            new_room = move(random_action, random_room)
            if new_room not in rooms:
                rooms.append(new_room)
            room_map[random_room][random_action] = new_room
            room_map[new_room][reverse_action(random_action)] = random_room
        self.rooms = rooms
        self.room_map = room_map

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
                    if action == MoveAction.UP:
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
                    elif action == MoveAction.DOWN:
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
                    elif action == MoveAction.LEFT:
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
                    elif action == MoveAction.RIGHT:
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
            agent = (agent[0] - 0.5, agent[1] - 0.5)
            if self.alive_agents[i] == 0:
                ax.add_patch(patches.Circle(agent, 0.3, color="black", alpha=0.3))
            else:
                if i == self.turn:
                    ax.add_patch(
                        patches.Rectangle(agent, 1, 1, color="green", alpha=0.3)
                    )
                else:
                    ax.add_patch(
                        patches.Rectangle(agent, 1, 1, color="gray", alpha=0.3)
                    )

        ax.set_aspect("equal")
        plt.gca().invert_yaxis()  # Invert y-axis to match the coordinate system used in your environment
        plt.axis("off")  # Optionally hide the axis
        plt.title(
            f"Maze with {self.n_rooms} rooms (random seed: {np.random.get_state()[1][0]})"
        )
        plt.show()
