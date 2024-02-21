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
        return (position[0], position[1]+1)
    elif action == MoveAction.DOWN:
        return (position[0], position[1]-1)
    elif action == MoveAction.LEFT:
        return (position[0]-1, position[1])
    elif action == MoveAction.RIGHT:
        return (position[0]+1, position[1])

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

        self.n_rooms = n_rooms
        self.n_agents = n_agents
        self._generate_rooms()
        self.reset()

    def reset(self):
        self.t = 0
        rooms_list = np.arange(self.n_rooms)
        agent_pos_indices = np.random.choice(rooms_list, size=self.n_agents, replace=True)
        self.agent_positions = [self.rooms[i] for i in agent_pos_indices]
        job_pos_indices = np.random.choice(rooms_list, size=self.n_agents, replace=True)
        self.job_positions = [self.rooms[i] for i in job_pos_indices]
        self.completed_jobs = np.zeros(self.n_agents)
        self.turn = 0
        # TODO: Should turn be a part of the state?
        return ((self.agent_positions, self.job_positions, self.completed_jobs), {})

    def step(self, action):
        raise NotImplementedError

    def _generate_rooms(self):
        rooms = [(0,0)]
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
            ax.add_patch(patches.Rectangle((room_center[0] - room_size / 2, room_center[1] - room_size / 2),
                                        room_size, room_size, fill=None, edgecolor='black'))
            
            # For each connected room, draw a doorway
            if room in self.room_map:
                for action, next_room in self.room_map[room].items():
                    if action == MoveAction.UP:
                        ax.add_patch(patches.Rectangle((room_center[0] - door_width / 2, room_center[1] + room_size / 2),
                                                    door_width, door_width, color='black'))
                    elif action == MoveAction.DOWN:
                        ax.add_patch(patches.Rectangle((room_center[0] - door_width / 2, room_center[1] - room_size / 2 - door_width),
                                                    door_width, door_width, color='black'))
                    elif action == MoveAction.LEFT:
                        ax.add_patch(patches.Rectangle((room_center[0] - room_size / 2 - door_width, room_center[1] - door_width / 2),
                                                    door_width, door_width, color='black'))
                    elif action == MoveAction.RIGHT:
                        ax.add_patch(patches.Rectangle((room_center[0] + room_size / 2, room_center[1] - door_width / 2),
                                                    door_width, door_width, color='black'))
       # Draw agents and jobs 
        for j, job in enumerate(self.job_positions):
            if self.completed_jobs[j] == 0:
                ax.add_patch(patches.Circle(job, 0.3, color='green'))
            else:
                ax.add_patch(patches.Circle(job, 0.3, color='red'))

        for agent in self.agent_positions:
            ax.add_patch(patches.Circle(agent, 0.3, color='black', alpha=0.3))
                    
        ax.set_aspect('equal')
        plt.gca().invert_yaxis()  # Invert y-axis to match the coordinate system used in your environment
        plt.axis('off')  # Optionally hide the axis
        plt.title(f'Maze with {self.n_rooms} rooms (random seed: {np.random.get_state()[1][0]})')
        plt.show()