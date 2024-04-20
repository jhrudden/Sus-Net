from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Optional, Set
import pygame
import pathlib

from src.featurizers import StateSequenceFeaturizer
from src.env import FourRoomEnvWithTagging

ASSETS_PATH = pathlib.Path(__file__).parent.parent / "assets"

# Constants for Pygame visualization
CELL_SIZE = 60  # Pixel size of each cell
BACKGROUND_COLOR = (30, 30, 30)  # Dark grey
GRID_COLOR = (200, 200, 200)  # Light grey
WALL_COLOR = (0, 0, 0)
CREW_COLOR = (0, 0, 255)  # Blue
IMPOSTOR_COLOR = (255, 0, 0)  # Red
FONT_SIZE = 16

COMPLETED_JOB_COLOR = (0, 255, 0, 100)  # Green
JOB_COlOR = (255, 255, 0)  # Yellow

BLOOD_SPLATTER_PATH = ASSETS_PATH / "blood_splatter.png"
IMPOSTER_PATH = ASSETS_PATH / "purple.png"
CREW_PATH = ASSETS_PATH / "blue.png"
GAME_PADDING = 50


class AmongUsVisualizer:
    def __init__(self, env: FourRoomEnvWithTagging):
        assert env.n_cols == env.n_rows, "Only square grids are supported"

        self.env = env
        self.grid_size = env.n_cols
        self.window_size = self.grid_size * CELL_SIZE + GAME_PADDING * 2
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.font = pygame.font.Font(None, FONT_SIZE)
        pygame.display.set_caption(
            f"Among Us - Four Room Environment {self.grid_size}x{self.grid_size}"
        )

        blood_spatter_image = pygame.image.load(BLOOD_SPLATTER_PATH)
        self.blood_spatter_image = pygame.transform.scale(
            blood_spatter_image, (CELL_SIZE, CELL_SIZE)
        )  # Scale it to fit the cell

        imposter_image = pygame.image.load(IMPOSTER_PATH)
        self.imposter_image = pygame.transform.scale(
            imposter_image, (CELL_SIZE, CELL_SIZE)
        )

        crew_image = pygame.image.load(CREW_PATH)
        self.crew_image = pygame.transform.scale(crew_image, (CELL_SIZE, CELL_SIZE))

    # NOTE: __enter__ and __exit__ methods allow the class to be used as a context manager
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def render(self):
        """
        Render the Four Room environment based on the current state
        """
        self.screen.fill(BACKGROUND_COLOR)  # clear the screen
        self._draw_grid()
        self._draw_voting()
        self._draw_jobs()
        self._draw_agents()
        if self.game_over:
            self._draw_win_text()
        pygame.display.flip()

    def _draw_win_text(self):
        big_font = pygame.font.Font(None, 36)  # Use a larger font size, e.g., 36

        _, win_reward = self.env.check_win_condition()
        win_text = "Sussy Victory!" if win_reward < 0 else "Crewmates win!"
        text_color = (255, 0, 0) if win_reward < 0 else (0, 255, 0)
        win_surface = big_font.render(win_text, True, text_color)
        win_rect = win_surface.get_rect(
            center=(self.window_size // 2, self.window_size // 2)
        )

        border_rect = win_rect.inflate(20, 20)  # Add padding around the text

        pygame.draw.rect(
            self.screen, text_color, border_rect
        )  # Border with the same color as the text
        pygame.draw.rect(
            self.screen, (0, 0, 0), border_rect.inflate(-4, -4)
        )  # Black background inside the border

        self.screen.blit(win_surface, win_rect)

    def _calculate_cell(self, x: int, y: int):
        """
        Calculate the cell position based on the x and y coordinates

        Args:
            x (int): X coordinate
            y (int): Y coordinate
        """

        return x * CELL_SIZE + GAME_PADDING, y * CELL_SIZE + GAME_PADDING

    def _calculate_center(self, x: int, y: int):
        """
        Calculate the center of the cell based on the x and y coordinates

        Args:
            x (int): X coordinate
            y (int): Y coordinate
        """

        return (
            x * CELL_SIZE + CELL_SIZE // 2 + GAME_PADDING,
            y * CELL_SIZE + CELL_SIZE // 2 + GAME_PADDING,
        )

    def _draw_grid(self):
        """
        Draw environment grid
        """
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                is_wall = ~self.env.grid[x, y]
                cell_color = WALL_COLOR if is_wall else GRID_COLOR
                border = int(~is_wall)
                cell_x, cell_y = self._calculate_cell(x, y)
                rect = pygame.Rect(cell_x, cell_y, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, cell_color, rect, border)

    def _draw_voting(self):
        """
        Draw voting state
        """
        vote_counts = self.env.tag_counts.flatten()
        vote_counts[self.env.alive_agents == 0] = -1
        voted = self.env.used_tag_actions.flatten() + 0
        voted[self.env.alive_agents == 0] = -1

        time_left_to_vote = self.env.tag_reset_interval - self.env.tag_reset_timer

        screen_width, screen_height = self.screen.get_size()
        votes_text = "Votes: " + " ".join(map(str, vote_counts))
        voted_text = "Voted: " + " ".join(map(str, voted))
        time_text = f"Vote counts reset in: {time_left_to_vote}"

        votes_surface = self.font.render(votes_text, True, (255, 255, 255))
        voted_surface = self.font.render(voted_text, True, (255, 255, 255))

        time_vote_surface = self.font.render(time_text, True, (255, 255, 255))

        text_width, text_height = votes_surface.get_size()

        time_text_width, time_text_height = time_vote_surface.get_size()

        bar_x = screen_width - text_width - 10
        bar_y = 10

        bar_left = 10

        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (bar_x - 5, bar_y - 5, text_width + 10, text_height * 2 + 10),
        )
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (bar_x - 5, bar_y - 5, text_width + 10, text_height * 2 + 10),
            1,
        )

        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (bar_left - 5, bar_y - 5, time_text_width + 10, time_text_height + 10),
        )
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (bar_left - 5, bar_y - 5, time_text_width + 10, time_text_height + 10),
            1,
        )

        # Blit the text onto the screen
        self.screen.blit(votes_surface, (bar_x, bar_y))
        self.screen.blit(voted_surface, (bar_x, bar_y + text_height))
        self.screen.blit(time_vote_surface, (bar_left, bar_y))

    def _draw_agents(self):
        """
        Draw agents on the grid
        """
        dead_agents = np.argwhere(~self.env.alive_agents).flatten()
        alive_agents = np.argwhere(self.env.alive_agents).flatten()

        # draw dead agents first
        for agent_idx in dead_agents:
            x, y = self.env.agent_positions[agent_idx]

            self._draw_agent(True, agent_idx, x, y)

        # draw alive agents second
        for agent_idx in alive_agents:
            x, y = self.env.agent_positions[agent_idx]

            self._draw_agent(False, agent_idx, x, y)

    def _draw_agent(self, is_dead: bool, agent_idx: int, x: int, y: int):
        """
        Draw a single agent on the grid

        Args:
            is_dead (bool): Whether the agent is dead
            agent_idx (int): Index of the agent
            x (int): X position of the agent
            y (int): Y position of the agent
        """
        is_imposter = self.env.imposter_mask[agent_idx]

        center_x, center_y = self._calculate_center(x, y)
        cell_x, cell_y = self._calculate_cell(x, y)

        if not is_dead:
            image = self.imposter_image if is_imposter else self.crew_image
            self.screen.blit(image, (cell_x, cell_y))
        else:
            self.screen.blit(self.blood_spatter_image, (cell_x, cell_y))

        text_surface = self.font.render(f"{agent_idx}", True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(center_x, center_y))
        self.screen.blit(text_surface, text_rect)

    def _draw_jobs(self):
        """
        Draw jobs on the grid
        """
        for job_idx in range(self.env.n_jobs):
            x, y = self.env.job_positions[job_idx]

            is_completed = self.env.completed_jobs[job_idx]

            # Draw jobs as triangles with black border
            center_x, center_y = self._calculate_center(x, y)

            job_color = COMPLETED_JOB_COLOR if is_completed else JOB_COlOR

            pygame.draw.polygon(
                self.screen,
                job_color,
                [
                    (center_x, center_y - CELL_SIZE // 4),
                    (center_x - CELL_SIZE // 4, center_y + CELL_SIZE // 4),
                    (center_x + CELL_SIZE // 4, center_y + CELL_SIZE // 4),
                ],
            )

    def step(self, actions: List[int]) -> Dict:
        """
        Take a step in the environment and render the new state

        Args:
            actions (list): List of actions for each agent

        Returns:
            dict: Dictionary containing the results from gym.Env.step method
        """
        if self.game_over:
            raise ValueError("Cannot call step on a completed episode")

        state, reward, done, truncated, info = self.env.step(actions)
        self.game_over = done
        self.render()

        return state, reward, done, truncated, info

    def reset(self) -> Dict:
        """
        Reset the environment and render the new state

        Returns:
            dict: Dictionary containing the results from gym.Env.reset method
        """
        initial = self.env.reset()
        self.game_over = False
        self.render()
        return initial

    def close(self):
        """
        Close the pygame window
        """
        pygame.quit()


class StateSequenceVisualizer:
    def __init__(self, featurizer: StateSequenceFeaturizer, cmap="Blues"):
        # TODO: revisit giving imposter_positions to constructor, this is a hack
        self.featurizer = featurizer
        self.cmap = cmap

    def visualize_global_state(self, imposters: torch.Tensor):
        for b, spatial in enumerate(torch.unbind(self.featurizer.spatial, dim=0)):
            imposters_locations = set(imposters[b].tolist())
            self._visualize_sequence(spatial, imposters_locations, title=f"Global State, Batch {b}")

    def visualize_perspectives(self):
        assert self.featurizer.batch_size == 1, "Visualizing perspectives only works for batch size 1"
        for agent_idx, (spatial, non_spatial) in enumerate(self.featurizer.generator()):
            self._visualize_sequence(
                spatial.squeeze(0), # remove batch dimension
                title=f"Agent {agent_idx}'s Perspective",
                description=f"Non-Spatial: \n{str(non_spatial.squeeze(0))}",
            )

    def _visualize_step(self, spatial: torch.Tensor, sequence_idx: int, imposters: Set[int] = None, ax=None):

        n_channels, n_rows, n_cols = spatial[sequence_idx, ...].shape
        n_agents = self.featurizer.env.n_agents

        assert n_rows == n_cols, "Only supports square grids"

        if ax is None:
            _, ax = plt.subplots(1, n_agents, figsize=(5 * n_agents, 5))

        cells = np.arange(n_rows)
        ticks = cells - 0.5

        for channel_idx in range(n_channels):
            if channel_idx < n_agents:
                is_imposter = channel_idx in imposters
                role = "Imposter" if is_imposter else "Crewmate"
                ax[channel_idx].set_title(f"{role} {channel_idx}")
            else:
                ax[channel_idx].set_title(f"Other {channel_idx}")

            rotated = np.flipud(spatial[sequence_idx, channel_idx].t().numpy())

            ax[channel_idx].imshow(rotated, cmap=self.cmap)
            ax[channel_idx].grid(True)
            ax[channel_idx].set_xticks(ticks, minor=False)
            ax[channel_idx].set_yticks(ticks, minor=False)
            ax[channel_idx].set_xticklabels([], minor=False)
            ax[channel_idx].set_yticklabels([], minor=False)
            ax[channel_idx].tick_params(axis="both", which="both", length=0)

            for y in cells:
                for x in cells:
                    is_colored = spatial[sequence_idx, channel_idx, x, y]
                    ax[channel_idx].text(
                        x,
                        n_rows - y - 1,
                        str((x, y)),
                        va="center",
                        ha="center",
                        fontsize=8,
                        color="white" if is_colored else "black",
                    )

    def _visualize_sequence(
        self, spatial: torch.Tensor, imposters: Set[int], title: str = None, description: str = None
    ):
        seq_len, n_channels, _, __ = spatial.size()

        _, ax = plt.subplots(seq_len, n_channels, figsize=(n_channels * 5, seq_len * 5))

        for seq_idx in range(seq_len):
            # add title to row
            label = (
                "S$_{t"
                + ("-" + str(seq_len - seq_idx - 1) if seq_idx < seq_len - 1 else "")
                + "}$"
            )
            ax[seq_idx][0].set_ylabel(label, rotation=0, labelpad=40, fontsize=20)
            self._visualize_step(spatial, seq_idx, ax=ax[seq_idx], imposters=imposters)

        if title is not None:
            plt.suptitle(title, fontsize=20)
        if description is not None:
            plt.figtext(
                0.5,
                0.01,
                description,
                wrap=True,
                horizontalalignment="center",
                fontsize=16,
            )
        plt.show()
