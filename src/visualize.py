from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import List, Dict, Set
import pygame
import pathlib
import json
import ipywidgets as widgets
from IPython.display import display

from src.replay_memory import ReplayBuffer
from src.models.dqn import ModelType, Q_Estimator
from src.metrics import SusMetrics
from src.features.model_ready import SequenceStateFeaturizer
from src.environment import FourRoomEnv

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
    def __init__(self, env: FourRoomEnv):
        assert env.n_cols == env.n_rows, "Only square grids are supported"

        self.env = env
        self.grid_size = env.n_cols
        self.window_size = self.grid_size * CELL_SIZE + GAME_PADDING * 2
        self.game_over = False
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
        if "Tagging" in self.env.__class__.__name__:
            self._draw_voting()
        self._draw_jobs()
        self._draw_agents()
        if self.game_over:
            self._draw_win_text()
        pygame.display.flip()

    def _draw_win_text(self):
        big_font = pygame.font.Font(None, 36)  # Use a larger font size, e.g., 36

        imposter_won = self.env.metrics.metrics[SusMetrics.IMPOSTER_WON]
        win_text = "Sussy Victory!" if imposter_won else "Crewmates win!"
        text_color = (255, 0, 0) if imposter_won else (0, 255, 0)
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
        if self.env.__dict__.get("tag_counts") is None:
            return
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
    def __init__(self, featurizer: SequenceStateFeaturizer, cmap="Blues"):
        # TODO: revisit giving imposter_positions to constructor, this is a hack
        self.featurizer = featurizer
        self.cmap = cmap

    def visualize_global_state(self, imposters: torch.Tensor):
        for b, spatial in enumerate(torch.unbind(self.featurizer.spatial, dim=0)):
            imposters_locations = set(imposters[b].tolist())
            self._visualize_sequence(
                spatial, imposters_locations, title=f"Global State, Batch {b}"
            )

    def visualize_perspectives(self, imposters: torch.Tensor):
        for agent_id, (batched_spatial, batch_non_spatial) in enumerate(
            self.featurizer.generator()
        ):
            B, *_ = batched_spatial.size()
            for b in range(B):
                self._visualize_sequence(
                    batched_spatial[b],  # remove batch dimension
                    title=f"Agent {agent_id}'s Perspective",
                    description=f"Non-Spatial: \n{str(batch_non_spatial[b])}",
                    imposters=set(imposters[b].tolist()),
                )

    def _visualize_step(
        self,
        spatial: torch.Tensor,
        sequence_idx: int,
        imposters: Set[int] = None,
        ax=None,
    ):

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
        self,
        spatial: torch.Tensor,
        imposters: Set[int],
        title: str = None,
        description: str = None,
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

def moving_average(data, *, window_size=50):
    """Smooths 1-D data array using a moving average."""
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel, 'valid') / window_size
    return smooth_data

def plot_experiment_metrics(exp, label_attr=None, label_name=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    for version_dir in sorted(exp.iterdir(), key=lambda x: x.name):
        if not version_dir.is_dir():
            continue

        config_path = version_dir / 'config.json'
        metrics_path = version_dir / 'metrics.json'

        if not config_path.exists() or not metrics_path.exists():
            continue

        config = json.loads(config_path.read_text())
        metrics = json.loads(metrics_path.read_text())
        label = f"{label_name}={config.get(label_attr, 0.0):.2f} | target_update_freq={config.get('target_update_interval', 1000)}"

        # Process and plot returns
        returns = np.array(metrics.get(SusMetrics.AVG_IMPOSTER_RETURNS, []))
        if returns.size > 0:
            returns_cumsum = np.cumsum(returns)
            axes[0].plot(returns_cumsum, label=label)

        # Process and plot episode lengths
        episode_lengths = np.array(metrics.get(SusMetrics.TOTAL_TIME_STEPS, []))
        if episode_lengths.size > 0:
            lengths = np.repeat(np.arange(len(episode_lengths)), episode_lengths)
            axes[1].plot(np.arange(len(lengths)), lengths, label=label)
             

        # Process and plot losses
        losses = np.array(metrics.get(SusMetrics.IMPOSTER_LOSS, []))
        if losses.size > 0:
            axes[2].plot(losses, alpha=0.35, label=label)

    # Setting common titles and labels
    axes[0].set_title('Expected Returns')
    axes[0].set_xlabel('Episode Number')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].legend()

    x_ticks = np.arange(0, len(lengths) + 100_000, 100_000)
    x_ticks_labels = [f'{x // 100_000}' for x in x_ticks]

    axes[1].set_title('Episode Length by Time Step')
    axes[1].set_xlabel('Time Step (x100k)')
    axes[1].set_ylabel('Episode Number')
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_ticks_labels)
    axes[1].legend()


    x_ticks = np.arange(0, len(losses) + 10_000, 10_000)
    x_ticks_labels = [f'{x // 10_000}' for x in x_ticks]
    axes[2].set_title('Imposter Loss')
    axes[2].set_xlabel('Batch Number (x10k)')
    axes[2].set_ylabel('Loss')
    axes[2].set_xticks(x_ticks)
    axes[2].set_xticklabels(x_ticks_labels)
    axes[2].legend()

    # add title based on experiment name
    fig.suptitle(exp.name)
    # save figure to experiment directory
    plt.savefig(exp / f'{exp.name}_metrics.png')

    plt.tight_layout()
    plt.show()


def run_game(
    env: FourRoomEnv,
    imposter_model: Q_Estimator,
    crew_model: Q_Estimator,
    featurizer: SequenceStateFeaturizer,
    sequence_length: int = 2,
    debug: bool = True,
):
    def reset_game(visualizer):
        state, _ = visualizer.reset()
        replay_memory = ReplayBuffer(
            max_size=10_000,
            trajectory_size=sequence_length,
            state_size=visualizer.env.flattened_state_size,
            n_imposters=visualizer.env.n_imposters,
            n_agents=visualizer.env.n_agents,
        )
        state_sequence = np.zeros(
            (replay_memory.trajectory_size, replay_memory.state_size)
        )
        for i in range(replay_memory.trajectory_size):
            state_sequence[i] = visualizer.env.flatten_state(state)
        return state, replay_memory, state_sequence
    
    with AmongUsVisualizer(env) as visualizer:
        state, replay_memory, state_sequence = reset_game(visualizer)

        stop_game = False
        done = False
        paused = False
        while not stop_game:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    stop_game = True
                    break
                # if you click r, reset the game
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    state, replay_memory, state_sequence = reset_game(visualizer)
                    paused = False
                    done = False

            if not done and not paused:
                featurizer.fit(state_sequence=torch.tensor(state_sequence).unsqueeze(0))
                actions = []
                action_strs = []

                for agent_idx, (agent_spatial, agent_non_spatial) in enumerate(
                    featurizer.generate_featurized_states()
                ):
                    if agent_idx in env.imposter_idxs:
                        action_probs = imposter_model(agent_spatial, agent_non_spatial)
                        action = action_probs.argmax().item()
                    else:
                        action = crew_model(agent_spatial, agent_non_spatial).argmax().item()
                    action_strs.append(env.compute_action(agent_idx, action))
                    
                    actions.append(action)

                next_state, reward, done, truncated, _ = visualizer.step(actions)

                if debug:
                    print(f'Actions: {action_strs}')

                next_state_sequence = np.roll(state_sequence.copy(), -1, axis=0)
                next_state_sequence[-1] = env.flatten_state(next_state)

                replay_memory.add(
                    state=state_sequence,
                    action=actions,
                    reward=reward,
                    done=done,
                    next_state=next_state_sequence,
                    imposters=env.imposter_idxs,
                )

                state = next_state
                state_sequence = next_state_sequence

            pygame.time.wait(250)
        visualizer.close()


def setup_experiment_buttons(base_path, identifier_attribute, experiments, featurizers):
    
    max_exp_name_length = max(len(exp) for exp in experiments)
    min_width = max_exp_name_length * 8  # Set a minimum width based on the longest experiment name

    def button_callback(button, buttons_list):
        # Disable all buttons in the same list to prevent multiple runs
        for b in buttons_list:
            b.disabled = True
        v_path = button.v_path
        config_path = v_path / 'config.json'
        exp = button.exp
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        imposter_model = ModelType.build(config['imposter_model_type'], pretrained_model_path= v_path / f'imposter_{config.get('imposter_model_type')}_100%.pt')
        crew_model = ModelType.build(config['crew_model_type'], n_actions=config['crew_model_args']['n_actions'])
        featurizer = featurizers[exp]
        env = featurizer.env
        run_game(env, imposter_model, crew_model, featurizer, sequence_length=config['sequence_length'])

        for b in buttons_list:
            b.disabled = False


    for exp in experiments:
        buttons = []
        exp_path = base_path / exp
        if not exp_path.exists():
            continue
        for version_dir in sorted(exp_path.iterdir(), key=lambda x: x.name):
            config_path = version_dir / 'config.json'
            if not config_path.exists():
                continue
            
            with open(config_path, 'r') as file:
                config = json.load(file)
            
            button_label = f"{identifier_attribute}={config.get(identifier_attribute, 'Unknown')}"
            button = widgets.Button(description=button_label)
            button.v_path = version_dir
            button.exp = exp
            buttons.append(button)
        
        # Pass the current list of buttons as a default argument to the lambda
        for button in buttons:
            button.on_click(lambda event, b=buttons: button_callback(event, b))
        
        buttons_box = widgets.HBox(buttons, layout=widgets.Layout(flex_flow='row wrap', align_items='flex-start'))
        
        # Set a fixed width for the label to align them
        exp_label = widgets.Label(value=f"Experiment: {exp}", 
                                  layout=widgets.Layout(width=f'{min_width}px'))
        
        row = widgets.HBox([exp_label, buttons_box], layout=widgets.Layout(align_items='center', justify_content='flex-start'))
        display(row)

def plot_episode_lengths(root_dir, separator_strings, seperator_labels=None):
    assert len(separator_strings) == len(seperator_labels) or seperator_labels is None, "Seperator labels must be the same length as the separator strings"
    root_dir = pathlib.Path(root_dir)
    episode_lengths_by_exp = defaultdict(list)
    filled = set()

    for experiment_dir in root_dir.iterdir():
        for separator in separator_strings:
            if separator in experiment_dir.name and experiment_dir.name not in filled:
                filled.add(experiment_dir.name)
                exp_name = experiment_dir.name
                label = None
                best_run_lengths = []
                max_length = 0

                for run_dir in experiment_dir.iterdir():
                    if run_dir.is_dir():
                        metrics_path = run_dir / 'metrics.json'
                        config_path = run_dir / 'config.json'
                        if metrics_path.exists() and config_path.exists():
                            metrics = json.loads(metrics_path.read_text())
                            episode_lengths = metrics.get(SusMetrics.TOTAL_TIME_STEPS, [])
                            if len(episode_lengths) > max_length:
                                config = json.loads(config_path.read_text())
                                label = f"$\\gamma$={config.get('gamma', 0.0):.2f} | target_update_interval={config.get('target_update_interval', 1000)}"
                                max_length = len(episode_lengths)
                                best_run_lengths = episode_lengths
                if best_run_lengths:
                    episode_lengths_by_exp[separator].append((best_run_lengths, label, exp_name))
                break  # Break out of the separator loop, as we've found a match
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 15))
    for sep_idx, separator in enumerate(episode_lengths_by_exp.keys()):
        for best_run_lengths, label, exp_name in episode_lengths_by_exp[separator]:
            lengths = np.repeat(np.arange(len(best_run_lengths)), best_run_lengths)
            ax[sep_idx].plot(np.arange(len(lengths)), lengths, label=f'{exp_name} {label}')
        
        if seperator_labels:
            separator_title = seperator_labels[sep_idx]
        else:
            separator_title = separator.replace('_', ' ').title()

        x_ticks = np.arange(0, len(lengths) + 100_000, 100_000)
        x_ticks_labels = [f'{x // 100_000}' for x in x_ticks]
        ax[sep_idx].set_title(f'Episode Length by Time Step ({separator_title})') 
        ax[sep_idx].set_ylabel('Episode Number')
        ax[sep_idx].set_xlabel('Time Step (x100k)')
        ax[sep_idx].set_xticks(x_ticks)
        ax[sep_idx].set_xticklabels(x_ticks_labels)
        ax[sep_idx].legend()


    plt.show()