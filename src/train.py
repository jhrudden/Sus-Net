from typing import List
from scheduler import ExponentialSchedule
from src.env import FourRoomEnv, AgentTypes, StateFields
from src.featurizers import PerspectiveFeaturizer, StateSequenceFeaturizer
from src.replay_memory import FastReplayBuffer
from src.models.dqn import SpatialDQN, RandomEquiprobable
import numpy as np
import tqdm
from torch import nn
import torch


def run_experiment(
    env: FourRoomEnv,
    num_steps: int,
    imposter_model_type: str = "spatial_dqn",
    crew_model_type: str = "spatial_dqn",
    featurizer_type: str = "perspective",
    sequence_length: int = 5,
    replay_buffer_size: int = 100_000,
    replay_prepopulate_steps: int = 1000,
    batch_size: int = 32,
    gamma: float = 0.99,
    scheduler: ExponentialSchedule = ExponentialSchedule(1.0, 0.05, 1_000_000),
    who_to_train: AgentTypes = AgentTypes.AGENT,
    imposter_pretrained_model_path: str | None = None,
    crew_pretrained_model_path: str | None = None,
    experiment_save_path: str | None = None,
    optimizer_type: str = "Adam",
    learning_rate: float = 0.0001,
):
    # initializing models
    if imposter_model_type == "spatial_dqn":
        # TODO: MODEL INIT
        imposter_model = None
    elif imposter_model_type == "random":
        imposter_model = RandomEquiprobable(env.n_imposter_actions)
    else:
        raise ValueError(f"Invalid model type: {imposter_model_type}")

    if imposter_model_type == "spatial_dqn":
        # TODO: MODEL INIT
        crew_model = None
    elif imposter_model_type == "random":
        crew_model = RandomEquiprobable(env.n_crew_actions)
    else:
        raise ValueError(f"Invalid model type: {imposter_model_type}")

    # determining which parameters to optimize: who are we training
    if who_to_train == AgentTypes.AGENT:
        params_to_train = [imposter_model.parameters(), crew_model.parameters()]
    elif who_to_train == AgentTypes.IMPOSTER:
        params_to_train = [imposter_model.parameters()]
    elif who_to_train == AgentTypes.CREW_MEMBER:
        params_to_train = [imposter_model.parameters()]

    # initializing optimizer
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(params=params_to_train, lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_type}")


def train(
    env: FourRoomEnv,
    num_steps: int,
    n_imposter_actions: int,
    n_crew_actions: int,
    replay_buffer: FastReplayBuffer,
    featurizer: StateSequenceFeaturizer,
    optimizer: torch.optim.optimizer,
    imposter_model: nn.Module,
    crew_model: nn.Module,
    save_directory_path: str,
    train_step_interval: int = 5,
    batch_size: int = 32,
    gamma: float = 0.99,
    scheduler: ExponentialSchedule = ExponentialSchedule(1.0, 0.05, 1_000_000),
    who_to_train: AgentTypes = AgentTypes.AGENT,  # trains both imposters and crew by default
    num_saves: int = 5,
):
    rewards = torch.empty(size=(num_steps, env.n_agents))
    returns = []
    game_lengths = []
    losses = []
    kills = []
    vote_outs = []
    game_results = []

    # Initialize structures to store the models at different stages of training
    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)

    i_episode = 0  # Use this to indicate the index of the current episode
    t_episode = 0  # Use this to indicate the time-step inside current episode

    state, info = env.reset()  # Initialize state of first episode
    current_trajectory = [env.flatten_state(state)] * replay_buffer.trajectory_size

    # Iterate for a total of `num_steps` steps
    pbar = tqdm.trange(num_steps)
    for t_total in pbar:

        # Save model
        if t_total in t_saves:
            # TODO: SAVE MODELS
            pass

        # Update Target DQNs
        if t_total % 10_000 == 0:
            imposter_target_dqn = crew_target_dqn = None
            if who_to_train in [AgentTypes.AGENT, AgentTypes.IMPOSTER]:
                imposter_target_dqn = imposter_model.copy()

            if who_to_train in [AgentTypes.AGENT, AgentTypes.CREW_MEMBER]:
                crew_target_dqn = crew_model.copy()

        # featurizing trajectory
        featurizer.fit(current_trajectory, env.imposter_idxs)

        # getting next action
        eps = scheduler.value(t_total)
        agent_actions = np.zeros(env.n_agents)
        alive_agents = state[env.state_fields[StateFields.ALIVE_AGENTS]]

        for agent_idx, (spatial, non_spatial) in enumerate(featurizer.generator()):

            # choose action for alive imposter
            if env.imposter_mask[agent_idx] and alive_agents[agent_idx]:
                if np.random.random() <= eps:
                    agent_actions[agent_idx] = np.random.randint(0, n_imposter_actions)
                else:
                    agent_actions[agent_idx] = int(
                        torch.argmax(imposter_model(spatial, non_spatial))
                    )

            # choose action for alive crew member
            elif alive_agents[agent_idx]:
                if np.random.random() <= eps:
                    agent_actions[agent_idx] = np.random.randint(0, n_crew_actions)
                else:
                    agent_actions[agent_idx] = int(
                        torch.argmax(crew_model(spatial, non_spatial))
                    )

        # TODO: USE INFO TO STORE game details
        next_state, reward, done, trunc, info = env.step(agent_actions=agent_actions)

        # updating current trajectory
        current_trajectory.pop(0)
        current_trajectory.append(env.flatten_state(next_state))

        if t_total % train_step_interval == 0:

            batch = replay_buffer.sample(batch_size=batch_size)

            # TODO: featurtize

            if who_to_train in [AgentTypes.AGENT, AgentTypes.IMPOSTER]:
                # UPDATE imposter model
                pass

            if who_to_train in [AgentTypes.AGENT, AgentTypes.CREW_MEMBER]:
                # UPDATE crew model
                pass

        if done or trunc:

            pbar.set_description(
                f"Episode: {i_episode} | Steps: {t_episode + 1} | Return: {G:5.2f} | Epsilon: {eps:4.2f} | Loss: {losses[-1]:4.2f}"
            )

            returns.append(G)
            game_lengths.append(t_episode)
            G = 0
            t_episode = 0
            i_episode += 1

            state, _ = env.reset()
            current_trajectory = [
                env.flatten_state(state)
            ] * replay_buffer.trajectory_size

        else:
            state = next_state
            t_episode += 1


def train_step(optimizer, batch, dqn_model, dqn_target, gamma) -> float:
    pass
