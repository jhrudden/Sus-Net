from typing import List
import os
from scheduler import ExponentialSchedule
from src.env import FourRoomEnv, AgentTypes, StateFields
from src.featurizers import (
    PerspectiveFeaturizer,
    StateSequenceFeaturizer,
    GlobalFeaturizer,
)
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
    scheduler_start_eps: float = 1.0,
    scheduler_end_eps: float = 0.05,
    scheduler_time_steps: int = 1_000_000,
    who_to_train: AgentTypes = AgentTypes.AGENT,
    imposter_pretrained_model_path: str | None = None,
    crew_pretrained_model_path: str | None = None,
    experiment_save_path: str | None = None,
    optimizer_type: str = "Adam",
    learning_rate: float = 0.0001,
    train_step_interval: int = 5,
    num_checkpoint_saves: int = 5,
):
    # initializing models
    if imposter_model_type == "spatial_dqn":

        imposter_model = SpatialDQN(
            input_image_size=env.n_cols,
            non_spatial_input_size=23,
            n_channels=[8, 5, 3],
            strides=[1, 1],
            paddings=[1, 1],
            kernel_sizes=[3, 3],
            rnn_layers=3,
            rnn_hidden_dim=64,
            rnn_dropout=0.2,
            mlp_hidden_layer_dims=[16, 16],
            n_actions=env.n_imposter_actions,
        )
    elif imposter_model_type == "random":
        imposter_model = RandomEquiprobable(env.n_imposter_actions)
    else:
        raise ValueError(f"Invalid model type: {imposter_model_type}")

    # loading model checkpoint if provided
    # NOTE: This currently over writes the model initialized above
    if imposter_pretrained_model_path is not None:
        imposter_model = SpatialDQN.load_from_checkpoint(crew_pretrained_model_path)

    if crew_model_type == "spatial_dqn":

        crew_model = SpatialDQN(
            input_image_size=env.n_cols,
            non_spatial_input_size=23,
            n_channels=[8, 5, 3],
            strides=[1, 1],
            paddings=[1, 1],
            kernel_sizes=[3, 3],
            rnn_layers=3,
            rnn_hidden_dim=64,
            rnn_dropout=0.2,
            mlp_hidden_layer_dims=[16, 16],
            n_actions=env.n_crew_actions,
        )
    elif crew_model_type == "random":
        crew_model = RandomEquiprobable(env.n_crew_actions)
    else:
        raise ValueError(f"Invalid model type: {imposter_model_type}")

    # loading model checkpoint if provided
    # NOTE: This currently over writes the model initialized above
    if crew_pretrained_model_path is not None:
        crew_model = SpatialDQN.load_from_checkpoint(crew_pretrained_model_path)

    # initializing optimizers
    if optimizer_type == "Adam":
        crew_optimizer = torch.optim.Adam(
            params=crew_model.parameters(), lr=learning_rate
        )
        imposter_optimizer = torch.optim.Adam(
            params=imposter_model.parameters(), lr=learning_rate
        )
    else:
        raise ValueError(f"Invalid optimizer: {optimizer_type}")

    # initialize scheduler
    scheduler = ExponentialSchedule(
        scheduler_start_eps, scheduler_end_eps, scheduler_time_steps
    )

    # initialize replay buffer and prepopulate it
    replay_buffer = FastReplayBuffer(
        max_size=replay_buffer_size,
        trajectory_size=sequence_length,
        state_size=env.flattened_state_size,
        n_imposters=env.n_imposters,
        n_agents=env.n_agents,
    )
    replay_buffer.populate(env=env, num_steps=replay_prepopulate_steps)

    # initialize featurizer
    if featurizer_type == "perspective":
        featurizer = PerspectiveFeaturizer(env=env, sequence_len=sequence_length)
    elif featurizer_type == "global":
        featurizer = GlobalFeaturizer(env=env, sequence_len=sequence_length)

    # run actual experiment
    train(
        env=env,
        num_steps=num_steps,
        featurizer=featurizer,
        crew_optimizer=crew_optimizer,
        imposter_optimizer=imposter_optimizer,
        imposter_model=imposter_model,
        crew_model=crew_model,
        save_directory_path=experiment_save_path,
        train_step_interval=train_step_interval,
        batch_size=batch_size,
        gamma=gamma,
        scheduler=scheduler,
        who_to_train=who_to_train,
        num_saves=num_checkpoint_saves,
    )

    # run experiment
    if experiment_save_path is not None:
        # TODO: SAVE some stuff!!!!
        pass


def train(
    env: FourRoomEnv,
    num_steps: int,
    replay_buffer: FastReplayBuffer,
    featurizer: StateSequenceFeaturizer,
    crew_optimizer: torch.optim.optimizer,
    imposter_optimizer: torch.optim.optimizer,
    imposter_model: nn.Module,
    crew_model: nn.Module,
    scheduler: ExponentialSchedule,
    save_directory_path: str,
    train_step_interval: int = 5,
    batch_size: int = 32,
    gamma: float = 0.99,
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
        featurizer.fit(replay_buffer.get_last_trajectory(), env.imposter_idxs)

        # getting next action
        eps = scheduler.value(t_total)
        agent_actions = np.zeros(env.n_agents)
        alive_agents = state[env.state_fields[StateFields.ALIVE_AGENTS]]

        for agent_idx, (spatial, non_spatial) in enumerate(featurizer.generator()):

            # choose action for alive imposter
            if env.imposter_mask[agent_idx] and alive_agents[agent_idx]:
                if np.random.random() <= eps:
                    agent_actions[agent_idx] = np.random.randint(
                        0, env.n_imposter_actions
                    )
                else:
                    agent_actions[agent_idx] = int(
                        torch.argmax(imposter_model(spatial, non_spatial))
                    )

            # choose action for alive crew member
            elif alive_agents[agent_idx]:
                if np.random.random() <= eps:
                    agent_actions[agent_idx] = np.random.randint(0, env.n_crew_actions)
                else:
                    agent_actions[agent_idx] = int(
                        torch.argmax(crew_model(spatial, non_spatial))
                    )

        # TODO: USE INFO TO STORE game details
        next_state, reward, done, trunc, info = env.step(agent_actions=agent_actions)

        # adding the timestep to replay buffer
        replay_buffer.add(
            state=state,
            action=agent_actions,
            reward=reward,
            next_state=next_state,
            done=done,
            imposters=env.imposter_idxs,
            is_start=t_episode == 0,
        )

        if t_total % train_step_interval == 0:

            batch = replay_buffer.sample(batch_size=batch_size)

            # TODO: featurtize: WHOLE BATCH!!
            featurizer.fit(batch.states, batch.imposters)

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
