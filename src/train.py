from collections import defaultdict
from typing import List
import os
import math
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
import torch.functional as F

from utils import add_info_to_episode_dict


def run_experiment(
    env: FourRoomEnv,
    num_steps: int,
    imposter_model_args: dict,
    crew_model_args: dict,
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
    who_to_train: AgentTypes | None = AgentTypes.AGENT,
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

        imposter_model = SpatialDQN(*imposter_model_args)
    elif imposter_model_type == "random":
        imposter_model = RandomEquiprobable(env.n_imposter_actions)
    else:
        raise ValueError(f"Invalid model type: {imposter_model_type}")

    # loading model checkpoint if provided
    # NOTE: This currently over writes the model initialized above
    if imposter_pretrained_model_path is not None:
        imposter_model = SpatialDQN.load_from_checkpoint(crew_pretrained_model_path)

    if crew_model_type == "spatial_dqn":
        crew_model = SpatialDQN(*crew_model_args)
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
    who_to_train: (
        AgentTypes | None
    ) = AgentTypes.AGENT,  # trains both imposters and crew by default
    num_saves: int = 5,
):
    returns = []
    game_lengths = []
    losses = []
    info_list = []  # keep track of events during each episode

    # Initialize structures to store the models at different stages of training
    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)

    i_episode = 0  # Use this to indicate the index of the current episode
    t_episode = 0  # Use this to indicate the time-step inside current episode

    state, info = env.reset()  # Initialize state of first episode

    episode_info_dict = defaultdict(list)
    add_info_to_episode_dict(episode_info_dict, info)

    G = torch.zeros(env.n_agents)

    # Iterate for a total of `num_steps` steps
    pbar = tqdm.trange(num_steps)
    for t_total in pbar:

        # Save model
        if t_total in t_saves and who_to_train is not None:
            percent_progress = math.round(t_total / num_steps * 100)
            imposter_model.dump_to_checkpoint(
                os.path.join(
                    save_directory_path, f"imposter_model_{percent_progress}%.pt"
                )
            )
            crew_model.dump_to_checkpoint(
                os.path.join(save_directory_path, f"crew_model_{percent_progress}%.pt")
            )

        # Update Target DQNs
        if t_total % 10_000 == 0:
            imposter_target_dqn = imposter_model.deepcopy()
            crew_target_dqn = crew_model.deepcopy()

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
        G = G * gamma + reward

        add_info_to_episode_dict(episode_info_dict, info)

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

        # Training update for imposters and/or crew
        if t_total % train_step_interval == 0:

            if who_to_train in [AgentTypes.AGENT, AgentTypes.IMPOSTER]:

                batch = replay_buffer.sample(batch_size)
                train_step(
                    optimizer=imposter_optimizer,
                    batch=batch,
                    dqn_model=imposter_model,
                    dqn_target_model=imposter_target_dqn,
                    gamma=gamma,
                    featurizer=featurizer,
                    who_to_train=AgentTypes.IMPOSTER,
                )

            if who_to_train in [AgentTypes.AGENT, AgentTypes.CREW_MEMBER]:

                batch = replay_buffer.sample(batch_size)
                train_step(
                    optimizer=crew_optimizer,
                    batch=batch,
                    dqn_model=crew_model,
                    dqn_target_model=crew_target_dqn,
                    gamma=gamma,
                    featurizer=featurizer,
                    who_to_train=AgentTypes.CREW_MEMBER,
                )

        # checking if the env needs to be reset
        if done or trunc:

            pbar.set_description(
                f"Episode: {i_episode} | Steps: {t_episode + 1} | Epsilon: {eps:4.2f} | Loss: {losses[-1]:4.2f}"
            )

            returns.append(G.tolist())
            game_lengths.append(t_episode)
            G = torch.zeros(env.n_agents)
            t_episode = 0
            i_episode += 1
            info_list.append(dict(episode_info_dict))
            episode_info_dict = defaultdict(list)

            state, _ = env.reset()

        else:
            state = next_state
            t_episode += 1

    # saving final model states
    imposter_model.dump_to_checkpoint(
        os.path.join(save_directory_path, f"imposter_model_100%.pt")
    )
    crew_model.dump_to_checkpoint(
        os.path.join(save_directory_path, f"crew_model_100%.pt")
    )

    return info_list


def train_step(
    optimizer, batch, dqn_model, dqn_target_model, gamma, featurizer, who_to_train
):
    # TODO: ensure whole batch can be featurized!!!
    featurizer.fit(batch)

    # dqn_model.train()
    # action_values = dqn_model(torch.tensor(batch.states))
    # actions = torch.tensor(batch.actions)
    # # print( actions.view(-1).shape)
    # values = torch.gather(action_values, 1, actions.view(-1).unsqueeze(-1)).view(-1)

    # dqn_target_model.eval()
    # with torch.no_grad():

    #     done_mask = torch.tensor(batch.dones).view(-1)
    #     rewards = torch.tensor(batch.rewards).view(-1)
    #     next_states = torch.tensor(batch.next_states)

    #     target_values = (
    #         rewards + gamma * torch.max(dqn_target_model(next_states), dim=1)[0]
    #     )
    #     target_values[done_mask] = rewards[done_mask]

    # # DO NOT EDIT

    # assert (
    #     values.shape == target_values.shape
    # ), "Shapes of values tensor and target_values tensor do not match."

    # # Testing that the values tensor requires a gradient,
    # # and the target_values tensor does not
    # assert values.requires_grad, "values tensor requires gradients"
    # assert (
    #     not target_values.requires_grad
    # ), "target_values tensor should not require gradients"

    # # Computing the scalar MSE loss between computed values and the TD-target
    # # DQN originally used Huber loss, which is less sensitive to outliers
    # loss = F.mse_loss(values, target_values)

    # optimizer.zero_grad()  # Reset all previous gradients
    # loss.backward()  # Compute new gradients
    # optimizer.step()  # Perform one gradient-descent step

    # return loss.item()
