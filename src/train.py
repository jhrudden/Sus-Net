from collections import defaultdict
from enum import StrEnum, auto
import os
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import copy
import tqdm

from src.utils import add_info_to_episode_dict
from src.scheduler import ExponentialSchedule
from src.env import FourRoomEnv, StateFields
from src.featurizers import (
    StateSequenceFeaturizer,
    FeaturizerType
)
from src.replay_memory import FastReplayBuffer
from src.models.dqn import ModelType, Q_Estimator

class OptimizerType(StrEnum):
    ADAM = auto()

    @staticmethod
    def build(optimizer_type: str, model: Q_Estimator, learning_rate: float):
        assert optimizer_type in [o.value for o in OptimizerType], f"Invalid optimizer type: {optimizer_type}"
        if model.model_type == ModelType.RANDOM:
            return None
        if optimizer_type == OptimizerType.ADAM:
            return torch.optim.Adam(params=model.parameters(), lr=learning_rate) # might need to make this more flexible in the future


class DQNTeamTrainer:

    def __init__(self, imposter_optimizer, crew_optimizer, gamma):
        self.imposter_optimizer = imposter_optimizer
        self.crew_optimizer = crew_optimizer
        self.gamma = gamma

        # wether or not this trainer is just a place holder!
        self.train = imposter_optimizer is not None or crew_optimizer is not None

    def train_step(
        self,
        batch,
        featurizer,
        imposter_model,
        imposter_target_model,
        crew_model,
        crew_target_model,
    ):

        accumulated_losses = [0, 0]

        if not self.train:
            return accumulated_losses

        # reset gradents for both optimizers
        for opt in [self.imposter_optimizer, self.crew_optimizer]:
            if opt is not None:
                opt.zero_grad()

        featurizer.fit(batch.states)

        for agent_idx, (spatial, non_spatial) in enumerate(featurizer.generator()):

            # samples in which agnets is an imposter/crew member
            imposter_samples = (batch.imposters == agent_idx).view(-1)
            crew_samples = ~imposter_samples

            # training via gradient accumulation
            for loss_idx, (opt, team_samples, team_model, team_model_target) in enumerate(
                [
                    (
                        self.imposter_optimizer,
                        imposter_samples,
                        imposter_model,
                        imposter_target_model,
                    ),
                    (self.crew_optimizer, crew_samples, crew_model, crew_target_model),
                ]
            ):
                if opt is not None and team_samples.sum() > 0:
                    team_model.train()
                    action_values = team_model(
                        spatial[team_samples, :-1, :, :, :].detach().clone(),
                        non_spatial[team_samples, :-1, :].detach().clone(),
                    )
                    actions = torch.tensor(batch.actions[team_samples, -1, agent_idx])
                    values = torch.gather(
                        action_values, 1, actions.view(-1).unsqueeze(-1)
                    ).view(-1)

                    with torch.no_grad():
                        done_mask = torch.tensor(batch.dones[team_samples, -1]).view(-1)
                        rewards = torch.tensor(
                            batch.rewards[team_samples, -2, agent_idx]
                        ).view(-1)

                        target_values = (
                            rewards
                            + self.gamma
                            * torch.max(
                                team_model_target(
                                    spatial[team_samples, 1:, :, :, :].detach().clone(),
                                    non_spatial[team_samples, 1:, :].detach().clone(),
                                ),
                                dim=1,
                            )[0]
                        )
                        target_values[done_mask] = rewards[done_mask]

                    loss = F.mse_loss(values, target_values)
                    accumulated_losses[loss_idx] += loss.item()
                    loss.backward()

        # use gradients to update models
        for opt in [self.imposter_optimizer, self.crew_optimizer]:
            if opt is not None:
                opt.step()
        return accumulated_losses


def run_experiment(
    env: FourRoomEnv,
    num_steps: int,
    imposter_model_args: dict,
    crew_model_args: dict,
    imposter_model_type: ModelType = ModelType.SPATIAL_DQN,
    crew_model_type: ModelType = ModelType.SPATIAL_DQN,
    featurizer_type: FeaturizerType = FeaturizerType.GLOBAL,
    sequence_length: int = 2,
    replay_buffer_size: int = 100_000,
    replay_prepopulate_steps: int = 1000,
    batch_size: int = 32,
    gamma: float = 0.99,
    scheduler_start_eps: float = 1.0,
    scheduler_end_eps: float = 0.05,
    scheduler_time_steps: int = 1_000_000,
    train_imposter: bool = True,
    train_crew: bool = True,
    experiment_save_path: str | None = None,
    optimizer_type: OptimizerType = OptimizerType.ADAM,
    learning_rate: float = 0.0001,
    train_step_interval: int = 5,
    num_checkpoint_saves: int = 5,
):
    # initializing models
    imposter_model = ModelType.build(imposter_model_type, **imposter_model_args)
    crew_model = ModelType.build(crew_model_type, **crew_model_args)

    # initializing optimizers
    crew_optimizer = imposter_optimizer = None
    if optimizer_type is not None:
        if train_imposter:
            imposter_optimizer = OptimizerType.build(optimizer_type, imposter_model, learning_rate)
        if train_crew:
            crew_optimizer = OptimizerType.build(optimizer_type, crew_model, learning_rate)

    # initializing trainer
    trainer = DQNTeamTrainer(
        imposter_optimizer=imposter_optimizer,
        crew_optimizer=crew_optimizer,
        gamma=gamma,
    )

    # initialize scheduler
    scheduler = ExponentialSchedule(
        scheduler_start_eps, scheduler_end_eps, scheduler_time_steps
    )

    # initialize replay buffer and prepopulate it
    replay_buffer = FastReplayBuffer(
        max_size=replay_buffer_size,
        trajectory_size=sequence_length + 1, # +1 so we always fetch both state and next state from the buffer
        state_size=env.flattened_state_size,
        n_imposters=env.n_imposters,
        n_agents=env.n_agents,
    )
    replay_buffer.populate(env=env, num_steps=replay_prepopulate_steps)

    # initialize featurizer
    featurizer = FeaturizerType.build(
        featurizer_type, env=env
    )

    # run actual experiment
    training_log = train(
        env=env,
        num_steps=num_steps,
        replay_buffer=replay_buffer,
        featurizer=featurizer,
        imposter_model = imposter_model,
        crew_model = crew_model,
        save_directory_path=experiment_save_path,
        train_step_interval=train_step_interval,
        batch_size=batch_size,
        gamma=gamma,
        scheduler=scheduler,
        trainer=trainer,
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
    imposter_model: Q_Estimator,
    crew_model: Q_Estimator,
    scheduler: ExponentialSchedule,
    save_directory_path: str,
    trainer: DQNTeamTrainer,
    train_step_interval: int = 5,
    batch_size: int = 32,
    gamma: float = 0.99,
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

    # adding dummy time step to replay buffer
    replay_buffer.add_start(env.flatten_state(state), env.imposter_idxs)

    G = torch.zeros(env.n_agents)

    # Iterate for a total of `num_steps` steps
    pbar = tqdm.trange(num_steps)
    for t_total in pbar:

        # Save model
        if t_total in t_saves and trainer.train:
            percent_progress = np.round(t_total / num_steps * 100)
            imposter_model.dump_to_checkpoint(
                os.path.join(
                    save_directory_path, f"imposter_{imposter_model.model_type}_{percent_progress}%.pt"
                )
            )
            crew_model.dump_to_checkpoint(
                os.path.join(save_directory_path, f"crew_{crew_model.model_type}_{percent_progress}%.pt")
            )

        # Update Target DQNs
        if t_total % 10_000 == 0:
            # TODO: is this the best way to do this?
            imposter_target_model = copy.deepcopy(imposter_model)
            crew_target_model = copy.deepcopy(crew_model)

        # featurizing current trajectory
        featurizer.fit(replay_buffer.get_last_trajectory().states)

        # getting next action
        eps = scheduler.value(t_total)
        agent_actions = np.zeros(env.n_agents, dtype=np.int32)
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
                        torch.argmax(
                            imposter_model(spatial[:, 1:, :, :, :], non_spatial[:, 1:, :])
                        )
                    )

            # choose action for alive crew member
            elif alive_agents[agent_idx]:
                if np.random.random() <= eps:
                    agent_actions[agent_idx] = np.random.randint(0, env.n_crew_actions)
                else:
                    agent_actions[agent_idx] = int(
                        torch.argmax(
                            crew_model(spatial[:, 1:, :, :, :], non_spatial[:, 1:, :])
                        )
                    )

        next_state, reward, done, trunc, info = env.step(agent_actions=agent_actions)
        G = G * gamma + reward
        add_info_to_episode_dict(episode_info_dict, info)

        # adding the timestep to replay buffer
        replay_buffer.add(
            state=env.flatten_state(state),
            action=agent_actions,
            reward=reward,
            done=done,
            imposters=env.imposter_idxs,
            is_start=False,
        )

        # Training update for imposters and/or crew
        if t_total % train_step_interval == 0:

            # get sample of trajectories to train on
            batch = replay_buffer.sample(batch_size)

            step_losses = trainer.train_step(
                batch=batch,
                featurizer=featurizer,
                imposter_model = imposter_model,
                imposter_target_model=imposter_target_model,
                crew_model=crew_model,
                crew_target_model=crew_target_model,
            )

            losses.append(step_losses)

        # checking if the env needs to be reset
        if done or trunc:

            pbar.set_description(
                f"Episode: {i_episode} | Steps: {t_episode + 1} | Epsilon: {eps:4.2f} | Imposter Loss: {losses[-1][0]:4.2f} | Crew Loss: {losses[-1][1]:4.2f}"
            )

            # resetting episode
            returns.append(G.tolist())
            game_lengths.append(t_episode)
            G = torch.zeros(env.n_agents)
            t_episode = 0
            i_episode += 1
            info_list.append(dict(episode_info_dict))
            episode_info_dict = defaultdict(list)

            state, info = env.reset()
            add_info_to_episode_dict(episode_info_dict, info)
            replay_buffer.add_start(env.flatten_state(state), env.imposter_idxs)

        else:
            state = next_state
            t_episode += 1

    # saving final model states
    imposter_model.dump_to_checkpoint(
        os.path.join(save_directory_path, f"imposter_dqn_100%.pt")
    )
    crew_model.dump_to_checkpoint(os.path.join(save_directory_path, f"crew_dqn_100%.pt"))

    return info_list, returns, losses
