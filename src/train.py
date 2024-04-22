from enum import StrEnum, auto
import os
import numpy as np
import pygame
import torch
import torch.nn.functional as F
import copy
import tqdm
from src.scheduler import ExponentialSchedule
from src.env import FourRoomEnv, StateFields
from src.featurizers import StateSequenceFeaturizer, FeaturizerType
from src.metrics import EpisodicMetricHandler, SusMetrics
from src.replay_memory import FastReplayBuffer
from src.models.dqn import ModelType, Q_Estimator
from src.visualize import AmongUsVisualizer


class OptimizerType(StrEnum):
    ADAM = auto()

    @staticmethod
    def build(optimizer_type: str, model: Q_Estimator, learning_rate: float):
        assert optimizer_type in [
            o.value for o in OptimizerType
        ], f"Invalid optimizer type: {optimizer_type}"
        if model.model_type == ModelType.RANDOM:
            return None
        if optimizer_type == OptimizerType.ADAM:
            return torch.optim.Adam(
                params=model.parameters(), lr=learning_rate
            )  # might need to make this more flexible in the future


class DQNTeamTrainer:

    def __init__(self, imposter_optimizer, crew_optimizer, gamma):
        self.imposter_optimizer = imposter_optimizer
        self.crew_optimizer = crew_optimizer
        self.gamma = gamma

        # whether or not this trainer is just a place holder!
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

        # for i, p in enumerate(imposter_model.cnn.parameters()):
        #     if i == 0:
        #         print("before ")
        #         print(p.grad)

        featurizer.fit(batch.states)

        for agent_idx, (spatial, non_spatial) in enumerate(featurizer.generator()):

            # samples in which agnets is an imposter/crew member
            imposter_samples = (batch.imposters == agent_idx).view(-1)
            crew_samples = ~imposter_samples

            # spatial.requires_grad = non_spatial.requires_grad = True

            # training via gradient accumulation
            for loss_idx, (
                opt,
                team_samples,
                team_model,
                team_model_target,
            ) in enumerate(
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
                    # compute the value of the actions taken by the agent (gradients are calculated here!)

                    # print("imposters ", batch.imposters)

                    # assert (
                    #     spatial[batch.imposters[0], :-1, :, :, :]
                    #     != spatial[batch.imposters[1], :-1, :, :, :]
                    # )
                    print("team samples: ", {team_samples})
                    print("NON SPACIAL!!!")
                    print(
                        non_spatial[:, :-1, :],
                    )

                    action_values = team_model(
                        spatial[team_samples, :-1, :, :, :],
                        non_spatial[team_samples, :-1, :],
                    )

                    print("Action values")
                    print(action_values)
                    actions = torch.tensor(batch.actions[team_samples, -2, agent_idx])
                    print("Actions form train step")
                    print(actions)

                    values = torch.gather(
                        action_values, 1, actions.view(-1).unsqueeze(-1)
                    ).view(-1)

                    print("Actions value")
                    print(values)

                    with torch.no_grad():
                        done_mask = batch.dones[team_samples, -2].view(-1)

                        rewards = torch.tensor(
                            batch.rewards[team_samples, -2, agent_idx]
                        ).view(-1)

                        print(f"Bitch rewards: {rewards}")

                        print(f"Bitch Dones: {done_mask}")

                        # calculate target values, no gradients here (notice the detach() calls
                        target_values = (
                            rewards
                            + self.gamma
                            * torch.max(
                                team_model_target(
                                    spatial[team_samples, 1:, :, :, :].detach(),
                                    non_spatial[team_samples, 1:, :].detach(),
                                ),
                                dim=1,
                            )[0]
                        )
                        target_values[done_mask] = rewards[done_mask]

                    loss = F.mse_loss(values, target_values)
                    loss.backward()
                    accumulated_losses[loss_idx] += loss.item()

        for i, p in enumerate(imposter_model.parameters()):
            print("after ")
            print(p.grad.sum())

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
            imposter_optimizer = OptimizerType.build(
                optimizer_type, imposter_model, learning_rate
            )
        if train_crew:
            crew_optimizer = OptimizerType.build(
                optimizer_type, crew_model, learning_rate
            )

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

    # initialize metric handlers
    metrics = EpisodicMetricHandler()

    # initialize replay buffer and prepopulate it
    replay_buffer = FastReplayBuffer(
        max_size=replay_buffer_size,
        trajectory_size=sequence_length
        + 1,  # +1 so we always fetch both state and next state from the buffer
        state_size=env.flattened_state_size,
        n_imposters=env.n_imposters,
        n_agents=env.n_agents,
    )
    replay_buffer.populate(env=env, num_steps=replay_prepopulate_steps)

    # initialize featurizer
    featurizer = FeaturizerType.build(featurizer_type, env=env)

    # run actual experiment
    all_metrics, _, __ = train(
        env=env,
        metrics=metrics,
        num_steps=num_steps,
        replay_buffer=replay_buffer,
        featurizer=featurizer,
        imposter_model=imposter_model,
        crew_model=crew_model,
        save_directory_path=experiment_save_path,
        train_step_interval=train_step_interval,
        batch_size=batch_size,
        gamma=gamma,
        scheduler=scheduler,
        trainer=trainer,
        num_saves=num_checkpoint_saves,
    )

    avg_metrics = metrics.compute()

    print(f"Average Metrics: {avg_metrics}")

    # run experiment
    if experiment_save_path is not None:
        metrics.save_metrics(save_file_path=experiment_save_path / "metrics.json")
    
    return all_metrics


def train(
    env: FourRoomEnv,
    metrics: EpisodicMetricHandler,
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

    # Initialize structures to store the models at different stages of training
    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)

    i_episode = 0  # Use this to indicate the index of the current episode
    t_episode = 0  # Use this to indicate the time-step inside current episode

    state, info = env.reset()  # Initialize state of first episode

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
                    save_directory_path,
                    f"imposter_{imposter_model.model_type}_{percent_progress}%.pt",
                )
            )
            crew_model.dump_to_checkpoint(
                os.path.join(
                    save_directory_path,
                    f"crew_{crew_model.model_type}_{percent_progress}%.pt",
                )
            )

        # Update Target DQNs
        if t_total % 1000 == 0:
            # TODO: is this the best way to do this?
            imposter_target_model = copy.deepcopy(imposter_model)
            crew_target_model = copy.deepcopy(crew_model)

        # featurizing current trajectory
        featurizer.fit(replay_buffer.get_last_trajectory().states)

        # getting next action
        eps = scheduler.value(t_total)
        agent_actions = np.zeros(env.n_agents, dtype=np.int32)
        alive_agents = state[env.state_fields[StateFields.ALIVE_AGENTS]]

        with torch.no_grad():
            for agent_idx, (spatial, non_spatial) in enumerate(featurizer.generator()):

                # choose action for alive imposter
                if env.imposter_mask[agent_idx] and alive_agents[agent_idx]:

                    print(
                        imposter_model(spatial[:, 1:, :, :, :], non_spatial[:, 1:, :])
                    )

                    if np.random.random() <= eps:
                        agent_actions[agent_idx] = np.random.randint(
                            0, env.n_imposter_actions
                        )
                    else:
                        agent_actions[agent_idx] = int(
                            torch.argmax(
                                imposter_model(
                                    spatial[:, 1:, :, :, :], non_spatial[:, 1:, :]
                                )
                            )
                        )

                # choose action for alive crew member
                elif alive_agents[agent_idx]:
                    if np.random.random() <= eps:
                        agent_actions[agent_idx] = np.random.randint(
                            0, env.n_crew_actions
                        )
                    else:
                        agent_actions[agent_idx] = int(
                            torch.argmax(
                                crew_model(
                                    spatial[:, 1:, :, :, :], non_spatial[:, 1:, :]
                                )
                            )
                        )

        next_state, reward, done, trunc, info = env.step(agent_actions=agent_actions)
        G = G * gamma + reward

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
                imposter_model=imposter_model,
                imposter_target_model=imposter_target_model,
                crew_model=crew_model,
                crew_target_model=crew_target_model,
            )

            losses.append(step_losses)

        # padding terminal sequence with dummy time step
        if done:
            replay_buffer.add_terminal(env.flatten_state(next_state), env.imposter_idxs)

        # checking if the env needs to be reset
        if done or trunc:

            imposter_return = G[env.imposter_mask].mean().item()
            crew_return = G[~env.imposter_mask].mean().item()

            info_with_returns = {
                **info,
                SusMetrics.AVG_IMPOSTER_RETURN: imposter_return,
                SusMetrics.AVG_CREW_RETURN: crew_return,
            }

            # update metrics
            metrics.step(info_with_returns)

            pbar.set_description(
                f"Episode: {i_episode} | Steps: {t_episode + 1} | Epsilon: {eps:4.2f} | Imposter Loss: {losses[-1][0]:4.2f} | Crew Loss: {losses[-1][1]:4.2f}"
            )

            # resetting episode
            returns.append(G.tolist())
            game_lengths.append(t_episode)
            G = torch.zeros(env.n_agents)
            t_episode = 0
            i_episode += 1

            state, _ = env.reset()
            replay_buffer.add_start(env.flatten_state(state), env.imposter_idxs)

        else:
            state = next_state
            t_episode += 1

    # saving final model states
    imposter_model.dump_to_checkpoint(
        os.path.join(save_directory_path, f"imposter_dqn_100%.pt")
    )
    crew_model.dump_to_checkpoint(
        os.path.join(save_directory_path, f"crew_dqn_100%.pt")
    )

    return metrics.metrics, returns, losses

def run_game(
    env: FourRoomEnv,
    imposter_model: Q_Estimator,
    crew_model: Q_Estimator,
    featurizer: StateSequenceFeaturizer,
    sequence_length: int = 2,
):
     with AmongUsVisualizer(env) as visualizer:
        state, _ = visualizer.reset()
        replay_memory = FastReplayBuffer(1_000, 4, env.flattened_state_size, env.n_agents, env.n_imposters)
        replay_memory.add_start(env.flatten_state(state), env.imposter_idxs)


        stop_game = False
        done = False
        paused = False
        while not stop_game:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    stop_game = True
                    break
            
            if not done and not paused:
                state = replay_memory.get_last_trajectory().states
                featurizer.fit(state)

                actions = []

                for agent_idx, (agent_spatial, agent_non_spatial) in enumerate(featurizer.generator()):
                    if agent_idx in env.imposter_idxs:
                        action = imposter_model(agent_spatial, agent_non_spatial).argmax().item()
                    else:
                        action = crew_model(agent_spatial, agent_non_spatial).argmax().item()
                    
                    actions.append(action)
                
                next_state, reward, done, truncated, _ = visualizer.step(actions)

                replay_memory.add(
                    env.flatten_state(next_state), 
                    actions,
                    reward, 
                    done, 
                    env.imposter_idxs)
            
            pygame.time.wait(1000)
        visualizer.close()

        