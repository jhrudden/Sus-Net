from typing import Dict, Optional, Tuple
import numpy as np
from gymnasium import spaces

from src.environment.base import FourRoomEnv, StateFields, Action
from src.metrics import SusMetrics


class FourRoomEnvWithTagging(FourRoomEnv):
    def __init__(
        self, *args, tag_reset_interval: int = 50, vote_reward: int = 3, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.state_fields = {
            field: idx
            for idx, field in enumerate(
                [
                    StateFields.AGENT_POSITIONS,
                    StateFields.JOB_POSITIONS,
                    StateFields.JOB_STATUS,
                    StateFields.ALIVE_AGENTS,
                    StateFields.USED_TAGS,
                    StateFields.TAG_COUNTS,
                    StateFields.TAG_RESET_COUNT,
                ]
            )
        }
        self.tag_counts = np.zeros(self.n_agents, dtype=int)
        self.used_tag_actions = np.zeros(self.n_agents, dtype=bool)
        self.tag_reset_timer = 0
        self.tag_reset_interval = tag_reset_interval
        self.vote_reward = vote_reward

        self.n_imposter_actions = self.n_imposter_actions + self.n_agents - 1
        self.n_crew_actions = self.n_crew_actions + self.n_agents - 1

        self.action_space = spaces.Discrete(
            len(Action) + self.n_agents
        )  # Add tagging action (1 for each agent)

        self.observation_space = spaces.Tuple(
            (
                spaces.Box(
                    low=0, high=self.n_rows, shape=(self.n_agents, 2), dtype=int
                ),  # Agent positions
                spaces.MultiBinary(self.n_agents),  # Alive agents
                spaces.Box(
                    low=0, high=self.n_rows, shape=(self.n_jobs, 2), dtype=int
                ),  # Job positions
                spaces.MultiBinary(self.n_jobs),  # Completed jobs
                spaces.MultiBinary(self.n_agents),  # Who has used their tag
                spaces.Box(
                    low=0, high=self.n_agents, shape=(self.n_agents,), dtype=int
                ),  # Tag counts
                spaces.Box(
                    low=1, high=self.tag_reset_interval, shape=(1,), dtype=int
                ),  # Time left for tag reset
            )
        )

    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Tuple, Dict]:
        state, _ = super().reset(seed, **kwargs)
        self.tag_counts = np.zeros(self.n_agents, dtype=int)
        self.used_tag_actions = np.zeros(self.n_agents, dtype=bool)
        self.tag_reset_timer = 0

        # updating agent_action_map to include tagging actions
        for agent_idx in range(self.n_agents):
            tag_actions = np.arange(self.n_agents)[
                np.arange(self.n_agents) != agent_idx
            ]
            self.agent_action_map[agent_idx] = np.hstack(
                [self.agent_action_map[agent_idx], tag_actions]
            )

        self.logger.debug(
            f"""
New Game Started!
-----------------
    Agent Positions: {list(map(tuple, self.agent_positions))}
    Imposters: {np.argwhere(self.imposter_mask).flatten()}
    Crew Members: {np.argwhere(~self.imposter_mask).flatten()}
    Alive Agents: {self.alive_agents}
    Job Positions: {list(map(tuple, self.job_positions))}
    Completed Jobs: {self.completed_jobs}
    Tag Counts: {self.tag_counts}
    Used Tag Actions: {self.used_tag_actions}
    Time Left for Tag Reset: {self.tag_reset_interval - self.tag_reset_timer}
-----------------
        """
        )

        state = (
            *state,
            self.used_tag_actions,
            self.tag_counts,
            self.tag_reset_interval - self.tag_reset_timer,
        )

        return state, {}

    def _agent_tag(self, agent_idx, agent_tagged):
        """Can only tag someone if your tag is unused and if the tagged agent is alive."""
        if (
            self.used_tag_actions[agent_idx] == 0
            and self.alive_agents[agent_tagged] > 0
        ):
            self.tag_counts[agent_tagged] += 1
            self.used_tag_actions[agent_idx] = 1

            self.logger.debug(
                f"""Agent {agent_idx} ({self.agent_positions[agent_idx]}) tagged Agent {agent_tagged} ({self.agent_positions[agent_tagged]})! {agent_tagged}'s new tag count: {self.tag_counts[agent_tagged]}"""
            )
        else:
            self.logger.debug(
                f"""Agent {agent_idx} tried to tag Agent {agent_tagged} but failed!"""
            )

    def step(self, agent_actions):
        """
        Executes a step in the environment by applying the actions of all agents and updating the environment's state accordingly.

        This function processes each agent's action in a specified or random order, updates the agents' positions,
        handles interactions such as kills, fixes, tagging, and sabotages, and determines whether the game reaches a
        terminal state based on the conditions of alive agents or completed objectives.

        Parameters:
        - agent_actions (list or dict): A collection of actions for each agent to take during this step.
            The actions should be indexed by the agent's index in the list.

        Returns:
        - tuple containing:
            - A tuple of (agent_position, job_positions, completed_jobs, alive_agents, used_tag_actions, agent_tag_count, time_till_vote_reset) reflecting the new state of the environment.
            - agent_rewards (numpy.ndarray): An array of rewards received by each agent during this step.
            - done (bool): A flag indicating whether the game has reached a terminal state.
            - truncated (bool): A flag indicating whether the episode was truncated (not applicable in this context, but included for API consistency).
            - info (dict): An empty dictionary that could be used for debugging or logging additional information in the future.

        Side Effects:
        - Updates `self.agent_positions` based on the actions that involve movement. Also updates `self.tag_reset_timer` at the start of the step.
        - Modifies `self.completed_jobs`, `self.tag_counts`, `self.used_tag_action` and `self.alive_agents` based on actions that involve fixing, sabotaging, tagging or killing.
        - Alters `self.agent_rewards` to reflect the rewards accumulated by each agent during this step.
        """
        assert (
            len(agent_actions) == self.n_agents
        ), f"Expected {self.n_agents} actions, got {len(agent_actions)}"
        assert all(
            action < self.action_space.n for action in agent_actions
        ), f"Invalid action(s) {agent_actions}"

        self.metrics.increment(SusMetrics.TOTAL_TIME_STEPS, 1)

        # kill agents who have been tagged too many times (tag count > half the number of agents)

        truncated = False
        done = False
        reset_tag_counts = False
        team_reward = 0

        # initialize the agent reward array before computing all agent rewards
        self.agent_rewards = np.ones(self.n_agents) * self.time_step_reward

        # getting the order in which agent actions will be performed
        agent_action_order = list(range(self.n_agents))
        if self.is_action_order_random:
            np.random.shuffle(agent_action_order)

        # perform action for each agent
        for agent_idx in agent_action_order:

            agent_action = self.agent_action_map[agent_idx][agent_actions[agent_idx]]

            if isinstance(agent_action, int):  # this is a tag action
                self._agent_tag(agent_idx=agent_idx, agent_tagged=agent_action)

            else:
                self._agent_step(agent_idx=agent_idx, agent_action=agent_action)

        self.tag_counts *= self.alive_agents  # reset tag counts for dead agents

        self.tag_reset_timer += 1

        if self.tag_reset_timer >= self.tag_reset_interval:
            highest_vote_idx = np.argmax(self.tag_counts)
            highest_vote = self.tag_counts[highest_vote_idx]

            quorum = (self.alive_agents.sum() + 1) // 2

            if highest_vote >= quorum:
                self.alive_agents[highest_vote_idx] = (
                    0  # kick out the agent with the highest vote
                )
                is_imposter = self.imposter_mask[highest_vote_idx]
                # Reward Crew if Imposter is voted out, else reward Imposters (team reward penalizes the team that lost a member)
                team_reward += self.vote_reward * (-1 if is_imposter else 1)

                if is_imposter:
                    self.metrics.increment(SusMetrics.IMP_VOTED_OUT, 1)
                else:
                    self.metrics.increment(SusMetrics.CREW_VOTED_OUT, 1)

                self.logger.debug(
                    f"""Agent {highest_vote_idx} got voted OUT! Tag Count / Alive Agents: {highest_vote} / {self.alive_agents.sum()}"""
                )

            self._reset_tagging_state()

        team_win, win_team_reward = self.check_win_condition()
        team_reward += win_team_reward
        done = done or team_win

        self.agent_rewards = self._merge_rewards(self.agent_rewards, team_reward)

        if self.t == self.max_time_steps - 1:
            truncated = True
        else:
            self.t += 1

        return (
            (
                self.agent_positions,
                self.alive_agents,
                self.job_positions,
                self.completed_jobs,
                self.used_tag_actions,  # Who has used their tag
                self.tag_counts,  # Tag counts
                self.tag_reset_interval
                - self.tag_reset_timer,  # Time left for tag reset
            ),
            self.agent_rewards,
            done,
            truncated,
            self.metrics.get_metrics(),
        )

    def _reset_tagging_state(self):
        self.tag_counts = np.zeros(self.n_agents, dtype=int)
        self.used_tag_actions = np.zeros(self.n_agents, dtype=bool)
        self.tag_reset_timer = 0
        self.logger.debug("Tagging state reset!")

    def compute_action(self, agent_idx, action_idx):
        if action_idx < len(Action):
            return str(Action(action_idx))
        else:
            players = np.arange(self.n_agents)
            player = players[players != agent_idx][action_idx - len(Action)]
            return f"Vote Player {player}"