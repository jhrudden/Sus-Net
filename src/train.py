from src.env import FourRoomEnv
from src.featurizers import SequenceStateFeaturizer
from src.replay_memory import FastReplayBuffer

def train_networks(
        env: FourRoomEnv,
        num_steps: int,
        *,
        sequence_length: int = 5,
        replay_buffer_size: int = 1000,
        replay_prepopulate_steps: int = 100,
        batch_size: int = 32,
        gamma: float = 0.99,
        scheduler,
):
    # setup models and optimizers

    # setup replay buffer
    replay_buffer = FastReplayBuffer(
        max_size=replay_buffer_size,
        trajectory_size=sequence_length,
        state_size=env.flattened_state_size,
        n_agents=env.n_agents,
        n_imposters=env.n_imposters,
    )

    # populate replay buffer
    replay_buffer.populate(env, replay_prepopulate_steps)
    
    raise NotImplementedError("Implement me!")