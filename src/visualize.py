from matplotlib import pyplot as plt
import numpy as np
from src.featurizers import SequenceStateFeaturizer

class SequenceStateVisualizer:
    def __init__(self, featurizer: SequenceStateFeaturizer, cmap='Blues'):
        self.featurizer = featurizer
        self.cmap = cmap
    
    def visualize_state(self, sequence_idx: int, ax: plt.Axes = None):
        spatial, states = self.featurizer.spatial, self.featurizer.states
        player_positions = states[sequence_idx][0]
        
        n_agents, n_rows, n_cols = spatial[sequence_idx, ...].shape

        assert n_rows == n_cols, "Only supports square grids"

        if ax is None:
            _, ax = plt.subplots(1, n_agents, figsize=(5 * n_agents, 5))

        cells = np.arange(n_rows)
        ticks = cells - 0.5

        imposters = set(self.featurizer.imposter_locations[sequence_idx].tolist())

        for agent_idx in range(n_agents):
            is_imposter = agent_idx in imposters
            role = "Imposter" if is_imposter else "Crewmate"
            agent_position = player_positions[agent_idx]

            rotated = np.flipud(spatial[sequence_idx, agent_idx].t().numpy())


            ax[agent_idx].imshow(rotated, cmap=self.cmap)
            ax[agent_idx].grid(True)
            ax[agent_idx].set_xticks(ticks, minor=False)
            ax[agent_idx].set_yticks(ticks, minor=False)
            ax[agent_idx].set_xticklabels([], minor=False)
            ax[agent_idx].set_yticklabels([], minor=False)
            ax[agent_idx].tick_params(axis=u'both', which=u'both', length=0)
            ax[agent_idx].set_title(f"{role} {agent_idx} - {agent_position}")

            for y in cells:
                for x in cells:
                    is_agent = x == player_positions[agent_idx][0] and y == player_positions[agent_idx][1]
                    ax[agent_idx].text(x, n_rows - y - 1, str((x,y)), va='center', ha='center',fontsize=8, color='white' if is_agent else 'black')


    def visualize_sequence(self):
        spatial, states = self.featurizer.spatial, self.featurizer.states
        states = [s[0] for s in states]
        seq_len, n_agents, _, __ = spatial.size()

        _, ax = plt.subplots(seq_len, n_agents, figsize=(n_agents * 5, seq_len * 5))

        for seq_idx in range(seq_len):
            # add title to row
            label = "S$_{t" + ("-" + str(seq_idx) if seq_idx > 0 else "") + "}$"
            ax[seq_idx][0].set_ylabel(label, rotation=0, labelpad=40, fontsize=20)
            self.visualize_state(seq_idx, ax[seq_idx])

        plt.tight_layout()
        plt.show()