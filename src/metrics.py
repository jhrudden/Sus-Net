from enum import StrEnum, auto
import json
from typing import Any, Dict, List
import json


class SusMetrics(StrEnum):
    IMP_KILLED_CREW = auto()
    IMP_VOTED_OUT = auto()
    CREW_VOTED_OUT = auto()
    SABOTAGED_JOBS = auto()
    COMPLETED_JOBS = auto()
    TOTAL_STALEMATES = auto()
    TOTAL_TIME_STEPS = auto()
    IMPOSTER_WON = auto()
    CREW_WON = auto()
    AVG_CREW_RETURNS = auto()
    AVG_IMPOSTER_RETURNS = auto()
    AVG_CREW_LOSS = auto()
    AVG_IMPOSTER_LOSS = auto()

    @classmethod
    def can_increment(cls, metric: str):
        return metric in [
            SusMetrics.IMP_KILLED_CREW,
            SusMetrics.IMP_VOTED_OUT,
            SusMetrics.CREW_VOTED_OUT,
            SusMetrics.SABOTAGED_JOBS,
            SusMetrics.COMPLETED_JOBS,
            SusMetrics.TOTAL_STALEMATES,
            SusMetrics.TOTAL_TIME_STEPS,
        ]


class EnvMetricHandler:
    def __init__(self):
        self.metrics = {metric: 0 for metric in SusMetrics}

    def increment(self, event, amount=1) -> None:
        """
        Increment the metric by the specified amount

        Args:
        - event (Metrics): The metric to increment
        """
        if SusMetrics.can_increment(event):
            self.metrics[event] += amount
        else:
            raise ValueError(f"Invalid metric: {event}")

    def update(self, event: SusMetrics, value: Any) -> None:
        if event not in SusMetrics:
            raise ValueError(f"Invalid metric: {event}")
        self.metrics[event] = value

    def reset(self) -> None:
        for key in self.metrics.keys():
            self.metrics[key] = 0

    def get_metrics(self) -> Dict[SusMetrics, int]:
        return {metric: self.metrics[metric] for metric in SusMetrics}

    def __repr__(self):
        return json.dumps(self.metrics, indent=4)


class EpisodicMetricHandler:
    """
    Averages metrics across multiple episodes
    """

    def __init__(self):
        self.metrics = {metric: [] for metric in SusMetrics}

    def step(self, metrics: Dict[SusMetrics, int]) -> None:
        for metric, value in metrics.items():
            self.metrics[metric].append(value)
    
    def set(self, metrics: Dict[SusMetrics, Any]) -> None:
        for metric, values in metrics.items():
            assert any(m.value == metric for m in SusMetrics), f"Invalid metric: {metric}"
            self.metrics[metric] = values

    def compute(self) -> Dict[SusMetrics, float]:
        return {
            metric: sum(values) / len(values) for metric, values in self.metrics.items()
        }

    def save_metrics(self, save_file_path):
        with open(save_file_path, "w") as f:
            json.dump(self.metrics, f)

    def load_metrics(self, metrics_file_path):
        with open(metrics_file_path, "r") as f:
            self.metrics = json.load(f)
