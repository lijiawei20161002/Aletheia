"""
Metrics tracking for training.
Computes and stores training metrics over time.
"""
from typing import List, Dict
from collections import defaultdict
import json
import os


class MetricsTracker:
    """
    Tracks metrics over training.
    """

    def __init__(self, save_path: str = "data/logs/metrics.json"):
        """
        Args:
            save_path: Path to save metrics
        """
        self.save_path = save_path
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.steps: List[int] = []

    def log_metric(self, name: str, value: float, step: int):
        """
        Log a metric value at a specific step.

        Args:
            name: Metric name
            value: Metric value
            step: Training step
        """
        if not self.steps or self.steps[-1] < step:
            self.steps.append(step)

        self.metrics[name].append(value)

    def log_metrics(self, metrics_dict: Dict[str, float], step: int):
        """Log multiple metrics at once."""
        if not self.steps or self.steps[-1] < step:
            self.steps.append(step)

        for name, value in metrics_dict.items():
            self.metrics[name].append(value)

    def get_metric(self, name: str) -> List[float]:
        """Get all values for a metric."""
        return self.metrics.get(name, [])

    def get_latest(self, name: str) -> float:
        """Get latest value for a metric."""
        values = self.metrics.get(name, [])
        return values[-1] if values else 0.0

    def get_average(self, name: str, last_n: int = 10) -> float:
        """Get average of last N values for a metric."""
        values = self.metrics.get(name, [])
        if not values:
            return 0.0
        relevant = values[-last_n:]
        return sum(relevant) / len(relevant)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "latest": values[-1],
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        return summary

    def save(self):
        """Save metrics to file."""
        data = {
            "steps": self.steps,
            "metrics": dict(self.metrics),
            "summary": self.summary()
        }

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load metrics from file."""
        if not os.path.exists(self.save_path):
            return

        with open(self.save_path, 'r') as f:
            data = json.load(f)

        self.steps = data.get("steps", [])
        self.metrics = defaultdict(list, data.get("metrics", {}))

    def print_summary(self):
        """Print summary of all metrics."""
        print("\nMetrics Summary:")
        print("-" * 60)
        summary = self.summary()
        for name, stats in summary.items():
            print(f"{name}:")
            print(f"  Latest: {stats['latest']:.4f}")
            print(f"  Mean:   {stats['mean']:.4f}")
            print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
        print("-" * 60)
