"""
Structured logging for training.
Logs to both console and file.
"""
import os
from datetime import datetime
from typing import Optional


class Logger:
    """
    Simple logger that writes to both console and file.
    """

    def __init__(self, log_dir: str = "data/logs", experiment_name: Optional[str] = None):
        """
        Args:
            log_dir: Directory to store log files
            experiment_name: Optional name for this experiment
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            filename = f"{experiment_name}_{timestamp}.log"
        else:
            filename = f"train_{timestamp}.log"

        self.log_file = os.path.join(log_dir, filename)

        # Write header
        self.log(f"{'='*60}")
        self.log(f"AutoConjecture Training Log")
        self.log(f"Started: {datetime.now().isoformat()}")
        self.log(f"{'='*60}\n")

    def log(self, message: str, level: str = "INFO"):
        """
        Log a message.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"

        # Print to console
        print(formatted_msg)

        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(formatted_msg + '\n')

    def info(self, message: str):
        """Log info message."""
        self.log(message, "INFO")

    def warning(self, message: str):
        """Log warning message."""
        self.log(message, "WARNING")

    def error(self, message: str):
        """Log error message."""
        self.log(message, "ERROR")

    def section(self, title: str):
        """Log a section header."""
        self.log(f"\n{'='*60}")
        self.log(title)
        self.log(f"{'='*60}")

    def subsection(self, title: str):
        """Log a subsection header."""
        self.log(f"\n{'-'*40}")
        self.log(title)
        self.log(f"{'-'*40}")

    def metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value."""
        if step is not None:
            self.log(f"Metric[{step}] {name} = {value:.4f}")
        else:
            self.log(f"Metric {name} = {value:.4f}")
