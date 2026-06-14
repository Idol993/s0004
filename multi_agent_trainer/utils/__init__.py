"""通用工具模块"""
import os
import time
import random
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import torch


class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"
    MEMORY = "memory"
    REFLECTOR = "reflector"


class MessageType(str, Enum):
    TASK = "task"
    PLAN = "plan"
    ACTION = "action"
    OBSERVATION = "observation"
    EVALUATION = "evaluation"
    MEMORY_QUERY = "memory_query"
    MEMORY_RESULT = "memory_result"
    REFLECTION = "reflection"
    ROLLBACK_REQUEST = "rollback_request"
    HEARTBEAT = "heartbeat"
    TERMINATE = "terminate"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class TrainingMetrics:
    total_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    communication_time: float = 0.0
    overhead_time: float = 0.0
    num_messages: int = 0
    num_episodes: int = 0
    success_count: int = 0
    failure_count: int = 0

    @property
    def overhead_ratio(self) -> float:
        if self.total_time == 0:
            return 0.0
        return (self.overhead_time + self.communication_time) / self.total_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time": self.total_time,
            "forward_time": self.forward_time,
            "backward_time": self.backward_time,
            "communication_time": self.communication_time,
            "overhead_time": self.overhead_time,
            "overhead_ratio": self.overhead_ratio,
            "num_messages": self.num_messages,
            "num_episodes": self.num_episodes,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "cuda") -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_str)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logger(name: str, log_dir: str = "./logs", level: str = "INFO") -> logging.Logger:
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
        fh.setLevel(getattr(logging, level.upper()))
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:16]


class Timer:
    def __init__(self):
        self._start: Optional[float] = None
        self._elapsed: float = 0.0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        if self._start is not None:
            self._elapsed += time.time() - self._start
            self._start = None

    @property
    def elapsed(self) -> float:
        if self._start is not None:
            return self._elapsed + (time.time() - self._start)
        return self._elapsed

    def reset(self) -> None:
        self._start = None
        self._elapsed = 0.0
