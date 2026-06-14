"""智能体基类 - 定义所有智能体的通用接口和行为"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import torch

from ..utils import AgentRole, MessageType, Timer, setup_logger
from ..communication import Message, MessageFactory, CentralMessageBus
from .llm_backbone import LLMConfig, CheckpointInfo


@dataclass
class ActionRecord:
    action_id: str
    agent_id: str
    agent_role: AgentRole
    step: int
    episode_id: str
    observation: Dict[str, Any]
    action: Dict[str, Any]
    reward: Optional[float] = None
    hidden_state: Optional[torch.Tensor] = None
    timestamp: float = field(default_factory=time.time)
    log_prob: Optional[float] = None


class BaseAgent(ABC):
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        llm_config: LLMConfig,
        message_bus: CentralMessageBus,
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
    ):
        self.agent_id = agent_id
        self.role = role
        self.llm_config = llm_config
        self.message_bus = message_bus
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.logger = setup_logger(f"agent_{agent_id}", log_dir)
        from .llm_backbone import LLMBackbone
        self.llm = LLMBackbone(llm_config, agent_id=agent_id, log_dir=log_dir)

        self._action_history: List[ActionRecord] = []
        self._state_snapshots: Dict[int, Dict[str, Any]] = {}
        self._current_episode: str = ""
        self._current_step: int = 0
        self._is_active: bool = False
        self._lock = threading.RLock()
        self._message_handler: Optional[Callable[[Message], None]] = None
        self._pending_messages: List[Message] = []
        self._local_reward: float = 0.0

        self.message_bus.register_agent(agent_id)
        self.message_bus.subscribe(
            agent_id,
            msg_type=MessageType.HEARTBEAT,
            callback=self._on_message_received,
        )
        self.message_bus.subscribe(
            agent_id,
            msg_type=MessageType.TERMINATE,
            callback=self._on_message_received,
        )

        self.logger.info(f"Agent '{agent_id}' ({role.value}) initialized")

    def _on_message_received(self, msg: Message) -> None:
        with self._lock:
            self._pending_messages.append(msg)

    def process_messages(self, timeout: float = 0.1) -> List[Message]:
        messages = []
        msg = self.message_bus.receive(self.agent_id, timeout=timeout)
        while msg is not None:
            messages.append(msg)
            msg = self.message_bus.receive(self.agent_id, timeout=0.01)

        with self._lock:
            messages.extend(self._pending_messages)
            self._pending_messages.clear()

        return messages

    def send_message(
        self,
        receiver: str,
        msg_type: MessageType,
        content: Dict[str, Any],
        requires_response: bool = False,
        in_response_to: Optional[str] = None,
    ) -> Message:
        msg = MessageFactory.create(
            sender=self.agent_id,
            receiver=receiver,
            msg_type=msg_type,
            content=content,
            episode_id=self._current_episode,
            step=self._current_step,
            requires_response=requires_response,
            in_response_to=in_response_to,
        )
        self.message_bus.send(msg)
        return msg

    def broadcast(self, msg_type: MessageType, content: Dict[str, Any]) -> Message:
        msg = MessageFactory.create(
            sender=self.agent_id,
            receiver="*",
            msg_type=msg_type,
            content=content,
            episode_id=self._current_episode,
            step=self._current_step,
        )
        self.message_bus.broadcast(msg)
        return msg

    @abstractmethod
    def perceive(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def decide(self, perceived_state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        perceived = self.perceive(observation)
        decision = self.decide(perceived)
        action_result = self.act(decision)

        record = ActionRecord(
            action_id=f"{self.agent_id}_ep{self._current_episode}_st{self._current_step}",
            agent_id=self.agent_id,
            agent_role=self.role,
            step=self._current_step,
            episode_id=self._current_episode,
            observation=observation,
            action=action_result,
        )
        self._action_history.append(record)

        self._state_snapshots[self._current_step] = self.llm.clone_state()

        self._current_step += 1
        return action_result

    def receive_reward(self, reward: float, step: Optional[int] = None) -> None:
        if step is not None and step < len(self._action_history):
            self._action_history[step].reward = reward
        elif self._action_history:
            self._action_history[-1].reward = reward
        self._local_reward += reward

    def save_checkpoint(self, episode: int, loss: float = 0.0, metric_score: float = 0.0) -> CheckpointInfo:
        is_best = metric_score > 0 and (
            not self.llm.best_checkpoint or metric_score > self.llm.best_checkpoint.metric_score
        )
        return self.llm.save_checkpoint(
            self.checkpoint_dir, episode, loss, metric_score, is_best
        )

    def rollback_to(self, step: int) -> bool:
        if step in self._state_snapshots:
            self.llm.restore_from_clone(self._state_snapshots[step])
            self._action_history = [a for a in self._action_history if a.step < step]
            keys_to_remove = [k for k in self._state_snapshots if k >= step]
            for k in keys_to_remove:
                del self._state_snapshots[k]
            self._current_step = step
            self.logger.info(f"Agent '{self.agent_id}' rolled back to step {step}")
            return True

        best = self.llm.best_checkpoint
        if best:
            self.llm.rollback_to_checkpoint(best)
            self.logger.info(f"Agent '{self.agent_id}' rolled back to best checkpoint")
            return True
        return False

    def rollback_to_best(self) -> bool:
        return self.llm.load_best_checkpoint(self.checkpoint_dir)

    def freeze(self) -> None:
        self.llm.freeze()

    def unfreeze(self) -> None:
        self.llm.unfreeze()

    @property
    def is_frozen(self) -> bool:
        return self.llm.is_frozen

    def get_action_at(self, step: int) -> Optional[ActionRecord]:
        for record in self._action_history:
            if record.step == step:
                return record
        return None

    def get_actions_in_episode(self, episode_id: str) -> List[ActionRecord]:
        return [a for a in self._action_history if a.episode_id == episode_id]

    def reset_episode(self, episode_id: str) -> None:
        self._current_episode = episode_id
        self._current_step = 0
        self._local_reward = 0.0

    @property
    def action_history(self) -> List[ActionRecord]:
        return list(self._action_history)

    @property
    def local_reward(self) -> float:
        return self._local_reward

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "is_frozen": self.is_frozen,
            "current_step": self._current_step,
            "current_episode": self._current_episode,
            "num_actions": len(self._action_history),
            "local_reward": self._local_reward,
        }
