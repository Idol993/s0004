"""中心通信协议模块 - 消息定义、消息总线和路由系统"""

import time
import queue
import threading
import zlib
import json
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

from ..utils import AgentRole, MessageType, TrainingMetrics, Timer, setup_logger


@dataclass
class Message:
    msg_id: str
    sender: str
    receiver: str
    msg_type: MessageType
    content: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    episode_id: Optional[str] = None
    step: int = 0
    priority: int = 0
    requires_response: bool = False
    in_response_to: Optional[str] = None
    compressed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["msg_type"] = self.msg_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        data["msg_type"] = MessageType(data["msg_type"])
        return cls(**data)

    def compress(self) -> "Message":
        if not self.compressed:
            content_str = json.dumps(self.content)
            compressed = zlib.compress(content_str.encode("utf-8"))
            self.content = {"_compressed": compressed.hex()}
            self.compressed = True
        return self

    def decompress(self) -> "Message":
        if self.compressed and "_compressed" in self.content:
            compressed = bytes.fromhex(self.content["_compressed"])
            content_str = zlib.decompress(compressed).decode("utf-8")
            self.content = json.loads(content_str)
            self.compressed = False
        return self

    @property
    def size_bytes(self) -> int:
        return len(json.dumps(self.to_dict()).encode("utf-8"))


class MessageFactory:
    _counter = 0
    _lock = threading.Lock()

    @classmethod
    def create(
        cls,
        sender: str,
        receiver: str,
        msg_type: MessageType,
        content: Dict[str, Any],
        episode_id: Optional[str] = None,
        step: int = 0,
        priority: int = 0,
        requires_response: bool = False,
        in_response_to: Optional[str] = None,
    ) -> Message:
        with cls._lock:
            cls._counter += 1
            msg_id = f"msg_{int(time.time() * 1000)}_{cls._counter}"
        return Message(
            msg_id=msg_id,
            sender=sender,
            receiver=receiver,
            msg_type=msg_type,
            content=content,
            episode_id=episode_id,
            step=step,
            priority=priority,
            requires_response=requires_response,
            in_response_to=in_response_to,
        )


class MessageQueue:
    def __init__(self, max_size: int = 1000):
        self._queue: "queue.PriorityQueue[Tuple[int, float, Message]]" = queue.PriorityQueue(maxsize=max_size)
        self._lock = threading.Lock()

    def put(self, msg: Message, timeout: Optional[float] = None) -> bool:
        try:
            priority_key = (-msg.priority, msg.timestamp)
            self._queue.put((priority_key, msg), timeout=timeout)
            return True
        except queue.Full:
            return False

    def get(self, timeout: Optional[float] = None) -> Optional[Message]:
        try:
            _, msg = self._queue.get(timeout=timeout)
            return msg
        except queue.Empty:
            return None

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()


class SubscriptionManager:
    def __init__(self):
        self._subscribers: Dict[str, Set[str]] = defaultdict(set)
        self._type_subscribers: Dict[MessageType, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def subscribe(self, agent_id: str, topic: Optional[str] = None, msg_type: Optional[MessageType] = None) -> None:
        with self._lock:
            if topic:
                self._subscribers[topic].add(agent_id)
            if msg_type:
                self._type_subscribers[msg_type].add(agent_id)

    def unsubscribe(self, agent_id: str, topic: Optional[str] = None, msg_type: Optional[MessageType] = None) -> None:
        with self._lock:
            if topic and agent_id in self._subscribers.get(topic, set()):
                self._subscribers[topic].discard(agent_id)
            if msg_type and agent_id in self._type_subscribers.get(msg_type, set()):
                self._type_subscribers[msg_type].discard(agent_id)

    def get_recipients(self, msg: Message) -> Set[str]:
        recipients = set()
        with self._lock:
            if msg.episode_id:
                recipients.update(self._subscribers.get(msg.episode_id, set()))
            recipients.update(self._type_subscribers.get(msg.msg_type, set()))
            if msg.receiver and msg.receiver != "*":
                recipients.add(msg.receiver)
        return recipients


class CentralMessageBus:
    def __init__(
        self,
        max_queue_size: int = 1000,
        message_timeout: float = 30.0,
        enable_logging: bool = True,
        compression_enabled: bool = True,
        log_dir: str = "./logs",
    ):
        self.max_queue_size = max_queue_size
        self.message_timeout = message_timeout
        self.enable_logging = enable_logging
        self.compression_enabled = compression_enabled

        self._queues: Dict[str, MessageQueue] = {}
        self._subscriptions = SubscriptionManager()
        self._lock = threading.RLock()
        self._metrics = TrainingMetrics()
        self._message_log: List[Dict[str, Any]] = []
        self._running = False
        self._workers: List[threading.Thread] = []
        self._callbacks: Dict[str, Callable[[Message], None]] = {}

        self.logger = setup_logger("message_bus", log_dir)

    def register_agent(self, agent_id: str) -> None:
        with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = MessageQueue(self.max_queue_size)
                self.logger.info(f"Agent '{agent_id}' registered on message bus")

    def unregister_agent(self, agent_id: str) -> None:
        with self._lock:
            self._queues.pop(agent_id, None)
            self.logger.info(f"Agent '{agent_id}' unregistered from message bus")

    def subscribe(
        self,
        agent_id: str,
        topic: Optional[str] = None,
        msg_type: Optional[MessageType] = None,
        callback: Optional[Callable[[Message], None]] = None,
    ) -> None:
        self._subscriptions.subscribe(agent_id, topic, msg_type)
        if callback:
            with self._lock:
                self._callbacks[agent_id] = callback

    def unsubscribe(
        self, agent_id: str, topic: Optional[str] = None, msg_type: Optional[MessageType] = None
    ) -> None:
        self._subscriptions.unsubscribe(agent_id, topic, msg_type)

    def send(self, msg: Message, use_compression: Optional[bool] = None) -> bool:
        t0 = time.time()
        if use_compression is None:
            use_compression = self.compression_enabled
        if use_compression and msg.size_bytes > 1024:
            msg.compress()

        with self._lock:
            recipients = self._subscriptions.get_recipients(msg)
            if not recipients:
                self.logger.warning(f"No recipients for message {msg.msg_id} of type {msg.msg_type}")
                self._metrics.communication_time += time.time() - t0
                return False

            success_count = 0
            for recipient in recipients:
                if recipient in self._queues:
                    if self._queues[recipient].put(msg):
                        success_count += 1
                    else:
                        self.logger.warning(f"Queue full for agent '{recipient}', dropping message {msg.msg_id}")

            self._metrics.num_messages += len(recipients)

            if self.enable_logging:
                self._message_log.append(
                    {
                        "msg_id": msg.msg_id,
                        "sender": msg.sender,
                        "receivers": list(recipients),
                        "msg_type": msg.msg_type.value,
                        "timestamp": msg.timestamp,
                        "episode_id": msg.episode_id,
                        "size_bytes": msg.size_bytes,
                    }
                )

            self._metrics.communication_time += time.time() - t0
            return success_count > 0

    def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]:
        t0 = time.time()
        with self._lock:
            if agent_id not in self._queues:
                self._metrics.communication_time += time.time() - t0
                return None
            q = self._queues[agent_id]

        msg = q.get(timeout=timeout)
        if msg:
            msg.decompress()
            with self._lock:
                if agent_id in self._callbacks:
                    try:
                        self._callbacks[agent_id](msg)
                    except Exception as e:
                        self.logger.error(f"Callback error for agent {agent_id}: {e}")
        self._metrics.communication_time += time.time() - t0
        return msg

    def broadcast(self, msg: Message) -> int:
        t0 = time.time()
        msg.receiver = "*"
        count = 0
        with self._lock:
            for agent_id in self._queues:
                msg_copy = Message.from_dict(msg.to_dict())
                msg_copy.receiver = agent_id
                if self._queues[agent_id].put(msg_copy):
                    count += 1
        self._metrics.num_messages += count
        self._metrics.communication_time += time.time() - t0
        return count

    def request_response(
        self, msg: Message, timeout: Optional[float] = None
    ) -> Optional[Message]:
        if timeout is None:
            timeout = self.message_timeout
        msg.requires_response = True
        self.send(msg)

        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.receive(msg.sender, timeout=0.1)
            if response and response.in_response_to == msg.msg_id:
                return response
        return None

    def get_message_log(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        with self._lock:
            if last_n:
                return self._message_log[-last_n:]
            return list(self._message_log)

    def get_metrics(self) -> TrainingMetrics:
        return self._metrics

    def reset_metrics(self) -> None:
        self._metrics = TrainingMetrics()

    def start(self) -> None:
        self._running = True
        self.logger.info("Central Message Bus started")

    def stop(self) -> None:
        self._running = False
        for t in self._workers:
            t.join(timeout=2.0)
        self._workers.clear()
        self.logger.info("Central Message Bus stopped")

    @property
    def registered_agents(self) -> List[str]:
        with self._lock:
            return list(self._queues.keys())
