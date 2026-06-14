"""记忆器智能体 - 负责经验存储、检索和知识管理"""

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import AgentRole, MessageType
from .base_agent import BaseAgent
from .llm_backbone import LLMConfig
from ..communication.message_bus import CentralMessageBus


class MemoryIndex:
    def __init__(self, embedding_dim: int = 256, max_memories: int = 10000):
        self.embedding_dim = embedding_dim
        self.max_memories = max_memories
        self._keys: List[torch.Tensor] = []
        self._values: List[Dict[str, Any]] = []
        self._timestamps: List[float] = []
        self._importance: List[float] = []

    def add(self, key_embedding: torch.Tensor, value: Dict[str, Any], importance: float = 0.5) -> None:
        if len(self._keys) >= self.max_memories:
            min_idx = self._importance.index(min(self._importance))
            self._keys.pop(min_idx)
            self._values.pop(min_idx)
            self._timestamps.pop(min_idx)
            self._importance.pop(min_idx)

        self._keys.append(key_embedding.detach().cpu())
        self._values.append(value)
        import time
        self._timestamps.append(time.time())
        self._importance.append(importance)

    def retrieve(
        self, query_embedding: torch.Tensor, top_k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not self._keys:
            return []

        keys_tensor = torch.stack(self._keys)
        query = query_embedding.detach().cpu()

        if keys_tensor.dim() > 2:
            keys_tensor = keys_tensor.squeeze(1)
        if query.dim() > 1:
            query = query.squeeze(0)

        min_dim = min(keys_tensor.shape[-1], query.shape[-1])
        keys_trunc = keys_tensor[..., :min_dim]
        query_trunc = query[..., :min_dim]

        if keys_trunc.dim() == 1:
            keys_trunc = keys_trunc.unsqueeze(0)

        similarities = F.cosine_similarity(keys_trunc, query_trunc.unsqueeze(0), dim=-1)
        top_k = min(top_k, len(similarities))
        top_values, top_indices = torch.topk(similarities, top_k)

        results = []
        for idx, score in zip(top_indices.tolist(), top_values.tolist()):
            results.append((self._values[idx], score))
        return results

    @property
    def size(self) -> int:
        return len(self._keys)


class MemoryAgent(BaseAgent):
    def __init__(
        self,
        agent_id: str = "memory",
        llm_config: Optional[LLMConfig] = None,
        message_bus: Optional[CentralMessageBus] = None,
        checkpoint_dir: str = "./checkpoints/memory",
        log_dir: str = "./logs",
        embedding_dim: int = 256,
        max_memories: int = 10000,
    ):
        if llm_config is None:
            llm_config = LLMConfig(model_name="gpt2-medium")
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.MEMORY,
            llm_config=llm_config,
            message_bus=message_bus,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
        )
        self._memory_index = MemoryIndex(embedding_dim, max_memories)
        self._short_term: List[Dict[str, Any]] = []
        self._embedding_dim = embedding_dim
        self._embedding_proj = None
        self._init_projection()

    def _init_projection(self) -> None:
        if self.llm.model is not None:
            hidden_size = self.llm.model.config.hidden_size
            self._embedding_proj = nn.Linear(hidden_size, self._embedding_dim).to(self.llm.device)
        else:
            self._embedding_proj = nn.Linear(1024, self._embedding_dim)

    def perceive(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        query = observation.get("query", "")
        query_type = observation.get("query_type", "retrieve")
        data_to_store = observation.get("data_to_store", None)

        return {
            "query": query,
            "query_type": query_type,
            "data_to_store": data_to_store,
            "prompt": f"Memory query: {query}\nQuery type: {query_type}",
        }

    def decide(self, perceived_state: Dict[str, Any]) -> Dict[str, Any]:
        query_type = perceived_state["query_type"]

        if query_type == "retrieve":
            query_embedding = self._encode_text(perceived_state["query"])
            results = self._memory_index.retrieve(query_embedding, top_k=5)
            return {
                "action": "retrieve",
                "results": [(v, s) for v, s in results],
                "query": perceived_state["query"],
            }
        elif query_type == "store" and perceived_state["data_to_store"]:
            data = perceived_state["data_to_store"]
            importance = self._compute_importance(data)
            key_embedding = self._encode_text(str(data)[:500])
            return {
                "action": "store",
                "data": data,
                "key_embedding": key_embedding,
                "importance": importance,
            }
        else:
            return {"action": "noop", "query": perceived_state["query"]}

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        if decision["action"] == "store":
            self._memory_index.add(
                decision["key_embedding"],
                decision["data"],
                decision["importance"],
            )
            self._short_term.append(decision["data"])

            self.send_message(
                receiver="*",
                msg_type=MessageType.MEMORY_RESULT,
                content={
                    "action": "stored",
                    "importance": decision["importance"],
                    "memory_size": self._memory_index.size,
                },
            )
            return {
                "action_type": "store",
                "success": True,
                "memory_size": self._memory_index.size,
            }

        elif decision["action"] == "retrieve":
            results = decision["results"]
            requester = decision.get("query", "")

            return {
                "action_type": "retrieve",
                "results": results,
                "num_retrieved": len(results),
            }

        return {"action_type": "noop"}

    def _encode_text(self, text: str) -> torch.Tensor:
        if self.llm.model is None:
            return torch.randn(self._embedding_dim)

        inputs = self.llm.encode(text)
        with torch.no_grad():
            outputs = self.llm.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        if self._embedding_proj is not None:
            embedding = self._embedding_proj(last_hidden)
        else:
            embedding = last_hidden[:, :self._embedding_dim]

        return embedding

    def _compute_importance(self, data: Dict[str, Any]) -> float:
        importance = 0.5
        if data.get("reward") is not None:
            importance += abs(data["reward"]) * 0.3
        if data.get("failed", False):
            importance += 0.2
        if data.get("novel", False):
            importance += 0.1
        return min(1.0, importance)

    @property
    def memory_size(self) -> int:
        return self._memory_index.size

    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()
        stats["memory_size"] = self._memory_index.size
        stats["short_term_size"] = len(self._short_term)
        return stats
