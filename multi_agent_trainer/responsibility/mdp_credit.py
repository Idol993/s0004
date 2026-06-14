"""基于MDP的时延信用分配网络

核心思想：
1. 将多智能体协作过程建模为一个MDP，其中状态 = 各智能体观测的联合，
   动作 = 各智能体动作的联合，奖励 = 任务结果。
2. 使用Temporal Difference (TD) 学习和广义优势估计(GAE)来处理
   时延奖励下的信用分配问题——即一个智能体的早期决策可能很久之后
   才体现其影响。
3. 结合价值函数分解(VDN/IQL风格)将全局奖励分解为各智能体的局部贡献。

时间复杂度：O(T * N) 其中T是时间步数，N是智能体数，远低于暴力方法的O(N^T)
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..agents.base_agent import BaseAgent, ActionRecord
from ..utils import AgentRole, Timer, setup_logger


@dataclass
class MDPTransition:
    state: torch.Tensor
    agent_id: str
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    step: int
    episode_id: str
    log_prob: Optional[float] = None
    hidden_state: Optional[torch.Tensor] = None


@dataclass
class MDPCreditConfig:
    gamma: float = 0.99
    lambda_gae: float = 0.95
    hidden_dim: int = 256
    num_layers: int = 2
    learning_rate: float = 1e-4
    state_dim: int = 128
    action_dim: int = 64
    num_agents: int = 5
    max_trajectory_length: int = 100


class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AgentValueNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class TemporalCreditNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.credit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, states: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        lstm_out, new_hidden = self.lstm(states, hidden)
        credits = self.credit_head(lstm_out)
        return credits, new_hidden


class GlobalValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_agents: int):
        super().__init__()
        input_dim = state_dim * num_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, agent_states: torch.Tensor) -> torch.Tensor:
        return self.net(agent_states)


class MDPCreditAssignmentNetwork:
    def __init__(
        self,
        config: MDPCreditConfig,
        agents: Dict[str, BaseAgent],
        log_dir: str = "./logs",
    ):
        self.config = config
        self.agents = agents
        self.agent_ids = list(agents.keys())
        self.num_agents = len(self.agent_ids)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger = setup_logger("mdp_credit", log_dir)

        self._build_networks()
        self._setup_optimizers()

        self._transition_buffer: List[MDPTransition] = []
        self._credit_history: Dict[str, List[float]] = {aid: [] for aid in self.agent_ids}
        self._timer = Timer()
        self._total_compute_time: float = 0.0

    def _build_networks(self) -> None:
        sd = self.config.state_dim
        ad = self.config.action_dim
        hd = self.config.hidden_dim

        self.state_encoder = StateEncoder(sd, hd, sd).to(self.device)
        self.action_encoder = ActionEncoder(ad, hd // 2, ad).to(self.device)
        self.temporal_credit = TemporalCreditNetwork(sd, hd, self.config.num_layers).to(self.device)
        self.global_value = GlobalValueNetwork(sd, hd, self.num_agents).to(self.device)

        self.agent_value_nets = nn.ModuleDict({
            aid: AgentValueNetwork(sd, ad, hd).to(self.device)
            for aid in self.agent_ids
        })

    def _setup_optimizers(self) -> None:
        all_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
            + list(self.temporal_credit.parameters())
            + list(self.global_value.parameters())
        )
        for net in self.agent_value_nets.values():
            all_params.extend(list(net.parameters()))

        self.optimizer = torch.optim.Adam(all_params, lr=self.config.learning_rate)

    def encode_observation(self, observation: Dict[str, Any]) -> torch.Tensor:
        features = []
        for key in sorted(observation.keys()):
            val = observation[key]
            if isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, str):
                hash_val = hash(val) % (10 ** 6) / (10 ** 6)
                features.append(hash_val)
            elif isinstance(val, bool):
                features.append(float(val))
            elif isinstance(val, dict):
                sub_features = self._flatten_dict(val)
                features.extend(sub_features[:10])
            elif isinstance(val, list):
                for item in val[:5]:
                    if isinstance(item, (int, float)):
                        features.append(float(item))

        while len(features) < self.config.state_dim:
            features.append(0.0)
        features = features[:self.config.state_dim]

        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        return self.state_encoder(tensor.unsqueeze(0)).squeeze(0)

    def encode_action(self, action: Dict[str, Any]) -> torch.Tensor:
        features = []
        self._extract_action_features(action, features)
        while len(features) < self.config.action_dim:
            features.append(0.0)
        features = features[:self.config.action_dim]

        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        return self.action_encoder(tensor.unsqueeze(0)).squeeze(0)

    def _extract_action_features(self, action: Any, features: List[float]) -> None:
        if isinstance(action, dict):
            for key in sorted(action.keys()):
                self._extract_action_features(action[key], features)
        elif isinstance(action, (int, float)):
            features.append(float(action))
        elif isinstance(action, str):
            features.append(hash(action) % (10 ** 6) / (10 ** 6))
        elif isinstance(action, bool):
            features.append(float(action))
        elif isinstance(action, list):
            for item in action[:5]:
                self._extract_action_features(item, features)

    def _flatten_dict(self, d: Dict[str, Any], depth: int = 0) -> List[float]:
        if depth > 3:
            return []
        features = []
        for key, val in d.items():
            if isinstance(val, (int, float)):
                features.append(float(val))
            elif isinstance(val, dict):
                features.extend(self._flatten_dict(val, depth + 1)[:5])
        return features

    def record_transition(
        self,
        agent_id: str,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_observation: Dict[str, Any],
        done: bool,
        step: int,
        episode_id: str,
    ) -> None:
        state = self.encode_observation(observation).detach()
        action_enc = self.encode_action(action).detach()
        next_state = self.encode_observation(next_observation).detach()

        transition = MDPTransition(
            state=state,
            agent_id=agent_id,
            action=action_enc,
            reward=reward,
            next_state=next_state,
            done=done,
            step=step,
            episode_id=episode_id,
        )
        self._transition_buffer.append(transition)

    def compute_gae(
        self,
        transitions: List[MDPTransition],
        agent_id: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        agent_transitions = [t for t in transitions if t.agent_id == agent_id]
        if not agent_transitions:
            return torch.tensor([]), torch.tensor([])

        values = []
        for t in agent_transitions:
            with torch.no_grad():
                v = self.agent_value_nets[agent_id](t.state.unsqueeze(0), t.action.unsqueeze(0))
                values.append(v.squeeze())

        values_tensor = torch.stack(values) if values else torch.tensor([0.0])
        rewards = torch.tensor([t.reward for t in agent_transitions], device=self.device)
        dones = torch.tensor([float(t.done) for t in agent_transitions], device=self.device)

        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values_tensor[t + 1].item()

            delta = rewards[t].item() + self.config.gamma * next_value * (1.0 - dones[t].item()) - values_tensor[t].item()
            gae = delta + self.config.gamma * self.config.lambda_gae * (1.0 - dones[t].item()) * gae
            advantages.insert(0, gae)

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = advantages_tensor + values_tensor

        return advantages_tensor, returns_tensor

    def compute_temporal_credits(
        self,
        trajectory: List[ActionRecord],
        episode_id: str,
    ) -> Dict[str, List[Tuple[int, float]]]:
        with self._timer:
            agent_trajectories: Dict[str, List[ActionRecord]] = {}
            for record in trajectory:
                if record.agent_id not in agent_trajectories:
                    agent_trajectories[record.agent_id] = []
                agent_trajectories[record.agent_id].append(record)

            all_credits: Dict[str, List[Tuple[int, float]]] = {}

            for agent_id, records in agent_trajectories.items():
                if not records:
                    all_credits[agent_id] = []
                    continue

                states = []
                for record in records:
                    state = self.encode_observation(record.observation)
                    states.append(state)

                states_tensor = torch.stack(states).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    credits, _ = self.temporal_credit(states_tensor)
                    credit_values = credits.squeeze(-1).squeeze(0).tolist()

                step_credits = [
                    (records[i].step, credit_values[i])
                    for i in range(len(records))
                ]
                all_credits[agent_id] = step_credits
                self._credit_history[agent_id].extend(credit_values)

            self._total_compute_time += self._timer.elapsed

        return all_credits

    def compute_delayed_credit_assignment(
        self,
        trajectory: List[ActionRecord],
        final_reward: float,
        episode_id: str,
    ) -> Dict[str, float]:
        temporal_credits = self.compute_temporal_credits(trajectory, episode_id)

        agent_credit_scores: Dict[str, float] = {}

        for agent_id, step_credits in temporal_credits.items():
            if not step_credits:
                agent_credit_scores[agent_id] = 0.0
                continue

            weighted_reward = 0.0
            total_weight = 0.0
            for step, credit in step_credits:
                time_decay = self.config.gamma ** (len(step_credits) - step_credits.index((step, credit)) - 1)
                weight = credit * time_decay
                weighted_reward += final_reward * weight
                total_weight += weight

            if total_weight > 0:
                agent_credit_scores[agent_id] = weighted_reward / total_weight
            else:
                agent_credit_scores[agent_id] = final_reward / max(len(temporal_credits), 1)

        return agent_credit_scores

    def value_decomposition(
        self,
        joint_state: torch.Tensor,
    ) -> Dict[str, float]:
        agent_values = {}
        with torch.no_grad():
            for agent_id in self.agent_ids:
                agent_state = joint_state
                default_action = torch.zeros(self.config.action_dim, device=self.device)
                v = self.agent_value_nets[agent_id](
                    agent_state.unsqueeze(0), default_action.unsqueeze(0)
                )
                agent_values[agent_id] = v.item()

        total = sum(agent_values.values())
        if abs(total) > 1e-8:
            for aid in agent_values:
                agent_values[aid] /= total

        return agent_values

    def train_credit_network(self, num_steps: int = 10) -> Dict[str, float]:
        if not self._transition_buffer:
            return {"loss": 0.0}

        total_loss = 0.0
        for _ in range(num_steps):
            batch_size = min(32, len(self._transition_buffer))
            indices = np.random.choice(len(self._transition_buffer), batch_size, replace=False)
            batch = [self._transition_buffer[i] for i in indices]

            loss = self._compute_training_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.state_encoder.parameters())
                + list(self.action_encoder.parameters())
                + list(self.temporal_credit.parameters())
                + list(self.global_value.parameters()),
                1.0,
            )
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(num_steps, 1)
        return {"loss": avg_loss}

    def _compute_training_loss(self, batch: List[MDPTransition]) -> torch.Tensor:
        value_loss = torch.tensor(0.0, device=self.device)
        td_errors = []

        for t in batch:
            v = self.agent_value_nets[t.agent_id](t.state.unsqueeze(0), t.action.unsqueeze(0))
            with torch.no_grad():
                next_v = torch.tensor(0.0, device=self.device)
                if not t.done:
                    next_v = self.agent_value_nets[t.agent_id](
                        t.next_state.unsqueeze(0), t.action.unsqueeze(0)
                    ).detach()
                target = t.reward + self.config.gamma * next_v

            td_error = v - target
            td_errors.append(td_error)
            value_loss += F.mse_loss(v, target)

        value_loss /= max(len(batch), 1)

        if td_errors and len(td_errors) > 1:
            td_tensor = torch.cat([e.flatten() for e in td_errors])
            consistency_loss = torch.var(td_tensor, unbiased=False)
        else:
            consistency_loss = torch.tensor(0.0, device=self.device)

        total_loss = value_loss + 0.1 * consistency_loss
        return total_loss

    def get_credit_summary(self, agent_id: str) -> Dict[str, Any]:
        credits = self._credit_history.get(agent_id, [])
        return {
            "agent_id": agent_id,
            "num_credits": len(credits),
            "mean_credit": float(np.mean(credits)) if credits else 0.0,
            "std_credit": float(np.std(credits)) if credits else 0.0,
            "compute_time": self._total_compute_time,
        }

    def clear_buffer(self) -> None:
        self._transition_buffer.clear()
