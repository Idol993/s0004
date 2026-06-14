"""LLM模型封装 - 支持微调、推理和检查点管理"""

import os
import copy
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from ..utils import get_device, ensure_dir, setup_logger


def _get_transformers():
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    return AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass
class LLMConfig:
    model_name: str = "gpt2-medium"
    device: str = "cuda"
    max_length: int = 512
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = False


@dataclass
class CheckpointInfo:
    path: str
    step: int
    episode: int
    loss: float
    metric_score: float
    timestamp: float
    is_best: bool = False


class LLMBackbone:
    def __init__(self, config: LLMConfig, agent_name: str = "default", log_dir: str = "./logs"):
        self.config = config
        self.agent_name = agent_name
        self.device = get_device(config.device)
        self.logger = setup_logger(f"llm_{agent_name}", log_dir)

        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None

        self._current_step: int = 0
        self._current_episode: int = 0
        self._checkpoints: List[CheckpointInfo] = []
        self._best_checkpoint: Optional[CheckpointInfo] = None
        self._is_frozen: bool = False

        self._build_model()

    def _build_model(self) -> None:
        AutoModelForCausalLM, AutoTokenizer, _ = _get_transformers()
        self.logger.info(f"Loading model {self.config.model_name} for agent '{self.agent_name}'")
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self._setup_optimizer()
        self.logger.info(f"Model loaded on {self.device}")

    def _setup_optimizer(self, total_steps: int = 10000) -> None:
        _, _, get_linear_schedule_with_warmup = _get_transformers()
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps,
        )

    def freeze(self) -> None:
        if self.model is None:
            return
        for param in self.model.parameters():
            param.requires_grad = False
        self._is_frozen = True
        self.logger.info(f"Agent '{self.agent_name}' model frozen")

    def unfreeze(self) -> None:
        if self.model is None:
            return
        for param in self.model.parameters():
            param.requires_grad = True
        self._is_frozen = False
        self.logger.info(f"Agent '{self.agent_name}' model unfrozen")

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    def encode(self, text: Union[str, List[str]], **kwargs) -> Dict[str, torch.Tensor]:
        max_len = min(self.config.max_length, 64)
        default_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": max_len,
        }
        default_kwargs.update(kwargs)
        encoded = self.tokenizer(text, **default_kwargs).to(self.device)
        if "token_type_ids" in encoded:
            del encoded["token_type_ids"]
        return encoded

    @torch.no_grad()
    def generate(self, prompt: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        if self.model is None:
            return self._mock_generate(prompt)
        try:
            self.model.eval()
            inputs = self.encode(prompt)
            default_kwargs = {
                "max_new_tokens": min(64, kwargs.pop("max_new_tokens", 256)),
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            default_kwargs.update(kwargs)
            outputs = self.model.generate(**inputs, **default_kwargs)
            input_len = inputs["input_ids"].shape[1]
            generated = outputs[:, input_len:]
            result = self.decode(generated)

            del outputs, inputs, generated
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if isinstance(prompt, str):
                return result[0] if isinstance(result, list) else result
            return result
        except (RuntimeError, MemoryError) as e:
            self.logger.warning(f"LLM generate failed ({e}), using mock output")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt: Union[str, List[str]]) -> Union[str, List[str]]:
        default_text = "Based on the input, here is a reasonable response."
        if isinstance(prompt, list):
            return [default_text for _ in prompt]
        return default_text

    def decode(self, token_ids: torch.Tensor, **kwargs) -> Union[str, List[str]]:
        if self.tokenizer is None:
            return "[mock decoded output]" if token_ids.dim() == 1 else ["[mock decoded output]"] * token_ids.shape[0]
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True, **kwargs)

    @torch.no_grad()
    def forward_pass(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.model.eval()
        outputs = self.model(**inputs, output_hidden_states=True)
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "loss": outputs.loss if "labels" in inputs else None,
        }

    def train_step(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        accumulation_steps: int = 1,
    ) -> Dict[str, float]:
        if self.model is None or self._is_frozen:
            return {"loss": 0.0}

        try:
            self.model.train()
            if labels is not None:
                inputs["labels"] = labels

            outputs = self.model(**inputs)
            loss = outputs.loss / accumulation_steps
            loss.backward()

            if (self._current_step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            self._current_step += 1
            loss_val = loss.item() * accumulation_steps

            del outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {"loss": loss_val}
        except (RuntimeError, MemoryError) as e:
            self.logger.warning(f"Train step failed: {e}")
            if self.optimizer:
                self.optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"loss": 0.0}

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "model_state": self.model.state_dict() if self.model else None,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "step": self._current_step,
            "episode": self._current_episode,
            "config": self.config.__dict__,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if self.model and state.get("model_state"):
            self.model.load_state_dict(state["model_state"])
        if self.optimizer and state.get("optimizer_state"):
            self.optimizer.load_state_dict(state["optimizer_state"])
        if self.scheduler and state.get("scheduler_state"):
            self.scheduler.load_state_dict(state["scheduler_state"])
        self._current_step = state.get("step", 0)
        self._current_episode = state.get("episode", 0)

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        episode: int,
        loss: float,
        metric_score: float = 0.0,
        is_best: bool = False,
    ) -> CheckpointInfo:
        ensure_dir(checkpoint_dir)
        import time

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}_st{self._current_step}.pt")
        state = self.get_state_dict()
        state["loss"] = loss
        state["metric_score"] = metric_score

        torch.save(state, checkpoint_path)

        info = CheckpointInfo(
            path=checkpoint_path,
            step=self._current_step,
            episode=episode,
            loss=loss,
            metric_score=metric_score,
            timestamp=time.time(),
            is_best=is_best,
        )
        self._checkpoints.append(info)

        if is_best or self._best_checkpoint is None or metric_score > self._best_checkpoint.metric_score:
            best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
            shutil.copy2(checkpoint_path, best_path)
            info.is_best = True
            self._best_checkpoint = info

        self._current_episode = episode
        self.logger.info(f"Checkpoint saved: {checkpoint_path} (loss={loss:.4f}, score={metric_score:.4f})")
        return info

    def load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(state)
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")

    def load_best_checkpoint(self, checkpoint_dir: str) -> bool:
        best_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        if os.path.exists(best_path):
            self.load_checkpoint(best_path)
            return True
        return False

    def rollback_to_checkpoint(self, checkpoint_info: CheckpointInfo) -> None:
        self.load_checkpoint(checkpoint_info.path)
        self.logger.info(
            f"Rolled back to checkpoint at episode {checkpoint_info.episode}, "
            f"step {checkpoint_info.step}"
        )

    def list_checkpoints(self) -> List[CheckpointInfo]:
        return list(self._checkpoints)

    @property
    def best_checkpoint(self) -> Optional[CheckpointInfo]:
        return self._best_checkpoint

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def current_episode(self) -> int:
        return self._current_episode

    def clone_state(self) -> Dict[str, Any]:
        if self.model is None:
            return {}
        return {
            "step": self._current_step,
            "episode": self._current_episode,
            "best_checkpoint_path": self._best_checkpoint.path if self._best_checkpoint else None,
            "latest_checkpoint_path": self._checkpoints[-1].path if self._checkpoints else None,
            "has_model_state": False,
        }

    def restore_from_clone(self, cloned_state: Dict[str, Any]) -> None:
        if not cloned_state:
            return
        if "model_state" in cloned_state and cloned_state.get("has_model_state", True) and self.model:
            self.model.load_state_dict(cloned_state["model_state"])
        self._current_step = cloned_state.get("step", self._current_step)
        self._current_episode = cloned_state.get("episode", self._current_episode)
