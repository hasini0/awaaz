"""Multi-task MuRIL: one shared Indian-language encoder, two classification heads."""
from __future__ import annotations

from pathlib import Path

from torch import nn
from transformers import AutoConfig, AutoModel

BASE_MODEL = "google/muril-base-cased"


class MultiTaskModel(nn.Module):
    def __init__(self, base_model: str, n_dept: int, n_urg: int, from_pretrained: bool = True,
                 config_dir: Path | None = None):
        super().__init__()
        if from_pretrained:
            self.encoder = AutoModel.from_pretrained(base_model)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(config_dir or base_model))
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.dept_head = nn.Linear(h, n_dept)
        self.urg_head = nn.Linear(h, n_urg)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # mean-pool over real tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = self.drop(pooled)
        return self.dept_head(pooled), self.urg_head(pooled)
