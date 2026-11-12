from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for ATLAS-IDNet. Install torch in the runtime.") from exc


@dataclass
class FeatureGroups:
    traffic_idx: list[int]
    protocol_idx: list[int]
    temporal_idx: list[int]


def infer_feature_groups(feature_names: list[str]) -> FeatureGroups:
    traffic_tokens = ["byte", "pkt", "packet", "ttl", "throughput", "length", "rate", "iat", "header"]
    protocol_tokens = ["proto", "port", "flag", "tcp", "udp", "icmp", "l7", "service"]
    temporal_tokens = ["duration", "time", "flow", "second", "min", "max", "avg", "std"]

    traffic, protocol, temporal = [], [], []
    for i, name in enumerate(feature_names):
        low = name.lower()
        if any(tok in low for tok in protocol_tokens):
            protocol.append(i)
        elif any(tok in low for tok in temporal_tokens):
            temporal.append(i)
        elif any(tok in low for tok in traffic_tokens):
            traffic.append(i)
        else:
            traffic.append(i)

    all_idx = set(range(len(feature_names)))
    if not protocol:
        protocol = sorted(list(all_idx))[: max(1, len(feature_names) // 4)]
    if not temporal:
        temporal = sorted(list(all_idx))[max(1, len(feature_names) // 4) : max(2, len(feature_names) // 2)]
    traffic = sorted(list(all_idx - set(protocol) - set(temporal))) or sorted(list(all_idx))
    return FeatureGroups(traffic, protocol, temporal)


def build_model(input_dim: int, num_classes: int, feature_names: list[str], hidden_dim: int = 128, dropout: float = 0.2):
    torch, nn, F = require_torch()
    groups = infer_feature_groups(feature_names)

    class ResidualBlock(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.BatchNorm1d(dim),
            )

        def forward(self, x):
            return F.gelu(x + self.net(x))

    class Branch(nn.Module):
        def __init__(self, in_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(hidden_dim),
                ResidualBlock(hidden_dim),
            )

        def forward(self, x):
            return self.net(x)

    class ATLASIDNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.groups = groups
            self.traffic = Branch(len(groups.traffic_idx))
            self.protocol = Branch(len(groups.protocol_idx))
            self.temporal = Branch(len(groups.temporal_idx))
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 3),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(hidden_dim),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x, return_attention: bool = False):
            traffic = self.traffic(x[:, self.groups.traffic_idx])
            protocol = self.protocol(x[:, self.groups.protocol_idx])
            temporal = self.temporal(x[:, self.groups.temporal_idx])
            stacked = torch.stack([traffic, protocol, temporal], dim=1)
            weights = torch.softmax(self.attn(torch.cat([traffic, protocol, temporal], dim=1)), dim=1)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
            logits = self.classifier(fused)
            if return_attention:
                return logits, weights
            return logits

    return ATLASIDNet()


def parameter_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def predict_proba(model, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    torch, _, F = require_torch()
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
            outputs.append(F.softmax(model(batch), dim=1).cpu().numpy())
    return np.vstack(outputs)
