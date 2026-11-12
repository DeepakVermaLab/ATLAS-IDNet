import numpy as np
import torch
from torch import nn


def grouping(names):
    """Partition flow features into reproducible semantic groups."""
    groups = [[], [], []]
    reasons = []
    for index, name in enumerate(names):
        low = name.lower()
        if any(token in low for token in ('proto', 'port', 'flag', 'tcp', 'udp', 'icmp', 'l7', 'service')):
            group, reason = 1, 'protocol/transport descriptor'
        elif any(token in low for token in ('iat', 'duration', 'time', 'second')):
            group, reason = 2, 'time-derived flow statistic; not sequence modeling'
        else:
            group, reason = 0, 'traffic statistic or remaining numeric descriptor'
        groups[group].append(index)
        reasons.append({'feature': name, 'branch': group, 'reason': reason})
    if any(not group for group in groups):
        raise ValueError('Empty semantic branch; review the feature-to-branch mapping.')
    return groups, reasons


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout, normalization='batch'):
        super().__init__()
        norm = nn.BatchNorm1d if normalization == 'batch' else nn.LayerNorm
        self.net = nn.Sequential(
            nn.Linear(width, width),
            norm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            norm(width),
        )

    def forward(self, inputs):
        return torch.nn.functional.gelu(inputs + self.net(inputs))


class PlainBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )

    def forward(self, inputs):
        return self.net(inputs)


class BranchEncoder(nn.Module):
    def __init__(self, input_dim, width, dropout, residual=True, normalization='layer'):
        super().__init__()
        norm = nn.BatchNorm1d if normalization == 'batch' else nn.LayerNorm
        first = ResidualBlock(width, dropout, normalization) if residual else PlainBlock(width, dropout)
        second = ResidualBlock(width, dropout, normalization) if residual else PlainBlock(width, dropout)
        self.net = nn.Sequential(
            nn.Linear(input_dim, width),
            norm(width),
            nn.GELU(),
            nn.Dropout(dropout),
            first,
            second,
        )

    def forward(self, inputs):
        return self.net(inputs)


class Detector(nn.Module):
    """Proposed detector and its controlled architectural variants.

    ``atlas`` uses semantic branches, residual encoders, branch attention and a
    gated global context path. ``legacy`` reproduces the submitted branch,
    attention and classifier topology. Other variants remove one mechanism or
    replace the semantic partition while keeping the training protocol fixed.
    """

    SUPPORTED_VARIANTS = {
        'atlas',
        'legacy',
        'no_global_context',
        'no_residual',
        'mean',
        'concat',
        'random_groups',
        'single_residual',
        'mlp',
    }

    def __init__(self, names, classes, variant='atlas', width=128, seed=42, dropout=0.15):
        super().__init__()
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(f'Unknown variant {variant!r}; expected one of {sorted(self.SUPPORTED_VARIANTS)}')

        semantic_groups, self.feature_map = grouping(names)
        groups = semantic_groups
        if variant in ('mlp', 'single_residual'):
            groups = [list(range(len(names)))]
        elif variant == 'random_groups':
            shuffled = np.random.default_rng(seed).permutation(len(names)).tolist()
            cut_points = np.cumsum([len(group) for group in semantic_groups])[:-1]
            groups = [part.tolist() for part in np.split(shuffled, cut_points)]

        self.groups = groups
        self.variant = variant
        self.width = width
        self.dropout = dropout
        legacy = variant == 'legacy'
        residual = variant not in ('mlp', 'no_residual')
        normalization = 'batch' if legacy else 'layer'

        self.branches = nn.ModuleList(
            BranchEncoder(len(group), width, dropout if not legacy else 0.2, residual, normalization)
            for group in groups
        )
        branch_count = len(groups)
        self.attention = nn.Sequential(
            nn.Linear(width * branch_count, width),
            nn.Tanh(),
            nn.Linear(width, branch_count),
        )

        self.global_context = None
        self.context_gate = None
        if variant == 'atlas':
            self.global_context = nn.Sequential(
                nn.Linear(len(names), width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
                ResidualBlock(width, dropout, 'layer'),
            )
            self.context_gate = nn.Sequential(nn.Linear(width * 2, width), nn.Sigmoid())

        if legacy:
            self.head = nn.Sequential(
                nn.Linear(width, width),
                nn.BatchNorm1d(width),
                nn.GELU(),
                nn.Dropout(0.2),
                ResidualBlock(width, 0.2, 'batch'),
                nn.Linear(width, classes),
            )
        elif variant == 'concat':
            self.head = nn.Sequential(
                nn.Linear(width * branch_count, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, classes),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(width, classes),
            )

    def forward(self, inputs, return_attention=False):
        encoded = [branch(inputs[:, indices]) for branch, indices in zip(self.branches, self.groups)]
        stacked = torch.stack(encoded, dim=1)
        concatenated = torch.cat(encoded, dim=1)

        if self.variant == 'concat':
            fused = concatenated
            weights = torch.full(
                (inputs.shape[0], len(encoded)),
                1.0 / len(encoded),
                dtype=inputs.dtype,
                device=inputs.device,
            )
        elif self.variant == 'mean' or len(encoded) == 1:
            weights = torch.full(
                (inputs.shape[0], len(encoded)),
                1.0 / len(encoded),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            fused = stacked.mean(dim=1)
        else:
            weights = self.attention(concatenated).softmax(dim=1)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)

        if self.global_context is not None:
            context = self.global_context(inputs)
            gate = self.context_gate(torch.cat([fused, context], dim=1))
            fused = gate * fused + (1.0 - gate) * context

        logits = self.head(fused)
        if return_attention:
            return logits, weights
        return logits


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def matched_width(names, classes, variant, target_parameters, seed=42, dropout=0.15, minimum=8, maximum=512):
    """Return the width whose trainable parameter count is closest to target."""
    best = None
    for width in range(minimum, maximum + 1):
        model = Detector(names, classes, variant, width, seed, dropout)
        count = parameter_count(model)
        distance = abs(count - target_parameters)
        if best is None or distance < best[0]:
            best = (distance, width, count)
    return {'width': best[1], 'parameters': best[2], 'target_parameters': int(target_parameters)}
