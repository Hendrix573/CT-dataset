from __future__ import annotations

import torch
from torch import nn

from .data import ORIENTATION_VOCAB, VIEW_VOCAB


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class LightweightEncoder(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128), blocks=(2, 2, 2, 2), out_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], 5, 2, 2, bias=False),
            nn.GroupNorm(_groups(channels[0]), channels[0]), nn.SiLU(),
        )
        stages = []
        in_channels = channels[0]
        for stage_index, (out_channels, count) in enumerate(zip(channels, blocks)):
            for block_index in range(count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                stages.append(ResidualBlock(in_channels, out_channels, stride))
                in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(channels[-1], out_dim)

    def forward(self, x):
        return self.projection(self.pool(self.stages(self.stem(x))).flatten(1))


class MetadataEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.view_embedding = nn.Embedding(len(VIEW_VOCAB), 16)
        self.orientation_embedding = nn.Embedding(len(ORIENTATION_VOCAB), 8)
        self.network = nn.Sequential(nn.Linear(27, 64), nn.SiLU(), nn.Linear(64, out_dim))

    def forward(self, view_ids, orientation_ids, numeric_metadata):
        return self.network(torch.cat((
            self.view_embedding(view_ids), self.orientation_embedding(orientation_ids), numeric_metadata
        ), dim=-1))


def study_mean(features, image_to_study, num_studies):
    pooled = features.new_zeros((num_studies, features.shape[-1]))
    pooled.index_add_(0, image_to_study, features)
    counts = torch.bincount(image_to_study, minlength=num_studies).clamp_min(1)
    return pooled / counts.to(features.dtype).unsqueeze(1)


class StudyMultiInputNet(nn.Module):
    """Local quadrants + global image + metadata, fused by class-query attention."""
    def __init__(self, feature_dim=128, metadata_dim=32, attention_heads=4, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.local_encoder = LightweightEncoder(blocks=(1, 1, 1, 1), out_dim=feature_dim)
        self.global_encoder = LightweightEncoder(blocks=(2, 2, 2, 2), out_dim=feature_dim)
        self.metadata_encoder = MetadataEncoder(metadata_dim)
        self.quadrant_embedding = nn.Parameter(torch.randn(4, feature_dim) * 0.02)
        self.metadata_to_token = nn.Linear(metadata_dim, feature_dim)
        self.class_queries = nn.Parameter(torch.randn(10, feature_dim) * 0.02)
        self.global_to_query = nn.Linear(feature_dim, feature_dim)
        self.metadata_to_query = nn.Linear(metadata_dim, feature_dim)
        self.query_norm = nn.LayerNorm(feature_dim)
        self.token_norm = nn.LayerNorm(feature_dim)
        self.cross_attention = nn.MultiheadAttention(
            feature_dim, attention_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        head_dim = feature_dim * 2 + metadata_dim
        self.classifier_weight = nn.Parameter(torch.empty(10, head_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(10))
        nn.init.xavier_uniform_(self.classifier_weight)

    @staticmethod
    def _pad_study_tokens(tokens, image_to_study, num_studies):
        lengths = [int((image_to_study == i).sum()) * 4 for i in range(num_studies)]
        max_length = max(lengths)
        padded = tokens.new_zeros((num_studies, max_length, tokens.shape[-1]))
        valid = torch.zeros((num_studies, max_length), dtype=torch.bool, device=tokens.device)
        for study_index, length in enumerate(lengths):
            selected = tokens[image_to_study == study_index].reshape(length, -1)
            padded[study_index, :length] = selected
            valid[study_index, :length] = True
        return padded, valid

    def forward(self, batch, return_attention=False):
        global_images = batch["global_images"]
        local_patches = batch["local_patches"]
        image_to_study = batch["image_to_study"]
        num_studies = int(batch["num_studies"])
        num_images = global_images.shape[0]

        local = self.local_encoder(local_patches.reshape(-1, *local_patches.shape[2:]))
        local = local.view(num_images, 4, self.feature_dim)
        metadata = self.metadata_encoder(
            batch["view_ids"], batch["orientation_ids"], batch["numeric_metadata"]
        )
        local = local + self.quadrant_embedding.unsqueeze(0) + self.metadata_to_token(metadata).unsqueeze(1)
        tokens, valid_tokens = self._pad_study_tokens(local, image_to_study, num_studies)

        image_global = self.global_encoder(global_images)
        study_global = study_mean(image_global, image_to_study, num_studies)
        study_metadata = study_mean(metadata, image_to_study, num_studies)
        queries = (
            self.class_queries.unsqueeze(0)
            + self.global_to_query(study_global).unsqueeze(1)
            + self.metadata_to_query(study_metadata).unsqueeze(1)
        )
        attended, attention = self.cross_attention(
            self.query_norm(queries), self.token_norm(tokens), self.token_norm(tokens),
            key_padding_mask=~valid_tokens, need_weights=return_attention,
            average_attn_weights=False,
        )
        hidden = queries + self.dropout(attended)
        hidden = hidden + self.dropout(self.ffn(self.ffn_norm(hidden)))
        combined = torch.cat((
            hidden,
            study_global.unsqueeze(1).expand(-1, 10, -1),
            study_metadata.unsqueeze(1).expand(-1, 10, -1),
        ), dim=-1)
        logits = torch.einsum("bcd,cd->bc", combined, self.classifier_weight) + self.classifier_bias
        output = {"logits": logits}
        if return_attention:
            output["attention"] = attention
            output["token_mask"] = valid_tokens
        return output
