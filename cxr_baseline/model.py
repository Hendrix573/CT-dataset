from __future__ import annotations

import torch
import torch.nn.functional as F
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
        self.projection = nn.Linear(channels[-1], out_dim)

    def forward_features(self, x):
        return self.stages(self.stem(x))

    def forward_tokens(self, x, grid_size=1):
        features = self.forward_features(x)
        features = F.adaptive_avg_pool2d(features, (grid_size, grid_size))
        features = features.flatten(2).transpose(1, 2)
        return self.projection(features)

    def forward(self, x):
        return self.forward_tokens(x, grid_size=1).squeeze(1)


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
    def __init__(self, feature_dim=128, metadata_dim=32, attention_heads=4, dropout=0.0,
                 local_token_grid=2, use_global_tokens=True,
                 use_study_mean_residual=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.local_token_grid = local_token_grid
        self.tokens_per_patch = local_token_grid ** 2
        self.local_tokens_per_image = 4 * self.tokens_per_patch
        self.use_global_tokens = use_global_tokens
        self.use_study_mean_residual = use_study_mean_residual
        if local_token_grid < 1:
            raise ValueError("local_token_grid must be positive")
        self.local_encoder = LightweightEncoder(blocks=(1, 1, 1, 1), out_dim=feature_dim)
        self.global_encoder = LightweightEncoder(blocks=(2, 2, 2, 2), out_dim=feature_dim)
        self.metadata_encoder = MetadataEncoder(metadata_dim)
        self.quadrant_embedding = nn.Parameter(torch.randn(4, feature_dim) * 0.02)
        self.patch_cell_embedding = nn.Parameter(
            torch.randn(self.tokens_per_patch, feature_dim) * 0.02
        )
        self.metadata_to_token = nn.Linear(metadata_dim, feature_dim)
        if use_global_tokens:
            # Keep view identity explicit on the full-image token. The local tokens already
            # receive the same information through metadata_to_token.
            self.global_token_embedding = nn.Parameter(torch.randn(feature_dim) * 0.02)
            self.global_view_embedding = nn.Embedding(len(VIEW_VOCAB), feature_dim)
            self.global_orientation_embedding = nn.Embedding(
                len(ORIENTATION_VOCAB), feature_dim
            )
            nn.init.normal_(self.global_view_embedding.weight, std=0.02)
            nn.init.normal_(self.global_orientation_embedding.weight, std=0.02)
        self.class_queries = nn.Parameter(torch.randn(10, feature_dim) * 0.02)
        if use_study_mean_residual:
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
        head_dim = feature_dim
        if use_study_mean_residual:
            head_dim += feature_dim + metadata_dim
        self.classifier_weight = nn.Parameter(torch.empty(10, head_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(10))
        nn.init.xavier_uniform_(self.classifier_weight)

    @staticmethod
    def _pad_study_tokens(tokens, image_to_study, num_studies):
        tokens_per_image = tokens.shape[1]
        lengths = [int((image_to_study == i).sum()) * tokens_per_image for i in range(num_studies)]
        max_length = max(lengths)
        padded = tokens.new_zeros((num_studies, max_length, tokens.shape[-1]))
        valid = torch.zeros((num_studies, max_length), dtype=torch.bool, device=tokens.device)
        for study_index, length in enumerate(lengths):
            selected = tokens[image_to_study == study_index].reshape(length, -1)
            padded[study_index, :length] = selected
            valid[study_index, :length] = True
        return padded, valid

    def _attention_token_metadata(self, batch, max_length):
        """Describe padded attention tokens so weights can be mapped back to views."""
        device = batch["image_to_study"].device
        num_studies = int(batch["num_studies"])
        tokens_per_image = self.local_tokens_per_image + int(self.use_global_tokens)
        descriptors = {
            "token_type_ids": torch.full(
                (num_studies, max_length), -1, dtype=torch.long, device=device
            ),
            "token_image_indices": torch.full(
                (num_studies, max_length), -1, dtype=torch.long, device=device
            ),
            "token_view_ids": torch.full(
                (num_studies, max_length), -1, dtype=torch.long, device=device
            ),
            "token_orientation_ids": torch.full(
                (num_studies, max_length), -1, dtype=torch.long, device=device
            ),
            "token_quadrant_ids": torch.full(
                (num_studies, max_length), -1, dtype=torch.long, device=device
            ),
            "token_cell_ids": torch.full(
                (num_studies, max_length), -1, dtype=torch.long, device=device
            ),
        }
        for study_index in range(num_studies):
            source_indices = torch.nonzero(
                batch["image_to_study"] == study_index, as_tuple=False
            ).flatten()
            for image_index, source_index in enumerate(source_indices.tolist()):
                start = image_index * tokens_per_image
                stop = start + tokens_per_image
                descriptors["token_image_indices"][study_index, start:stop] = image_index
                descriptors["token_view_ids"][study_index, start:stop] = batch["view_ids"][source_index]
                descriptors["token_orientation_ids"][study_index, start:stop] = (
                    batch["orientation_ids"][source_index]
                )
                local_start = start
                if self.use_global_tokens:
                    descriptors["token_type_ids"][study_index, start] = 0  # global
                    local_start += 1
                descriptors["token_type_ids"][study_index, local_start:stop] = 1  # local
                descriptors["token_quadrant_ids"][study_index, local_start:stop] = (
                    torch.arange(4, device=device).repeat_interleave(self.tokens_per_patch)
                )
                descriptors["token_cell_ids"][study_index, local_start:stop] = (
                    torch.arange(self.tokens_per_patch, device=device).repeat(4)
                )
        return descriptors

    def forward(self, batch, return_attention=False):
        global_images = batch["global_images"]
        local_patches = batch["local_patches"]
        image_to_study = batch["image_to_study"]
        num_studies = int(batch["num_studies"])
        num_images = global_images.shape[0]

        local = self.local_encoder.forward_tokens(
            local_patches.reshape(-1, *local_patches.shape[2:]),
            grid_size=self.local_token_grid,
        )
        local = local.view(num_images, 4, self.tokens_per_patch, self.feature_dim)
        metadata = self.metadata_encoder(
            batch["view_ids"], batch["orientation_ids"], batch["numeric_metadata"]
        )
        local = (
            local
            + self.quadrant_embedding[None, :, None, :]
            + self.patch_cell_embedding[None, None, :, :]
            + self.metadata_to_token(metadata)[:, None, None, :]
        )
        image_global = self.global_encoder(global_images)
        local = local.flatten(1, 2)
        if self.use_global_tokens:
            global_tokens = (
                image_global
                + self.global_token_embedding[None, :]
                + self.global_view_embedding(batch["view_ids"])
                + self.global_orientation_embedding(batch["orientation_ids"])
            )
            image_tokens = torch.cat((global_tokens.unsqueeze(1), local), dim=1)
        else:
            image_tokens = local
        tokens, valid_tokens = self._pad_study_tokens(
            image_tokens, image_to_study, num_studies
        )

        queries = self.class_queries.unsqueeze(0).expand(num_studies, -1, -1)
        if self.use_study_mean_residual:
            study_global = study_mean(image_global, image_to_study, num_studies)
            study_metadata = study_mean(metadata, image_to_study, num_studies)
            queries = (
                queries
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
        combined = hidden
        if self.use_study_mean_residual:
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
            output.update(self._attention_token_metadata(batch, tokens.shape[1]))
        return output
