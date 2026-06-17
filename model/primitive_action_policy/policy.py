"""Primitive-action policy model for ResPlan-conditioned autoregressive training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.primitive_action_policy.config import PrimitiveActionPolicyConfig
from model.primitive_action_policy.layers import make_image_encoder, masked_mean


class PrimitiveActionPolicyModel(nn.Module):
    """Predict the next primitive action from plan, screenshot, floorplan, and action history."""

    def __init__(self, config: PrimitiveActionPolicyConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_size

        self.image_encoder = make_image_encoder(config.image_channels, h)
        self.global_floorplan_encoder = make_image_encoder(config.image_channels, h)
        self.floorplan_cross_attention = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.floorplan_cross_norm = nn.LayerNorm(h)

        self.wall_encoder = nn.Sequential(
            nn.Linear(config.wall_feature_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
        )
        self.insertion_encoder = nn.Sequential(
            nn.Linear(config.insertion_feature_dim, h),
            nn.GELU(),
            nn.Linear(h, h),
            nn.LayerNorm(h),
        )
        self.plan_projection = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.LayerNorm(h))
        self.progress_encoder = nn.Sequential(
            nn.Linear(config.progress_feature_dim, h),
            nn.GELU(),
            nn.LayerNorm(h),
        )

        self.action_type_embedding = nn.Embedding(config.num_action_types, h)
        self.high_level_embedding = nn.Embedding(config.num_high_level_actions, h)
        self.gui_action_embedding = nn.Embedding(config.num_gui_actions, h)
        self.key_embedding = nn.Embedding(config.num_keys + 1, h)
        self.coordinate_frame_embedding = nn.Embedding(config.num_coordinate_frames, h)
        self.primitive_projection = nn.Sequential(
            nn.Linear(config.primitive_action_dim, h),
            nn.GELU(),
            nn.LayerNorm(h),
        )
        self.history_projection = nn.Sequential(nn.Linear(h * 6, h), nn.GELU(), nn.LayerNorm(h))

        self.plan_token = nn.Parameter(torch.zeros(1, 1, h))
        self.observation_token = nn.Parameter(torch.zeros(1, 1, h))
        self.progress_token = nn.Parameter(torch.zeros(1, 1, h))
        self.step_token = nn.Parameter(torch.zeros(1, 1, h))
        self.query_token = nn.Parameter(torch.zeros(1, 1, h))
        self.position_embedding = nn.Embedding(config.history_length + 5, h)
        self.step_index_embedding = nn.Embedding(config.max_step_index + 1, h)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.num_attention_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_transformer_layers)
        self.final_norm = nn.LayerNorm(h)

        self.action_type_head = nn.Linear(h, config.num_action_types)
        self.high_level_head = nn.Linear(h, config.num_high_level_actions)
        self.gui_action_head = nn.Linear(h, config.num_gui_actions)
        self.coordinate_frame_head = nn.Linear(h, config.num_coordinate_frames)
        self.xy_head = nn.Linear(h, 2)
        self.key_head = nn.Linear(h, config.num_keys)
        self.repeat_head = nn.Linear(h, config.max_repeat_count + 1)
        self.interval_head = nn.Linear(h, 1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode_plan(self, plan: dict[str, torch.Tensor]) -> torch.Tensor:
        walls = plan["walls"]
        wall_mask = plan["wall_mask"]
        insertions = plan["insertions"]
        insertion_mask = plan["insertion_mask"]

        wall_features = self.wall_encoder(walls) if walls.shape[1] else torch.zeros(
            walls.shape[0], 0, self.config.hidden_size, device=walls.device, dtype=walls.dtype
        )
        insertion_features = self.insertion_encoder(insertions) if insertions.shape[1] else torch.zeros(
            insertions.shape[0], 0, self.config.hidden_size, device=insertions.device, dtype=insertions.dtype
        )
        wall_context = masked_mean(wall_features, wall_mask)
        insertion_context = masked_mean(insertion_features, insertion_mask)
        return self.plan_projection(torch.cat([wall_context, insertion_context], dim=-1))

    def encode_history(self, history: dict[str, torch.Tensor]) -> torch.Tensor:
        primitive = history["primitive_action"].float().clone()
        primitive[..., 0] = primitive[..., 0].clamp(min=0) / max(self.config.num_action_types - 1, 1)
        primitive[..., 3] = primitive[..., 3].clamp(min=0) / max(self.config.num_keys - 1, 1)
        primitive[..., 4] = primitive[..., 4].clamp(min=0) / max(self.config.max_repeat_count, 1)
        primitive[..., 5] = primitive[..., 5].clamp(min=0) / 1000.0
        action_type = history["action_type_id"].clamp(min=0, max=self.config.num_action_types - 1)
        high_level = history["high_level_id"].clamp(min=0, max=self.config.num_high_level_actions - 1)
        gui_action = history["gui_action_id"].clamp(min=0, max=self.config.num_gui_actions - 1)
        key = history["key_id"].clamp(min=-1, max=self.config.num_keys - 1) + 1
        coordinate_frame = history["coordinate_frame_id"].clamp(min=0, max=self.config.num_coordinate_frames - 1)
        pieces = [
            self.primitive_projection(primitive),
            self.action_type_embedding(action_type),
            self.high_level_embedding(high_level),
            self.gui_action_embedding(gui_action),
            self.key_embedding(key),
            self.coordinate_frame_embedding(coordinate_frame),
        ]
        return self.history_projection(torch.cat(pieces, dim=-1))

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        observation = batch.get("observation")
        if observation is None:
            raise KeyError("PrimitiveActionPolicyModel expects batch['observation']")
        observation = observation.to(dtype=self.plan_token.dtype)
        plan = batch["plan"]
        history = batch["history"]

        batch_size = observation.shape[0]
        plan_embedding = self.encode_plan(plan) + self.plan_token.squeeze(1)
        observation_embedding = self.image_encoder(observation) + self.observation_token.squeeze(1)
        progress = batch.get("progress")
        if progress is None:
            progress = torch.zeros(
                (batch_size, self.config.progress_feature_dim),
                dtype=observation.dtype,
                device=observation.device,
            )
        progress_embedding = self.progress_encoder(progress.float()) + self.progress_token.squeeze(1)
        step_index = batch.get("step_index")
        if step_index is None:
            step_index = torch.zeros((batch_size,), dtype=torch.long, device=observation.device)
        step_embedding = (
            self.step_index_embedding(step_index.long().clamp(min=0, max=self.config.max_step_index))
            + self.step_token.squeeze(1)
        )
        history_embeddings = self.encode_history(history)
        query_embedding = self.query_token.expand(batch_size, -1, -1).squeeze(1)

        tokens = torch.cat(
            [
                plan_embedding.unsqueeze(1),
                observation_embedding.unsqueeze(1),
                progress_embedding.unsqueeze(1),
                step_embedding.unsqueeze(1),
                history_embeddings,
                query_embedding.unsqueeze(1),
            ],
            dim=1,
        )
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        tokens = tokens + self.position_embedding(positions)

        key_padding_mask = torch.cat(
            [
                torch.zeros((batch_size, 4), dtype=torch.bool, device=tokens.device),
                ~history["mask"].to(torch.bool),
                torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        hidden = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        query_hidden = self.final_norm(hidden[:, -1])
        global_floorplan = batch.get("global_floorplan")
        if global_floorplan is not None:
            global_floorplan = global_floorplan.to(dtype=self.plan_token.dtype)
            floorplan_embedding = self.global_floorplan_encoder(global_floorplan)
            floorplan_available = batch.get("global_floorplan_available")
            if floorplan_available is None:
                floorplan_available = torch.ones((batch_size,), dtype=torch.bool, device=floorplan_embedding.device)
            floorplan_available_f = floorplan_available.to(floorplan_embedding.dtype).view(batch_size, 1)
            floorplan_embedding = floorplan_embedding * floorplan_available_f
            floorplan_context, _ = self.floorplan_cross_attention(
                query=query_hidden.unsqueeze(1),
                key=floorplan_embedding.unsqueeze(1),
                value=floorplan_embedding.unsqueeze(1),
                need_weights=False,
            )
            query_hidden = self.floorplan_cross_norm(query_hidden + floorplan_context.squeeze(1) * floorplan_available_f)

        xy = self.xy_head(query_hidden)
        if self.config.xy_output_activation == "sigmoid":
            xy = torch.sigmoid(xy)
        elif self.config.xy_output_activation not in {"identity", "none", None}:
            raise ValueError(f"Unsupported xy_output_activation: {self.config.xy_output_activation}")

        return {
            "action_type_logits": self.action_type_head(query_hidden),
            "high_level_logits": self.high_level_head(query_hidden),
            "gui_action_logits": self.gui_action_head(query_hidden),
            "coordinate_frame_logits": self.coordinate_frame_head(query_hidden),
            "xy": xy,
            "key_logits": self.key_head(query_hidden),
            "repeat_logits": self.repeat_head(query_hidden),
            "key_interval": self.interval_head(query_hidden).squeeze(-1),
        }

    def _latest_history_wall_move_xy(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history = batch["history"]
        wall_move = history.get("is_wall_move")
        if wall_move is None:
            batch_size = history["mask"].shape[0]
            xy = history["xy"].new_zeros((batch_size, 2))
            empty = torch.zeros((batch_size,), dtype=torch.bool, device=xy.device)
            return xy, empty, empty

        valid = wall_move.bool() & history["mask"].bool()
        has_previous = valid.any(dim=1)
        positions = torch.arange(valid.shape[1], device=valid.device).view(1, -1)
        latest_index = (valid.long() * positions).max(dim=1).values
        gather_index = latest_index.view(-1, 1, 1).expand(-1, 1, 2)
        previous_xy = history["xy"].gather(dim=1, index=gather_index).squeeze(1)

        history_valid = history["mask"].bool()
        has_immediate_previous = history_valid.any(dim=1)
        immediate_index = (history_valid.long() * positions).max(dim=1).values
        immediate_action = history["action_type_id"].gather(dim=1, index=immediate_index.view(-1, 1)).squeeze(1)
        previous_action_was_click = has_immediate_previous & (immediate_action == 2)
        return previous_xy, has_previous, previous_action_was_click

    def _wall_geometry_losses(
        self,
        batch: dict[str, Any],
        outputs: dict[str, torch.Tensor],
        zero: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target = batch["target"]
        current_wall_move = target.get("is_wall_move")
        if current_wall_move is None:
            return {
                "loss_wall_orthogonal": zero,
                "loss_wall_length": zero,
                "loss_wall_endpoint": zero,
            }

        previous_xy, has_previous, previous_action_was_click = self._latest_history_wall_move_xy(batch)
        current_wall_move = current_wall_move.bool()
        pair_mask = current_wall_move & has_previous
        if not pair_mask.any():
            return {
                "loss_wall_orthogonal": zero,
                "loss_wall_length": zero,
                "loss_wall_endpoint": zero,
            }

        cfg = self.config
        target_delta = target["xy"] - previous_xy
        target_length = torch.linalg.vector_norm(target_delta, dim=-1)
        endpoint_mask = pair_mask & (target_length <= cfg.wall_endpoint_epsilon)
        segment_mask = pair_mask & previous_action_was_click & (target_length > cfg.wall_endpoint_epsilon)

        losses = {
            "loss_wall_orthogonal": zero,
            "loss_wall_length": zero,
            "loss_wall_endpoint": zero,
        }
        if endpoint_mask.any():
            losses["loss_wall_endpoint"] = F.smooth_l1_loss(
                outputs["xy"][endpoint_mask],
                previous_xy[endpoint_mask],
                beta=cfg.xy_smooth_l1_beta,
            )

        if segment_mask.any():
            pred_delta = outputs["xy"] - previous_xy
            horizontal = target_delta[:, 0].abs() >= target_delta[:, 1].abs()
            orthogonal_error = torch.where(horizontal, pred_delta[:, 1], pred_delta[:, 0])
            losses["loss_wall_orthogonal"] = F.smooth_l1_loss(
                orthogonal_error[segment_mask],
                torch.zeros_like(orthogonal_error[segment_mask]),
                beta=cfg.xy_smooth_l1_beta,
            )

            pred_length = torch.linalg.vector_norm(pred_delta, dim=-1)
            target_segment_length = target_length[segment_mask]
            pred_segment_length = pred_length[segment_mask]
            length_match = F.smooth_l1_loss(
                pred_segment_length,
                target_segment_length,
                beta=cfg.xy_smooth_l1_beta,
            )
            required_length = torch.maximum(
                target_segment_length * 0.85,
                torch.full_like(target_segment_length, cfg.wall_min_segment_length),
            )
            short_wall_penalty = F.smooth_l1_loss(
                torch.clamp(pred_segment_length, max=required_length),
                required_length,
                beta=cfg.xy_smooth_l1_beta,
            )
            losses["loss_wall_length"] = length_match + short_wall_penalty
        return losses

    def compute_loss(self, batch: dict[str, Any], outputs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        if outputs is None:
            outputs = self.forward(batch)
        target = batch["target"]
        cfg = self.config

        action_type = target["action_type_id"].long()
        zero = outputs["xy"].sum() * 0.0
        losses: dict[str, torch.Tensor] = {
            "loss_action_type": F.cross_entropy(outputs["action_type_logits"], action_type),
            "loss_high_level": F.cross_entropy(outputs["high_level_logits"], target["high_level_id"].long()),
            "loss_gui_action": F.cross_entropy(outputs["gui_action_logits"], target["gui_action_id"].long()),
        }

        is_move = target["is_move"].bool()
        is_key_action = target["is_key_action"].bool()
        losses["loss_xy"] = (
            F.smooth_l1_loss(outputs["xy"][is_move], target["xy"][is_move], beta=cfg.xy_smooth_l1_beta)
            if is_move.any()
            else zero
        )
        losses.update(self._wall_geometry_losses(batch, outputs, zero))

        key_target = target["key_id"].long()
        valid_key = is_key_action & (key_target >= 0)
        losses["loss_key"] = (
            F.cross_entropy(outputs["key_logits"][valid_key], key_target[valid_key]) if valid_key.any() else zero
        )

        repeat_target = target["key_repeat_count"].long().clamp(min=0, max=cfg.max_repeat_count)
        valid_repeat = is_key_action & (target["key_repeat_count"] >= 0)
        losses["loss_repeat"] = (
            F.cross_entropy(outputs["repeat_logits"][valid_repeat], repeat_target[valid_repeat])
            if valid_repeat.any()
            else zero
        )

        if cfg.ignore_interval_loss:
            losses["loss_interval"] = zero
        else:
            valid_interval = is_key_action & (target["key_interval"] >= 0)
            losses["loss_interval"] = (
                F.smooth_l1_loss(
                    outputs["key_interval"][valid_interval],
                    target["key_interval"][valid_interval] / 1000.0,
                )
                if valid_interval.any()
                else zero
            )

        losses["loss"] = (
            cfg.loss_action_type_weight * losses["loss_action_type"]
            + cfg.loss_high_level_weight * losses["loss_high_level"]
            + cfg.loss_gui_action_weight * losses["loss_gui_action"]
            + cfg.loss_xy_weight * losses["loss_xy"]
            + cfg.loss_wall_orthogonal_weight * losses["loss_wall_orthogonal"]
            + cfg.loss_wall_length_weight * losses["loss_wall_length"]
            + cfg.loss_wall_endpoint_weight * losses["loss_wall_endpoint"]
            + cfg.loss_key_weight * losses["loss_key"]
            + cfg.loss_repeat_weight * losses["loss_repeat"]
            + cfg.loss_interval_weight * losses["loss_interval"]
        )
        return losses

    @torch.no_grad()
    def decode_action(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        action_type_id = outputs["action_type_logits"].argmax(dim=-1)
        high_level_id = outputs["high_level_logits"].argmax(dim=-1)
        gui_action_id = outputs["gui_action_logits"].argmax(dim=-1)
        coordinate_frame_id = outputs["coordinate_frame_logits"].argmax(dim=-1)
        x = outputs["xy"][:, 0]
        y = outputs["xy"][:, 1]
        key_id = outputs["key_logits"].argmax(dim=-1)
        repeat_count = outputs["repeat_logits"].argmax(dim=-1)
        if self.config.ignore_interval_loss:
            key_interval = torch.full_like(outputs["key_interval"], float(self.config.default_key_interval_ms))
        else:
            key_interval = outputs["key_interval"] * 1000.0
        primitive_action = torch.stack(
            [action_type_id.float(), x, y, key_id.float(), repeat_count.float(), key_interval],
            dim=-1,
        )
        non_move = action_type_id != 1
        primitive_action[non_move, 1:3] = -1
        non_key = ~torch.isin(action_type_id, torch.tensor([3, 4], device=action_type_id.device))
        primitive_action[non_key, 3:] = -1
        decoded = {
            "primitive_action": primitive_action,
            "action_type_id": action_type_id,
            "high_level_id": high_level_id,
            "gui_action_id": gui_action_id,
            "coordinate_frame_id": coordinate_frame_id,
        }
        return decoded
