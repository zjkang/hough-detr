import copy
import math
from typing import Tuple

import torch
import torchvision
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import PositionEmbeddingLearned, get_sine_pos_embed
from util.misc import inverse_sigmoid

from torchvision.ops import boxes as box_ops


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1),
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


class HoughTransformer(TwostageTransformer):
    def __init__(
        self,
        encoder: nn.Module,
        neck: nn.Module,
        decoder: nn.Module,
        num_classes: int,
        num_feature_levels: int = 4,
        two_stage_num_proposals: int = 900,
        level_filter_ratio: Tuple = (0.25, 0.5, 1.0, 1.0),
        layer_filter_ratio: Tuple = (1.0, 0.8, 0.6, 0.6, 0.4, 0.2),
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        # salience parameters
        self.register_buffer("level_filter_ratio", torch.Tensor(level_filter_ratio))
        self.register_buffer("layer_filter_ratio", torch.Tensor(layer_filter_ratio))
        self.alpha = nn.Parameter(torch.Tensor(3), requires_grad=True)

        # model structure
        self.encoder = encoder
        self.neck = neck
        self.decoder = decoder
        self.tgt_embed = nn.Embedding(two_stage_num_proposals, self.embed_dim)
        self.encoder_class_head = nn.Linear(self.embed_dim, num_classes)
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.encoder.enhance_mcsp = self.encoder_class_head
        self.enc_mask_predictor = MaskPredictor(self.embed_dim, self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize embedding layers
        nn.init.normal_(self.tgt_embed.weight)
        # initialize encoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.encoder_class_head.bias, bias_value)
        # initiailize encoder regression layers
        nn.init.constant_(self.encoder_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.encoder_bbox_head.layers[-1].bias, 0.0)
        # initialize alpha
        self.alpha.data.uniform_(-0.3, 0.3)

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        noised_label_query,
        noised_box_query,
        attn_mask, # query mask from denoising generation self.denoising_generator(gt_labels_list, gt_boxes_list)[2]
    ):
        # get input for encoder
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)

        backbone_output_memory = self.gen_encoder_output_proposals(
            feat_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes
        )[0]

        # calculate filtered tokens numbers for each feature map
        reverse_multi_level_masks = [~m for m in multi_level_masks]
        valid_token_nums = torch.stack([m.sum((1, 2)) for m in reverse_multi_level_masks], -1)
        focus_token_nums = (valid_token_nums * self.level_filter_ratio).int()
        level_token_nums = focus_token_nums.max(0)[0]
        focus_token_nums = focus_token_nums.sum(-1)

        # from high level to low level
        batch_size = feat_flatten.shape[0]
        selected_score = []
        selected_inds = []
        salience_score = []
        for level_idx in range(spatial_shapes.shape[0] - 1, -1, -1):
            start_index = level_start_index[level_idx]
            end_index = level_start_index[level_idx + 1] if level_idx < spatial_shapes.shape[0] - 1 else None
            level_memory = backbone_output_memory[:, start_index:end_index, :]
            mask = mask_flatten[:, start_index:end_index]
            # update the memory using the higher-level score_prediction
            if level_idx != spatial_shapes.shape[0] - 1:
                upsample_score = torch.nn.functional.interpolate(
                    score,
                    size=spatial_shapes[level_idx].unbind(),
                    mode="bilinear",
                    align_corners=True,
                )
                upsample_score = upsample_score.view(batch_size, -1, spatial_shapes[level_idx].prod())
                upsample_score = upsample_score.transpose(1, 2)
                level_memory = level_memory + level_memory * upsample_score * self.alpha[level_idx]
            # predict the foreground score of the current layer
            score = self.enc_mask_predictor(level_memory)
            valid_score = score.squeeze(-1).masked_fill(mask, score.min())
            score = score.transpose(1, 2).view(batch_size, -1, *spatial_shapes[level_idx])

            # get the topk salience index of the current feature map level
            level_score, level_inds = valid_score.topk(level_token_nums[level_idx], dim=1)
            level_inds = level_inds + level_start_index[level_idx]
            salience_score.append(score)
            selected_inds.append(level_inds)
            selected_score.append(level_score)

        selected_score = torch.cat(selected_score[::-1], 1)
        index = torch.sort(selected_score, dim=1, descending=True)[1]
        selected_inds = torch.cat(selected_inds[::-1], 1).gather(1, index)

        # create layer-wise filtering
        num_inds = selected_inds.shape[1]
        # change dtype to avoid shape inference error during exporting ONNX
        cast_dtype = num_inds.dtype if torchvision._is_tracing() else torch.int64
        layer_filter_ratio = (num_inds * self.layer_filter_ratio).to(cast_dtype)
        selected_inds = [selected_inds[:, :r] for r in layer_filter_ratio]
        salience_score = salience_score[::-1]
        foreground_score = self.flatten_multi_level(salience_score).squeeze(-1)
        foreground_score = foreground_score.masked_fill(mask_flatten, foreground_score.min())

        # transformer encoder
        memory = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # salience input
            foreground_score=foreground_score,
            focus_token_nums=focus_token_nums,
            foreground_inds=selected_inds,
            multi_level_masks=multi_level_masks,
        )

        if self.neck is not None:
            feat_unflatten = memory.split(spatial_shapes.prod(-1).unbind(), dim=1)
            feat_unflatten = dict((
                i,
                feat.transpose(1, 2).contiguous().reshape(-1, self.embed_dim, *spatial_shape),
            ) for i, (feat, spatial_shape) in enumerate(zip(feat_unflatten, spatial_shapes)))
            feat_unflatten = list(self.neck(feat_unflatten).values())
            memory = torch.cat([feat.flatten(2).transpose(1, 2) for feat in feat_unflatten], dim=1)

        # get encoder output, classes and coordinates
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        enc_outputs_class = self.encoder_class_head(output_memory)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        if torchvision._is_tracing():
            topk = torch.min(torch.tensor(self.two_stage_num_proposals * 4), enc_outputs_class.shape[1])
        else:
            topk = min(self.two_stage_num_proposals * 4, enc_outputs_class.shape[1])
        topk_scores, topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)
        topk_index = self.nms_on_topk_index(
            topk_scores, topk_index, spatial_shapes, level_start_index, iou_threshold=0.3
        ).unsqueeze(-1)
        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, self.num_classes))
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()
        target = self.tgt_embed.weight.expand(multi_level_feats[0].shape[0], -1, -1)

        # combine with noised_label_query and noised_box_query for denoising training
        if noised_label_query is not None and noised_box_query is not None:
            target = torch.cat([noised_label_query, target], 1)
            reference_points = torch.cat([noised_box_query.sigmoid(), reference_points], 1)

        # decoder
        outputs_classes, outputs_coords = self.decoder(
            query=target,
            value=memory,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            value_pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )

        return outputs_classes, outputs_coords, enc_outputs_class, enc_outputs_coord, salience_score

    @staticmethod
    def fast_repeat_interleave(input, repeats):
        """torch.Tensor.repeat_interleave is slow for one-dimension input for unknown reasons. 
        This is a simple faster implementation. Notice the return shares memory with the input.

        :param input: input Tensor
        :param repeats: repeat numbers of each element in the specified dim
        :param dim: the dimension to repeat, defaults to None
        """
        # the following inplementation runs a little faster under one-dimension settings
        return torch.cat([aa.expand(bb) for aa, bb in zip(input, repeats)])

    @torch.no_grad()
    def nms_on_topk_index(
        self, topk_scores, topk_index, spatial_shapes, level_start_index, iou_threshold=0.3
    ):
        batch_size, num_topk = topk_scores.shape
        if torchvision._is_tracing():
            num_pixels = spatial_shapes.prod(-1).unbind()
        else:
            num_pixels = spatial_shapes.prod(-1).tolist()

        # flatten topk_scores and topk_index for batched_nms
        topk_scores, topk_index = map(lambda x: x.view(-1), (topk_scores, topk_index))

        # get level coordinates for queries and construct boxes for them
        level_index = torch.arange(level_start_index.shape[0], device=level_start_index.device)
        feat_width, start_index, level_idx = map(
            lambda x: self.fast_repeat_interleave(x, num_pixels)[topk_index],
            (spatial_shapes[:, 1], level_start_index, level_index),
        )
        topk_spatial_index = topk_index - start_index
        x = topk_spatial_index % feat_width
        y = torch.div(topk_spatial_index, feat_width, rounding_mode="trunc")
        coordinates = torch.stack([x - 1.0, y - 1.0, x + 1.0, y + 1.0], -1)

        # get unique idx for queries in different images and levels
        image_idx = torch.arange(batch_size).repeat_interleave(num_topk, 0)
        image_idx = image_idx.to(level_idx.device)
        idxs = level_idx + level_start_index.shape[0] * image_idx

        # perform batched_nms
        indices = torchvision.ops.batched_nms(coordinates, topk_scores, idxs, iou_threshold)

        # stack valid index
        results_index = []
        if torchvision._is_tracing():
            min_num = torch.tensor(self.two_stage_num_proposals)
        else:
            min_num = self.two_stage_num_proposals
        # get indices in each image
        for i in range(batch_size):
            topk_index_per_image = topk_index[indices[image_idx[indices] == i]]
            if torchvision._is_tracing():
                min_num = torch.min(topk_index_per_image.shape[0], min_num)
            else:
                min_num = min(topk_index_per_image.shape[0], min_num)
            results_index.append(topk_index_per_image)
        return torch.stack([index[:min_num] for index in results_index])


class HoughTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        dropout=0.1,
        n_heads=8,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
        # focus parameter
        topk_sa=300,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.topk_sa = topk_sa

        # pre attention
        self.pre_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout, batch_first=True)
        self.pre_dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.pre_attention.in_proj_weight)
        nn.init.xavier_uniform_(self.pre_attention.out_proj.weight)
        # initilize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, query):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(query))))
        query = query + self.dropout3(src2)
        query = self.norm2(query)
        return query

    def forward(
        self,
        query,
        query_pos,
        value,  # focus parameter
        reference_points,
        spatial_shapes,
        level_start_index,
        query_key_padding_mask=None,
        # focus parameter
        score_tgt=None,
        foreground_pre_layer=None,
    ):
        mc_score = score_tgt.max(-1)[0] * foreground_pre_layer
        select_tgt_index = torch.topk(mc_score, self.topk_sa, dim=1)[1]
        select_tgt_index = select_tgt_index.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        select_tgt = torch.gather(query, 1, select_tgt_index)
        select_pos = torch.gather(query_pos, 1, select_tgt_index)
        query_with_pos = key_with_pos = self.with_pos_embed(select_tgt, select_pos)
        tgt2 = self.pre_attention(
            query_with_pos,
            key_with_pos,
            select_tgt,
        )[0]
        select_tgt = select_tgt + self.pre_dropout(tgt2)
        select_tgt = self.pre_norm(select_tgt)
        query = query.scatter(1, select_tgt_index, select_tgt)

        # self attention
        src2 = self.self_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=query_key_padding_mask,
        )
        query = query + self.dropout1(src2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class HoughTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: nn.Module, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.embed_dim = encoder_layer.embed_dim

        # learnt background embed for prediction
        self.background_embedding = PositionEmbeddingLearned(200, num_pos_feats=self.embed_dim // 2)

        self.init_weights()

    def init_weights(self):
        # initialize encoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            ref = torch.stack((ref_x, ref_y), -1)  # [n, h*w, 2]
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [n, s, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [n, s, l, 2]
        return reference_points

    def forward(
        self,
        query,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        query_pos=None,
        query_key_padding_mask=None,
        # salience input
        foreground_score=None,
        focus_token_nums=None,
        foreground_inds=None,
        multi_level_masks=None,
    ):
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=query.device)
        b, n, s, p = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = query_pos
        value = output = query
        for layer_id, layer in enumerate(self.layers):
            inds_for_query = foreground_inds[layer_id].unsqueeze(-1).expand(-1, -1, self.embed_dim)
            query = torch.gather(output, 1, inds_for_query)
            query_pos = torch.gather(ori_pos, 1, inds_for_query)
            foreground_pre_layer = torch.gather(foreground_score, 1, foreground_inds[layer_id])
            reference_points = torch.gather(
                ori_reference_points.view(b, n, -1), 1,
                foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, s * p)
            ).view(b, -1, s, p)
            score_tgt = self.enhance_mcsp(query)
            query = layer(
                query,
                query_pos,
                value,
                reference_points,
                spatial_shapes,
                level_start_index,
                query_key_padding_mask,
                score_tgt,
                foreground_pre_layer,
            )
            outputs = []
            for i in range(foreground_inds[layer_id].shape[0]):
                foreground_inds_no_pad = foreground_inds[layer_id][i][:focus_token_nums[i]]
                query_no_pad = query[i][:focus_token_nums[i]]
                outputs.append(
                    output[i].scatter(
                        0,
                        foreground_inds_no_pad.unsqueeze(-1).repeat(1, query.size(-1)),
                        query_no_pad,
                    )
                )
            output = torch.stack(outputs)

        # add learnt embedding for background
        if multi_level_masks is not None:
            background_embedding = [
                self.background_embedding(mask).flatten(2).transpose(1, 2) for mask in multi_level_masks
            ]
            background_embedding = torch.cat(background_embedding, dim=1)
            background_embedding.scatter_(1, inds_for_query, 0)
            background_embedding *= (~query_key_padding_mask).unsqueeze(-1)
            output = output + background_embedding

        return output


class HoughTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        d_ffn=1024,
        n_heads=8,
        dropout=0.1,
        activation=nn.ReLU(inplace=True),
        n_levels=4,
        n_points=4,
        cross_attn_type='sin_scale' # ['def', 'sin_scale', 'mul_scale', 'other']
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads

        # cross attention
        # v1 original
        if cross_attn_type == 'def':
            self.cross_attn = MultiScaleDeformableAttention(
                embed_dim, n_levels, n_heads, n_points)
        # v2 cascade single scale
        elif cross_attn_type == 'sin_scale':
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, n_heads, dropout=dropout, batch_first=True)
        # v3 cascade multi-scale
        elif cross_attn_type == 'mul_scale':
            self.cross_attn = nn.ModuleList(
                [nn.MultiheadAttention(
                    embed_dim, n_heads, dropout=dropout, batch_first=True)
                    for _ in range(n_levels)])
            # se module for level wise attn
            self.lvl_attn = nn.Linear(embed_dim, n_levels)

        self.cross_attn_type = cross_attn_type

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        # self attention
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # ffn
        self.linear1 = nn.Linear(embed_dim, d_ffn)
        self.activation = activation
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, embed_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize self_attention
        nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        # initialize Linear layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        query,
        query_pos,
        reference_points,
        value,
        spatial_shapes,
        level_start_index,
        value_pos=None,
        self_attn_mask=None,
        memory_mask=None,
        key_padding_mask=None,
    ):
        # self attention
        query_with_pos = key_with_pos = self.with_pos_embed(query, query_pos)
        query2 = self.self_attn(
            query=query_with_pos,
            key=key_with_pos,
            value=query,
            attn_mask=self_attn_mask,
        )[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        # cross attention
        if self.cross_attn_type == 'def':
             # v1 original
            query2 = self.cross_attn(
                query=self.with_pos_embed(query, query_pos),
                reference_points=reference_points,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
            )
        elif self.cross_attn_type == 'sin_scale':
            # v2 single scale
            query2 = self.cross_attn(
                query=self.with_pos_embed(query, query_pos),
                key=self.with_pos_embed(value, value_pos),
                value=value,
                attn_mask=memory_mask,
                key_padding_mask=key_padding_mask,
            )[0]
        # v3 multi-scale
        elif self.cross_attn_type == 'mul_scale':
            tmp_query = self.with_pos_embed(query, query_pos)
            query2_list = []
            for idx, (start, end) in enumerate(zip(level_start_index, level_start_index[1:] + [value.shape[1]])):
                value_level = value[:, start:end]
                value_pos_level = value_pos[:, start:end] if value_pos is not None else None
                key_padding_mask_level = key_padding_mask[:, start:end] if key_padding_mask is not None else None
                memory_mask_level = memory_mask[idx] if memory_mask is not None else None

                query2_i, _ = self.cross_attn_list[idx](
                    query=tmp_query,
                    key=self.with_pos_embed(value_level, value_pos_level),
                    value=value_level,
                    attn_mask=memory_mask_level,
                    key_padding_mask=key_padding_mask_level,
                )
                # maybe no need extra to proj
                # query2_i = self.ca_out_v_proj(query2_i)
                # query2_i: (batch_size, num_queries, embed_dim)
                query2_list.append(query2_i)

            # level_weight: (batch_size, num_queries, n_levels) -> (n_levels, batch_size, num_queries, 1)
            level_weight = self.lvl_attn(tmp_query).softmax(-1).permute(2, 0, 1).unsqueeze(-1)

            # (a) simple: 对多个尺度级别的注意力输出进行平均
            # query2 = sum(query2_list) / len(query2_list)
            # (b) or use level_weights query2_all: (n_levels, batch_size, num_queries, embed_dim)
            query2_all = torch.stack(query2_list)
            # query2: (batch_size, num_queries, embed_dim)
            query2 = (query2_all * level_weight).sum(0)


        query = query + self.dropout1(query2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query


class HoughTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, num_classes):
        super().__init__()
        # parameters
        self.embed_dim = decoder_layer.embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.cross_attn_type = decoder_layer.cross_attn_type # ['def', 'sin_scale', 'mul_scale', 'other']

        # decoder layers and embedding
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.ref_point_head = MLP(2 * self.embed_dim, self.embed_dim, self.embed_dim, 2)

        # iterative bounding box refinement
        self.class_head = nn.ModuleList([nn.Linear(self.embed_dim, num_classes) for _ in range(num_layers)])
        self.bbox_head = nn.ModuleList([MLP(self.embed_dim, self.embed_dim, 4, 3) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

    def init_weights(self):
        # initialize decoder layers
        for layer in self.layers:
            if hasattr(layer, "init_weights"):
                layer.init_weights()
        # initialize decoder classification layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_head in self.class_head:
            nn.init.constant_(class_head.bias, bias_value)
        # initiailize decoder regression layers
        for bbox_head in self.bbox_head:
            nn.init.constant_(bbox_head.layers[-1].weight, 0.0)
            nn.init.constant_(bbox_head.layers[-1].bias, 0.0)

    def forward(
        self,
        query,
        reference_points,
        value, # memory by default
        spatial_shapes,
        level_start_index,
        valid_ratios,
        value_pos=None,
        key_padding_mask=None,
        attn_mask=None, # self-attention mask
        memory_mask=None, # cross-attention mask
    ):
        outputs_classes = []
        outputs_coords = []
        # (batch_size, 1, num_levels, 4)
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            # reference_points: (batch_size, num_queries, 4)
            # 乘法的效果：
            # 假设 reference_points 中有一个坐标 [1.0, 1.0]（特征图的右下角）。
            # 如果 valid_ratios 为 [0.8, 0.9]，乘法后坐标变为 [0.8, 0.9]。
            # 这个新坐标现在正确地指向原始图像内容的右下角，而不是填充后的特征图右下角
            # reference_points_input: (batch_size, num_queries, num_levels, 4) -> 4: (x_c,y_c,w,h)
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            if self.cross_attn_type == 'def':
                pass
            elif self.cross_attn_type == 'sin_scale':
                # single scale
                memory_mask = gen_memory_mask(
                    reference_points_input, spatial_shapes, valid_ratios, layer.num_heads)
                memory_mask = torch.cat(memory_mask, dim=-1)
            elif self.cross_attn_type == 'mul_scale':
                memory_mask = gen_memory_mask(
                    reference_points_input, spatial_shapes, valid_ratios, layer.num_heads)


            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                value_pos=value_pos,
                key_padding_mask=key_padding_mask,
                self_attn_mask=attn_mask,
                memory_mask=memory_mask,
            )

            # get output, reference_points are not detached for look_forward_twice
            output_class = self.class_head[layer_idx](self.norm(query))
            output_coord = self.bbox_head[layer_idx](self.norm(query)) + inverse_sigmoid(reference_points)
            output_coord = output_coord.sigmoid()
            outputs_classes.append(output_class)
            outputs_coords.append(output_coord)

            if layer_idx == self.num_layers - 1:
                break

            # iterative bounding box refinement
            # 通过多层迭代，reference_points 逐步细化，每一层都基于前一层的结果进行调整
            reference_points = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points.detach())
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords


# gen_memory_mask 函数的主要作用是为多尺度特征图生成内存掩码(memory mask)
# memory: encoder outut
# 用于在注意力机制中限制每个查询只关注其相关的区域 for cross-attention
def gen_memory_mask(
        reference_points, # 参考点，通常是预测的边界框 (batch_size, num_queries, num_levels, 4) -> 4: (x_c,y_c,w,h)
        spatial_shapes, # [(h_i,w_i)]
        valid_ratios, # (batch_size, num_levels, 2)
        num_heads):
    # num_heads: 注意力头的数量
    # 掩码生成逻辑:
    # 对于每个参考点(边界框),生成一个二进制掩码
    # 掩码中，边界框内的区域为 False(允许注意),边界框外的区域为 True(阻止注意)
    with torch.no_grad():
        mem_mask_list = []
        for level in range(len(spatial_shapes)):
            # mem_mask_shape 计算掩码的形状: (batch_size, num_queries, h_i, w_i)
            mem_mask_shape = (reference_points.shape[0],reference_points.shape[1]) + tuple(spatial_shapes[level].tolist())
            # 初始化为全True
            mem_mask = torch.zeros(mem_mask_shape).to(reference_points) < 1
            # mem_mask = torch.zeros(mem_mask_shape) < 1
            for bs in range(reference_points.shape[0]):
                # 获取当前批次的参考点
                # ref: (num_queries, 4)
                ref = reference_points[bs,:,level,:]
                # 获取当前批次和级别的源图像高度和宽度
                h,w = spatial_shapes[level]
                ref = box_ops._box_cxcywh_to_xyxy(ref)
                # 将边界框坐标转换为整数像素坐标，并确保在有效范围内
                ref[:,0] = torch.clamp(torch.floor(ref[:,0]*w),min=0,max=w-1)
                ref[:,1] = torch.clamp(torch.floor(ref[:,1]*h),min=0,max=h-1)
                ref[:,2] = torch.clamp(torch.ceil(ref[:,2]*w),min=1,max=w)
                ref[:,3] = torch.clamp(torch.ceil(ref[:,3]*h),min=1,max=h)
                # 创建高度方向的掩码，True表示在边界框外 hMask: (num_queries, h_i)
                hMask = torch.logical_or(torch.arange(mem_mask_shape[2]).unsqueeze(0).to(ref)<ref[:, 1, None], torch.arange(mem_mask_shape[2]).unsqueeze(0).to(ref)>=ref[:, 3, None])
                # 创建宽度方向的掩码，True表示在边界框外
                wMask = torch.logical_or(torch.arange(mem_mask_shape[3]).unsqueeze(0).to(ref)<ref[:, 0, None], torch.arange(mem_mask_shape[3]).unsqueeze(0).to(ref)>=ref[:, 2, None])
                # 合并高度和宽度掩码，得到最终的2D掩码
                mem_mask[bs] = torch.logical_or(hMask.unsqueeze(2), wMask.unsqueeze(1))

            # 将掩码从4D展平为3D (batch_size, num_queries, h_i * w_i)
            mem_mask = mem_mask.flatten(2)
            # https://stackoverflow.com/questions/68205894/how-to-prepare-data-for-tpytorchs-3d-attn-mask-argument-in-multiheadattention
            # 重复掩码以匹配多头注意力的形状
            mem_mask = torch.repeat_interleave(mem_mask, num_heads, dim=0)
            # 将当前级别的掩码添加到列表中
            mem_mask_list.append(mem_mask)
    # mem_mask_list: [(batch_size * num_heads, num_queries, h_i * w_i)]
    return mem_mask_list
