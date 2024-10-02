import copy
import math
from typing import Tuple

import torch
import torchvision
from torch import nn

from models.bricks.base_transformer import TwostageTransformer
from models.bricks.basic import MLP
from models.bricks.hough import Hough
from models.bricks.ms_deform_attn import MultiScaleDeformableAttention
from models.bricks.position_encoding import PositionEmbeddingLearned, get_sine_pos_embed

from util.misc import inverse_sigmoid


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


BN_MOMENTUM = 0.1

class HeatmapPredictor(nn.Module):
    def __init__(
        self,
        num_classes,
        embed_dim,
        region_num,
        vote_field_size=17,
        head_conv=64):
        super().__init__()
        self.inplanes = embed_dim # default=64
        self.region_num = region_num
        self.num_classes = num_classes
        self.vote_field_size = vote_field_size
        self.head_conv = head_conv
        self.deconv_layers = self._make_deconv_layer2(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )
        # voting-map vs voting diff
        # voting-map is learnable, voting is aggregated result from Hough
        # learn to generate voting-map
        num_output = self.num_classes * self.region_num
        self.voting_map_hm = nn.Sequential(
            nn.Conv2d(self.inplanes, self.head_conv,
                kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, self.num_classes,
                kernel_size=1, stride=1, padding=0))
        self.voting_hm = Hough(
            region_num=self.region_num,
            vote_field_size=self.vote_field_size,
            num_classes=self.num_classes
        )

        self.init_weights()  # 在__init__的最后调用初始化方法

    def _make_deconv_layer2(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = nn.Conv2d(self.inplanes, planes,
                    kernel_size=3, stride=1,
                    padding=1, dilation=1, bias=False)
            self.fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def fill_fc_weights(layers):
        # layers.modules() 返回该模块中的所有子模块（如卷积层、全连接层、激活函数等）
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # x: (b, c=embed_dim, h, w)
    def forward(self, x):
        # x: (b, region_num * num_classes, h, w)
        # NOTE!!! does it need to add this?
        # x = self.deconv_layers(x)
        # x: (b, region_num * num_classes, h, w)
        x = self.voting_map_hm(x)
        # out: (b, num_classes, h, w)
        out = self.voting_hm(x)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


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
        region_num: int = 17,
        vote_field_size: int = 65
    ):
        super().__init__(num_feature_levels, encoder.embed_dim)
        # model parameters
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes
        # hough parameters
        self.region_num = region_num
        self.vote_field_size = vote_field_size
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
        # input dimension: embed_dim, hidden dimension: embed_dim
        # 4: x,y,width,height, 3: # of layers
        self.encoder_bbox_head = MLP(self.embed_dim, self.embed_dim, 4, 3)
        self.encoder.enhance_mcsp = self.encoder_class_head
        self.enc_mask_voting_hm_predictor = HeatmapPredictor(
            self.num_classes, self.embed_dim, self.vote_field_size)

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
        multi_level_feats, # [(b,c,h_i,w_i)]
        multi_level_masks, # [(b,h_i,w_i)]
        multi_level_pos_embeds, # [(b,c,h_i,w_i)]
        noised_label_query, # (b,n_noise,c), n_noise: Number of noised queries
        noised_box_query, # (b,n_noise,4)
        attn_mask, # ???(b,n_q,n_q), n_q: number of queries
    ):
        # get input for encoder
        # feat_flatten: (b, sum(h_i * w_i), c=embed_dim)
        feat_flatten = self.flatten_multi_level(multi_level_feats)
        # mask_flatten: (b, sum(h_i * w_i))
        mask_flatten = self.flatten_multi_level(multi_level_masks)
        # lvl_pos_embed_flatten: (b, sum(h_i * w_i), c)
        lvl_pos_embed_flatten = self.get_lvl_pos_embed(multi_level_pos_embeds)
        # spatial_shapes: (L, 2), level_start_index: (L+1), valid_ratios: (b, L, 2)
        spatial_shapes, level_start_index, valid_ratios = self.multi_level_misc(multi_level_masks)
        # backbone_output_memory: (b, sum(h_i * w_i), c)
        backbone_output_memory = self.gen_encoder_output_proposals(
            feat_flatten + lvl_pos_embed_flatten, mask_flatten, spatial_shapes
        )[0]

        # calculate filtered tokens numbers for each feature map
        # reverse_multi_level_masks: [(b, h_i, w_i)]
        reverse_multi_level_masks = [~m for m in multi_level_masks]
        # valid_token_nums: (b, L), L is the number of feature levels
        valid_token_nums = torch.stack([m.sum((1, 2)) for m in reverse_multi_level_masks], -1)
        # focus_token_nums: (b, L)
        focus_token_nums = (valid_token_nums * self.level_filter_ratio).int()
        # level_token_nums: (L,)
        level_token_nums = focus_token_nums.max(0)[0]
        # focus_token_nums: (b,)
        focus_token_nums = focus_token_nums.sum(-1)

        # generate hm voting for each level
        batch_size = feat_flatten.shape[0]
        heat_maps = []
        selected_score = []
        selected_inds = []
        salience_score = []
        for level_idx in range(spatial_shapes.shape[0] - 1, -1, -1):
            start_index = level_start_index[level_idx]
            end_index = level_start_index[level_idx + 1] if level_idx < spatial_shapes.shape[0] - 1 else None
            # level_memory: (b, h_i * w_i, c)
            level_memory = backbone_output_memory[:, start_index:end_index, :]
            # mask: (b, h_i * w_i)
            mask = mask_flatten[:, start_index:end_index]
            # reshape level_memory to match spatial dimensions (b, c, h_i, w_i)
            level_memory = level_memory.permute(0, 2, 1).view(batch_size, -1, *spatial_shapes[level_idx])

            if level_idx != spatial_shapes.shape[0] - 1:
                # Upsample to the first level's resolution
                heat_map = torch.nn.functional.interpolate(
                    heat_map,
                    size=spatial_shapes[0].unbind(),
                    mode="bilinear",
                    align_corners=True
                )

            # heat_map:(b, num_classes, h_i, w_i)
            heat_map = self.enc_mask_voting_hm_predictor(level_memory)
            # heat_map:(b, num_classes, h_i, w_i) -> (b, h_i*w_i, num_classes)
            heat_map = heat_map.permute(0, 2, 3, 1).view(batch_size, -1, self.num_classes)

            # 方法1：简单求和
            # foreground_score = heat_map.sum(dim=-1)

            # # 方法2：归一化后求和
            # normalized_heat_map = F.softmax(heat_map, dim=-1)
            # foreground_score = normalized_heat_map.sum(dim=-1)

            # # 方法3：取最大值
            # foreground_score, _ = heat_map.max(dim=-1)

            # # 方法4：加权求和（假设我们有一个权重向量）
            # weights = torch.tensor([w1, w2, ..., wn]).to(heat_map.device)
            # foreground_score = (heat_map * weights).sum(dim=-1)

            # # 方法5：阈值处理后求和
            # threshold = 0.5
            # thresholded_heat_map = heat_map * (heat_map > threshold)
            # foreground_score = thresholded_heat_map.sum(dim=-1)
            # score:(b, h_i*w_i, 1)
            score = heat_map.max(dim=-1, keepdim=True)
            # valid_score:(b, h_i*w_i)
            valid_score = score.squeeze(-1).masked_fill(mask, score.min())
            # score:(b, h_i*w_i, 1) -> (b, 1, h_i, w_i)
            score = score.transpose(1, 2).view(
                batch_size, -1, *spatial_shapes[level_idx])

            # level_score:(b, k), level_inds:(b, k)
            level_score, level_inds = valid_score.topk(
                level_token_nums[level_idx], dim=1)
            # level_inds:(b, k) -> to flatten index
            level_inds = level_inds + level_start_index[level_idx]

            # heat_map: (b, num_classes, h_i, w_i)
            heat_map.transpose(1, 2).view(batch_size, -1, *spatial_shapes[level_idx])
            heat_maps.append(heat_map)

            salience_score.append(score)
            selected_inds.append(level_inds)
            selected_score.append(level_score)

        # 从低层级到高层级排列, 沿着维度 1（列方向）将所有层级的得分连接起来
        # selected_score: (b, sum(k_i))
        selected_score = torch.cat(selected_score[::-1], 1)
        index = torch.sort(selected_score, dim=1, descending=True)[1]
        # selected_inds: (b, sum(k_i)) 按显著性得分从高到低排序的 token 索引
        selected_inds = torch.cat(selected_inds[::-1], 1).gather(1, index)

        # Combine heatmaps from all levels
        # combined_voting_hm: (b, sum(h_i x w_i), num_classes)
        # combined_voting_hm = torch.cat(heat_maps[::-1], dim=1)
        # combined_voting_hm = combined_voting_hm / combined_voting_hm.max()

        # create layer-wise filtering
        num_inds = selected_inds.shape[1]
        # change dtype to avoid shape inference error during exporting ONNX
        cast_dtype = num_inds.dtype if torchvision._is_tracing() else torch.int64
        layer_filter_ratio = (num_inds * self.layer_filter_ratio).to(cast_dtype)
        # transformer encode: 对每一层，根据计算出的保留数量 r，从 selected_inds 中选择前 r 个索引
        # selected_inds: [[d1,d2,d_r]]: selected foreground index each layer
        selected_inds = [selected_inds[:, :r] for r in layer_filter_ratio]
        # 反转 salience_score 列表，使其从低层级到高层级排列
        # salience_score: [(b, 1, h_i, w_i)]
        salience_score = salience_score[::-1]
        # foreground_score: (b,sum(h_i*w_i))
        foreground_score = self.flatten_multi_level(salience_score).squeeze(-1)
        # foreground_score: (b,sum(h_i*w_i))
        foreground_score = foreground_score.masked_fill(mask_flatten, foreground_score.min())

        # heat_maps: [(b, num_classes, h_i, w_i)]
        heat_maps = heat_maps[::-1]
        # foreground_heat_map: (b, sum(h_i*w_i), num_classes)
        foreground_heat_map = self.flatten_multi_level(heat_maps)

        # transformer encoder
        # memory: (batch_size, sum(h_i * w_i), embed_dim)
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
            # heat map inputs
            heat_maps=foreground_heat_map
        )

        # neck特征融合是在encoder之后做的
        if self.neck is not None:
            # feat_unflatten: tuple((b, embed_dim, h_i*w_i),)
            feat_unflatten = memory.split(spatial_shapes.prod(-1).unbind(), dim=1)
            # feat_unflatten: dict(i, (b, embed_dim, h_i*w_i))
            feat_unflatten = dict((
                i,
                feat.transpose(1, 2).contiguous().reshape(-1, self.embed_dim, *spatial_shape),
            ) for i, (feat, spatial_shape) in enumerate(zip(feat_unflatten, spatial_shapes)))
            # feat_unflatten: [(b, embed_dim, h_i*w_i)]
            feat_unflatten = list(self.neck(feat_unflatten).values())
            # memory: (b, sum(h_i*w_i), embed_dim)
            memory = torch.cat([feat.flatten(2).transpose(1, 2) for feat in feat_unflatten], dim=1)

        # get encoder output, classes and coordinates
        # output_memory: (b, sum(h_i*w_i), embed_dim)
        # output_proposals: (b, sum(h_i*w_i), 4)
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # enc_outputs_class: (b, sum(h_i*w_i), num_classes)
        enc_outputs_class = self.encoder_class_head(output_memory)
        # enc_outputs_coord: (b, sum(h_i*w_i), 4)
        enc_outputs_coord = self.encoder_bbox_head(output_memory) + output_proposals
        # enc_outputs_coord: (b, sum(h_i*w_i), 4)
        enc_outputs_coord = enc_outputs_coord.sigmoid()

        # get topk output classes and coordinates
        if torchvision._is_tracing():
            topk = torch.min(torch.tensor(self.two_stage_num_proposals * 4), enc_outputs_class.shape[1])
        else:
            topk = min(self.two_stage_num_proposals * 4, enc_outputs_class.shape[1])
        # topk_scores: (b, topk), topk_index: (b, topk) 降序
        topk_scores, topk_index = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)
        # 使用 NMS 算法对 topk_scores 和 topk_index 进行处理，以抑制重复的预测
        # topk_index: (b, min_num): min_num 是每个图像中 NMS 后保留的预测数量 < topk
        topk_index = self.nms_on_topk_index(
            topk_scores, topk_index, spatial_shapes, level_start_index, iou_threshold=0.3
        ).unsqueeze(-1)
        # enc_outputs_class: (b, min_num, num_classes)
        enc_outputs_class = enc_outputs_class.gather(1, topk_index.expand(-1, -1, self.num_classes))
        # enc_outputs_coord: (b, min_num, 4)
        enc_outputs_coord = enc_outputs_coord.gather(1, topk_index.expand(-1, -1, 4))

        # get target and reference points
        reference_points = enc_outputs_coord.detach()
        # initial input query for the decoder: "object queries" or "target queries"
        # self.tgt_embed.weight 的形状通常是 (num_queries, embed_dim); b=multi_level_feats[0].shape[0]
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
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_mask=attn_mask,
        )

        return outputs_classes, outputs_coords,\
            enc_outputs_class, enc_outputs_coord,\
            salience_score, heat_maps

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
        # topk_scores: (batch_size, num_topk)
        # topk_index: (batch_size, num_topk)
        batch_size, num_topk = topk_scores.shape
        if torchvision._is_tracing():
            num_pixels = spatial_shapes.prod(-1).unbind()
        else:
            num_pixels = spatial_shapes.prod(-1).tolist()

        # flatten topk_scores and topk_index for batched_nms
        # topk_scores: (batch_size * num_topk),topk_index: (batch_size * num_topk)
        topk_scores, topk_index = map(lambda x: x.view(-1), (topk_scores, topk_index))

        # get level coordinates for queries and construct boxes for them
        level_index = torch.arange(level_start_index.shape[0], device=level_start_index.device)
        # feat_width: 用于后续计算 x 和 y 坐标。
        # start_index: 用于计算每个预测在其特征层中的相对位置。
        # level_idx: 用于区分不同特征层的预测，这在执行 NMS 时很重要
        # feat_width: (batch_size * num_topk,)
        # start_index: (batch_size * num_topk,)
        # level_idx: (batch_size * num_topk,)
        feat_width, start_index, level_idx = map(
            lambda x: self.fast_repeat_interleave(x, num_pixels)[topk_index],
            (spatial_shapes[:, 1], level_start_index, level_index),
        )
        # 有两个特征层：
        # 第一层大小为 10x10 (100个元素)
        # 第二层大小为 5x5 (25个元素)
        # level_start_index 为 [0, 100]
        # 0 是第一层的起始索引
        # 100 是第二层的起始索引   
        #  topk_spatial_index: [50 - 0, 105 - 100, 20 - 0, 110 - 100, 80 - 0] = [50, 5, 20, 10, 80]
        topk_spatial_index = topk_index - start_index
        x = topk_spatial_index % feat_width
        y = torch.div(topk_spatial_index, feat_width, rounding_mode="trunc")
        # 生成的 coordinates 将用于非极大值抑制（NMS）操作
        coordinates = torch.stack([x - 1.0, y - 1.0, x + 1.0, y + 1.0], -1)

        # get unique idx for queries in different images and levels
        # image_idx: (batch_size, num_topk):  batch_size=2, num_topk=3，结果会是 [0,0,0,1,1,1]
        image_idx = torch.arange(batch_size).repeat_interleave(num_topk, 0)
        image_idx = image_idx.to(level_idx.device)
        # idxs: (batch_size * num_topk)
        idxs = level_idx + level_start_index.shape[0] * image_idx

        # perform batched_nms
        # 低 IoU 阈值导致 NMS 更"激进"，更容易将边界框视为重复并抑制
        indices = torchvision.ops.batched_nms(coordinates, topk_scores, idxs, iou_threshold)

        # stack valid index
        results_index = []
        if torchvision._is_tracing():
            min_num = torch.tensor(self.two_stage_num_proposals)
        else:
            min_num = self.two_stage_num_proposals
        # get indices in each image
        for i in range(batch_size):
            # M 是第 i 个图像中 NMS 后保留的预测数量 形状: (M,)，其中 M ≤ N，且 M ≤ num_topk
            topk_index_per_image = topk_index[indices[image_idx[indices] == i]]
            if torchvision._is_tracing():
                min_num = torch.min(topk_index_per_image.shape[0], min_num)
            else:
                min_num = min(topk_index_per_image.shape[0], min_num)
            results_index.append(topk_index_per_image)
        # 将每个图像的结果索引列表堆叠成一个张量，并只保留前 min_num 个元素 每个图像选择相同数量的预测
        #
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

        # pre attention: self-attention for foreground
        self.pre_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout, batch_first=True)
        self.pre_dropout = nn.Dropout(dropout)
        self.pre_norm = nn.LayerNorm(embed_dim)

        # self attention: multi-scale deformable attention for foreground
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
        # multi-class score: (b, num_queries, num_classes)
        mc_score = score_tgt.max(-1)[0] * foreground_pre_layer
        # select_tgt_index: (b, topk_sa)
        select_tgt_index = torch.topk(mc_score, self.topk_sa, dim=1)[1]
        # select_tgt_index: (b, topk_sa) -> (b, topk_sa, embed_dim)
        select_tgt_index = select_tgt_index.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        # select_tgt: (b, topk_sa, embed_dim)
        select_tgt = torch.gather(query, 1, select_tgt_index)
        # select_pos: (b, topk_sa, embed_dim)
        select_pos = torch.gather(query_pos, 1, select_tgt_index)
        # query_with_pos, key_with_pos: (b, topk_sa, embed_dim)
        query_with_pos = key_with_pos = self.with_pos_embed(select_tgt, select_pos)
        # query_with_pos: The query tensor, which includes positional encoding.
        # key_with_pos: The key tensor, which also includes positional encoding.
        # select_tgt: The value tensor, which does not include positional encoding.
        # All these tensors have the shape (b, self.topk_sa, self.embed_dim)
        # tgt2: (b, topk_sa, embed_dim)
        tgt2 = self.pre_attention(
            query_with_pos,
            key_with_pos,
            select_tgt,
        )[0]
        select_tgt = select_tgt + self.pre_dropout(tgt2)
        select_tgt = self.pre_norm(select_tgt)
        # query: (b, num_queries, embed_dim)
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
        # valid_ratios: (b, num_levels, 2)
        reference_points_list = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            # 创建一个覆盖整个特征图的均匀网格
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            # 将坐标展平并归一化
            # ref_y, ref_x: (1, h_i*w_i); valid_ratios[:]引入了批次
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * w)
            # ref:(b, h*w, 2)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # reference_points： (b, sum(h_i*w_i), 2)
        reference_points = torch.cat(reference_points_list, 1)
        # reference_points: (n, sum(h_i*w_i), l, 2)
        # 参考点实际上代表的是图像空间中的位置，而不仅仅是特定特征图上的像素
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
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
        # heat_maps input (b, sum(h_i*w_i), num_classes)
        heat_maps=None
    ):
        # reference_points: (b, num_total_queries, num_levels, 2(x,y)), normalized to [0,1]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=query.device)
        b, n, s, p = reference_points.shape
        ori_reference_points = reference_points
        ori_pos = query_pos
        value = output = query
        # each encoder layer
        for layer_id, layer in enumerate(self.layers):
            # foreground_inds[layer_id]: (b, num_selected_queries)
            # .unsqueeze(-1): (b, num_selected_queries, 1)
            # .expand(): (b, num_selected_queries, embed_dim)
            # inds_for_query: (b, num_queries_for_layer, embed_dim)
            inds_for_query = foreground_inds[layer_id].unsqueeze(-1).expand(-1, -1, self.embed_dim)
            # query: (b, num_queries_for_layer, embed_dim)
            query = torch.gather(output, 1, inds_for_query)
            # query_pos: (b, num_queries_for_layer, embed_dim)
            query_pos = torch.gather(ori_pos, 1, inds_for_query)
            # 获取前一层的前景分数 (b, num_queries_for_layer)
            foreground_pre_layer = torch.gather(foreground_score, 1, foreground_inds[layer_id])
            # reference_points: (b, num_queries_for_layer, num_levels, 2)
            reference_points = torch.gather(
                ori_reference_points.view(b, n, -1), 1,
                foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, s * p)
            ).view(b, -1, s, p)
            # 增强多类别单点预测 score_tgt:(b, num_queries_for_layer, num_classes)
            #???可能可以利用hough进行优化
            # score_tgt = self.enhance_mcsp(query)
            score_tgt = torch.gather(heat_maps, 1, foreground_inds[layer_id].unsqueeze(-1).repeat(1, 1, heat_maps.shape[-1]))
            # 创建一个新的张量并赋值给 query，而 output 仍然指向原始张量
            # 当前 Transformer 层处理查询 query:(b, num_queries_for_layer, embed_dim)
            query = layer(
                query, # query: (b, num_queries_for_layer, embed_dim)
                query_pos, # query_pos: (b, num_queries_for_layer, embed_dim)
                value, # value: (b, sum(h_i * w_i), embed_dim)
                reference_points, # reference_points: (b, num_queries_for_layer, num_levels, 2)
                spatial_shapes, # spatial_shapes: (num_levels, 2)
                level_start_index, # level_start_index: (num_levels,)
                query_key_padding_mask, # query_key_padding_mask: (b, sum(h_i w_i))
                score_tgt, # score_tgt: (b, num_queries_for_layer, num_classes)
                foreground_pre_layer, # foreground_pre_layer: (b, num_queries_for_layer)
            )
            outputs = []
            # 遍历批次中的每个样 batch_size
            # query 现在包含了更新后的特征，但只针对选定的前景查询点
            # output 仍然保持着原始的、完整的特征图
            # 我们需要将更新后的前景查询点特征（在 query 中）合并回完整的特征图（在 output 中）
            for i in range(foreground_inds[layer_id].shape[0]):
                # (focus_tokens_for_sample,)
                foreground_inds_no_pad = foreground_inds[layer_id][i][:focus_token_nums[i]]
                # (focus_tokens_for_sample, embed_dim)
                query_no_pad = query[i][:focus_token_nums[i]]
                # query_no_pad 中的值散布到 output[i] 中，位置由 foreground_inds_no_pad 决定
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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        # cross attention
        self.cross_attn = MultiScaleDeformableAttention(embed_dim, n_levels, n_heads, n_points)
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
        self_attn_mask=None,
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
        query2 = self.cross_attn(
            query=self.with_pos_embed(query, query_pos),
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
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
        value,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        key_padding_mask=None,
        attn_mask=None,
    ):
        outputs_classes = []
        outputs_coords = []
        valid_ratio_scale = torch.cat([valid_ratios, valid_ratios], -1)[:, None]

        for layer_idx, layer in enumerate(self.layers):
            reference_points_input = reference_points.detach()[:, :, None] * valid_ratio_scale
            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            # relation embedding
            query = layer(
                query=query,
                query_pos=query_pos,
                reference_points=reference_points_input,
                value=value,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask,
                self_attn_mask=attn_mask,
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
            reference_points = self.bbox_head[layer_idx](query) + inverse_sigmoid(reference_points.detach())
            reference_points = reference_points.sigmoid()

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        return outputs_classes, outputs_coords
