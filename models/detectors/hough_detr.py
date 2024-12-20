from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops

from models.bricks.denoising import GenerateCDNQueries
from models.bricks.losses import sigmoid_focal_loss, weighted_multi_class_focal_loss
from models.detectors.base_detector import DNDETRDetector


# class SalienceCriterion(nn.Module):
#     def __init__(
#         self,
#         limit_range: Tuple = ((-1, 64), (64, 128), (128, 256), (256, 99999)),
#         noise_scale: float = 0.0, 
#         alpha: float = 0.25,
#         gamma: float = 2.0,
#     ):
#         super().__init__()
#         self.limit_range = limit_range
#         self.noise_scale = noise_scale
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, foreground_mask, targets, feature_strides, image_sizes):
#         # foreground_mask: [(batch_size, 1, H_i, W_i)]
#         # gt_boxes_list: [boxes] len() = batch, boxes: the boxes in one image
#         gt_boxes_list = []
#         for t, (img_h, img_w) in zip(targets, image_sizes):
#             boxes = t["boxes"]
#             boxes = box_ops._box_cxcywh_to_xyxy(boxes)
#             scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
#             gt_boxes_list.append(boxes * scale_factor)

#         mask_targets = []
#         for level_idx, (mask, feature_stride) in enumerate(zip(foreground_mask, feature_strides)):
#             feature_shape = mask.shape[-2:]
#             # coord mapping to original image
#             coord_x, coord_y = self.get_pixel_coordinate(feature_shape, feature_stride, device=mask.device)
#             masks_per_level = []
#             for gt_boxes in gt_boxes_list:
#                 # mask: (h*w, confidence)
#                 mask = self.get_mask_single_level(coord_x, coord_y, gt_boxes, level_idx)
#                 masks_per_level.append(mask)
#             # mask_per_level: (batch_size, h*w)
#             masks_per_level = torch.stack(masks_per_level)
#             mask_targets.append(masks_per_level)
#         # mask_targets: (batch_size, sum(h_i*w_i))
#         mask_targets = torch.cat(mask_targets, dim=1)
#         # foreground_mask init: [(batch_size, 1, h_i, w_i)]
#         # then (batch_size, 1, h_i * w_i)
#         foreground_mask = torch.cat([e.flatten(-2) for e in foreground_mask], -1)
#         # foreground_mask: (batch_size, sum(h_i * w_i))
#         foreground_mask = foreground_mask.squeeze(1)
#         num_pos = torch.sum(mask_targets > 0.5 * self.noise_scale).clamp_(min=1)
#         salience_loss = (
#             sigmoid_focal_loss(
#                 foreground_mask,
#                 mask_targets,
#                 num_pos,
#                 alpha=self.alpha,
#                 gamma=self.gamma,
#             ) * foreground_mask.shape[1]
#         )
#         return {"loss_salience": salience_loss}

#     def get_pixel_coordinate(self, feature_shape, stride, device):
#         height, width = feature_shape
#         coord_y, coord_x = torch.meshgrid(
#             torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride[0],
#             torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride[1],
#             indexing="ij",
#         )
#         coord_y = coord_y.reshape(-1)
#         coord_x = coord_x.reshape(-1)
#         return coord_x, coord_y

#     def get_mask_single_level(self, coord_x, coord_y, gt_boxes, level_idx):
#         # m = num_objects
#         # gt_label: (num_objects,) gt_boxes: (num_objects, 4)
#         # coord_x: (h*w, )
#         left_border_distance = coord_x[:, None] - gt_boxes[None, :, 0]  # (h*w, m)
#         top_border_distance = coord_y[:, None] - gt_boxes[None, :, 1]
#         right_border_distance = gt_boxes[None, :, 2] - coord_x[:, None]
#         bottom_border_distance = gt_boxes[None, :, 3] - coord_y[:, None]
#         border_distances = torch.stack(
#             [left_border_distance, top_border_distance, right_border_distance, bottom_border_distance],
#             dim=-1,
#         )  # [h*w, m, 4]

#         # the foreground queries must satisfy two requirements:
#         # 1. the quereis located in bounding boxes
#         # 2. the distance from queries to the box center match the feature map stride
#         min_border_distances = torch.min(border_distances, dim=-1)[0]  # [h*w, m]
#         max_border_distances = torch.max(border_distances, dim=-1)[0]
#         mask_in_gt_boxes = min_border_distances > 0
#         min_limit, max_limit = self.limit_range[level_idx]
#         # 每个特征层级负责检测特定大小范围的目标。
#         # 较浅的层级（分辨率高）负责检测小目标。
#         # 较深的层级（分辨率低）负责检测大目标。
#         # 优势：
#         # 提高检测效率：每个层级只需要关注特定大小的目标，减少了计算量。
#         # 改善检测准确性：通过在适当的尺度上检测目标，可以提高检测的准确性
#         mask_in_level = (max_border_distances > min_limit) & (max_border_distances <= max_limit)
#         mask_pos = mask_in_gt_boxes & mask_in_level # [h*w, m]: bool

#         # scale-independent salience confidence
#         row_factor = left_border_distance + right_border_distance
#         col_factor = top_border_distance + bottom_border_distance
#         delta_x = (left_border_distance - right_border_distance) / row_factor
#         delta_y = (top_border_distance - bottom_border_distance) / col_factor
#         confidence = torch.sqrt(delta_x**2 + delta_y**2) / 2

#         # region in the target is confidence, then 0 outside
#         confidence_per_box = 1 - confidence # [h*w, m]
#         confidence_per_box[~mask_in_gt_boxes] = 0

#         # process positive coordinates
#         # 对于每个空间位置，在所有目标中选择最高的置信度值
#         # 结果 mask 的形状将是 (HW,)，代表每个位置的最高置信度
#         # 上下文考虑：
#         # 这通常用于生成前景掩码或注意力图
#         # 高置信度区域更可能包含重要的目标信息
#         if confidence_per_box.numel() != 0:
#             mask = confidence_per_box.max(-1)[0]
#         else:
#             mask = torch.zeros(coord_y.shape, device=confidence.device, dtype=confidence.dtype)

#         # process negative coordinates
#         mask_pos = mask_pos.long().sum(dim=-1) >= 1
#         # 选择没有有效目标的位置 设为 0
#         mask[~mask_pos] = 0

#         # add noise to add randomness
#         mask = (1 - self.noise_scale) * mask + self.noise_scale * torch.rand_like(mask)
#         return mask # confidence score & 0


# NOTE: potential improvement:
# 1. heat map design
#    (1): distance -> confidence (x)
#    (2): gaussian conv
# 2. loss function
#    (1) focal loss with positive / negative (x)
#    (2) focal loss with weights
#    (3) cornernet: only center or few points as positive, negative with weights
# 3. Do I need to select tokens (pixels) in multi-scales
#       high resolution (small object); low (large object)
class HoughCriterion(nn.Module):
    def __init__(
        self,
        limit_range: Tuple = ((-1, 64), (64, 128), (128, 256), (256, 99999)),
        noise_scale: float = 0.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        num_classes: int = 91
    ):
        super().__init__()
        self.limit_range = limit_range
        self.noise_scale = noise_scale
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, heat_maps, targets, feature_strides, image_sizes):
        # heat_maps(foreground_mask): [(b, num_classes, h_i, w_i)]
        # gt_boxes_list: [{boxes, labels}] len() = batch, boxes: the boxes in one image
        # ground-truth target real coords
        gt_boxes_list = []
        for t, (img_h, img_w) in zip(targets, image_sizes):
            boxes = t["boxes"]
            labels = t["labels"]
            boxes = box_ops._box_cxcywh_to_xyxy(boxes)
            scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
            gt_boxes_list.append(((boxes * scale_factor), labels))

        mask_targets = []
        for level_idx, (heat_map, feature_stride) in enumerate(zip(heat_maps, feature_strides)):
            feature_shape = heat_map.shape[-2:]
            coord_x, coord_y = self.get_pixel_coordinate(
                feature_shape, feature_stride, device=heat_map.device)
            masks_per_level = []
            for gt_boxes, labels in gt_boxes_list:
                # mask: (h_i * w_i, num_classes)
                mask = self.get_mask_single_level(coord_x, coord_y, gt_boxes, labels, level_idx)
                masks_per_level.append(mask)
            # (masks_per_level: (batch_size, h_i * w_i, num_classes)
            masks_per_level = torch.stack(masks_per_level)
            # masks_per_level: (batch_size, num_classes, h_i * w_i)
            masks_per_level = masks_per_level.permute(0, 2, 1)
            mask_targets.append(masks_per_level)
        # mask_targets: (batch_size, num_classes, sum(h_i * w_i))
        mask_targets = torch.cat(mask_targets, dim=2)

        # heat_maps: (batch_size, num_classes, sum(h_i * w_i))
        heat_maps = torch.cat([e.flatten(-2) for e in heat_maps], -1)
        # num_pos: (batch_size, num_classes)
        num_pos = torch.sum(mask_targets > 0.5 * self.noise_scale, dim=2).clamp_(min=1)
        heat_map_loss = (
            weighted_multi_class_focal_loss(
                heat_maps,
                mask_targets,
                num_pos,
                alpha=self.alpha,
                gamma=self.gamma,
            )
            # weighted_multi_class_focal_loss(
            #     heat_maps,
            #     mask_targets,
            #     num_pos,
            #     alpha=self.alpha,
            #     gamma=self.gamma,
            # ) * heat_maps.shape[2]
        )
        return {"loss_hough": heat_map_loss}

    def get_pixel_coordinate(self, feature_shape, stride, device):
        height, width = feature_shape
        coord_y, coord_x = torch.meshgrid(
            torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride[0],
            torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride[1],
            indexing="ij",
        )
        coord_y = coord_y.reshape(-1)
        coord_x = coord_x.reshape(-1)
        return coord_x, coord_y

    def get_mask_single_level(self, coord_x, coord_y, gt_boxes, labels, level_idx):
        # TODO: alternative gaussian to calculate point confidence
        # labels: (m,) gt_boxes: (m, 4)
        # coord_x: (h*w, )
        left_border_distance = coord_x[:, None] - gt_boxes[None, :, 0]  # (h*w, m)
        top_border_distance = coord_y[:, None] - gt_boxes[None, :, 1]
        right_border_distance = gt_boxes[None, :, 2] - coord_x[:, None]
        bottom_border_distance = gt_boxes[None, :, 3] - coord_y[:, None]
        border_distances = torch.stack(
            [left_border_distance, top_border_distance, right_border_distance, bottom_border_distance],
            dim=-1,
        )  # [h*w, m, 4]

        # the foreground queries must satisfy two requirements:
        # 1. the quereis located in bounding boxes
        # 2. the distance from queries to the box center match the feature map stride
        min_border_distances = torch.min(border_distances, dim=-1)[0]  # [h*w, m]
        max_border_distances = torch.max(border_distances, dim=-1)[0]
        mask_in_gt_boxes = min_border_distances > 0
        min_limit, max_limit = self.limit_range[level_idx]
        mask_in_level = (max_border_distances > min_limit) & (max_border_distances <= max_limit)
        mask_pos = mask_in_gt_boxes & mask_in_level

        # scale-independent salience confidence
        row_factor = left_border_distance + right_border_distance
        col_factor = top_border_distance + bottom_border_distance
        # delta_x, delta_y, range [-1,1]
        delta_x = (left_border_distance - right_border_distance) / row_factor
        delta_y = (top_border_distance - bottom_border_distance) / col_factor
        # confidence range [0, 0.707], related the relative dist from point to boundary rather than absolute dist
        confidence = torch.sqrt(delta_x**2 + delta_y**2) / 2

        # confidence_per_box: (h * w, m), m is # of boxes
        confidence_per_box = 1 - confidence
        confidence_per_box[~mask_in_gt_boxes] = 0

        # 创建一个包含所有类别的掩码
        mask = torch.zeros((coord_y.shape[0], self.num_classes), device=confidence.device, dtype=confidence.dtype)

        # 为每个类别分别处理
        for class_id in range(self.num_classes):
            class_mask = labels == class_id # [T,F,...T,F]
            if class_mask.sum() > 0:
                class_confidence = confidence_per_box[:, class_mask]
                class_mask_pos = mask_pos[:, class_mask]

                if class_confidence.numel() != 0:
                    class_mask_values = class_confidence.max(-1)[0]
                else:
                    class_mask_values = torch.zeros(coord_y.shape, device=confidence.device, dtype=confidence.dtype)

                class_mask_pos = class_mask_pos.long().sum(dim=-1) >= 1
                class_mask_values[~class_mask_pos] = 0

                mask[:, class_id] = class_mask_values

        # add noise to add randomness
        # mask: (h_i * w_i, num_classes)
        mask = (1 - self.noise_scale) * mask + self.noise_scale * torch.rand_like(mask)
        return mask


class HoughDETR(DNDETRDetector):
    def __init__(
        # model structure
        self,
        backbone: nn.Module,
        neck: nn.Module,
        position_embedding: nn.Module,
        transformer: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        hough_criterion: nn.Module,
        # model parameters
        # 数据集有 80 个已标注的物体类别（类）。然而，有些代码中可能会设置为 91 个类别，这是因为 COCO 数据集中存在 11 个
        # 没有被使用或没有标注的背景类（或者称为未使用的类别 ID）。这些额外的类别 ID 从 81 到 91，包括了某些没有被标注的物体
        num_classes: int = 91,
        num_queries: int = 900,
        denoising_nums: int = 100,
        # model variants
        aux_loss: bool = True,
        min_size: int = None,
        max_size: int = None,
    ):
        super().__init__(min_size, max_size)
        # define model parameters
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        embed_dim = transformer.embed_dim

        # define model structures
        self.backbone = backbone
        self.neck = neck
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.denoising_generator = GenerateCDNQueries(
            num_queries=num_queries,
            num_classes=num_classes,
            label_embed_dim=embed_dim,
            denoising_nums=denoising_nums,
            label_noise_prob=0.5,
            box_noise_scale=1.0,
        )
        self.hough_criterion = hough_criterion

    def forward(self, images: List[Tensor], targets: List[Dict] = None):
        # get original image sizes, used for postprocess
        # original_image_sizes: [(h_1,w_1), ..., (h_b, w_b)], b is batch size
        original_image_sizes = self.query_original_sizes(images)
        # different h,w will align into h_max, w_max with padding
        # images: (b,c,h,w): ImageList object,image_size: original size of the image
        # targets: [{boxes:,labels:}]: 列表长度等于批次大小，每个字典包含 "boxes"、"labels" 等键,
        #          "boxes": 多个 cxcywh的表示,并且根据图片h,w做归一化
        # mask: (b,h,w): image区域是0,padding是1
        images, targets, mask = self.preprocess(images, targets)

        # extract features
        multi_level_feats = self.backbone(images.tensors)
        # multi_level_feats: [(b, c_i, h_i, w_i)], h_i > h_(i+1) && w_i > w_(i+1)
        multi_level_feats = self.neck(multi_level_feats)

        # multi_level_masks: [(b, h_i, w_i)], boolean
        multi_level_masks = []
        # multi_level_position_embeddings: [(b, c, h_i, w_i)],
        # c is 2 num_pos_feats (default is 2 64 = 128) -> (x,y)
        multi_level_position_embeddings = []
        for feature in multi_level_feats:
            multi_level_masks.append(F.interpolate(mask[None], size=feature.shape[-2:]).to(torch.bool)[0])
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        if self.training:
            # collect ground truth for denoising generation
            gt_labels_list = [t["labels"] for t in targets]
            gt_boxes_list = [t["boxes"] for t in targets]
            noised_results = self.denoising_generator(gt_labels_list, gt_boxes_list)
            noised_label_query = noised_results[0]
            noised_box_query = noised_results[1]
            attn_mask = noised_results[2]
            denoising_groups = noised_results[3]
            max_gt_num_per_image = noised_results[4]
        else:
            noised_label_query = None
            noised_box_query = None
            attn_mask = None
            denoising_groups = None
            max_gt_num_per_image = None

        # feed into transformer
        # outputs_class: (num_layers, b, num_queries, num_classes)
        #    num_layers 是解码器的层数
        #    num_queries 是查询的数量，通常等于目标检测的最大对象数
        # outputs_coord: (num_layers, b, num_queries, 4)
        # enc_class: (b, num_proposals, num_classes)
        # enc_coord: (b, num_proposals, 4)
        # foreground_mask: salience_score [(b, 1, h_i, w_i)]
        # foreground_heat_map: [(b, num_classes, h_i, w_i)]
        outputs_class, outputs_coord, enc_class, enc_coord, foreground_mask, heat_map = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            noised_label_query,
            noised_box_query,
            attn_mask=attn_mask,
        )
        # hack implementation for distributed training
        outputs_class[0] += self.denoising_generator.label_encoder.weight[0, 0] * 0.0

        # denoising postprocessing
        if denoising_groups is not None and max_gt_num_per_image is not None:
            dn_metas = {
                "denoising_groups": denoising_groups,
                "max_gt_num_per_image": max_gt_num_per_image,
            }
            outputs_class, outputs_coord = self.dn_post_process(outputs_class, outputs_coord, dn_metas)

        # prepare for loss computation
        # last level to the header
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        # enc_class 和 enc_coord 是从编码器（encoder）输出的初步预测结果
        # 这些初步预测结果会在后续的解码器（decoder）阶段被进一步精炼。通过包含这些编码器输出，
        # 模型可以在训练过程中对这个初步预测进行监督，从而提高整体的检测性能
        output["enc_outputs"] = {"pred_logits": enc_class, "pred_boxes": enc_coord}

        if self.training:
            # compute loss
            loss_dict = self.criterion(output, targets)
            dn_losses = self.compute_dn_loss(dn_metas, targets)
            loss_dict.update(dn_losses)

            # compute focus loss
            feature_stride = [(
                images.tensors.shape[-2] / feature.shape[-2],
                images.tensors.shape[-1] / feature.shape[-1],
            ) for feature in multi_level_feats]

            # ignore focus loss
            # focus_loss = self.focus_criterion(foreground_mask, targets, feature_stride, images.image_sizes)
            # loss_dict.update(focus_loss)

            # compute hough loss
            hough_loss = self.hough_criterion(heat_map, targets,feature_stride, images.image_sizes)
            loss_dict.update(hough_loss)

            # loss reweighting
            weight_dict = self.criterion.weight_dict
            loss_dict = dict((k, loss_dict[k] * weight_dict[k]) for k in loss_dict.keys() if k in weight_dict)
            return loss_dict

        detections = self.postprocessor(output, original_image_sizes)
        return detections
