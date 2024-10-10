from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops
from collections import defaultdict

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
#    (1) focal loss with positive / negative (x): 
#    (2) focal loss with weights
#    (3) cornernet: only center or few points as positive, negative with weights
# 3. Do I need to select tokens (pixels) in multi-scales
#       high resolution (small object); low (large object)

import torch
import torch.nn as nn
import numpy as np

# v2: project targets [{'boxes', 'labels'}] to heat_maps [[(b, num_classes, h_i, w_i)]]
class HoughCriterion(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def gaussian2D(shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        h = h / h.max()  # 归一化，确保最大值为 1
        return torch.from_numpy(h).float()

    def draw_gaussian(self, heatmap, center, radius):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        
        x, y = int(center[0]), int(center[1])
        height, width = heatmap.shape[-2:]
        
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[..., y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian.to(heatmap.device), out=masked_heatmap)
        return heatmap

    def forward(self, heat_maps, targets, feature_strides, image_sizes):
        batch_size = len(targets)
        device = heat_maps[0].device
        
        target_maps = []
        for level, (heat_map, stride) in enumerate(zip(heat_maps, feature_strides)):
            h, w = heat_map.shape[-2:]
            target_map = torch.zeros_like(heat_map)
            for b in range(batch_size):
                # 获取原始图像尺寸
                orig_h, orig_w = image_sizes[b]
                # 计算缩放因子
                scale_x = w / orig_w
                scale_y = h / orig_h
                
                for box, label in zip(targets[b]['boxes'], targets[b]['labels']):
                    # 将归一化的坐标转换回原始图像尺寸，然后应用缩放
                    cx, cy, bw, bh = box * torch.tensor([orig_w, orig_h, orig_w, orig_h], device=device)
                    cx = cx * scale_x
                    cy = cy * scale_y
                    bw = bw * scale_x
                    bh = bh * scale_y
                    
                    ct = torch.tensor([cx, cy], device=device)
                    ct_int = ct.long()
                    
                    radius = max(1, int(((bw + bh) / 2) / 2))
                    self.draw_gaussian(target_map[b, label], ct_int, radius)
            target_maps.append(target_map)
        
        # 将所有层级的热图和目标图拼接
        heat_maps = torch.cat([m.flatten(2) for m in heat_maps], dim=2)
        target_maps = torch.cat([m.flatten(2) for m in target_maps], dim=2)
        
        # 计算 Focal Loss
        pos_inds = target_maps.eq(1).float()
        neg_inds = target_maps.lt(1).float()
        
        neg_weights = torch.pow(1 - target_maps, self.beta)
        
        heat_maps = torch.clamp(heat_maps.sigmoid_(), min=1e-4, max=1-1e-4)
        pos_loss = torch.log(heat_maps) * torch.pow(1 - heat_maps, self.alpha) * pos_inds
        neg_loss = torch.log(1 - heat_maps) * torch.pow(heat_maps, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.sum(dim=2, keepdim=True)
        pos_loss = pos_loss.sum(dim=2, keepdim=True)
        neg_loss = neg_loss.sum(dim=2, keepdim=True)

        loss = torch.where(num_pos == 0, -neg_loss, -(pos_loss + neg_loss) / num_pos)
        loss = torch.clamp(loss, max=10)
        
        # 梯度裁剪
        # loss = torch.where(num_pos == 0, -neg_loss, -(pos_loss + neg_loss) / num_pos)
        # loss = torch.clamp(loss, max=10)  # 限制最大损失值
        # return {"loss_hough": loss.mean()}

        loss_hough = torch.log1p(loss.mean()) / (1 + torch.log1p(loss.mean()))
        return {"loss_hough": loss_hough}


# v1: project heat_maps [(b, num_classes, h_i, w_i)] to targets [{}] original dimension
# loss is calculated based on the shape of heat_map, true label is assigned based on the distance to the center of the target
# def cal_multi_class_focal_loss(mask_targets, heat_maps, noise_scale, alpha, gamma):
#     # num_pos: (batch_size, num_classes)
#     num_pos = torch.sum(mask_targets > 0.5 * noise_scale, dim=2).clamp_(min=1)
#     hough_loss = (
#         weighted_multi_class_focal_loss(
#             heat_maps,
#             mask_targets,
#             num_pos,
#             alpha=alpha,
#             gamma=gamma,
#         )
#         # weighted_multi_class_focal_loss(
#         #     heat_maps,
#         #     mask_targets,
#         #     num_pos,
#         #     alpha=self.alpha,
#         #     gamma=self.gamma,
#         # ) * heat_maps.shape[2]
#     )
#     return hough_loss

# class HoughCriterion(nn.Module):
#     def __init__(
#         self,
#         limit_range: Tuple = ((-1, 64), (64, 128), (128, 256), (256, 99999)),
#         noise_scale: float = 0.0,
#         alpha: float = 0.25,
#         gamma: float = 2.0,
#         num_classes: int = 91
#     ):
#         super().__init__()
#         self.limit_range = limit_range
#         self.noise_scale = noise_scale
#         self.alpha = alpha
#         self.gamma = gamma
#         self.num_classes = num_classes

#     def forward(self, heat_maps, targets, feature_strides, image_sizes):
#         # heat_maps(foreground_mask): [(b, num_classes, h_i, w_i)]
#         # gt_boxes_list: [{boxes, labels}] len() = batch, boxes: the boxes in one image
#         # ground-truth target real coords
#         gt_boxes_list = []
#         for t, (img_h, img_w) in zip(targets, image_sizes):
#             boxes = t["boxes"]
#             labels = t["labels"]
#             boxes = box_ops._box_cxcywh_to_xyxy(boxes)
#             scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
#             gt_boxes_list.append(((boxes * scale_factor), labels))

#         mask_targets = []
#         for level_idx, (heat_map, feature_stride) in enumerate(zip(heat_maps, feature_strides)):
#             feature_shape = heat_map.shape[-2:]
#             coord_x, coord_y = self.get_pixel_coordinate(
#                 feature_shape, feature_stride, device=heat_map.device)
#             masks_per_level = []
#             for gt_boxes, labels in gt_boxes_list:
#                 # mask: (h_i * w_i, num_classes)
#                 mask = self.get_mask_single_level(coord_x, coord_y, gt_boxes, labels, level_idx)
#                 masks_per_level.append(mask)
#             # (masks_per_level: (batch_size, h_i * w_i, num_classes)
#             masks_per_level = torch.stack(masks_per_level)
#             # masks_per_level: (batch_size, num_classes, h_i * w_i)
#             masks_per_level = masks_per_level.permute(0, 2, 1)
#             mask_targets.append(masks_per_level)
#         # mask_targets: (batch_size, num_classes, sum(h_i * w_i))
#         mask_targets = torch.cat(mask_targets, dim=2)

#         # heat_maps: (batch_size, num_classes, sum(h_i * w_i))
#         heat_maps = torch.cat([e.flatten(-2) for e in heat_maps], -1)

#         # multi-class focal loss
#         loss_hough = cal_multi_class_focal_loss(
#             mask_targets, heat_maps, self.noise_scale, self.alpha, self.gamma)

#         return {"loss_hough": loss_hough}

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

#     def get_mask_single_level(self, coord_x, coord_y, gt_boxes, labels, level_idx):
#         # TODO: alternative gaussian to calculate point confidence
#         # labels: (m,) gt_boxes: (m, 4)
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
#         mask_in_level = (max_border_distances > min_limit) & (max_border_distances <= max_limit)
#         mask_pos = mask_in_gt_boxes & mask_in_level

#         # scale-independent salience confidence
#         row_factor = left_border_distance + right_border_distance
#         col_factor = top_border_distance + bottom_border_distance
#         # delta_x, delta_y, range [-1,1]
#         delta_x = (left_border_distance - right_border_distance) / row_factor
#         delta_y = (top_border_distance - bottom_border_distance) / col_factor
#         # confidence range [0, 0.707], related the relative dist from point to boundary rather than absolute dist
#         confidence = torch.sqrt(delta_x**2 + delta_y**2) / 2

#         # confidence_per_box: (h * w, m), m is # of boxes
#         confidence_per_box = 1 - confidence
#         confidence_per_box[~mask_in_gt_boxes] = 0

#         # 创建一个包含所有类别的掩码
#         mask = torch.zeros((coord_y.shape[0], self.num_classes), device=confidence.device, dtype=confidence.dtype)

#         # 为每个类别分别处理
#         for class_id in range(self.num_classes):
#             class_mask = labels == class_id # [T,F,...T,F]
#             if class_mask.sum() > 0:
#                 class_confidence = confidence_per_box[:, class_mask]
#                 class_mask_pos = mask_pos[:, class_mask]

#                 if class_confidence.numel() != 0:
#                     class_mask_values = class_confidence.max(-1)[0]
#                 else:
#                     class_mask_values = torch.zeros(coord_y.shape, device=confidence.device, dtype=confidence.dtype)

#                 class_mask_pos = class_mask_pos.long().sum(dim=-1) >= 1
#                 class_mask_values[~class_mask_pos] = 0

#                 mask[:, class_id] = class_mask_values

#         # add noise to add randomness
#         # mask: (h_i * w_i, num_classes)
#         mask = (1 - self.noise_scale) * mask + self.noise_scale * torch.rand_like(mask)
#         return mask



# TODO: plot features points within targets in levels
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

import random
import string

def get_size(level):
    base_size = 2
    return base_size * (level + 2)

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = image.clone()  # 创建副本以避免修改原始数据
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)  # 反标准化
    return image

def tensor_to_pil(image_tensor):
    image = denormalize(image_tensor)
    image = image.mul(255).clamp(0, 255).byte()  # 转换到 0-255 范围
    image = image.cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(image)

def visualize_points_in_targets(points_in_targets, image):
    plt.figure(figsize=(12, 12))
    
    # Convert image tensor to PIL Image
    image_pil = tensor_to_pil(image)
    
    # Create a new tensor for drawing bounding boxes
    draw_image = torch.tensor(np.array(image_pil)).permute(2, 0, 1)
    
    boxes = []
    labels = []
    colors = []
    
    for item in points_in_targets:
        points = item['points'].cpu().numpy()
        box = item['box'].cpu()
        label = item['label'].item()
        level = item['level']
        size = get_size(level)
        
        # Plot points
        plt.scatter(points[:, 0], points[:, 1], s=size, alpha=0.5)
        
        # Prepare box and label for drawing
        boxes.append(box)
        labels.append(f"Label: {label}, Level: {level}")
        colors.append("red")

    # Draw all bounding boxes at once
    draw_image = draw_bounding_boxes(draw_image, torch.stack(boxes), labels, colors, width=2)
    
    # Convert back to PIL and display
    plt.imshow(to_pil_image(draw_image))
    plt.axis('off')
    letters = string.ascii_lowercase
    random_str = ''.join(random.choice(letters) for i in range(4))
    filename = f'visualization_{random_str}.png'
    plt.savefig(filename)
    plt.close()
    # plt.show()

# def visualize_points_in_targets(points_in_targets, image):
#     plt.figure(figsize=(12, 12))
#     plt.imshow(image)

#     for item in points_in_targets:
#         points = item['points']
#         box = item['box']
#         label = item['label']
#         level = item['level']
#         item['']
#         # 绘制点
#         plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
#         # 绘制边界框
#         rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1],
#                              fill=False, edgecolor='r', linewidth=2)
#         plt.gca().add_patch(rect)
#         # 添加标签
#         # plt.text(box[0], box[1], f'Label: {label}', color='r', fontsize=8)
#     plt.axis('off')
#     plt.show()

def get_pixel_coordinate(feature_shape, stride, device):
    height, width = feature_shape
    coord_y, coord_x = torch.meshgrid(
        torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device) * stride[0],
        torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device) * stride[1],
        indexing="ij",
    )
    coord_y = coord_y.reshape(-1)
    coord_x = coord_x.reshape(-1)
    return coord_x, coord_y

def get_targets_info(feature_maps, targets, feature_strides, image_sizes):
    # gt_boxes_list: [{boxes, labels}] len() = batch, boxes: the boxes in one image
    # ground-truth target real coords
    gt_boxes_list = []
    for t, (img_h, img_w) in zip(targets, image_sizes):
        boxes = t["boxes"]
        labels = t["labels"]
        boxes = box_ops._box_cxcywh_to_xyxy(boxes)
        scale_factor = torch.tensor([img_w, img_h, img_w, img_h], device=boxes.device)
        gt_boxes_list.append(((boxes * scale_factor), labels))

    points_in_targets_map = defaultdict(list)
    for level_idx, (feat_map, feature_stride) in enumerate(zip(feature_maps, feature_strides)):
        feature_shape = feat_map.shape[-2:]
        coord_x, coord_y = get_pixel_coordinate(
            feature_shape, feature_stride, device=feat_map.device)

        for img_idx, (gt_boxes, labels) in enumerate(gt_boxes_list):
            points_per_image = []
            # (h_i*w_i, m)
            mask_in_gt_boxes, coord_x, coord_y = filter_mask_in_target(coord_x, coord_y, gt_boxes)
            # 对每个目标框单独处理
            for box_idx in range(gt_boxes.shape[0]):
                mask_for_box = mask_in_gt_boxes[:, box_idx]
                points_in_box_x = coord_x[mask_for_box]
                points_in_box_y = coord_y[mask_for_box]
                points_in_box = torch.stack([points_in_box_x, points_in_box_y], dim=1)
                points_per_image.append({
                    'points': points_in_box,
                    'box': gt_boxes[box_idx],
                    'label': labels[box_idx],
                    'level': level_idx,
                    'boxIdx': box_idx,
                })
            points_in_targets_map[img_idx].extend(points_per_image)

    return points_in_targets_map

def filter_mask_in_target(coord_x, coord_y, gt_boxes):
    left_border_distance = coord_x[:, None] - gt_boxes[None, :, 0]  # (h*w, m)
    top_border_distance = coord_y[:, None] - gt_boxes[None, :, 1]
    right_border_distance = gt_boxes[None, :, 2] - coord_x[:, None]
    bottom_border_distance = gt_boxes[None, :, 3] - coord_y[:, None]
    border_distances = torch.stack(
        [left_border_distance, top_border_distance, right_border_distance, bottom_border_distance],
        dim=-1,
    )  # (h*w, m, 4)
    min_border_distances = torch.min(border_distances, dim=-1)[0]  # (h*w, m)
    mask_in_gt_boxes = min_border_distances > 0
    return mask_in_gt_boxes, coord_x, coord_y

def get_single_image_from_image_list(image_list, index: int):
    batched_images = image_list.tensors
    # 获取原始图像尺寸
    original_size = image_list.image_sizes[index]
    # 提取单张图像
    single_image = batched_images[index]
    # 裁剪到原始尺寸
    single_image = single_image[:, :original_size[0], :original_size[1]]
    return single_image

def plot_targets(images, targets, multi_level_feats):
    feature_strides = [(
        images.tensors.shape[-2] // feature.shape[-2],
        images.tensors.shape[-1] // feature.shape[-1],
    ) for feature in multi_level_feats]

    target_map = get_targets_info(
        multi_level_feats, targets, feature_strides, images.image_sizes)

    for img_idx in range(len(images.image_sizes)):
        box_idx, level_idx = -1, -1
        # plot by targets and levels
        points_in_target = []
        for l_tgt in target_map[img_idx]:
            cur_level = l_tgt['level']
            cur_box_idx = l_tgt['boxIdx']
            if (box_idx == -1 or cur_box_idx == box_idx) and \
                (level_idx == -1 or cur_level == level_idx):
                points_in_target.append(l_tgt)

        image = get_single_image_from_image_list(images, img_idx)
        visualize_points_in_targets(points_in_target, image)
    print("Visualization saved as 'visualization.png'")




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

        # plot the targets
        # plot_targets(images, targets, multi_level_feats)

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
