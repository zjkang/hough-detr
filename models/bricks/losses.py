import torch
from torch.nn import functional as F

# inputs, targets: (b, num_classes)
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    target_score = targets.to(inputs.dtype)
    weight = (1 - alpha) * prob**gamma * (1 - targets) + targets * alpha * (1 - prob)**gamma
    # according to original implementation, sigmoid_focal_loss keep gradient on weight
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, reduction="none")
    loss = loss * weight
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def weighted_multi_class_focal_loss(
        heat_maps, mask_targets, num_pos, alpha=0.25, gamma=2.0, sigma=1.0):
    # Uncomment the following line if you want to use distance-based weights
    # distance_weights = generate_distance_weights(mask_targets, sigma)
    distance_weights = torch.ones_like(mask_targets)

    total_loss = 0
    total_positive = 0

    for c in range(heat_maps.shape[1]):
        if isinstance(num_pos, torch.Tensor):
            if num_pos.dim() == 2:  # (batch_size, num_classes)
                class_num_pos = num_pos[:, c].sum()
            elif num_pos.dim() == 1:  # (num_classes,)
                class_num_pos = num_pos[c]
            else:
                raise ValueError(f"Unexpected shape for num_pos: {num_pos.shape}")
        else:
            class_num_pos = num_pos
        total_positive += class_num_pos

        class_loss = sigmoid_focal_loss(
            heat_maps[:, c],  # shape: (batch_size, height * width)
            mask_targets[:, c],  # shape: (batch_size, height * width)
            class_num_pos,
            alpha=alpha,
            gamma=gamma
        )  # class_loss is a scalar

        # Apply the mean distance weight for this class
        weighted_class_loss = class_loss * distance_weights[:, c].mean()
        total_loss += weighted_class_loss

    # Normalize by total number of positive samples
    return total_loss / max(total_positive, 1)  # Avoid division by zero


def generate_distance_weights(mask_targets, sigma=1.0):
    """
    生成基于距离的权重
    mask_targets: (batch_size, num_classes, H, W)
    """
    batch_size, num_classes, H, W = mask_targets.shape

    # 初始化 distance_weights 为全 1 张量
    distance_weights = torch.ones_like(mask_targets)

    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack([y_coords, x_coords], dim=-1).to(mask_targets.device)

    for b in range(batch_size):
        for c in range(num_classes):
            center = torch.nonzero(mask_targets[b, c] == 1).float().mean(0) if mask_targets[b, c].sum() > 0 else None
            if center is not None:
                distances = torch.norm(coords.float() - center, dim=-1)
                distance_weights[b, c] = torch.exp(-distances**2 / (2 * sigma**2))
            # 如果没有中心点，保持该类别的权重为 1

    return distance_weights


def vari_sigmoid_focal_loss(inputs, targets, gt_score, num_boxes, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid().detach()  # pytorch version of RT-DETR has detach while paddle version not
    target_score = targets * gt_score.unsqueeze(-1)
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + target_score
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes


def ia_bce_loss(inputs, targets, gt_score, num_boxes, k: float = 0.25, alpha: float = 0, gamma: float = 2):
    prob = inputs.sigmoid().detach()
    # calculate iou_aware_score and constrain the value following original implementation
    iou_aware_score = prob**k * gt_score.unsqueeze(-1)**(1 - k)
    iou_aware_score = iou_aware_score.clamp(min=0.01)
    target_score = targets * iou_aware_score
    weight = (1 - alpha) * prob.pow(gamma) * (1 - targets) + targets
    loss = F.binary_cross_entropy_with_logits(inputs, target_score, weight=weight, reduction="none")
    # we use sum/num to replace mean to avoid NaN
    return (loss.sum(1) / max(loss.shape[1], 1)).sum() / num_boxes
