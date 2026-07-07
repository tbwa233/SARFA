import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def l2_regularisation(m):
    l2_reg = None
    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def calculate_dice_loss(inputs, targets, num_masks = 5):
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def calculate_sigmoid_focal_loss(inputs, targets, num_masks = 5, alpha = 0.25, gamma = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction = "none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def iou(pred, true):
    pred_bool = pred.bool().detach().cpu()
    true_bool = true.bool().detach().cpu()
    intersection = (pred_bool & true_bool).float().sum()
    union = (pred_bool | true_bool).float().sum()
    if union == 0 and intersection == 0:
        return 1
    else:
        return intersection / union

def mask_IoU(prediction, groundtruth):
    prediction = prediction.detach().cpu().numpy()
    groundtruth = groundtruth.detach().cpu().numpy()
    intersection = np.logical_and(groundtruth, prediction)
    union = np.logical_or(groundtruth, prediction)
    if np.sum(union) == 0:
        return 1
    return np.sum(intersection) / np.sum(union)

def generalized_energy_distance_iou(predictions, masks):
    n = predictions.shape[0]
    m = masks.shape[0]
    d1 = d2 = d3 = 0

    for i in range(n):
        for j in range(m):
            d1 += (1 - mask_IoU(predictions[i], masks[j]))

    for i in range(n):
        for j in range(n):
            d2 += (1 - mask_IoU(predictions[i], predictions[j]))

    for i in range(m):
        for j in range(m):
            d3 += (1 - mask_IoU(masks[i], masks[j]))

    d1 *= 2 / (n * m)
    d2 *= 1 / (n * n)
    d3 *= 1 / (m * m)

    return d1 - d2 - d3, mask_IoU(predictions[0], masks[0])

def hm_iou_cal(preds, trues):
    num_preds = len(preds)
    num_trues = len(trues)
    cost_matrix = torch.zeros((num_preds, num_trues))
    for i, pred in enumerate(preds):
        for j, true in enumerate(trues):
            cost_matrix[i, j] = 1 - iou(pred, true)
    row_ind, col_ind = linear_sum_assignment(cost_matrix.numpy())
    matched_iou = [iou(preds[i], trues[j]) for i, j in zip(row_ind, col_ind)]
    return torch.tensor(matched_iou).mean().item(), matched_iou

def dice_max_cal2(pred_eval, label_four):
    dice_max = -1
    for i in range(pred_eval.shape[0]):
        for j in range(label_four.shape[0]):
            intersection = (pred_eval[i] & label_four[j]).sum()
            union = pred_eval[i].sum() + label_four[j].sum()
            dice = 2 * intersection / union if union > 0 else 1
            dice_max = max(dice_max, dice)
    return dice_max