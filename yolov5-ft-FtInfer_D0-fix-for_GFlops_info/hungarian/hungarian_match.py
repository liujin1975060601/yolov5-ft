
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from hungarian.ious import bbox_overlaps

def bbox_iou(boxA, boxB):
    """
    Compute the Intersection Over Union value between two bounding boxes
    Each box is defined by its center, width and height: [x_center, y_center, width, height]
    """

    # Convert from center to exact coordinates
    boxA = [boxA[0] - boxA[2] / 2.0, boxA[1] - boxA[3] / 2.0, boxA[0] + boxA[2] / 2.0, boxA[1] + boxA[3] / 2.0]
    boxB = [boxB[0] - boxB[2] / 2.0, boxB[1] - boxB[3] / 2.0, boxB[0] + boxB[2] / 2.0, boxB[1] + boxB[3] / 2.0]

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute IOU
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def hungarian_match(predicted_boxes, true_boxes):
    ntruth = true_boxes.size(0)  # true_boxes[nt,4] Number of true boxes
    npred = predicted_boxes.size(0)  # predicted_boxes[np,4] Number of predicted boxes

    #assert(ntruth<=npred)
    #print(ntruth, npred)

    # Compute pairwise IoU between true and predicted boxes
    ious = bbox_overlaps(true_boxes,predicted_boxes, mode='iou')
    '''
    ious = torch.zeros(ntruth, npred)
    for i in range(ntruth):
        for j in range(npred):
            ious[i, j] = bbox_iou(true_boxes[i], predicted_boxes[j])
    '''
    # Calculate the differences in center points
    true_centers = true_boxes[:, :2].unsqueeze(1)   # Shape: [nt, 1, 2]
    pred_centers = predicted_boxes[:, :2].unsqueeze(0)  # Shape: [1, np, 2]
    center_diffs = (true_centers - pred_centers).pow(2).sum(dim=-1).sqrt()  # Shape: [nt, np]
    center_diffs /= center_diffs.max()# Normalize the differences (this can be adjusted as needed)
    # Calculate the differences in width and height

    # Incorporate these differences into the IoU matrix
    modified_ious = center_diffs - ious

    # Use Hungarian algorithm to find best matches, and then compute loss
    row_idx, col_idx = linear_sum_assignment(modified_ious.detach().cpu().numpy())  # Minimize negative IoU
    assert(row_idx.shape[0] == col_idx.shape[0])
    n_matched = min(ntruth,npred)
    assert(row_idx.shape[0]==n_matched)
    matched_ious = ious[row_idx, col_idx]
    assert(matched_ious.shape[0]==n_matched)

    return ious,matched_ious,(row_idx, col_idx)

def hungarian_loss(predicted_boxes, true_boxes):
    ious,matched_ious,(row_idx, col_idx) = hungarian_match(predicted_boxes, true_boxes)

    # Here we use (1 - IoU) as loss for matched pairs.
    loss = (1 - matched_ious).sum()

    column_selected = torch.zeros(ious.shape[1], dtype=torch.float32)
    column_selected[col_idx] = 1

    return loss,column_selected,(row_idx, col_idx)

# from model import coords
# # 初始化MSELoss函数
# loss_fn = nn.MSELoss()
# criterion_sfm = nn.CrossEntropyLoss()  # 注意：这里的CrossEntropyLoss已经包括了softmax
# def batch_loss(predicted_boxes_batch, true_boxes_batch):
#     batch_size = predicted_boxes_batch.size(0)
#     total_loss = 0.0
    
#     losses = {'lobj': 0.0,'lbox': 0.0,'lcls': 0.0}
#     for i in range(batch_size):
#         predicted_boxes = predicted_boxes_batch[i,:,1:1+coords]#predicted_boxes[nt,coords]
#         true_boxes = true_boxes_batch[i][:,1:1+coords]#true_boxes[max_num_boxes,coords]

#         # Here we call the hungarian_loss function for each sample in the batch
#         lbox,tobj,(row_idx, col_idx) = hungarian_loss(predicted_boxes, true_boxes)
#         assert(col_idx.shape[0]==row_idx.shape[0])
#         losses['lbox']+=lbox

#         objs = predicted_boxes_batch[i,:,0]
#         assert(objs.shape == tobj.shape)
#         lobj = loss_fn(objs,tobj.to(objs.device))
#         losses['lobj']+=lobj

#         predicted_classes = predicted_boxes_batch[i,:,1+coords:]#[N]范围在(0,classes-1)的ids
#         true_ids = true_boxes_batch[i][row_idx,0].long()#[ntruth,classes]
#         assert(true_ids.shape[0]==true_boxes.shape[0])
#         pred_cls = predicted_classes[col_idx]
#         lcls = criterion_sfm(pred_cls,true_ids)
#         losses['lcls'] += lcls
        
#     losses['lobj']/=batch_size
#     losses['lbox']/=batch_size
#     losses['lcls']/=batch_size
#     total_loss = 0.4*losses['lobj'] + 0.3*losses['lbox'] + 0.3*losses['lcls']
        
#     # You can either return the mean loss or the sum of losses
#     # Here we return the mean
#     return total_loss, losses