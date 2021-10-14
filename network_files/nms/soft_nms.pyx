import torch


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou

def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.3):
    """
    https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[x1, y1, x2, y2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
    # Return
        the index of the selected boxes
    """
    device = dets.device
    # Indexes concatenate boxes with the last column
    indexes = torch.arange(0, dets.shape[0], dtype=torch.float).view(dets.shape[0], 1).to(device)
    dets = torch.cat((dets, indexes), dim=1)
    scores = box_scores.clone()
    N = dets.shape[0]//60
    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1
        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
        # IoU calculate
        ovr = box_iou(dets[i, :4].view(1, -1), dets[pos:, :4])
        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]
    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].long()
    return keep
