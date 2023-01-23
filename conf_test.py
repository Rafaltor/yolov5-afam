import os
from utils.torch_utils import select_device, time_sync
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.metrics import ConfusionMatrix, ap_per_class, ap_per_size, box_iou, afam_per_class, plot_pr_comparison
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, emojis, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.dataloaders import create_dataloader
from utils.callbacks import Callbacks
from models.common import DetectMultiBackend

import argparse
import json
import sys
from pathlib import Path
from utils.tools import box_area, get_intersection_from_list, get_union_from_list, get_intersection_between_list, \
    boxes_area, compute_afam
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib
import torchvision
from Image_labeler import yolo2bbox

from utils.tools import boxes_iou, gt_box_iou, get_centroids, clustering
from sklearn.cluster import DBSCAN




def plot_box(image, bboxes, t):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape

    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = box[:4]
        # denormalize the coordinates
        xmin = int(x1)
        ymin = int(y1)
        xmax = int(x2)
        ymax = int(y2)

        if t == 'gt':
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color= (0, 255, 0), thickness=1)
        elif t == 'pred':
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(int(box[4]*255), 0, 0), thickness=1)

        font_scale = min(1, max(3, int(w / 500)))
        font_thickness = min(2, max(10, int(w / 50)))

        p1, p2 = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        # Text width and height

    return image

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
        iouv (Array[10,1]), iou thresholds
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]

    for i in range(len(iouv)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return correct



out = torch.load(Path('prediction.pt'))
targets = torch.load(Path('targets.pt'))
im = torch.load(Path('im.pt'))
device = 'cpu'
iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95

conf_thres = 0.001
iou_thres = 0.6

im = im.float()  # uint8 to fp16/32
nb, _, height, width = im.shape  # batch size, channels, height, width
n = 13

targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)

stats = []
boxes = out[n, out[n, :, 4] > 0.001]
gt_box = xywh2xyxy(targets[targets[:, 0] == n, 2:])
centroids = clustering(boxes)

image = np.array(im[n]*255).transpose((1,2,0)).astype(np.uint8).copy()


boxes[:, 5:] *= boxes[:, 4:5]
box = xywh2xyxy(boxes[:, :4])
i, j = (boxes[:, 5:] > conf_thres).nonzero(as_tuple=False).T
x = torch.cat((box[i], boxes[i, j+5, None], j[:, None].float()), 1)
whole = torch.cat((box[boxes[:, 5:].max(1)[0] > 0.001, :4],
                   boxes[boxes[:, 5:].max(1)[0] > 0.001, 5:].max(1, keepdim=True)[0],
                        boxes[boxes[:, 5:].max(1)[0] > 0.001, 5:].max(1, keepdim=True)[1]), 1)


centroids = centroids[centroids[:, 4].sort()[1]]
x = x[x[:, 4].sort()[1]]

c = centroids[:, 5:6] * 7680  # classes
boxes, scores = centroids[:, :4] + c, centroids[:, 4]  # boxes (offset by class), scores
i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
if i.shape[0] > 300:  # limit detections

    i = i[:300]








labels = torch.cat((targets[targets[:, 0] == n, 1:2], gt_box), 1)
predn = centroids[i]


correct = process_batch(predn, labels, iouv)

stats.append((correct, predn[:, 4], predn[:, 5], labels[:, 0]))

stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
tp, fp, p, r, f1, ap, ap_class, py = ap_per_class(*stats, plot=False, save_dir= Path('runs'))

print('new',p.mean(), r.mean(),ap.mean(1).mean())
c = x[:, 5:6] * 7680  # classes
boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
if i.shape[0] > 300:  # limit detections

    i = i[:300]

stats = []
labels = torch.cat((targets[targets[:, 0] == n, 1:2], gt_box), 1)
correct_2 = process_batch(x[i], labels, iouv)
stats.append((correct_2, x[i][:, 4], x[i][:, 5], labels[:, 0]))

stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
tp, fp, p, r, f1, ap, ap_class, py = ap_per_class(*stats, plot=False, save_dir=Path('runs'))
print('classique',p.mean(), r.mean(), ap.mean(1).mean())


matplotlib.use('TkAgg')
image = np.array(im[n]*255).transpose((1,2,0)).astype(np.uint8).copy()
image = plot_box(image, centroids, 'pred')
images = plot_box(image, gt_box, 'gt')
plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(image)

plt.subplot(1,2,2)
image = np.array(im[n]*255).transpose((1,2,0)).astype(np.uint8).copy()
image = plot_box(image, x, 'pred')
images = plot_box(image, gt_box, 'gt')

plt.imshow(image)
plt.show()


non_max_suppression(out, conf_thres, iou_thres, labels=[], multi_label=True, agnostic=False, targets=targets)
print(iop, iog)