# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:15:13 2022

@author: jamyl and Rafaltor

"""

import torch as th
import random
import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN, Birch, AffinityPropagation

OPENING = +1
CLOSING = -1


class CoverQuery:
    """Segment tree to maintain a set of integer intervals
    and permitting to query the size of their union.
    """

    def __init__(self, L):
        """creates a structure, where all possible intervals
        will be included in [0, L - 1].
        """
        assert L != []  # L is assumed sorted
        self.N = 1
        while self.N < len(L):
            self.N *= 2
        self.c = [0] * (2 * self.N)  # --- covered
        self.s = [0] * (2 * self.N)  # --- score
        self.w = [0] * (2 * self.N)  # --- length
        for i, _ in enumerate(L):
            self.w[self.N + i] = L[i]
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def cover(self):
        """:returns: the size of the union of the stored intervals
        """
        return self.s[1]

    def change(self, i, k, offset):
        """when offset = +1, adds an interval [i, k],
        when offset = -1, removes it
        :complexity: O(log L)
        """
        self._change(1, 0, self.N, i, k, offset)

    def _change(self, p, start, span, i, k, offset):
        if start + span <= i or k <= start:  # --- disjoint
            return
        if i <= start and start + span <= k:  # --- included
            self.c[p] += offset
        else:
            self._change(2 * p, start, span // 2, i, k, offset)
            self._change(2 * p + 1, start + span // 2, span // 2,
                         i, k, offset)
        if self.c[p] == 0:
            if p >= self.N:  # --- leaf
                self.s[p] = 0
            else:
                self.s[p] = self.s[2 * p] + self.s[2 * p + 1]
        else:
            self.s[p] = self.w[p]


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def boxes_area(boxes):
    # boxes = xyxy(4,n)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def get_intersection_from_tuple(rect1, rect2, class_exigence=False):
    """
    Computes the intersection of two rectangles and returns the caracteristics
    of the resulting rectangle

    Parameters
    ----------
    rect1 : Array[5]
        contains (x1, y1, x2, y2, class): the coordinates of the top left and bottom
        right corners of the first rectangle
    rect2 : Array[5]
        idem from the second rectangle
    class_exigence : Boolean
        When true, two overlapping boxes will get an intersection surface of 0
        If they have a different class (label)

    Returns
    -------
    intersection : Array[4]
        contains (x1, y1, x2, y2) of the resulting intersection

    """
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])

    x2 = min(rect1[2], rect2[2])
    y2 = min(rect1[3], rect2[3])

    if x1 <= x2 and y1 <= y2 and not class_exigence:
        intersect = th.tensor([x1, y1, x2, y2], device=rect1.device)
    elif x1 <= x2 and y1 <= y2 and class_exigence and rect1[4] == rect2[4]:
        intersect = th.tensor([x1, y1, x2, y2, rect1[4]], device=rect1.device)
    else:
        intersect = None
    return intersect


def get_intersection_from_list(rectangles, rect2, class_exigence=False):
    """
       Computes the intersection of a list of rectangles and one rectangle and returns the caracteristics list
       of the resulting rectangles

       Parameters
       ----------
       rectangles : list of Array[5]
           contains (x1, y1, x2, y2, class): the coordinates of the top left and bottom
           right corners of the first rectangle
       rect2 : Array[5]
           idem from the second rectangle
       class_exigence : Boolean
            When true, two overlapping boxes will get an intersection surface of 0
            If they have a different class (label)

       Returns
       -------
       intersection : list Array[4]
           contains a list of (x1, y1, x2, y2) of the resulting intersection

    """

    inter_boxes = []

    for box in rectangles:

        intersect = get_intersection_from_tuple(
            box, rect2, class_exigence)
        if intersect is not None:
            inter_boxes.append(intersect)
    return inter_boxes


def get_intersection_between_list(rectangles1, rectangles2, class_exigence=False):
    """
       Computes the intersection of a list of rectangles and another list of rectangle and returns the list
       of the resulting rectangles

       Parameters
       ----------
       rectangles1 : list of Array[5]
           contains (x1, y1, x2, y2, class): the coordinates of the top left and bottom
           right corners of the first rectangle
       rectangles2 : list of Array[5]
           idem from the second rectangle
       class_exigence : Boolean
            When true, two overlapping boxes will get an intersection surface of 0
            If they have a different class (label)

       Returns
       -------
       intersection : list Array[4]
           contains a list of (x1, y1, x2, y2) of the resulting intersection

    """
    inter_boxes = []

    for box in rectangles2:
        inter_boxes = inter_boxes + (get_intersection_from_list(rectangles1, box, class_exigence))

    return inter_boxes


def get_intersection_area_from_tuple(rect1, rect2, class_exigence=False):
    """
    Returns the area of the intersection between rect1 and rect2

    Parameters
    ----------
    rect1 : Array([5])
        x1, y1, x2, y2, class
    rect2 : Array([5])
        Idem
    class_exigence : Boolean
        When true, two overlapping boxes will get an intersection surface of 0
        If they have a different class (label)

    Returns
    -------
    intersec_area : float
        area of the intersection
    """

    intersection_area = max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])) * \
                        max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    return intersection_area


def union_area_tuple(rect1, rect2, class_exigence=False):
    return box_area(rect1) + box_area(rect2) - get_intersection_area_from_tuple(rect1, rect2, class_exigence)


def get_union_from_list(R):
    """Area of union of rectangles
    Source: https://github.com/jilljenn/tryalgo.
    :param R: list of rectangles defined by (x1, y1, x2, y2)
       where (x1, y1) is top left corner and (x2, y2) bottom right corner
    :returns: area
    :complexity: :math:`O(n \\log n)`
    """

    if not R:  # segment tree would fail on an empty list
        return 0
    X = set()  # set of all x coordinates in the input
    events = []  # events for the sweep line
    for Rj in R:
        (x1, y1, x2, y2) = Rj
        assert x1 <= x2 and y1 <= y2
        X.add(x1)
        X.add(x2)
        events.append((y1, OPENING, x1, x2))
        events.append((y2, CLOSING, x1, x2))
    i_to_x = list(sorted(X))
    # inverse dictionary
    x_to_i = {i_to_x[i]: i for i in range(len(i_to_x))}
    L = [i_to_x[i + 1] - i_to_x[i] for i in range(len(i_to_x) - 1)]
    C = CoverQuery(L)
    area = 0
    previous_y = 0  # arbitrary initial value,
    #                 because C.cover() is 0 at first iteration
    for y, offset, x1, x2 in sorted(events):
        area += (y - previous_y) * C.cover()
        i1 = x_to_i[x1]
        i2 = x_to_i[x2]
        C.change(i1, i2, offset)
        previous_y = y
    return area


def box_ioa(box1, box2):
    """
    Returns the intersection over box2 area given box1, box2.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Array[4])
        box2 (Array[4])
    Returns:
        ioa (int): the IoA values for boxes1 and boxes2
    """

    inter = get_intersection_area_from_tuple(box1, box2)

    # IoA = inter / (area1)
    return inter / box_area(box1)


def box_is_in(box, box_list):
    for ib, boxes in enumerate(box_list):
        if box_ioa(box, boxes) == 1:
            return True

    return False


def boxes_iou(boxes):
    # Returns Intersection over Union (IoU) of boxes1(n,5) to boxes2(n,5)

    boxes = boxes.expand(boxes.shape[0], boxes.shape[0], boxes.shape[1])
    boxesT = boxes.transpose(0, 1)
    #correct_classes = boxes[:, :, 5] == boxesT[:, :, 5]
    #conf = boxes[:, :, 4]

    inter_area = np.maximum(0, np.minimum(boxes[:, :, 3], boxesT[:, :, 3])
                            - np.maximum(boxes[:, :, 1], boxesT[:, :, 1])) * \
                 np.maximum(0, np.minimum(boxes[:, :, 2], boxesT[:, :, 2])
                            - np.maximum(boxes[:, :, 0], boxesT[:, :, 0]))

    boxes_area = (boxes[:, :, 3] - boxes[:, :, 1]) * (boxes[:, :, 2] - boxes[:, :, 0])
    boxesT_area = (boxesT[:, :, 3] - boxesT[:, :, 1]) * (boxesT[:, :, 2] - boxesT[:, :, 0])

    union = boxesT_area + boxes_area - inter_area

    iou = ( inter_area ) / union
    iop = ( inter_area ) / boxes_area
    iog = ( inter_area ) / boxesT_area
    return iou, iop, iog


def gt_box_iou(box1, box2):
    """
    Return intersection-over-union of between box1 and box2.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 5])
        box2 (Tensor[M, 5])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2, a3), (b1, b2, b3) = box1[:, None].chunk(3, 2), box2.chunk(3, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    correct_class = (a3 == b3)[:, :, 0]
    # IoU = inter / (area1 + area2 - inter)
    return (inter * correct_class) / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def get_centroids(boxes, labels):
    """
    Return the centroids of clusters of boxes.
    Boxes is expected to be in (x1, y1, x2, y2, conf, class) format.
    Arguments:
        boxes (Tensor[N, 6])
        labels (Tensor[N, 1]) "the index of the cluster of which each box belong"
    Returns:
        centroids (Tensor[M, 6]): the Mx6 matrix containing the centroids of each cluster
    """
    unique_cluster, nc = np.unique(labels, return_counts=True)
    centroids = torch.zeros(0, 6)

    for i in range(len(nc) - 1):
        ind = gt_box_iou(boxes[labels == i], boxes[labels == i].mean(0, keepdims=True)).max(0)[1]
        centre = boxes[labels == i][ind]
        centre[0, 4] = boxes[labels == i, 4].max()
        #centre[0, :4] = boxes[labels == i, :4].mean(0, keepdims=True)
        if centre[0, 4] > 0.001:
            centroids = torch.cat((centroids, centre), 0)

    return centroids


def xywh2xyxy(x):
    # from utils.general
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def clustering(boxes, iou_t=0.3):
    """
    Return the boxes after clustering filtering .
    Boxes is expected to be in (x1, y1, x2, y2, conf, conf_classes) format.
    Arguments:
        boxes (Tensor[N, 5 + len(class)])
    Returns:
        centroids (Tensor[M, 6]): the Mx6 matrix containing the best guesses
    """

    boxes[:, 5:] *= boxes[:, 4:5]
    box = xywh2xyxy(boxes[:, :4])

    #i, j = (boxes[:, 5:] > 0.001).nonzero(as_tuple=False).T
    #box = torch.cat((box[i], boxes[i, j + 5, None], j[:, None].float()), 1)

    box = torch.cat((box, boxes[:, 5:].max(1, keepdim=True)[0], boxes[:, 5:].max(1, keepdim=True)[1]), 1)

    iou, iop, iog = boxes_iou(box)
    dist = (iou + iog + iop)/3

    cluster = DBSCAN(eps=0.1, min_samples=3, metric='precomputed').fit(dist)
    cluster = AffinityPropagation(damping=0.9, affinity='precomputed').fit(dist)
    #centroids = get_centroids(box, cluster.labels_)
    centroids = box[cluster.cluster_centers_indices_]


    return centroids


def compute_afam(detections, labels):
    """
    Return correct and incorrect prediction matrix according to the AFA metrics.
    Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
        ioav (Array[19,1]) 19 ioas level for threshold
        accurate_metrics (Boolean) light or big calcul for the metrics
                False during training, True for validation
    Returns:
        tp_labels (Array[N/min(N;M),19]), TP for the recall
        fp_prediction (Array[N/min(N;M),19]), FP (and TP) for the precision
        conf (Array[N/min(N;M),1]), conf ordered and reduced
    """

    # Sort by confidence

    conf = detections[:, 4]
    order = np.argsort(-conf.cpu())
    conf, detections = conf[order], detections[order]

    # Computation vectors
    iogs = torch.zeros(detections.shape[0], labels.shape[0], device=labels.device)  # Input over Ground Truth
    correct_class = detections[:, 5] == labels[:, 5]

    for i, label in enumerate(labels[:, :4]):
        indexes = torch.where(correct_class[i])
        label_area = box_area(label)
        union_preds_labels = labels[:i, :4][labels[:i, 5] == labels[i, 5]]

        for j, pred in enumerate(detections[correct_class[i]]):
            intersection_pred_label = get_intersection_from_list([pred], label)
            if intersection_pred_label and (iogs[correct_class[i], i].sum()) < 0.99 * label_area:

                iogs[indexes[0][j], i] = box_area(intersection_pred_label[0]) - get_union_from_list(
                    get_intersection_between_list(union_preds_labels, intersection_pred_label))
                if iogs[indexes[0][j], i] > 10:
                    union_preds_labels = torch.cat((union_preds_labels, intersection_pred_label[0].reshape(1, 4)))

    iops = torch.t(torch.t(iogs) / boxes_area(detections[:, :4]))
    iogs = iogs / boxes_area(labels[:, :4])

    return iops, iogs
