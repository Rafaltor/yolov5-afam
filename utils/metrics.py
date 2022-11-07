# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Model validation metrics
"""

import math
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    # Box filter of fraction f
    # number of filter elements (must be odd)
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def calc_custom_indicator(p, r, eps):
    return p * r / (20 * p + r + eps)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Confidence value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by Confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting

    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros(
        (nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Computes the custom metrics to achieve a satisfying tradeoff between
    # p and r
    f1 = 2 * p * r / (p + r + eps)
    # list: only classes that have data
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, y_label='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, y_label='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, y_label='Recall')

        torch.save(p, Path(save_dir) / 'precision.pt')
        torch.save(r, Path(save_dir) / 'recall.pt')
        torch.save(f1, Path(save_dir) / 'f1.pt')
        torch.save(px, Path(save_dir) / 'confidence.pt')

    i = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def ap_per_class_afam(tp_recall, tp_precision, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Confidence value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by Confidence
    i = np.argsort(-conf)
    tp_recall, tp_precision, conf, pred_cls = tp_recall[i], tp_precision[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting

    ap, p, r = np.zeros((nc, tp_recall.shape[1])), np.zeros(
        (nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs

        tpc_recall = tp_recall[i].cumsum(0)
        tpc_precision = tp_precision[i].cumsum(0)
        fpc_precision = (1 - tp_precision[i]).cumsum(0)

        # Recall
        recall = tpc_recall / (n_l + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc_precision / (tpc_precision + fpc_precision)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp_recall.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Computes the custom metrics to achieve a satisfying tradeoff between
    # p and r
    f1 = 2 * p * r / (p + r + eps)
    # list: only classes that have data
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve_afam.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve_afam.png', names, y_label='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve_afam.png', names, y_label='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve_afam.png', names, y_label='Recall')

        torch.save(p, Path(save_dir) / 'precision.pt')
        torch.save(r, Path(save_dir) / 'recall.pt')
        torch.save(f1, Path(save_dir) / 'f1.pt')
        torch.save(px, Path(save_dir) / 'confidence.pt')

    i = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    return p, r, f1, ap

def avam_per_class(ta, fa, conf, labels_area, plot=False, save_dir='.', names=(), eps=1e-16):
    """
    Compute the Average Area metrics, given the True Area et False Area matrix.
    Arguments:
        ta : True area (nparray[N,1]).
        fa : False area (nparray[N,1])
        conf : Confidence value from 0-1 (nparray[N,1]).
        labels_area : (Float) Total area of the labels boxes
        plot : (Boolean) Plot curves
        names : (Tuple) Legend for plot
        save_dir  (Path) Plot save directory
    # Returns
        The average area precision and recall
    """

    # Sort by confidence
    order = np.argsort(-conf)
    ta, fa, conf = ta[order], fa[order], conf[order]

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, avap, avar = np.zeros((1, ta.shape[1])), np.zeros((1, 1000)), np.zeros((1, 1000))

    tac = ta.cumsum(0)
    fac = fa.cumsum(0)

    # Recall
    recall = tac / (labels_area + eps)  # recall curve
    # negative x, xp because xp decreases
    avar[0] = np.interp(-px, -conf, recall[:, 0], left=0)

    # Precision
    precision = tac / (fac + tac)  # precision curve
    avap[0] = np.interp(-px, -conf, precision[:, 0], left=1)  # p at pr_score

    # AP from recall-precision curve
    for j in range(ta.shape[1]):
        ap[0, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
        if plot and j == 0:
            py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1
    f1 = 2 * avap * avar / (avap + avar + eps)

    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve_avam.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve_avam.png', names, y_label='F1')
        plot_mc_curve(px, avap, Path(save_dir) / 'P_curve_avam.png', names, y_label='AVAP')
        plot_mc_curve(px, avar, Path(save_dir) / 'R_curve_avam.png', names, y_label='AVAR')

        torch.save(avap, Path(save_dir) / 'precision_area.pt')
        torch.save(avar, Path(save_dir) / 'recall_area.pt')
        torch.save(f1, Path(save_dir) / 'f1_area.pt')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    avap, avar = avap[:, i], avar[:, i]

    return avap, avar


def afam_per_class(tp_labels, tp_pred, conf, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    # Arguments
        tp_labels  True positives regarding the labels (nparray[N,len(ioav)])
        fp_pred  False positives regarding the prediction (nparray[N,len(ioav)])
        conf  Confidence value from 0-1 (nparray[N,1])
        target_cls  True object classes (nparray[M,1]).
        plot  Plot curves
        save_dir  Plot save directory
    # Returns
        The average "Any For Any" precision and recall
    """

    # Sort by confidence
    order = np.argsort(-conf)
    tp_labels, tp_pred, conf = tp_labels[order], tp_pred[order], conf[order]
    ntot = target_cls.shape[0]

    # Create Precision-Recall curve
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, afap, afar = np.zeros((tp_labels.shape[0], tp_labels.shape[1])), np.zeros(
        (tp_labels.shape[1], 1000)), np.zeros((tp_labels.shape[1], 1000))

    tp_labelsc = tp_labels.cumsum(0)
    tp_predc = tp_pred.cumsum(0)
    fp_predc = (1-tp_pred).cumsum(0)

    # Recall
    recall = tp_labelsc / (ntot + eps)
    # Precision
    precision = tp_predc / (fp_predc + tp_predc)

    # Compute precision and recall curve for each ioa thresholds
    for i in range(tp_labels.shape[1]):
        # Recall
        afar[i, :] = np.interp(-px, -conf, recall[:, i], left=0)
        # Precision
        afap[i, :] = np.interp(-px, -conf, precision[:, i], left=1)  # p at pr_score

    # AP from recall-precision curve
    for j in range(tp_labels.shape[1]):
        ap[0, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
        if plot:
            py.append(np.interp(px, mrec, mpre))

    # Compute F1
    f1 = 2 * afar * afap / (afap + afar + eps)
    print(afar.shape)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve_afam.png', names=names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve_afam.png', names=names, y_label='F1')
        plot_mc_curve(px, afap, Path(save_dir) / 'P_curve_afam.png', names=names, y_label='AFAP')
        plot_mc_curve(px, afar, Path(save_dir) / 'R_curve_afam.png', names=names, y_label='AFAR')
        plot_mc_curve(list(names), np.transpose(afar[:,np.linspace(0,999,11).astype(int)]), Path(save_dir) / 'afar.png',
                      names=tuple(np.round(np.linspace(0, 1, 11), 2).tolist()),
                      y_label='Recall',
                      x_label='Ioa threshold')
        plot_mc_curve(list(names), np.transpose(afap[:,np.linspace(0,999,11).astype(int)]), Path(save_dir) / 'afap.png',
                      names=tuple(np.round(np.linspace(0, 1, 11), 2).tolist()),
                      y_label='Precision',
                      x_label='Ioa threshold')
        plot_mc_curve(list(names), [afap[:, 200],afar[:,200]], Path(save_dir) / 'afa_at_035.png',
                      names=('AFAP','AFAR'),
                      x_label='Ioa threshold')

        torch.save(afap, Path(save_dir) / 'custom_precision.pt')
        torch.save(afar, Path(save_dir) / 'custom_recall.pt')
        torch.save(f1, Path(save_dir) / 'custom_f1.pt')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    afap, afar = afap[:, i], afar[:, i]

    return afap, afar


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        # points where x axis (recall) changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat(
                (torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
            # don't annotate (would appear as 0.00)
            array[array < 0.005] = np.nan

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            nc, nn = self.nc, len(names)  # number of classes, names
            sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
            labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
            with warnings.catch_warnings():
                # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                warnings.simplefilter('ignore')
                sn.heatmap(array,
                           annot=nc < 30,
                           annot_kws={
                               "size": 8},
                           cmap='Blues',
                           fmt='.2f',
                           square=True,
                           vmin=0.0,
                           xticklabels=names + tuple(['background FP']) if labels else "auto",
                           yticklabels=names + tuple(['background FN']) if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, 1), box2.chunk(4, 1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or GIoU or DIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 +
                                                            b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        # GIoU https://arxiv.org/pdf/1902.09630.pdf
        return iou - (c_area - union) / c_area
    return iou  # IoU


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def box_ioa(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Returns the intersection over box2 area given box1, box2.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        ioa (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoA = inter / (area2)
    return inter / (box_area(box2.T))


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    # iou = inter / (area1 + area2 - inter)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)


# Plots ----------------------------------------------------------------------------------------------------------------


def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            # plot(recall, precision)
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue',
    label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), x_label='Confidence', y_label='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            # plot(confidence, metric)
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')
    else:
        # plot(confidence, metric)
        ax.plot(px, py.T, linewidth=1, color='grey')

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()
