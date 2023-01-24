# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Metrics AFAM by Rafaltor

"""
Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python path/to/val.py --weights yolov5s.pt                 # PyTorch
                                      yolov5s.torchscript        # TorchScript
                                      yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                      yolov5s.xml                # OpenVINO
                                      yolov5s.engine             # TensorRT
                                      yolov5s.mlmodel            # CoreML (macOS-only)
                                      yolov5s_saved_model        # TensorFlow SavedModel
                                      yolov5s.pb                 # TensorFlow GraphDef
                                      yolov5s.tflite             # TensorFlow Lite
                                      yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import os
from utils.torch_utils import select_device, time_sync
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.metrics import ConfusionMatrix, cover_per_conf, box_iou, afam_per_class, plot_pr_comparison, box_ioa
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
from utils.tools import boxes_iou, get_intersection_from_list, get_union_from_list, get_intersection_between_list
import numpy as np
import torch
from tqdm import tqdm
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


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


def scale_boxes(pred, ax, ay):
    '''
    Returns scaled rectangle of pred:
        box = lam * pred, (Array[N, 4])
    Arguments:
        pred (Array[N, 4]), x1, y1, x2, y2
        ax, ay 2 * (Array[N, 1]), scale factors
    Returns:
        box (Array[N, 4]), scaled prediction
    '''
    box = torch.clone(pred)
    cx, cy = (box[:, 2] + box[:, 0]) / 2, (box[:, 3] + box[:, 1]) / 2
    ratioX = (box[:, 2] - box[:, 0]) * ax / 2
    ratioY = (box[:, 3] - box[:, 1]) * ay / 2
    box[:, 0] = cx - ratioX
    box[:, 1] = cy - ratioY
    box[:, 2] = cx + ratioX
    box[:, 3] = cy + ratioY
    return box


def ious(gt, pred):
    """
        Returns IoU, IoP, IoG between ground truth and prediction
        Arguments:
        gt (Array[N, 4]), x1, y1, x2, y2
        pred (Array[N, 4]), x1, y1, x2, y2
    Returns:
        iop (Array[N, 1]), IoP between gt and pred two by two
    """
    inter = np.maximum(0, torch.minimum(gt[:, 3], pred[:, 3]) - torch.maximum(gt[:, 1], pred[:, 1])) * \
            np.maximum(0, torch.minimum(gt[:, 2], pred[:, 2]) - torch.maximum(gt[:, 0], pred[:, 0]))
    pred_areas = (pred[:, 3] - pred[:, 1]) * (pred[:, 2] - pred[:, 0])
    gt_areas = (gt[:, 3] - gt[:, 1]) * (gt[:, 2] - gt[:, 0])

    iou = inter / (pred_areas + gt_areas - inter)
    iog = inter / gt_areas
    iop = inter / pred_areas
    return iou, iog, iop


def find_scales(gt, pred):
    """
    Return the extremum scale factors for which the box pred contains fully the ground truth box and is
    contained fully by the gt box
    Both boxes are in (x1, y1, x2, y2) format.
    Arguments:
        gt (Array[N, 4]), x1, y1, x2, y2
        pred (Array[N, 4]), x1, y1, x2, y2
    Returns:
        am, aM (Array[N, 2]), scale factor for iop = 1 and iog = 1
    """
    scale = torch.zeros(pred.shape[0], 4)
    cx, cy = (pred[:, 2] + pred[:, 0]) / 2, (pred[:, 3] + pred[:, 1]) / 2
    ax1, ax2 = 2 * (gt[:, 2] - cx) / (pred[:, 2] - pred[:, 0]), 2 * (cx - gt[:, 0]) / (pred[:, 2] - pred[:, 0])
    ay1, ay2 = 2 * (gt[:, 3] - cy) / (pred[:, 3] - pred[:, 1]), 2 * (cy - gt[:, 1]) / (pred[:, 3] - pred[:, 1])

    scale[:, 0], scale[:, 1] = torch.minimum(ax1, ax2), torch.minimum(ay1, ay2)
    scale[:, 2], scale[:, 3] = torch.maximum(ax1, ax2), torch.maximum(ay1, ay2)

    return scale


def process_score(detections, labels, iou_thres, n_conf, n_class, nc):
    """
        Return the score for each label according to the conformal learning theory.
        Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
            iou_thres (float) iou threshold for the score
            n_conf (int) number of conf interval
            n_class (int) number of class interval
            nc (int) number of total class
        Returns:
            scores (Array[M, 1]), Scores for each label
    """
    conf = detections[:, 4]
    order = np.argsort(-conf.cpu())
    conf, detections = conf[order], detections[order]

    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]

    ind = torch.nonzero((iou > iou_thres) * iou * correct_class)
    labels_matched = labels[ind[:, 0], 1:]
    detection_matched = detections[ind[:, 1], :4]

    scale = find_scales(labels_matched, detection_matched)

    # pred_area = boxes_area(detection_matched[:, :4])
    # size_int = torch.linspace(0, 200, n_size)
    # pred_size = (pred_area > torch.t(size_int.expand(1, n_size)) ** 2).sum(0) - 1

    classes = detections[ind[:, 1], 5]
    class_int = torch.linspace(0, n_class, n_class, device=classes.device)
    pred_class = (classes*n_class/nc >= torch.t(class_int.expand(1, n_class))).sum(0) - 1

    conf = detections[ind[:, 1], 4]
    conf_int = torch.linspace(0, 1, n_conf, device=conf.device)
    pred_conf = (conf >= torch.t(conf_int.expand(1, n_conf))).sum(0) - 1

    return scale, pred_class, pred_conf



def coverage(detections, labels, scale):
    """
        Return the coverage rates for each class and conf_class of detections.
        Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
            amx, amy, aMx, aMy 4 * (Array[N, 1]), scale factors
        Returns:
            cov_rate (Array[n_conf, n_classes]), cover rate for each conf and class
    """
    smallB, bigB = scale_boxes(detections[:, :4], scale[0, :], scale[1, :]), \
                   scale_boxes(detections[:, :4], scale[2, :], scale[3, :])

    iop = box_ioa(labels[:, 1:], smallB)
    iog = box_ioa(bigB, labels[:, 1:])
    iou = box_iou(labels[:, 1:], detections[:, :4])
    if len(labels):
        covered = (iop.t() + iog).max(1)[0] / 2 == 1
        return covered
    else:
        return torch.zeros(len(detections), device=labels.device)


@torch.no_grad()
def run(
        data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        plots=True,
        callbacks=Callbacks(),
        risk=0.95,
        iou_conformal=0.5,
        n_conf=10,
        n_class=1,
        split=0.5

):
    # Initialize/load model and set device

    training = model is not None

    if training:  # called by train.py
        # get model device, PyTorch model
        device, pt, jit, engine = next(
            model.parameters()).device, True, False, False
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Load model
        model = DetectMultiBackend(
            weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1
                LOGGER.info(
                    f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        # Data
        data = check_dataset(data)  # check

    # Configure
    model.eval()
    cuda = device.type != 'cpu'
    is_coco = isinstance(data.get('val'), str) and data['val'].endswith(
        f'coco{os.sep}val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()



    # Dataloader
    if not training:
        if pt and not single_cls:  # check --weights are trained on --data
            ncm = model.model.nc
            assert ncm == nc, f'{weights[0]} ({ncm} classes) trained on different --data than what you passed ({nc} ' \
                              f'classes). Pass correct combination of --weights and --data that are trained together.'
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task in ('speed', 'benchmark') else 0.5
        rect = False if task == 'benchmark' else pt  # square inference for benchmarks
        # path to train/val/test images
        task = task if task in ('train', 'val', 'test') else 'val'
        cal, val = create_dataloader(data[task],
                                     imgsz,
                                     batch_size,
                                     stride,
                                     single_cls,
                                     pad=pad,
                                     rect=rect,
                                     workers=workers,
                                     prefix=colorstr(f'{task}: '),
                                     split=split)


    seen = 0
    names = {k: v for k, v in enumerate(
        model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P',
                                 'R', 'mAP@.5', 'mAP@.5:.95')

    loss = torch.zeros(3, device=device)
    stats = []

    callbacks.run('on_val_start')
    pbar = tqdm(cal, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs

        # NMS
        # to pixels
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls,
                                  targets=targets)

        # Split conformal prediction
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            # number of labels, predictions
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # native-space labels
                labels = torch.cat((labels[:, 0:1], tbox), 1)

            scale, classes, conf = process_score(predn, labels, iou_conformal, n_conf, n_class, nc)

            stats.append((scale, classes, conf))


            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        callbacks.run('on_val_batch_end')

    # Compute metrics

    scale, classes, conf = [torch.cat(x, 0).cpu() for x in zip(*stats)]  # to numpy




    qalpha = torch.ones(4, n_conf, n_class, device=predn.device)

    for co in range(qalpha.shape[1]):
        for cl in range(qalpha.shape[2]):
            if len(scale[torch.mul(conf == co, classes == cl), :]):
                qalpha[0, co, cl] = np.quantile(scale[torch.mul(conf == co, classes == cl), 0], 1 - risk)
                qalpha[1, co, cl] = np.quantile(scale[torch.mul(conf == co, classes == cl), 1], 1 - risk)
                qalpha[2, co, cl] = np.quantile(scale[torch.mul(conf == co, classes == cl), 2], risk)
                qalpha[3, co, cl] = np.quantile(scale[torch.mul(conf == co, classes == cl), 3], risk)

    #Plot relevants curves
    '''
    if plots:
        plt.figure(1)
        plt.subplot(2, 2, 1)
        plt.plot(qmx.max(1)[0])
        plt.plot(qmx.min(1)[0])
        plt.subplot(2, 2, 2)
        plt.plot(qmy.max(1)[0])
        plt.plot(qmy.min(1)[0])
        plt.subplot(2, 2, 3)
        plt.plot(qMx.max(1)[0])
        plt.plot(qMx.min(1)[0])
        plt.subplot(2, 2, 4)
        plt.plot(qMy.max(1)[0])
        plt.plot(qMy.min(1)[0])
        plt.show()

        plt.figure(3)
        plt.subplot(2, 2, 1)
        plt.plot(qmx.max(0)[0])
        plt.plot(qmx.min(0)[0])
        plt.subplot(2, 2, 2)
        plt.plot(qmy.max(0)[0])
        plt.plot(qmy.min(0)[0])
        plt.subplot(2, 2, 3)
        plt.plot(qMx.max(0)[0])
        plt.plot(qMx.min(0)[0])
        plt.subplot(2, 2, 4)
        plt.plot(qMy.max(0)[0])
        plt.plot(qMy.min(0)[0])
        plt.show()

        conf_int = np.linspace(0, 1, n_conf)
        classes_int = np.linspace(0, n_class + 1, n_class)
        classes_int, conf_int = np.meshgrid(classes_int, conf_int)
        plt.figure(2)
        plt.subplot(2, 2, 1)
        plt.pcolormesh(conf_int, classes_int, qmx)
        plt.colorbar()
        plt.subplot(2, 2, 2)
        plt.pcolormesh(conf_int, classes_int, qMx)
        plt.colorbar()
        plt.subplot(2, 2, 3)
        plt.pcolormesh(conf_int, classes_int, qmy)
        plt.colorbar()
        plt.subplot(2, 2, 4)
        plt.pcolormesh(conf_int, classes_int, qMy)
        plt.colorbar()

        plt.figure(4)
        plt.subplot(1, 2, 1)
        plt.pcolormesh(conf_int, classes_int, abs(qmy - qmx))
        plt.subplot(1, 2, 2)
        plt.pcolormesh(conf_int, classes_int, abs(qMy - qMx))
        plt.show()
        print(1)
    '''

    stats = []
    pbar = tqdm(val, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs

        # NMS
        # to pixels
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls,
                                  targets=targets)

        # Split conformal prediction
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            # number of labels, predictions
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]

            seen += 1

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # native-space labels
                labels = torch.cat((labels[:, 0:1], tbox), 1)

            conf_int = torch.linspace(0, 1, n_conf, device=predn.device)
            conf = (predn[:, 4] >= torch.t(conf_int.expand(1, n_conf))).sum(0) - 1

            class_int = torch.linspace(0, n_class, n_class, device=predn.device)
            classes = (predn[:, 5].long() * n_class / nc >= torch.t(class_int.expand(1, n_class))).sum(0) - 1

            scale = qalpha[:, conf, classes]
            covered = coverage(predn, labels, scale)
            stats.append((conf, classes, covered))

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy

    cover = cover_per_conf(*stats, n_conf, n_class)
    plt.figure()
    plt.imshow(cover)
    plt.show()
    print(qalpha.transpose(0, 2)[:, :-1, :], np.transpose(cover))

    return qalpha.transpose(0, 2)[:, :-1, :], np.transpose(cover)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--iou-conformal', type=float, default=0.5, help='IoU threshold for conformal prediction')
    parser.add_argument('--split', type=float, default=0.5, help='Split factor of dataset')
    parser.add_argument('--n-conf', type=int, default=10, help='Number of confidence interval')
    parser.add_argument('--n-class', type=int, default=1, help='Number of class interval')
    parser.add_argument('--risk', type=float, default=0.95, help='Coverage Rate')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        if opt.conf_thres > 0.001:  # https://github.com/ultralytics/yolov5/issues/1466
            LOGGER.info(emojis(f'WARNING: confidence threshold {opt.conf_thres} > 0.001 produces invalid results ⚠️'))
        run(**vars(opt))

    else:
        weights = opt.weights if isinstance(opt.weights, list) else [opt.weights]
        opt.half = True  # FP16 for fastest results
        if opt.task == 'speed':  # speed benchmarks
            # python val.py --task speed --data coco.yaml --batch 1 --weights yolov5n.pt yolov5s.pt...
            opt.conf_thres, opt.iou_thres, opt.save_json = 0.25, 0.45, False
            for opt.weights in weights:
                run(**vars(opt), plots=False)

        elif opt.task == 'study':  # speed vs mAP benchmarks
            # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n.pt yolov5s.pt...
            for opt.weights in weights:
                # filename to save to
                f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'
                # x axis (image sizes), y axis
                x, y = list(range(256, 1536 + 128, 128)), []
                for opt.imgsz in x:  # img-size
                    LOGGER.info(f'\nRunning {f} --imgsz {opt.imgsz}...')
                    r, _, t = run(**vars(opt), plots=False)
                    y.append(r + t)  # results and times
                np.savetxt(f, y, fmt='%10.4g')  # save
            os.system('zip -r study.zip study_*.txt')
            plot_val_study(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
