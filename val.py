# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Rafalor metrics for AFAM

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

from utils.torch_utils import select_device, time_sync
from utils.plots import output_to_target, plot_images, plot_val_study
from utils.metrics import ConfusionMatrix, ap_per_class, box_iou, afam_per_class, avam_per_class, plot_mc_curve, \
    ap_per_class_afam
from utils.general import (LOGGER, check_dataset, check_img_size, check_requirements, check_yaml,
                           coco80_to_coco91_class, colorstr, emojis, increment_path, non_max_suppression, print_args,
                           scale_coords, xywh2xyxy, xyxy2xywh)
from utils.dataloaders import create_dataloader
from utils.callbacks import Callbacks
from models.common import DetectMultiBackend
import argparse
import json
import os
import sys
from pathlib import Path
import tools
import numpy as np
import torch
from tqdm import tqdm
from time import time

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


def process_batch_avam(detections, labels, labels_area, accurate_metrics=False):
    """
    Return correct and incorrect area prediction matrix according to the AVA metrics.
    Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
        labels_area (Float) Total area of labels boxes
        accurate_metrics (Boolean) light or big calcul for the metrics
                False during training, True for validation
    Returns:
        correct_area (Array[min(N;M+3)/min(N;M+30),1]), vector of the correct area predicted
        correct_area (Array[min(N;M+3)/min(N;M+30),1]), vector of the incorrect area predicted
        conf (Array[min(N;M+3)/min(N;M+30),1]), conf ordered and reduced
    """

    conf = detections[:, 4]
    order = np.argsort(-conf.cpu())
    conf, detections = conf[order], detections[order]
    # FA + TA for recall
    labels_area += tools.get_union_from_list(list(labels[:, 1:5]))

    # Reduction of the detections vector
    if accurate_metrics:
        conf = conf[:min(len(detections), len(labels) + 30)]
        detections = detections[:min(len(detections), len(labels) + 30)]
    else:
        conf = conf[:min(len(detections), len(labels) + 3)]
        detections = detections[:min(len(detections), len(labels) + 3)]

    # Output vectors
    correct_area, incorrect_area = torch.zeros(len(
        detections), 1, device=labels.device), torch.zeros(len(detections), 1, device=labels.device)

    for i, box in enumerate(detections):
        correct_area[i] = tools.get_union_from_list(tools.get_intersection_between_list(labels[:, 1:], [box])) \
                          - tools.get_union_from_list(tools.get_intersection_between_list(
            tools.get_intersection_between_list(detections[:i, :4], [box]), labels[:, 1:]))

        incorrect_area[i] = tools.box_area(box) - correct_area[i] - tools.get_union_from_list(
            tools.get_intersection_between_list(detections[:i, :4], [box]))

    return correct_area, incorrect_area, conf, labels_area


def process_batch_afam(detections, labels, ioav, accurate_metrics):
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

    # Reduction of the detections vector for compute time issue
    if not accurate_metrics:
        conf, detections = conf[:min(len(detections), len(labels))], detections[:min(len(detections), len(labels))]

    # Output vectors
    tp_precision = torch.ones(detections.shape[0], len(ioav), dtype=torch.bool, device=labels.device)
    tp_recall = torch.zeros(detections.shape[0], len(ioav), device=labels.device)
    # Computation vectors
    iogs = torch.zeros(detections.shape[0], labels.shape[0], device=labels.device)  # Input over Ground Truth
    iops = torch.zeros(detections.shape[0], labels.shape[0], device=labels.device)  # Input over Predictions
    box_found = []
    correct_class = detections[:, 5] == labels[:, 0:1]

    for i, label in enumerate(labels[:, 1:5]):
        indexes = torch.where(correct_class[i])

        for j, pred in enumerate(detections[correct_class[i]]):
            if tools.get_intersection_area_from_tuple(pred, label) != 0:
                intersection_pred_label = tools.get_intersection_from_list([pred], label)[0]

                union_preds_labels = torch.cat((labels[:i, 1:][labels[:i, 0] == labels[i, 0]],
                                                detections[correct_class[i]][:j, :4]))
                iogs[indexes[0][j], i] = tools.box_area(intersection_pred_label) - tools.get_union_from_list(
                    tools.get_intersection_from_list(union_preds_labels, intersection_pred_label))

    iops = torch.t(torch.t(iogs) / tools.boxes_area(detections[:, :4]))
    iogs = iogs / tools.boxes_area(labels[:, 1:])

    tp_precision = iops.sum(1).reshape(-1, 1) > ioav.expand(len(detections), len(ioav))
    tp_recall = torch.diff((torch.transpose(iogs.cumsum(0).expand(len(ioav), len(detections), len(labels)), 0, 2) >
                            ioav.expand(len(detections), len(ioav))).sum(0), 1, 0,
                           prepend=torch.zeros(1, len(ioav), device=labels.device))

    return tp_recall, tp_precision, conf, detections[:, 5]


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
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        final_epoch=True,
        compute_avam=True,
        compute_afam=True
):
    compute_avam = 0
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

        # Directories
        save_dir = increment_path(
            Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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
    iouv_afam = torch.linspace(0.5, 0.95, 10, device=device)  # iou vector for mAP@0.5:0.95
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
        dataloader = create_dataloader(data[task],
                                       imgsz,
                                       batch_size,
                                       stride,
                                       single_cls,
                                       pad=pad,
                                       rect=rect,
                                       workers=workers,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0

    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(
        model.names if hasattr(model, 'names') else model.module.names)}
    afam_names = tuple(np.round(np.array(iouv.cpu()), 2).tolist())
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 10) % ('Class', 'Images', 'Labels', 'P',
                                  'R', 'mAP@.5', 'mAP@.5:.95', 'afam_P', 'afam_R', 'avam_P', 'avam_R')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    mr_afam, mp_afam, mr_avam, mp_avam, map50_afam, map_afam = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # AFA and AVA metrics
    labels_area = 0.0  # AVA metrics (FA + TA)
    loss = torch.zeros(3, device=device)
    totimg_full, totimg_empty = 0, 0
    jdict, stats, ap, ap_class, ap50 = [], [], [], [], []
    afam_stats, avam_stats, conf_classify, fp_classify, fn_classify, afam_stats_ap = [], [], [], [], [], []
    callbacks.run('on_val_start')
    pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
    for batch_i, (im, targets, paths, shapes) in enumerate(pbar):
        callbacks.run('on_val_batch_start')
        t1 = time_sync()
        if cuda:
            im = im.to(device, non_blocking=True)
            targets = targets.to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = im.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        out, train_out = model(im) if training else model(im, augment=augment, val=True)  # inference, loss outputs
        dt[1] += time_sync() - t2

        # Loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # NMS
        # to pixels
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3

        # Metrics
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            # number of labels, predictions
            nl, npr = labels.shape[0], pred.shape[0]
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
            seen += 1

            if npr == 0:
                if nl:
                    stats.append(
                        (correct, *torch.zeros((5, 0), device=device)))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(im[si].shape[1:], predn[:, :4], shape,
                         shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # native-space labels
                labels = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labels, iouv)

                if plots:
                    confusion_matrix.process_batch(predn, labels)

                if predn.shape[0]:
                    conf_classify.append(torch.max(predn[:, 4]))
                    fn_classify.append(1)
                    fp_classify.append(0)
                totimg_full += 1
            else:
                if predn.shape[0]:
                    conf_classify.append(torch.max(predn[:, 4]))
                    fp_classify.append(1)
                    fn_classify.append(0)

                totimg_empty += 1

            # Compute AVA and AFA metrics
            if compute_avam:
                # TA = True area, FA = False area, conf
                correct_avam, incorrect_avam, conf, labels_area = process_batch_avam(predn, labels, labels_area,
                                                                                     final_epoch)
                avam_stats.append((correct_avam, incorrect_avam, conf))

            if compute_afam:
                # TP for recall, FP=(1-TP) for precision, conf
                correct_rec_afam, correct_prec_afam, conf, pred_class = process_batch_afam(predn, labels, iouv_afam,
                                                                                           final_epoch)
                afam_stats.append((correct_rec_afam, correct_prec_afam, conf, labels[:, 0]))
                afam_stats_ap.append((correct_rec_afam, correct_prec_afam, conf, pred_class, labels[:, 0]))

            # (correct, conf, pcls, tcls)

            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                # append to COCO-JSON dictionary
                save_one_json(predn, jdict, path, class_map)
            callbacks.run('on_val_image_end', pred, predn, path, names, im[si])

        # Plot images
        if plots and batch_i < 3:
            plot_images(im, targets, paths, save_dir /
                        f'val_batch{batch_i}_labels.jpg', names)  # labels
            plot_images(im, output_to_target(out), paths, save_dir /
                        f'val_batch{batch_i}_pred.jpg', names)  # pred

        callbacks.run('on_val_batch_end')

    # Compute metrics

    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    avam_stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*avam_stats)] if compute_avam else []  # to numpy
    afam_stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*afam_stats)] if compute_afam else []  # to numpy
    afam_stats_ap = [torch.cat(x, 0).cpu().numpy() for x in zip(*afam_stats_ap)] if compute_afam else []  # to numpy
    conf_classify, fp_classify, fn_classify = np.array(conf_classify), np.array(fp_classify), np.array(fn_classify)
    order = np.argsort(conf_classify)

    conf_classify, fp_classify, fn_classify = conf_classify[order], fp_classify[order], fn_classify[order]
    fp_classify = fp_classify[::-1].cumsum(0)[::-1] / (totimg_empty + totimg_full)
    fn_classify = fn_classify.cumsum(0) / (totimg_full + totimg_empty)
    # plot_mc_curve(conf_classify, [1 - fp_classify, fp_classify, fn_classify, 1 - fn_classify],
    # Path(save_dir) / 'classify.png', names=("TN", "FP", "FN", "TP"), x_label='Confidence')
    if len(stats) and stats[0].any():
        if compute_afam:
            afap, afar, afamf1, ap_afam = ap_per_class_afam(*afam_stats_ap, plot=plots, save_dir=save_dir, names=names)

        if compute_avam:
            avap, avar = avam_per_class(*avam_stats, labels_area=labels_area.cpu(), plot=plots, save_dir=save_dir,
                                        names=names)
            mp_avam, mr_avam = avap.mean(), avar.mean()
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        ap50_afam, ap_afam = ap_afam[:, 0], ap_afam.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map, = p.mean(), r.mean(), ap50.mean(), ap.mean()
        mp_afam, mr_afam, map50_afam, map_afam, = afap.mean(), afar.mean(), ap50_afam.mean(), ap_afam.mean()
        # number of targets per class
        nt = np.bincount(stats[3].astype(int), minlength=nc)
    else:
        nt = torch.zeros(1)

    # Print results
    if compute_afam and compute_avam:
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 8  # print format
        LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, mp_afam, mr_afam, mp_avam, mr_avam))
    elif compute_afam:
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 8  # print format
        LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map, mp_afam, mr_afam, map50_afam, map_afam))
    else:
        pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
        LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
            LOGGER.info(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        LOGGER.info(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                # image IDs to evaluate
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.im_files]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            # update results (mAP@0.5:0.95, mAP@0.5)
            map, map50 = eval.stats[:2]
        except Exception as e:
            LOGGER.info(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        if c > maps.shape[0]:
            print(
                "A class present on a validation label was not present during training. Please ensure that the validation dataset is correct")
        maps[c] = ap[i]

    return (mp, mr, map50, map, mr_afam, mp_afam, mr_avam, mp_avam, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--final-epoch', type=int, default=0, help='Accurate or fast validation')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
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