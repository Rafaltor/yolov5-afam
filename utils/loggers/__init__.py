# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Logging utils
"""

import os
import warnings

import pkg_resources as pkg
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.general import colorstr, cv2, emojis
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.plots import plot_images, plot_results
from utils.torch_utils import de_parallel
import numpy as np

LOGGERS = ('csv', 'tb', 'wandb')  # text-file, TensorBoard, Weights & Biases
RANK = int(os.getenv('RANK', -1))

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in {0, -1}:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None


class Loggers():
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.logger = logger  # for printing results to console
        self.include = include
        self.keys = [
            'train/box_loss',
            'train/obj_loss',
            'train/cls_loss',  # train loss
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',
            'metrics/custom_recall',
            'metrics/custom_precision',
            'metrics/areaverage_recall',
            'metrics/areaverage_precision',  # metrics
            'val/box_loss',
            'val/obj_loss',
            'val/cls_loss',  # val loss
            'x/lr0',
            'x/lr1',
            'x/lr2']  # params
        self.best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP_0.5', 'best/mAP_0.5:0.95',
                          'best/area_precision', 'best/area_recall', 'best/areaverage_precision', 'best/areaverage_recall']
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv

        # Message
        if not wandb:
            prefix = colorstr('Weights & Biases: ')
            s = f"{prefix}run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)"
            self.logger.info(emojis(s))

        # TensorBoard
        s = self.save_dir
        if 'tb' in self.include and not self.opt.evolve:
            prefix = colorstr('TensorBoard: ')
            self.logger.info(
                f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/")
            self.tb = SummaryWriter(str(s))

        # W&B
        if wandb and 'wandb' in self.include:
            wandb_artifact_resume = isinstance(
                self.opt.resume, str) and self.opt.resume.startswith('wandb-artifact://')
            run_id = torch.load(self.weights).get(
                'wandb_id') if self.opt.resume and not wandb_artifact_resume else None
            self.opt.hyp = self.hyp  # add hyperparameters
            self.wandb = WandbLogger(self.opt, run_id)
            # temp warn. because nested artifacts not supported after 0.12.10
            if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.11'):
                self.logger.warning(
                    "YOLOv5 temporarily requires wandb version 0.12.10 or below. Some features may not work as expected."
                )
        else:
            self.wandb = None

    def on_train_start(self):
        # Callback runs on train start
        pass

    def on_pretrain_routine_end(self):
        # Callback runs on pre-train routine end
        paths = self.save_dir.glob('*labels*.jpg')  # training labels
        if self.wandb:
            self.wandb.log(
                {"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})

    def on_train_batch_end(self, ni, model, imgs, targets, paths, plots):
        # Callback runs on train batch end
        if plots:
            if ni == 0:
                if not self.opt.sync_bn:  # --sync known issue https://github.com/ultralytics/yolov5/issues/3754
                    with warnings.catch_warnings():
                        # suppress jit trace warning
                        warnings.simplefilter('ignore')
                        self.tb.add_graph(torch.jit.trace(
                            de_parallel(model), imgs[0:1], strict=False), [])
            if ni < 3:
                f = self.save_dir / f'train_batch{ni}.jpg'  # filename
                plot_images(imgs, targets, paths, f)
            if self.wandb and ni == 10:
                files = sorted(self.save_dir.glob('train*.jpg'))
                self.wandb.log(
                    {'Mosaics': [wandb.Image(str(f), caption=f.name) for f in files if f.exists()]})

    def on_train_epoch_end(self, epoch):
        # Callback runs on train epoch end
        if self.wandb:
            self.wandb.current_epoch = epoch + 1

    def on_val_image_end(self, pred, predn, path, names, im):
        # Callback runs on val image end
        if self.wandb:
            self.wandb.val_one_image(pred, predn, path, names, im)

    def on_val_end(self):
        # Callback runs on val end
        if self.wandb:
            files = sorted(self.save_dir.glob('val*.jpg'))
            self.wandb.log(
                {"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        # Callback runs at the end of each fit (train+val) epoch
        x = dict(zip(self.keys, vals))

        if self.csv:
            file = self.save_dir / 'results.csv'
            n = len(x) + 1  # number of cols
            s = '' if file.exists() else (
                ('%20s,' * n % tuple(['epoch'] + self.keys)).rstrip(',') + '\n')  # add header
            with open(file, 'a') as f:
                f.write(s + ('%20.5g,' * n %
                        tuple([epoch] + vals)).rstrip(',') + '\n')

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)

        if self.wandb:
            if best_fitness == fi:
                best_results = [epoch] + vals[3:11]
                for i, name in enumerate(self.best_keys):
                    # log best results in the summary
                    self.wandb.wandb_run.summary[name] = best_results[i]
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        # Callback runs on model save event
        if self.wandb:
            if ((epoch + 1) % self.opt.save_period == 0 and not final_epoch) and self.opt.save_period != -1:
                self.wandb.log_model(last.parent, self.opt,
                                     epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best, plots, epoch, results):
        # Callback runs on training end
        if plots:
            # save results.png
            plot_results(file=self.save_dir / 'results.csv')
        files = ['results.png', 'confusion_matrix.png', *
                 (f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R', 'custom_metrics'))]
        files = [(self.save_dir / f)
                 for f in files if (self.save_dir / f).exists()]  # filter
        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        if self.tb:
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(
                    str(f))[..., ::-1], epoch, dataformats='HWC')

        if self.wandb:
            self.wandb.log(dict(zip(self.keys[3:12], results)))
            self.wandb.log(
                {"Results": [wandb.Image(str(f), caption=f.name) for f in files]})

            p = torch.load(self.save_dir / 'precision.pt')[0]
            r = torch.load(self.save_dir / 'recall.pt')[0]
            f1 = torch.load(self.save_dir / 'f1.pt')[0]
            confidence = torch.load(self.save_dir / 'confidence.pt')


            avap = torch.load(self.save_dir / 'precision_area.pt')[0]
            avar = torch.load(self.save_dir / 'recall_area.pt')[0]
            f1_area = torch.load(self.save_dir / 'f1_area.pt')[0]


            custom_p = torch.load(self.save_dir / 'custom_precision.pt')[5]
            custom_r = torch.load(self.save_dir / 'custom_recall.pt')[5]
            custom_f1 = torch.load(self.save_dir / 'custom_f1.pt')[5]


            data = [
                [prec.item(), prec_area.item(), cust_prec.item(),
                 rec.item(), rec_area.item(), cust_rec.item(),
                 f1_score.item(), f1_score_area.item(), f1_score_custom.item(),
                 conf.item()] for (
                    prec, prec_area, cust_prec,
                    rec, rec_area, cust_rec,
                    f1_score, f1_score_area, f1_score_custom,
                    conf) in zip(
                    p, avap, custom_p,
                    r, avar, custom_r,
                    f1, f1_area, custom_f1,
                    confidence)]
            table = wandb.Table(data=data, columns=[
                                "precision", "area precision", "custom precision",
                                "recall", "area recall", "custom recall",
                                "f1", "area f1", "custom f1",
                                "confidence"])
            original = np.stack(
                (p, r, f1, confidence), axis=1)
            area = np.stack((avap, avar, f1_area,
                            confidence), axis=1)
            custom = np.stack(
                (custom_p, custom_r, custom_f1, confidence), axis=1)
            data2 = np.concatenate((original, area, custom))
            names = ['original']*(p.shape[0])\
                + ['area']*(avap.shape[0])\
                + ['custom']*(custom_p.shape[0])

            table_2 = wandb.Table(columns=['precision', 'recall',
                                           'f1', 'confidence'], data=data2)

            table_2.add_column(name="metric type", data=names)
            wandb.log({"metric Table": table_2})

            wandb.log({"Basic Precision": wandb.plot.line(table, "confidence", "precision",
                                                          title="Precision")})
            wandb.log({"Area Metric Precision": wandb.plot.line(table, "confidence", "area precision",
                                                                title="Area Precision")})
            wandb.log({"Custom Metric Precision": wandb.plot.line(table, "confidence", "custom precision",
                                                                  title="Custom Precision")})

            wandb.log({"Basic Metric Recall": wandb.plot.line(table, "confidence", "recall",
                                                              title="Recall")})
            wandb.log({"Area Metric Recall": wandb.plot.line(table, "confidence", "area recall",
                                                             title="Area Recall")})
            wandb.log({"Custom Metric Recall": wandb.plot.line(table, "confidence", "custom recall",
                                                               title="Custom Recall")})

            wandb.log({"Basic Precision Recall": wandb.plot.line(table, "recall", "precision",
                                                                 title="Precision Recall")})
            wandb.log({"Area Metric Precision Recall": wandb.plot.line(table, "area recall", "area precision",
                                                                       title="Area Precision Recall")})
            wandb.log({"Custom Metric Precision Recall": wandb.plot.line(table, "custom recall", "custom precision",
                                                                         title="Custom Precision Recall")})

            wandb.log({"Basic F1": wandb.plot.line(table, "confidence", "f1",
                                                   title="F1")})
            wandb.log({"Area Metric F1": wandb.plot.line(table, "confidence", "area f1",
                                                         title="Area F1")})
            wandb.log({"Custom Metric F1": wandb.plot.line(table, "confidence", "custom f1",
                                                           title="custom F1")})


            # Calling wandb.log. TODO: Refactor this into WandbLogger.log_model
            if not self.opt.evolve:
                wandb.log_artifact(str(best if best.exists() else last),
                                   type='model',
                                   name=f'run_{self.wandb.wandb_run.id}_model',
                                   aliases=['latest', 'best', 'stripped'])
            self.wandb.finish_run()
    def on_params_update(self, params):
        # Update hyperparams or configs of the experiment
        # params: A dict containing {param: value} pairs
        if self.wandb:
            self.wandb.wandb_run.config.update(params, allow_val_change=True)
