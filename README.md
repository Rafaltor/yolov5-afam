# YOLOv5 with AFAM

Ultralytics YOLOv5, plus a per-class AFAM score on validation. The metric lives in `afam_per_class` (`utils/metrics.py`) and follows the toolkit in [Object-Detection-Metrics](https://github.com/Rafaltor/Object-Detection-Metrics). `conformal.py` sits next to that score.

Train, detect, and export keep the usual YOLOv5 entry points.

## Validate

```bash
pip install -r requirements.txt
python val.py --weights yolov5s.pt --data coco128.yaml --img 640
```

`yolov5s.pt`, `yolov5l.pt`, and `codebrim.pt` are weight files, not source.

License of the YOLOv5 base: GPL-3.0.
