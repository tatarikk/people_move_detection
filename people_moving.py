import argparse

import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path

import pandas as pd
from collections import Counter
from collections import deque

import warnings

warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend

try:
    from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
except:
    import sys

    sys.path.append('yolov5/utils')

from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args,
                                  check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


class ObjectTracker:
    def __init__(self):
        self.line_position = 300
        self.enter_line_crossed = 0
        self.exit_line_crossed = 0
        self.last_known_position = {}
        self.threshold = 100

    @torch.no_grad()
    def run(
            self,
            source='0',
            yolo_weights=WEIGHTS / 'yolov5m.pt',
            strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',
            config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
            imgsz=(640, 640),
            conf_thres=0.5,
            iou_thres=0.45,
            max_det=1000,
            device='',
            show_vid=False,
            save_txt=False,
            save_conf=False,
            save_crop=False,
            save_vid=False,
            nosave=False,
            classes=None,
            agnostic_nms=False,
            augment=False,
            visualize=False,
            update=False,
            project=ROOT / 'runs/track',
            name='exp',
            exist_ok=False,
            line_thickness=2,
            hide_labels=False,
            hide_conf=False,
            hide_class=False,
            half=False,
            dnn=False,
            count=False,
            draw=False,
            speed=2,
            line_position=300,
    ):
        self.line_position = line_position
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)
        if not isinstance(yolo_weights, list):
            exp_name = yolo_weights.stem
        elif type(yolo_weights) is list and len(yolo_weights) == 1:
            exp_name = Path(yolo_weights[0]).stem
        else:
            exp_name = 'ensemble'
        exp_name = name if name else exp_name + "_" + strong_sort_weights.stem
        save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)
        (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        device = select_device(device)
        model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)
        trajectory = {}
        counted = {}
        if webcam:
            show_vid = check_imshow()
            cudnn.benchmark = True
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            nr_sources = len(dataset)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            nr_sources = 1
        vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources
        cfg = get_config()
        cfg.merge_from_file(config_strongsort)
        strongsort_list = []
        for i in range(nr_sources):
            strongsort_list.append(
                StrongSORT(
                    strong_sort_weights,
                    device,
                    max_dist=cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.STRONGSORT.MAX_AGE,
                    n_init=cfg.STRONGSORT.N_INIT,
                    nn_budget=cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

                )
            )
        outputs = [None] * nr_sources
        model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            if frame_idx % speed != 0:
                continue
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]
            t2 = time_sync()
            dt[0] += t2 - t1
            visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            for i, det in enumerate(pred):
                seen += 1
                if webcam:
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    p = Path(p)
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(save_dir / p.name)
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    p = Path(p)
                    if source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(save_dir / p.name)
                    else:
                        txt_file_name = p.parent.name
                        save_path = str(save_dir / p.parent.name)
                curr_frames[i] = im0
                txt_path = str(save_dir / 'tracks' / txt_file_name)
                s += '%gx%g ' % im.shape[2:]
                imc = im0.copy() if save_crop else im0
                annotator = Annotator(im0, line_width=2, pil=not ascii)
                cv2.line(imc, (0, self.line_position), (imc.shape[1], self.line_position),
                         (0, 255, 0), 1)
                if cfg.STRONGSORT.ECC:
                    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
                if det is not None and len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]
                    t4 = time_sync()
                    outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4
                    if len(outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            if cls != 0:
                                continue
                            center = (
                                (int(bboxes[0]) + int(bboxes[2])) // 2, (int(bboxes[1]) + int(bboxes[3])) // 2)
                            if draw:
                                if id not in trajectory:
                                    trajectory[id] = []
                                    counted[id] = False
                                trajectory[id].append(center)
                                for i1 in range(1, len(trajectory[id])):
                                    if trajectory[id][i1 - 1] is None or trajectory[id][i1] is None:
                                        continue
                                    thickness = 1
                                    try:
                                        cv2.line(im0, trajectory[id][i1 - 1], trajectory[id][i1], (0, 0, 255),
                                                 thickness)
                                        if not counted[id] and trajectory[id][i1 - 1][1] < self.line_position <= \
                                                trajectory[id][i1][1]:
                                            self.enter_line_crossed += 1
                                            message = "Человек зашел"
                                            out = max(0, self.enter_line_crossed - self.exit_line_crossed)
                                            LOGGER.info(f"Person ID: {id} {message}. "
                                                        f"Внутри людей: {out}. ")
                                            # f"Total: {self.enter_line_crossed}/{self.exit_line_crossed}")
                                            counted[id] = True
                                        elif not counted[id] and trajectory[id][i1 - 1][1] > self.line_position >= \
                                                trajectory[id][i1][1]:
                                            self.exit_line_crossed += 1
                                            message = "Человек вышел"
                                            out = max(0, self.enter_line_crossed - self.exit_line_crossed)
                                            LOGGER.info(f"Person ID: {id} {message}. "
                                                        f"Внутри людей: {out}. ")
                                            # f"Total: {self.enter_line_crossed}/{self.exit_line_crossed}")
                                            counted[id] = True
                                    except:
                                        pass
                            if save_vid or save_crop or show_vid:
                                c = int(cls)
                                id = int(id)
                                label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                                      (
                                                                          f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                                annotator.box_label(bboxes, label, color=colors(c, True))
                                if save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[
                                        c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                else:
                    strongsort_list[i].increment_ages()
                if show_vid:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):
                        break
                if save_vid:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
                prev_frames[i] = curr_frames[i]
        t = tuple(x / seen * 1E3 for x in dt)
        if save_txt or save_vid:
            s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(yolo_weights)
        print(f'Done. Exited: {self.exit_line_crossed} Entered: {self.enter_line_crossed}')


def main():
    tracker = ObjectTracker()
    tracker.run(
        # source="demo.avi",
        source="people_detection/people_count.avi",
        yolo_weights=WEIGHTS / "yolov5n6.pt",
        show_vid=True,
        draw=True,
        speed=2,
        line_position=750,
    )


if __name__ == "__main__":
    main()
