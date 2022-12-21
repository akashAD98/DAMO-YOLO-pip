# class YOLOV6:
#     def __init__(
#         self, 
#         weights = 'weights/yolov6s.pt',
#         device = 'cpu',
#         half = False,
#         conf_thres = 0.25,
#         iou_thresh = 0.45,
#         classes = None,
#         agnostic_nms = False,
#         max_det = 1000,
#         save_dir = 'inference/output',
#         save_txt = False,
#         save_img = True,
#         hide_labels = False,
#         hide_conf = False,
#         view_img = False
#     ):

#         self.__dict__.update(locals())
#         self.weights = weights
#         self.device = torch.device('cpu' if device == 'cpu' else 'cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.half = half
        
#         # Load model
#         model = self.load_model()
#         self.stride = model.stride
        
#         # Model Parameters
#         self.conf_thres = conf_thres
#         self.iou_thresh = iou_thresh
#         self.classes = classes
#         self.agnostic_nms = agnostic_nms
#         self.max_det = max_det
#         self.save_dir = save_dir
#         self.save_txt = save_txt
#         self.save_img = save_img
#         self.hide_labels = hide_labels
#         self.hide_conf = hide_conf
#         self.view_img = view_img

    
#     def load_model(self):
#         # Init model
#         model = DetectBackend(self.weights, device=self.device)
        
#         # Switch model to deploy status
#         model_switch(model.model)

#         # Half precision
#         if self.half & (self.device != 'cpu'):
#             model.model.half()
#         else:
#             model.model.float()
#             self.half = False

#         self.model = model
#         return model


#     def predict(
#         self, 
#         source,
#         yaml,
#         img_size,
#     ):
#         ''' Model Inference and results visualization '''
#         files = LoadData(source)
#         class_names = load_yaml(yaml)['names']
#         img_size = check_img_size(img_size, s=self.stride)
#         if self.device != 'cpu':
#             self.model(torch.zeros(1, 3, *img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

#         vid_path, vid_writer, windows = None, None, []
#         fps_calculator = CalcFPS()
#         for img_src, img_path, vid_cap in tqdm(files):
#             img, img_src = Inferer.precess_image(img_src, img_size, self.stride, self.half)
#             img = img.to(self.device)
#             if len(img.shape) == 3:
#                 img = img[None]
#                 # expand for batch dim
            
#             t1 = time.time()
#             pred_results = self.model(img)
#             det = non_max_suppression(pred_results, self.conf_thres, self.iou_thresh, classes=self.classes, agnostic=self.agnostic_nms, max_det=self.max_det)[0]
#             t2 = time.time()
            
#             # Create output files in nested dirs that mirrors the structure of the images' dirs
#             rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(source))
#             save_path = osp.join(self.save_dir, rel_path, osp.basename(img_path))  # im.jpg
#             txt_path = osp.join(self.save_dir, rel_path, osp.splitext(osp.basename(img_path))[0])
#             os.makedirs(osp.join(self.save_dir, rel_path), exist_ok=True)

#             gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
#             img_ori = img_src.copy()

#             # check image and font
#             assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
#             Inferer.font_check()

#             det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
#             for *xyxy, conf, cls in reversed(det):
#                 if self.save_txt:  # Write to file
#                     xywh = (Inferer.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
#                     line = (cls, *xywh, conf)
#                     with open(txt_path + '.txt', 'a') as f:
#                         f.write(('%g ' * len(line)).rstrip() % line + '\n')

#                 if self.save_img or self.view_img:  # Add bbox to image
#                     class_num = int(cls)  # integer class
#                     label = None if self.hide_labels else (class_names[class_num] if self.hide_conf else f'{class_names[class_num]} {conf:.2f}')

#                     Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))

 
#             img_src = np.asarray(img_ori)

#             # FPS counter
#             fps_calculator.update(1.0 / (t2 - t1))
#             avg_fps = fps_calculator.accumulate()
#             if files.type == 'video':
#                 Inferer.draw_text(
#                     img_src,
#                     f'FPS: {avg_fps:.2f}',
#                     pos=(20, 20),
#                     font_scale=1.0,
#                     text_color=(204, 85, 17),
#                     text_color_bg=(255, 255, 255),
#                     font_thickness=2,
#                 )

#             if self.view_img:
#                 if img_path not in windows:
#                     windows.append(img_path)
#                     cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
#                     cv2.resizeWindow(str(img_path), img_src.shape[1], img_src.shape[0])
#                 cv2.imshow(str(img_path), img_src)
#                 cv2.waitKey(0)  # 1 millisecond
                
            
#             # Save results (image with detections)
#             if self.save_img:
#                 if files.type == 'image':
#                     cv2.imwrite(save_path, img_src)
#                 else:  # 'video' or 'stream' 
#                     if vid_path != save_path:  # new video
#                         vid_path = save_path
#                         if isinstance(vid_writer, cv2.VideoWriter):
#                             vid_writer.release()  # release previous video writer
#                         if vid_cap:  # video
#                             fps = vid_cap.get(cv2.CAP_PROP_FPS)
#                             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                         else:  # stream
#                             fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
#                         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
#                         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#                     vid_writer.write(img_src)
                    
#!/usr/bin/env python3
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from damo.base_models.core.ops import RepConv
from damo.config.base import parse_config
from damo.detectors.detector import build_local_model
from damo.utils import get_model_info, vis
from damo.utils.demo_utils import transform_img


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


COCO_CLASSES = []
for i in range(80):
    COCO_CLASSES.append(str(i))
COCO_CLASSES = tuple(COCO_CLASSES)

class DAMO_YOLO:
    
    def make_parser():
        parser = argparse.ArgumentParser('damo eval')

        parser.add_argument(
            '-f',
            '--config_file',
            default=None,
            type=str,
            help='pls input your config file',
        )
        parser.add_argument('-p',
                            '--path',
                            default='./assets/dog.jpg',
                            type=str,
                            help='path to image')
        parser.add_argument('-c',
                            '--ckpt',
                            default=None,
                            type=str,
                            help='ckpt for eval')
        parser.add_argument('--conf',
                            default=0.6,
                            type=float,
                            help='conf of visualization')

        parser.add_argument('--img_size',
                            default=640,
                            type=int,
                            help='test img size')
        return parser


    @logger.catch
    def main():
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        args = make_parser().parse_args()

        origin_img = np.asarray(Image.open(args.path).convert('RGB'))
        config = parse_config(args.config_file)

        config.dataset.size_divisibility = args.img_size
        img = transform_img(origin_img, args.img_size,
                            **config.test.augment.transform)
        img = img.to(device)

        model = build_local_model(config, device)

        ckpt_file = args.ckpt
        logger.info('loading checkpoint from {}'.format(ckpt_file))
        loc = 'cuda:{}'.format(0)
        ckpt = torch.load(ckpt_file, map_location=loc)
        new_state_dict = {}
        for k, v in ckpt['model'].items():
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)

        for layer in model.modules():
            if isinstance(layer, RepConv):
                layer.switch_to_deploy()

        model.eval()
        logger.info('Model Summary: {}'.format(
            get_model_info(model, (args.img_size, args.img_size))))
        logger.info('loaded checkpoint done.')

        output_folder = './demo'
        mkdir(output_folder)

        output = model(img)

        ratio = min(origin_img.shape[0] / img.image_sizes[0][0],
                    origin_img.shape[1] / img.image_sizes[0][1])

        bboxes = output[0].bbox * ratio
        scores = output[0].get_field('scores')
        cls_inds = output[0].get_field('labels')

        out_img = vis(origin_img,
                      bboxes,
                      scores,
                      cls_inds,
                      conf=args.conf,
                      class_names=COCO_CLASSES)

        output_path = os.path.join(output_folder, os.path.split(args.path)[-1])
        logger.info('saved torch inference result into {}'.format(output_path))
        cv2.imwrite(output_path, out_img[:, :, ::-1])


    
if __name__ == '__main__':
    model = DAMO_YOLO(
        weights='kadirnar/yolov6t-v2.0',
        device='cuda:0',
        half=False
    )

    model = model.predict(
        source='data/images/',
        yaml='data/coco.yaml',
        img_size=640,
    )
