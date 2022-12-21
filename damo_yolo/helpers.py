# import os.path
# import torch

# from bsrgan.utils import utils_image as util
# from bsrgan.models.network_rrdbnet import RRDBNet 
# from bsrgan.main_download_pretrained_models import attempt_download_from_hub

# class BSRGAN:
#     def __init__(self, model_path):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model_path = attempt_download_from_hub(model_path, hf_token=None)
#         self.save = True
#         self.load_model()
    
#     def load_model(self):
        
#         model_name = os.path.splitext(os.path.basename(self.model_path))[0]
#         if [model_name] in ['BSRGANx2']:
#             sf = 2
            
#         model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)  # define network
#         model.load_state_dict(torch.load(self.model_path), strict=True)
#         model.eval()
        
#         for k, v in model.named_parameters():
#             v.requires_grad = False
            
#         model = model.to(self.device)
        
#         self.model_name = model_name
#         self.model = model
        
    
#     def predict(self, img_path):
#         img = util.imread_uint(img_path, n_channels=3)
#         img = util.uint2tensor4(img)
#         img = img.to(self.device)
#         img = self.model(img)
#         img = util.tensor2uint(img)
        
#         if self.save:
#             save_path = os.path.join('data/images_results')
#             util.mkdir(save_path)
#             result = util.imsave(img, os.path.join(save_path, self.model_name+'.png'))
#             return result



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
    main()
