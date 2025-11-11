# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
from tqdm import tqdm
import time
import torch

parser = ArgumentParser()
parser.add_argument('--img', default='', help='Image file')
parser.add_argument('--config', default='', help='Config file')
parser.add_argument('--checkpoint', default='', help='Checkpoint file')
parser.add_argument('--out-file', default=None, help='Path to output file')
parser.add_argument(
    '--device', default='cuda', help='Device used for inference')
parser.add_argument(
    '--opacity',
    type=float,
    default=0.5,
    help='Opacity of painted segmentation map. In (0, 1] range.')
parser.add_argument(
    '--with-labels',
    action='store_true',
    default=False,
    help='Whether to display the class labels.')
parser.add_argument(
    '--title', default='result', help='The image identifier.')
args = parser.parse_args()

args.device = 'cuda'
args.config = 'segformer_b4_fashion10k_add_background_2_74.23/segformer_mit-b4_8xb2-160k_fashion10k-512x512.py'
args.checkpoint = 'segformer_b4_fashion10k_add_background_2_74.23/best_mIoU_iter_90120.pth'

# build the model from a config file and a checkpoint file
model = init_model(args.config, args.checkpoint, device=args.device)
if 'fashion10k' in args.checkpoint or 'depth-anything' in args.checkpoint:
    model.dataset_meta['palette'] = [
        [0, 0, 0], [255, 0, 0], [0, 128, 0], [0, 0, 255],
        [0, 128, 128], [238, 130, 238], [128, 128, 128], [255, 255, 0],
        [255, 153, 18], [255, 125, 64], [127, 255, 0], [175, 238, 238],
        [138, 43, 226], [210, 105, 30], [0, 0, 139], [72, 61, 139],
        [255, 20, 147], [255, 192, 203], [205, 92, 92], [32, 178, 170],
        [132, 112, 255], [160, 82, 45], [255, 222, 173], [240, 230, 140],
        ]

input_shape = (512, 512)
input_tensor = torch.randn(1, 3, *input_shape).cuda()
onnx_file = 'depth-anything_seg.onnx'
model.onnx_export(input_tensor, onnx_file)
