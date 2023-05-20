import sys

sys.path.append('./model/RAFT/core')

import argparse
import torch
import cv2
import os
import numpy as np
from data.flow_viz import trend_plus_vis
from glob import glob
from os.path import join, exists
from raft import RAFT
from utils.utils import InputPadder

def vis(img, trend, win_name='img-trend', save_path=None, img_show=False):
    # map trend to rgb image
    vis = trend_plus_vis(trend)
    img_trend = np.concatenate([img, vis], axis=1)
    img_trend = img_trend[:, :, ::-1]
    if img_show:
        cv2.imshow(win_name, img_trend)
        cv2.waitKey(1)
    if save_path is not None:
        img_trend = img_trend.astype(np.uint8)
        cv2.imwrite(save_path, img_trend)

@torch.no_grad()
def gen_flow(img0, img1):
    img0 = torch.from_numpy(img0).permute(2, 0, 1).float()[None].to(device)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float()[None].to(device)
    padder = InputPadder(img0.shape)
    img0, img1 = padder.pad(img0, img1)
    flow_low, flow_up = model(img0, img1, iters=30, test_mode=True)
    flow_up = padder.unpad(flow_up)
    return flow_up[0].permute(1, 2, 0).cpu().numpy()

def gen_trend_dir(dir_path, trend_name, threshold_base=0.2, flow_ratio=0.5, mode='max'):
    gt_dir = join(dir_path, 'sharp')
    # gt_dir = join(dir_path, 'sharp_stab')
    blur_dir = join(dir_path, 'blur')
    trend_dir = join(dir_path, trend_name)
    flow_dir = join(dir_path, 'flow')
    # flow_dir = join(dir_path, 'flow_stab_z')
    if not exists(trend_dir):
        os.makedirs(trend_dir)
    if not exists(flow_dir):
        os.makedirs(flow_dir)
    threshold = threshold_base * flow_ratio
    num_gts = 7
    gt_indices = list(range(num_gts - 1))
    img_files = glob(join(gt_dir, '*.png'))
    num_imgs = int(len(img_files) / num_gts)
    # ! corresponding partial range to /data/dataset.py
    range_start = 0
    range_stop = num_imgs
    for i in range(range_start, range_stop):
        trend = []
        img_blur_path = join(blur_dir, '{:08d}.png'.format(i))
        img_blur = np.ascontiguousarray(cv2.imread(img_blur_path)[:, :, ::-1])  # rgb
        trend_path = join(flow_dir, '{:08d}_flow.npy'.format(i))
        if not exists(trend_path):
            for j in gt_indices:
                img_ref_path = join(gt_dir, '{:08d}_{:03d}.png'.format(i, j))
                img_ref = np.ascontiguousarray(cv2.imread(img_ref_path)[:, :, ::-1])
                img_tgt_path = join(gt_dir, '{:08d}_{:03d}.png'.format(i, j + 1))
                img_tgt = np.ascontiguousarray(cv2.imread(img_tgt_path)[:, :, ::-1])
                flow = gen_flow(img_tgt, img_ref)  # backward flow
                flow = flow * (-1.)
                size = (int(flow_ratio * flow.shape[1]), int(flow_ratio * flow.shape[0]))
                # ! resizing flow needs to time ratio
                flow = flow_ratio * cv2.resize(flow, size, interpolation=cv2.INTER_AREA)
                trend.append(flow)
            trend = np.concatenate(trend, axis=-1)
            np.save(trend_path, trend)
        else:
            # print('load {}'.format(trend_path))
            trend = np.load(trend_path)
        trend_x = trend[:, :, 0::2]
        trend_y = trend[:, :, 1::2]
        if mode == 'max':
            trend_x_idx = np.argmax(abs(trend_x), axis=-1)
            trend_x = np.take_along_axis(trend_x, np.expand_dims(trend_x_idx, axis=-1), axis=-1)
            trend_y_idx = np.argmax(abs(trend_y), axis=-1)
            trend_y = np.take_along_axis(trend_y, np.expand_dims(trend_y_idx, axis=-1), axis=-1)
        elif mode == 'avg':
            trend_x = np.mean(trend_x, axis=-1, keepdims=True)
            trend_y = np.mean(trend_y, axis=-1, keepdims=True)
        else:
            raise ValueError
        # trend_x[abs(trend_x) < threshold] = 0
        # trend_y[abs(trend_x) < threshold] = 0
        trend_x_temp = trend_x.copy()
        trend_y_temp = trend_y.copy()
        trend_x[np.sqrt((trend_x_temp ** 2) + (trend_y_temp ** 2)) < threshold] = 0
        trend_y[np.sqrt((trend_x_temp ** 2) + (trend_y_temp ** 2)) < threshold] = 0
        trend_x[trend_x > 0] = 1
        trend_x[trend_x < 0] = -1
        trend_y[trend_y > 0] = 1
        trend_y[trend_y < 0] = -1
        trend_x[(trend_x == 0) & (trend_y == 1)] = 1
        trend_x[(trend_x == 0) & (trend_y == -1)] = -1
        trend_y[(trend_y == 0) & (trend_x == 1)] = -1
        trend_y[(trend_y == 0) & (trend_x == -1)] = 1
        trend = np.concatenate([trend_x, trend_y], axis=-1)
        trend = trend.astype(np.int8)
        img_blur = cv2.resize(img_blur, (trend.shape[1], trend.shape[0]), interpolation=cv2.INTER_AREA)
        trend_vis_path = join(trend_dir, '{:08d}_trend.png'.format(i))
        vis(img_blur, trend, save_path=trend_vis_path)
        np.save(join(trend_dir, '{:08d}_trend.npy'.format(i)), trend)

if __name__ == '__main__':
      parser = argparse.ArgumentParser()
      parser.add_argument('-vd', '--video_dirs', nargs='+', help='paths of video directors', required=True)
      parser.add_argument('-tn', '--trend_name', type=str, default='trend+', help='name of trend dir')
      parser.add_argument('-tb', '--threshold_base', type=float, default=0.2, help='base of threshold')
      parser.add_argument('--mode', type=str, default='max', help='mode to quantify the opticla flows')
      # Arguments for RAFT
      parser.add_argument('-mp', '--model_path', default='./checkpoints/raft-sintel.pth', help="restore checkpoint")
      parser.add_argument('--small', action='store_true', help='use small model')
      parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
      parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
      args = parser.parse_args()
      device = 'cuda'
      model = torch.nn.DataParallel(RAFT(args))
      model.load_state_dict(torch.load(args.model_path))
      model = model.cuda()
      model.eval()
      for video_dir in args.video_dirs:
          gen_trend_dir(video_dir, trend_name=args.trend_name, threshold_base=args.threshold_base, flow_ratio=0.5,
                        mode=args.mode)
          print('finish generating flows for {}'.format(video_dir))
      print('finish all')