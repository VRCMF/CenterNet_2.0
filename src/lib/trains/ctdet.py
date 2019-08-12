from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import ctdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import ctdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer
import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from wider_eval.bbox import bbox_overlaps
from IPython import embed

from utils import bbox_helper
import logging
import os
import cv2
import math

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import json

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt):
    super(CtdetLoss, self).__init__()
    self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
    self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
              RegLoss() if opt.reg_loss == 'sl1' else None
    self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
              NormRegL1Loss() if opt.norm_wh else \
              RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
    self.opt = opt

  def forward(self, outputs, batch):
    opt = self.opt
    hm_loss, wh_loss, off_loss = 0, 0, 0
    for s in range(opt.num_stacks):
      output = outputs[s]
      if not opt.mse_loss:
        output['hm'] = _sigmoid(output['hm'])

      if opt.eval_oracle_hm:
        output['hm'] = batch['hm']
      if opt.eval_oracle_wh:
        output['wh'] = torch.from_numpy(gen_oracle_map(
          batch['wh'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
      if opt.eval_oracle_offset:
        output['reg'] = torch.from_numpy(gen_oracle_map(
          batch['reg'].detach().cpu().numpy(), 
          batch['ind'].detach().cpu().numpy(), 
          output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

      hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
      if opt.wh_weight > 0:
        if opt.dense_wh:
          mask_weight = batch['dense_wh_mask'].sum() + 1e-4
          wh_loss += (
            self.crit_wh(output['wh'] * batch['dense_wh_mask'],
            batch['dense_wh'] * batch['dense_wh_mask']) / 
            mask_weight) / opt.num_stacks
        elif opt.cat_spec_wh:
          wh_loss += self.crit_wh(
            output['wh'], batch['cat_spec_mask'],
            batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
        else:
          wh_loss += self.crit_reg(
            output['wh'], batch['reg_mask'],
            batch['ind'], batch['wh']) / opt.num_stacks
      
      if opt.reg_offset and opt.off_weight > 0:
        off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                             batch['ind'], batch['reg']) / opt.num_stacks
        
    loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
           opt.off_weight * off_loss
    loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                  'wh_loss': wh_loss, 'off_loss': off_loss}
    return loss, loss_stats

class CtdetTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    super(CtdetTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
    loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
    loss = CtdetLoss(opt)
    return loss_states, loss

  def debug(self, batch, output, iter_id):
    opt = self.opt
    reg = output['reg'] if opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=opt.cat_spec_wh, K=opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets[:, :, :4] *= opt.down_ratio
    dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
    dets_gt[:, :, :4] *= opt.down_ratio
    for i in range(1):
      debugger = Debugger(
        dataset=opt.dataset, ipynb=(opt.debug==3), theme=opt.debugger_theme)
      img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
      img = np.clip(((
        img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
      pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
      gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
      debugger.add_blend_img(img, pred, 'pred_hm')
      debugger.add_blend_img(img, gt, 'gt_hm')
      debugger.add_img(img, img_id='out_pred')
      for k in range(len(dets[i])):
        if dets[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                 dets[i, k, 4], img_id='out_pred')

      debugger.add_img(img, img_id='out_gt')
      for k in range(len(dets_gt[i])):
        if dets_gt[i, k, 4] > opt.center_thresh:
          debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                 dets_gt[i, k, 4], img_id='out_gt')

      if opt.debug == 4:
        debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
      else:
        debugger.show_all_imgs(pause=True)

  def save_result(self, output, batch, results):
    reg = output['reg'] if self.opt.reg_offset else None
    dets = ctdet_decode(
      output['hm'], output['wh'], reg=reg,
      cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
    dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
    dets_out = ctdet_post_process(
      dets.copy(), batch['meta']['c'].cpu().numpy(),
      batch['meta']['s'].cpu().numpy(),
      output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
    results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]


    """-----------------------the first modification-----------------------------------"""

  def get_gt_boxes(self, gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list

  def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
      f = open(cache_file, 'rb')
      boxes = pickle.load(f)
      f.close()
      return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
      if state == 0 and '--' in line:
        state = 1
        current_name = line
        continue
      if state == 1:
        state = 2
        continue

      if state == 2 and '--' in line:
        state = 1
        boxes[current_name] = np.array(current_boxes).astype('float32')
        current_name = line
        current_boxes = []
        continue

      if state == 2:
        box = [float(x) for x in line.split(' ')[:4]]
        current_boxes.append(box)
        continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes

  def read_pred_file(self, filepath):

    with open(filepath, 'r') as f:
      lines = f.readlines()
      img_file = lines[0].rstrip('\n\r')
      lines = lines[2:]

    boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes

  def get_preds(self, pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)

    for event in pbar:
      pbar.set_description('Reading Predictions ')
      event_dir = os.path.join(pred_dir, event)
      event_images = os.listdir(event_dir)
      current_event = dict()
      for imgtxt in event_images:
        imgname, _boxes = self.read_pred_file(os.path.join(event_dir, imgtxt))
        current_event[imgname.rstrip('.jpg')] = _boxes
      boxes[event] = current_event
    return boxes

  def norm_score(self, pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
      for _, v in k.items():
        if len(v) == 0:
          continue
        _min = np.min(v[:, -1])
        _max = np.max(v[:, -1])
        max_score = max(_max, max_score)
        min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, k in pred.items():
      for _, v in k.items():
        if len(v) == 0:
          continue
        v[:, -1] = (v[:, -1] - min_score) / diff

  def image_eval(self, pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy()
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):

      gt_overlap = overlaps[h]
      max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
      if max_overlap >= iou_thresh:
        if ignore[max_idx] == 0:
          recall_list[max_idx] = -1
          proposal_list[h] = -1
        elif recall_list[max_idx] == 0:
          recall_list[max_idx] = 1

      r_keep_index = np.where(recall_list == 1)[0]
      pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list

  def img_pr_info(self, thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

      thresh = 1 - (t + 1) / thresh_num
      r_index = np.where(pred_info[:, 4] >= thresh)[0]
      if len(r_index) == 0:
        pr_info[t, 0] = 0
        pr_info[t, 1] = 0
      else:
        r_index = r_index[-1]
        p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
        pr_info[t, 0] = len(p_index)
        pr_info[t, 1] = pred_recall[r_index]
    return pr_info

  def dataset_pr_info(self, thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
      _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
      _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve

  def voc_ap(self, rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

  def evaluation(self, pred, gt_path, iou_thresh=0.5):
    pred = self.get_preds(pred)
    self.norm_score(pred)
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = self.get_gt_boxes(gt_path)
    event_num = len(event_list)
    thresh_num = 1000
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    for setting_id in range(3):
      # different setting
      gt_list = setting_gts[setting_id]
      count_face = 0
      pr_curve = np.zeros((thresh_num, 2)).astype('float')
      # [hard, medium, easy]
      pbar = tqdm.tqdm(range(event_num))
      for i in pbar:
        pbar.set_description('Processing {}'.format(settings[setting_id]))
        event_name = str(event_list[i][0][0])
        img_list = file_list[i][0]
        pred_list = pred[event_name]
        sub_gt_list = gt_list[i][0]
        # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
        gt_bbx_list = facebox_list[i][0]

        for j in range(len(img_list)):
          pred_info = pred_list[str(img_list[j][0][0])]

          gt_boxes = gt_bbx_list[j][0].astype('float')
          keep_index = sub_gt_list[j][0]
          count_face += len(keep_index)

          if len(gt_boxes) == 0 or len(pred_info) == 0:
            continue
          ignore = np.zeros(gt_boxes.shape[0])
          if len(keep_index) != 0:
            ignore[keep_index - 1] = 1
          pred_recall, proposal_list = self.image_eval(pred_info, gt_boxes, ignore, iou_thresh)

          _img_pr_info = self.img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

          pr_curve += _img_pr_info
      pr_curve = self.dataset_pr_info(thresh_num, pr_curve, count_face)

      propose = pr_curve[:, 0]
      recall = pr_curve[:, 1]

      ap = self.voc_ap(recall, propose)
      aps.append(ap)
    print("Easy AP: {} || Medium AP: {} || Hard AP: {}".format(aps[0], aps[1], aps[2]))

    return aps