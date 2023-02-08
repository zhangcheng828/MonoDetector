import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, List, Dict, Any
from .base_dense_heads import BaseDenseHeads

warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses import *

from utils.tensor_ops import (extract_input, 
                              extract_target, 
                              transpose_and_gather_feat,
                              get_local_maximum, 
                              get_topk_from_heatmap)
from utils.target_generator import TargetGenerator
from utils.kitti_convert_utils import convert_to_kitti_2d, convert_to_kitti_3d

# Constants
EPS = 1e-12


class MonoConDenseHeads(BaseDenseHeads):
    def __init__(self,
                 in_ch: int = 64,
                 feat_ch: int = 64,
                 num_kpts: int = 9,
                 num_alpha_bins: int = 12,
                 num_classes: int = 3,
                 max_objs: int = 30,
                 test_config: Dict[str, Any] = None):
        
        super().__init__(in_ch=64, feat_ch=64, num_alpha_bins=12,
                        num_classes=3,max_objs=30, test_config=test_config)
        
        
        # Target Generator
        self.target_generator = TargetGenerator(
            num_classes=num_classes,
            max_objs=max_objs,
            num_kpt=num_kpts,
            num_alpha_bins=num_alpha_bins)


        """
        Prediction Heads
        """
        
        self.heatmap_head = self._build_head(in_ch, feat_ch, num_classes)
        self.wh_head = self._build_head(in_ch, feat_ch, 2)
        self.offset_head = self._build_head(in_ch, feat_ch, 2)
        
        # Heads for 2D-3D Properties
        self.center2kpt_offset_head = self._build_head(in_ch, feat_ch, (self.num_kpts * 2))
        self.kpt_heatmap_head = self._build_head(in_ch, feat_ch, self.num_kpts)
        self.kpt_heatmap_offset_head = self._build_head(in_ch, feat_ch, 2)
        
        # Heads for 3D Properties
        self.dim_head = self._build_head(in_ch, feat_ch, 3)
        self.depth_head = self._build_head(in_ch, feat_ch, (1 + 1))
        self.dir_feat, self.dir_cls, self.dir_reg = self._build_dir_head(in_ch, feat_ch)
        
        self.init_weights()


        """
        Criterions
        """

        # Losses for 2D Properties
        self.crit_center_heatmap = GaussianFocalLoss(loss_weight=1.0)
        self.crit_wh = L1Loss(loss_weight=0.1)
        self.crit_offset = L1Loss(loss_weight=1.0)
        
        # Losses for Keypoint Properties
        self.crit_center2kpt_offset = L1Loss(loss_weight=1.0)
        self.crit_kpt_heatmap = GaussianFocalLoss(loss_weight=1.0)
        self.crit_kpt_heatmap_offset = L1Loss(loss_weight=1.0)
        
        # Losses for 3D Properties
        self.crit_dim = DimAwareL1Loss(loss_weight=1.0)
        self.crit_depth = LaplacianAleatoricUncertaintyLoss(loss_weight=1.0)
        self.crit_alpha_cls = CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0)
        self.crit_alpha_reg = L1Loss(loss_weight=1.0)


    def init_weights(self, prior_prob: float = 0.1) -> None:
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        self.kpt_heatmap_head[-1].bias.data.fill_(bias_init)
        
        for head in [self.wh_head, self.offset_head, self.center2kpt_offset_head, self.kpt_heatmap_offset_head,
                     self.depth_head, self.dim_head, self.dir_feat, self.dir_cls, self.dir_reg]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    if hasattr(m, 'weight') and (m.weight is not None):
                        nn.init.normal_(m.weight, 0.0, 0.001)
                    if hasattr(m, 'bias') and (m.bias is not None):
                        nn.init.constant_(m.bias, 0.0)
     


    def _get_predictions(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:

        # (1) HeatMap
        heat_min, heat_max = 1e-4, (1. - 1e-4)
        center_heatmap_pred = torch.clamp(torch.sigmoid(self.heatmap_head(feat)), heat_min, heat_max)
        kpt_heatmap_pred = torch.clamp(torch.sigmoid(self.kpt_heatmap_head(feat)), heat_min, heat_max)

        # (2) Offset
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)
        kpt_heatmap_offset_pred = self.kpt_heatmap_offset_head(feat)
        center2kpt_offset_pred = self.center2kpt_offset_head(feat)
        
        # (3) Dimension
        dim_pred = self.dim_head(feat)
        
        # (4) Depth
        depth_pred = self.depth_head(feat)
        depth_pred[:, 0, :, :] = (1. / (torch.sigmoid(depth_pred[:, 0, :, :]) + EPS)) - 1

        # (5) Direction
        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat)
        alpha_offset_pred = self.dir_reg(alpha_feat)
        
        return {
            'center_heatmap_pred': center_heatmap_pred,
            'kpt_heatmap_pred': kpt_heatmap_pred,
            'wh_pred': wh_pred,
            'offset_pred': offset_pred,
            'kpt_heatmap_offset_pred': kpt_heatmap_offset_pred,
            'center2kpt_offset_pred': center2kpt_offset_pred,
            'dim_pred': dim_pred,
            'depth_pred': depth_pred,
            'alpha_cls_pred': alpha_cls_pred,
            'alpha_offset_pred': alpha_offset_pred}
    
        
    def _get_losses(self, 
                    pred_dict: Dict[str, torch.Tensor], 
                    target_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Indices and Mask
        indices = target_dict['indices']
        indices_kpt = target_dict['indices_kpt']
        
        device = indices.device
        batch_size = indices.shape[0]

        mask_target = target_dict['mask_target'].to(device)
        mask_center2kpt_offset = target_dict['mask_center2kpt_offset']
        mask_kpt_heatmap_offset = target_dict['mask_kpt_heatmap_offset']
        
        
        # Offset Loss
        offset_pred = extract_input(pred_dict['offset_pred'], indices, mask_target)
        offset_target = extract_target(target_dict['offset_target'], mask_target)
        offset_loss = self.crit_offset(offset_pred, offset_target)
        
        
        # WH Loss
        wh_pred = extract_input(pred_dict['wh_pred'], indices, mask_target)
        wh_target = extract_target(target_dict['wh_target'], mask_target)
        wh_loss = self.crit_wh(wh_pred, wh_target)
        
        
        # Dim Loss
        dim_pred = extract_input(pred_dict['dim_pred'], indices, mask_target)
        dim_target = extract_target(target_dict['dim_target'], mask_target)
        dim_loss = self.crit_dim(dim_pred, dim_target, dim_pred)
        
        
        # Depth Loss
        depth_pred = extract_input(pred_dict['depth_pred'], indices, mask_target)
        depth_target = extract_target(target_dict['depth_target'], mask_target)
        
        depth_pred, depth_log_var = depth_pred[:, 0:1], depth_pred[:, 1:2]
        depth_loss = self.crit_depth(depth_pred, depth_target, depth_log_var)
        
        
        #
        # Heatmap Losses
        #
        
        ct_heat_loss = self.crit_center_heatmap(pred_dict['center_heatmap_pred'], target_dict['center_heatmap_target'])
        kpt_heat_loss = self.crit_kpt_heatmap(pred_dict['kpt_heatmap_pred'], target_dict['kpt_heatmap_target'])
        
        
        #
        # Keypoint Losses
        #
        
        center2kpt_offset_pred = extract_input(pred_dict['center2kpt_offset_pred'], indices, mask_target)
        center2kpt_offset_target = extract_target(target_dict['center2kpt_offset_target'], mask_target)
        mask_center2kpt_offset = extract_target(target_dict['mask_center2kpt_offset'], mask_target)
        
        center2kpt_offset_pred = (center2kpt_offset_pred * mask_center2kpt_offset)
        center2kpt_offset_loss = self.crit_center2kpt_offset(center2kpt_offset_pred,
                                                             center2kpt_offset_target,
                                                             avg_factor=(mask_center2kpt_offset.sum() + EPS))
        
        kpt_heatmap_offset_pred = transpose_and_gather_feat(pred_dict['kpt_heatmap_offset_pred'], indices_kpt)
        kpt_heatmap_offset_pred = kpt_heatmap_offset_pred.reshape(batch_size, self.max_objs, (self.num_kpts * 2))
        kpt_heatmap_offset_pred = extract_target(kpt_heatmap_offset_pred, mask_target)
        
        kpt_heatmap_offset_target = extract_target(target_dict['kpt_heatmap_offset_target'], mask_target)
        mask_kpt_heatmap_offset = extract_target(mask_kpt_heatmap_offset, mask_target)
        
        kpt_heatmap_offset_loss = self.crit_kpt_heatmap_offset(kpt_heatmap_offset_pred,
                                                               kpt_heatmap_offset_target,
                                                               avg_factor=(mask_kpt_heatmap_offset.sum() + EPS))
        
        #
        # Alpha Losses
        #
        
        # Bin Classification
        alpha_cls_pred = extract_input(pred_dict['alpha_cls_pred'], indices, mask_target)
        alpha_cls_target = extract_target(target_dict['alpha_cls_target'], mask_target).type(torch.LongTensor)
        alpha_cls_onehot_target = alpha_cls_target\
            .new_zeros([len(alpha_cls_target), self.num_alpha_bins])\
            .scatter_(1, alpha_cls_target.view(-1, 1), 1).to(device)
        
        if mask_target.sum() > 0:
            loss_alpha_cls = self.crit_alpha_cls(alpha_cls_pred, alpha_cls_onehot_target)
        else:
            loss_alpha_cls = 0.0
        
        # Bin Offset Regression
        alpha_offset_pred = extract_input(pred_dict['alpha_offset_pred'], indices, mask_target)
        alpha_offset_pred = torch.sum(alpha_offset_pred * alpha_cls_onehot_target, 1, keepdim=True)
        alpha_offset_target = extract_target(target_dict['alpha_offset_target'], mask_target)
        
        loss_alpha_reg = self.crit_alpha_reg(alpha_offset_pred, alpha_offset_target)
        
        return {
            'loss_center_heatmap': ct_heat_loss,
            'loss_wh': wh_loss,
            'loss_offset': offset_loss,
            'loss_dim': dim_loss,
            'loss_center2kpt_offset': center2kpt_offset_loss,
            'loss_kpt_heatmap': kpt_heat_loss,
            'loss_kpt_heatmap_offset': kpt_heatmap_offset_loss,
            'loss_alpha_cls': loss_alpha_cls,
            'loss_alpha_reg': loss_alpha_reg,
            'loss_depth': depth_loss}
        
        
    def _get_bboxes(self, 
                    data_dict: Dict[str, Any],
                    pred_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        
        bboxes_2d, bboxes_3d, labels = self.decode_heatmap(data_dict, pred_dict)
        
        # Convert origin of 'bboxes_3d' from (0.5, 0.5, 0.5) to (0.5, 1.0, 0.5)
        for box_idx in range(len(bboxes_3d)):
            
            bbox_3d = bboxes_3d[box_idx]            # (K, 7)
            
            dst = bbox_3d.new_tensor((0.5, 1.0, 0.5))
            src = bbox_3d.new_tensor((0.5, 0.5, 0.5))
            
            bbox_3d[:, :3] += (bbox_3d[:, 3:6] * (dst - src))
            bboxes_3d[box_idx] = bbox_3d
        return bboxes_2d, bboxes_3d, labels
    
    
    # Data used for kitti evaluation
    def _get_eval_formats(self,
                          data_dict: Dict[str, Any],
                          pred_dict: Dict[str, torch.Tensor],
                          get_vis_format: bool = False) -> Dict[str, Any]:
        
        bboxes_2d, bboxes_3d, labels = self._get_bboxes(data_dict, pred_dict)
        
        #
        # (1) Convert the detection results to a list of numpy arrays.
        #     Results from this stage will be used for visualization.
        #
        
        results_2d, results_3d = [], []
        
        for bbox_2d, bbox_3d, label in zip(bboxes_2d, bboxes_3d, labels):
            results_2d.append(self.bbox_2d_to_result(bbox_2d, label, self.num_classes))
            
            score = bbox_2d[:, -1]
            results_3d.append(self.bbox_3d_to_result(bbox_3d, score, label))
        
        batch_size = data_dict['img'].shape[0]
        result_list = [dict() for _ in range(batch_size)]
        
        for result_dict, pred_2d, pred_3d in zip(result_list, results_2d, results_3d):
            result_dict['img_bbox'] = pred_3d
            result_dict['img_bbox2d'] = pred_2d
            
        if get_vis_format:
            return result_list
            
        #
        # (2) Convert to kitti format for evaluation and test submissions.
        #

        collected_2d = [result['img_bbox2d'] for result in result_list]
        kitti_2d = convert_to_kitti_2d(collected_2d, data_dict['img_metas'])
        
        collected_3d = [result['img_bbox'] for result in result_list]
        kitti_3d = convert_to_kitti_3d(collected_3d, data_dict['img_metas'], data_dict['calib'])

        kitti_format = {
            'img_bbox': kitti_3d,
            'img_bbox2d': kitti_2d}
        return kitti_format
        
    
    
    def decode_heatmap(self, 
                       data_dict: Dict[str, Any], 
                       pred_dict: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor]]:
        
        img_h, img_w = data_dict['img_metas']['pad_shape'][0]
        
        center_heatmap_pred = pred_dict['center_heatmap_pred']
        batch, _, feat_h, feat_w = center_heatmap_pred.shape
        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, 
            kernel=self.local_maximum_kernel)
        
        # (B, K)
        scores, indices, topk_labels, ys, xs = \
            get_topk_from_heatmap(center_heatmap_pred, k=self.topk)
        
        # Get 2D Predictions from Predicted Heatmap
        wh = transpose_and_gather_feat(pred_dict['wh_pred'], indices)               # (B, K, 2)
        offset = transpose_and_gather_feat(pred_dict['offset_pred'], indices)       # (B, K, 2)
        
        topk_xs = xs + offset[..., 0]
        topk_ys = ys + offset[..., 1]
        
        x1 = (topk_xs - wh[..., 0] / 2.) * (img_w / feat_w)
        y1 = (topk_ys - wh[..., 1] / 2.) * (img_h / feat_h)
        x2 = (topk_xs + wh[..., 0] / 2.) * (img_w / feat_w)
        y2 = (topk_ys + wh[..., 1] / 2.) * (img_h / feat_h)
        
        bboxes_2d = torch.stack([x1, y1, x2, y2], dim=2)
        bboxes_2d = torch.cat([bboxes_2d, scores[..., None]], dim=-1)               # (B, K, 5)
        
        
        # Get 3D Predictions from Predicted Heatmap
        # 'sigma' represents uncertainty.
        
        # Convert bin class and offset to alpha.
        alpha_cls = transpose_and_gather_feat(pred_dict['alpha_cls_pred'], indices)         # (B, K, 12)
        alpha_offset = transpose_and_gather_feat(pred_dict['alpha_offset_pred'], indices)   # (B, K, 12)
        alpha = self.decode_alpha(alpha_cls, alpha_offset)                                  # (B, K, 1)
        
        depth_pred = transpose_and_gather_feat(pred_dict['depth_pred'], indices)            # (B, K, 2)
        sigma = torch.exp(-depth_pred[:, :, 1])                                             # (B, K)
        bboxes_2d[..., -1] = (bboxes_2d[..., -1] * sigma)
        
        center2kpt_offset = transpose_and_gather_feat(
            pred_dict['center2kpt_offset_pred'],
            indices)
        center2kpt_offset = center2kpt_offset.view(batch, self.topk, (self.num_kpts * 2))[..., -2:]     # (B, K, 2)
        
        x_offset = xs.view(batch, self.topk, 1)
        x_scale = (img_w / feat_w)
        
        y_offset = ys.view(batch, self.topk, 1)
        y_scale = (img_h / feat_h)
        
        center2kpt_offset[..., ::2] = (center2kpt_offset[..., ::2] + x_offset) * x_scale
        center2kpt_offset[..., 1::2] = (center2kpt_offset[..., 1::2] + y_offset) * y_scale
        
        center2d = center2kpt_offset
        rot_y = self.calculate_roty(center2d, alpha, batched_calib=data_dict['calib'])      # (B, K, 1)
        
        depth = depth_pred[:, :, 0:1]                                                           # (B, K, 1)
        center3d = torch.cat([center2d, depth], dim=-1)                                         # (B, K, 3)
        center3d = self.convert_pts2D_to_pts3D(center3d, batched_calib=data_dict['calib'])      # (B, K, 3)
        
        dim = transpose_and_gather_feat(pred_dict['dim_pred'], indices)
        bboxes_3d = torch.cat([center3d, dim, rot_y], dim=-1)
        
        box_mask = (bboxes_2d[..., -1] > self.test_thres)                                   # (B, K)
        
        # Decoded Results
        ret_bboxes_2d = [
            bbox_2d[mask] 
            for bbox_2d, mask in zip(bboxes_2d, box_mask)]
        
        ret_bboxes_3d = [
            bbox_3d[mask] 
            for bbox_3d, mask in zip(bboxes_3d, box_mask)]
        
        ret_labels = [
            label[mask] 
            for label, mask in zip(topk_labels, box_mask)]
        
        return ret_bboxes_2d, ret_bboxes_3d, ret_labels

