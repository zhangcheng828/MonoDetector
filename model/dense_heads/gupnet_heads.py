import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, List, Dict, Any

warnings.filterwarnings('ignore')
from .base_dense_heads import BaseDenseHeads
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses import *
from model import AttnBatchNorm2d

from utils.tensor_ops import (extract_input, 
                              extract_target, 
                              transpose_and_gather_feat,
                              get_local_maximum, 
                              get_topk_from_heatmap)
from utils.gupnet_target_generator import GUPNetTargetGenerator
from utils.kitti_convert_utils import convert_to_kitti_3d
import torchvision.ops.roi_align as roi_align

# Constants
EPS = 1e-12

class GUPNetDenseHeads(BaseDenseHeads):
    def __init__(self,
                 in_ch: int = 64,
                 feat_ch: int = 64,
                 num_alpha_bins: int = 12,
                 num_classes: int = 3,
                 max_objs: int = 30,
                 test_config: Dict[str, Any] = None):
        
        super().__init__(in_ch=64, feat_ch=64, num_alpha_bins=12, num_classes=3,max_objs=30, test_config=test_config)
        
        self.is_training = False

        # Target Generator
        self.target_generator = GUPNetTargetGenerator(
            num_classes=num_classes,
            max_objs=max_objs,
            num_alpha_bins=num_alpha_bins)

        """
        Prediction Heads
        """
        
        # Heads for 2D Properties
        self.heatmap_head = self._build_head(in_ch, feat_ch, num_classes) #类别热力图
        self.wh_head = self._build_head(in_ch, feat_ch, 2) #2D包围盒的尺寸
        self.offset2d_head = self._build_head(in_ch, feat_ch, 2) #投影2D点和像素点的偏移（量化误差）   
        
        # Heads for 3D Properties
        self.offset3d_head = self._build_roi_head(in_ch + self.num_classes, feat_ch, 2) #投影2D点和像素点的偏移（量化误差）
        self.dim_head = self._build_roi_head(in_ch + self.num_classes, feat_ch, 4)
        self.depth_head = self._build_roi_head(in_ch + self.num_classes, feat_ch, (1 + 1))
        self.dir_feat, self.dir_cls, self.dir_reg = self._build_dir_head(in_ch + self.num_classes, feat_ch)
        
        self.init_weights()


        """
        Criterions
        """

        # Losses for 2D Properties
        self.crit_center_heatmap = GaussianFocalLoss(loss_weight=1.0)
        self.crit_wh = L1Loss(loss_weight=0.6)
        self.crit_offset2d = L1Loss(loss_weight=1.0)

        self.weight = 1.0
        # Losses for 3D ROI Properties
        self.crit_offset3d = L1Loss(loss_weight=self.weight)
        self.crit_dim = L1Loss(loss_weight=self.weight)
        self.crit_depth = LaplacianAleatoricUncertaintyLoss(loss_weight=self.weight)
        self.crit_h3d_logvar = LaplacianAleatoricUncertaintyLoss(loss_weight=self.weight) # 尺寸头的最后一维

        self.crit_alpha_cls = CrossEntropyLoss(use_sigmoid=True, loss_weight=self.weight)
        self.crit_alpha_reg = L1Loss(loss_weight=self.weight)


    def _build_dir_head(self, in_ch: int, feat_ch: int) -> Tuple[nn.Module]:
        dir_feat = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1),
            AttnBatchNorm2d(feat_ch, 10, momentum=0.03, eps=0.001),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        dir_cls = nn.Sequential(nn.Conv2d(feat_ch, self.num_alpha_bins, kernel_size=1))
        dir_reg = nn.Sequential(nn.Conv2d(feat_ch, self.num_alpha_bins, kernel_size=1))
        
        return dir_feat, dir_cls, dir_reg


    def init_weights(self, prior_prob: float = 0.1) -> None:
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        
        for head in [self.offset2d_head, self.depth_head, self.dim_head, self.offset3d_head,
                    self.wh_head, self.dir_feat, self.dir_cls, self.dir_reg]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    if hasattr(m, 'weight') and (m.weight is not None):
                        nn.init.normal_(m.weight, 0.0, 0.001)
                    if hasattr(m, 'bias') and (m.bias is not None):
                        nn.init.constant_(m.bias, 0.0)

    def get_two_stage_pred(self, pred_dict, indices, 
                            mask_target, labels, calibs) -> None:
        device = indices.device
        batch_size = indices.shape[0]
        num_masked_bin = mask_target.sum()
        box2d_maps = pred_dict['box2d_maps']
        feat = pred_dict['feat']
        box2d_masked = extract_input(box2d_maps, indices, mask_target)
        roi_feature_masked = roi_align(feat, box2d_masked,[7,7]) #[k, 64, 7, 7]

        # cls_ids = extract_target(labels, mask_target)
        cls_hots = torch.zeros(num_masked_bin, self.num_classes).to(device)
        cls_hots[torch.arange(num_masked_bin).to(device), labels.long()] = 1.0
        
        roi_feature_masked = torch.cat([roi_feature_masked, 
                                        cls_hots.unsqueeze(-1).unsqueeze(-1).repeat([1,1,7,7])],1)
        
        offset_3d_pred = self.offset3d_head(roi_feature_masked).squeeze()
        alpha_feat = self.dir_feat(roi_feature_masked)
        alpha_cls_pred = self.dir_cls(alpha_feat).squeeze() #[k, 12]
        alpha_offset_pred = self.dir_reg(alpha_feat).squeeze() #[k, 12]


        box2d_masked = box2d_masked[:, 1:].clone()
        box2d_masked *= 4.
        h_2d = torch.clamp(box2d_masked[:,3] - box2d_masked[:,1], min=1.0)

        # (3) Dimension
        dim_pred = self.dim_head(roi_feature_masked).squeeze() #[k, 4]
        size_3d, h_3d_log_std = dim_pred[:, :3], dim_pred[:,3]

        lw_pred = size_3d[:, [0, 2]]

        h_3d_pred = size_3d[:, 1]

        depth_geo = h_3d_pred / h_2d.squeeze() * calibs

        depth_net_out = self.depth_head(roi_feature_masked).squeeze()
        depth_geo_log_std = (h_3d_log_std.squeeze() + 2 * (calibs.log() - h_2d.log())).unsqueeze(-1)
        depth_net_log_std = torch.logsumexp(torch.cat([depth_net_out[:,1:2],\
                            depth_geo_log_std], -1), -1, keepdim=True)

        depth_pred = (1. / (depth_net_out[:,0:1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1)

        pred_dict['offset_3d_pred'] = offset_3d_pred
        pred_dict['alpha_cls_pred'] = alpha_cls_pred
        pred_dict['alpha_offset_pred'] = alpha_offset_pred
        pred_dict['lw_pred'] = lw_pred
        pred_dict['h_3d_pred'] = h_3d_pred
        pred_dict['h_3d_log_std'] = h_3d_log_std
        pred_dict['depth_pred'] = depth_pred
        pred_dict['depth_net_log_std'] = depth_net_log_std
        

    def _get_predictions(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        device_id = feat.device
        batch_size, _, h, w = feat.size()
        #[8, 64, 96, 312]
        # (1) HeatMap
        heat_min, heat_max = 1e-4, (1. - 1e-4)
        center_heatmap_pred = torch.clamp(torch.sigmoid(self.heatmap_head(feat)), heat_min, heat_max) #[8, 3, 96, 312]

        # 2D
        offset2d_pred = self.offset2d_head(feat) #[8, 2, 96, 312]
        wh_pred = self.wh_head(feat) #[8, 2, 96, 312]

        coord_map = torch.cat([torch.arange(w).unsqueeze(0).repeat([h,1]).unsqueeze(0),\
                        torch.arange(h).unsqueeze(-1).repeat([1,w]).unsqueeze(0)],0).unsqueeze(0).\
                        repeat([batch_size,1,1,1]).type(torch.float).to(device_id)
        box2d_centre = coord_map + offset2d_pred
        box2d_maps = torch.cat([box2d_centre - wh_pred / 2, box2d_centre + wh_pred / 2], 1)
        box2d_maps = torch.cat([torch.arange(batch_size).view(-1, 1, 1, 1).repeat([1, 1, h, w]).type(torch.float).to(device_id),
                                box2d_maps],1)
 
        return {
            'center_heatmap_pred': center_heatmap_pred,
            'feat': feat,
            'offset2d_pred': offset2d_pred,
            'wh_pred': wh_pred,
            'box2d_maps': box2d_maps}
    
    def decode_heatmap(self, 
                       data_dict: Dict[str, Any], 
                       pred_dict: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor]]:
        
        img_h, img_w = data_dict['img_metas']['pad_shape'][0]

        center_heatmap_pred = pred_dict['center_heatmap_pred']
        device_id = center_heatmap_pred.device
        batch, _, feat_h, feat_w = center_heatmap_pred.shape
        down_ratio = img_h / feat_h
        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, 
            kernel=self.local_maximum_kernel)
        
        # (B, K)
        scores, indices, topk_labels, ys, xs = \
            get_topk_from_heatmap(center_heatmap_pred, k=self.topk)


        masks = torch.ones(indices.size()).type(torch.uint8).to(device_id)
        calib_f = torch.cat([torch.tensor(calib.fu).unsqueeze(0).\
                            repeat(1, self.topk) for calib in data_dict['calib']], dim=0).to(device_id).view(-1)

        self.get_two_stage_pred(pred_dict, indices, masks, topk_labels.view(-1), calib_f.view(-1))

        points = torch.cat([xs.view(-1, 1),
                    ys.view(-1, 1).float()],
                    dim=1)
        points = (points + pred_dict['offset_3d_pred']) * down_ratio
        center2d = points.reshape(batch, self.topk, -1)

        #direction: local alpha

        alpha = self.decode_alpha(pred_dict['alpha_cls_pred'].view(batch, self.topk, -1), 
                                    pred_dict['alpha_offset_pred'].view(batch, self.topk, -1))       # (B, K, 1)


        depth_score = (-(0.5 * pred_dict['depth_net_log_std'].view(batch, self.topk, -1)).exp()).exp()

        scores = scores[..., None] * depth_score
        
        rot_y = self.calculate_roty(center2d, alpha, batched_calib=data_dict['calib'])      # (B, K, 1)

        depth_pred = pred_dict['depth_pred'].view(batch, self.topk, 1)                                                      # (B, K, 1)
        center3d = torch.cat([center2d, depth_pred], dim=-1)                                         # (B, K, 3)
        center3d = self.convert_pts2D_to_pts3D(center3d, batched_calib=data_dict['calib'])      # (B, K, 3)
        size_3d = torch.zeros(batch, self.topk, 3).to(device_id)
        size_3d[..., [0, 2]] = pred_dict['lw_pred'].view(batch, self.topk, 2)
        size_3d[..., 1] = pred_dict['h_3d_pred'].view(batch, self.topk)

        bboxes_3d = torch.cat([center3d, size_3d, rot_y, scores], dim=-1)
        
        box_mask = (scores[..., -1] > self.test_thres)                                   # (B, K)
        
        topk_labels = topk_labels.view(batch, self.topk, 1)

        ret_bboxes_3d = [
            bbox_3d[mask] 
            for bbox_3d, mask in zip(bboxes_3d, box_mask)]
        
        ret_labels = [
            label[mask] 
            for label, mask in zip(topk_labels, box_mask)]
        
        return ret_bboxes_3d, ret_labels


    def _get_losses(self, 
                    pred_dict: Dict[str, torch.Tensor], 
                    target_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        # Indices and Mask
        indices = target_dict['indices']   
        device = indices.device
        batch_size = indices.shape[0]
        mask_target = target_dict['mask_target'].to(device)
        cls_ids = extract_target(target_dict['cls_ids'], mask_target)
        calibs = extract_target(target_dict['calib_f'], mask_target)
        self.get_two_stage_pred(pred_dict, indices, mask_target, cls_ids, calibs)
        #
        # Heatmap Losses
        #
        
        ct_heat_loss = self.crit_center_heatmap(pred_dict['center_heatmap_pred'], 
                                                target_dict['center_heatmap_target'])    
        
        # 2D loss

        offset2d_pred = extract_input(pred_dict['offset2d_pred'], indices, mask_target)
        offset2d_target = extract_target(target_dict['offset_2d'], mask_target)        
        offset_2d_loss = self.crit_offset2d(offset2d_pred, offset2d_target)

        wh_pred = extract_input(pred_dict['wh_pred'], indices, mask_target)
        wh_target = extract_target(target_dict['wh_target'], mask_target)
        wh_loss = self.crit_wh(wh_pred, wh_target)


        # offset3d
        offset_3d_target = extract_target(target_dict['offset_3d'], mask_target)
        offset_3d_loss = self.crit_offset3d(pred_dict['offset_3d_pred'], offset_3d_target)

        # Bin Classification
        alpha_cls_target = extract_target(target_dict['alpha_cls_target'], mask_target).type(torch.LongTensor)
        alpha_cls_onehot_target = alpha_cls_target\
            .new_zeros([len(alpha_cls_target), self.num_alpha_bins])\
            .scatter_(1, alpha_cls_target.view(-1, 1), 1).to(device)
        
        if mask_target.sum() > 0:
            loss_alpha_cls = self.crit_alpha_cls(pred_dict['alpha_cls_pred'], alpha_cls_onehot_target)
        else:
            loss_alpha_cls = 0.0
        
        # Bin Offset Regression
        alpha_offset_pred = torch.sum(pred_dict['alpha_offset_pred'] * alpha_cls_onehot_target, 1, keepdim=True)
        alpha_offset_target = extract_target(target_dict['alpha_offset_target'], mask_target)
        
        loss_alpha_reg = self.crit_alpha_reg(alpha_offset_pred, alpha_offset_target)
        

        # (3) Dimension
        dim_target = extract_target(target_dict['dim_target'], mask_target)

        depth_target = extract_target(target_dict['depth_target'], mask_target)

        depth_loss = self.crit_depth(pred_dict['depth_pred'], depth_target, pred_dict['depth_net_log_std'])

        lw_loss = self.crit_dim(pred_dict['lw_pred'], dim_target[:, [0, 2]])
        h3d_loss = self.crit_h3d_logvar(pred_dict['h_3d_pred'], dim_target[:, 1], pred_dict['h_3d_log_std']) 
        dim_loss = (lw_loss * 2 / 3) + (h3d_loss * 1 / 3)

        return {
            'loss_center_heatmap': ct_heat_loss,
            'offset_2d_loss': offset_2d_loss,
            'wh_loss':wh_loss,
            'offset_3d_loss': offset_3d_loss,
            'loss_heading': loss_alpha_reg + loss_alpha_cls,
            'loss_dim': dim_loss,
            'loss_depth': depth_loss,
            }
        

    def _get_bboxes(self, 
                    data_dict: Dict[str, Any],
                    pred_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        
        bboxes_3d, labels = self.decode_heatmap(data_dict, pred_dict)
        
        # Convert origin of 'bboxes_3d' from (0.5, 0.5, 0.5) to (0.5, 1.0, 0.5)
        for box_idx in range(len(bboxes_3d)):
            
            bbox_3d = bboxes_3d[box_idx]            # (K, 8)
            
            dst = bbox_3d.new_tensor((0.5, 1.0, 0.5))
            src = bbox_3d.new_tensor((0.5, 0.5, 0.5))
            
            bbox_3d[:, :3] += (bbox_3d[:, 3:6] * (dst - src))
            bboxes_3d[box_idx] = bbox_3d
        return bboxes_3d, labels
     
    # Data used for kitti evaluation
    def _get_eval_formats(self,
                          data_dict: Dict[str, Any],
                          pred_dict: Dict[str, torch.Tensor],
                          get_vis_format: bool = False) -> Dict[str, Any]:
        
        bboxes_3d, labels = self._get_bboxes(data_dict, pred_dict)
        
        #
        # (1) Convert the detection results to a list of numpy arrays.
        #     Results from this stage will be used for visualization.
        #
        
        results_3d = []
        
        for bbox_3d, label in zip(bboxes_3d, labels):
            
            score = bbox_3d[:, -1]
            results_3d.append(self.bbox_3d_to_result(bbox_3d[:, :-1], score, label))
        
        batch_size = data_dict['img'].shape[0]
        result_list = [dict() for _ in range(batch_size)]
        
        for result_dict, pred_3d in zip(result_list, results_3d):
            result_dict['img_bbox'] = pred_3d
            
        if get_vis_format:
            return result_list
            
        #
        # (2) Convert to kitti format for evaluation and test submissions.
        #

        
        collected_3d = [result['img_bbox'] for result in result_list]
        kitti_3d = convert_to_kitti_3d(collected_3d, data_dict['img_metas'], data_dict['calib'])

        kitti_format = {
            'img_bbox': kitti_3d}
        return kitti_format
             
