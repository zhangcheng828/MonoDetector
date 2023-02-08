import os
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, List, Dict, Any

warnings.filterwarnings('ignore')
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from losses import *
from .base_dense_heads import BaseDenseHeads
from utils.tensor_ops import (extract_input, 
                              extract_target, 
                              transpose_and_gather_feat,
                              get_local_maximum, 
                              get_topk_from_heatmap)
from utils.monoflex_target_generator import MonoFlexTargetGenerator
from utils.kitti_convert_utils import convert_to_kitti_3d

# Constants
EPS = 1e-12

class MonoFlexDenseHeads(BaseDenseHeads):
    def __init__(self,
                 in_ch: int = 64,
                 feat_ch: int = 64,
                 num_kpts: int = 10,
                 num_alpha_bins: int = 12,
                 num_classes: int = 3,
                 max_objs: int = 30,
                 test_config: Dict[str, Any] = None):
        
        super().__init__(in_ch=64, feat_ch=64, num_kpts=num_kpts, num_alpha_bins=12,
                        num_classes=3,max_objs=30, test_config=test_config)
        
        
        # Target Generator
        self.target_generator = MonoFlexTargetGenerator(
            num_classes=num_classes,
            max_objs=max_objs,
            num_kpt=num_kpts,
            num_alpha_bins=num_alpha_bins)

        self.uncertainty_range = [-10, 10]
        """
        Prediction Heads
        """
        
        self.heatmap_head = self._build_head(in_ch, feat_ch, num_classes) #类别热力图
        self.wh_head = self._build_head(in_ch, feat_ch, 2) #2D包围盒的尺寸
        self.offset_head = self._build_head(in_ch, feat_ch, 2) #投影2D点和像素点的偏移（量化误差）
        
        self.center2kpt_offset_head = self._build_head(in_ch, feat_ch, (self.num_kpts * 2)) #投影关键点（角点，中心点）到中心像素点的偏移（量化误差）
        self.depth_uncertainty = self._build_head(in_ch, feat_ch, 4) #关键点解析出的深度不确定性，共有四组
        
        # Heads for 3D Properties
        self.dim_head = self._build_head(in_ch, feat_ch, 3)
        self.direct_depth_head = self._build_head(in_ch, feat_ch, 1) # direct depth
        self.dir_feat, self.dir_cls, self.dir_reg = self._build_dir_head(in_ch, feat_ch)



        """
        Criterions
        """

        # Losses for 2D Properties
        self.crit_center_heatmap = GaussianFocalLoss(loss_weight=1.0)
        self.crit_wh = L1Loss(loss_weight=0.1)
        self.crit_offset = L1Loss(loss_weight=1.0)
        
        # Losses for Keypoint Properties
        self.crit_center2kpt_offset = L1Loss(loss_weight=1.0)
        self.crit_indirect_depth = L1Loss(loss_weight=1.0)

        # Losses for 3D Properties
        self.crit_dim = DimAwareL1Loss(loss_weight=1.0)
        self.crit_direct_depth = L1Loss(loss_weight=1.0)
        
        self.crit_combine_depth = L1Loss(loss_weight=1.0)
        self.crit_alpha_cls = CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0)
        self.crit_alpha_reg = L1Loss(loss_weight=1.0)



    def init_weights(self, prior_prob: float = 0.1) -> None:
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        self.heatmap_head[-1].bias.data.fill_(bias_init)
        
        for head in [self.offset_head, self.wh_head, self.center2kpt_offset_head, self.indirect_depth_uncertainty,
                     self.direct_depth_head, self.dim_head, self.dir_feat, self.dir_cls, self.dir_reg]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    if hasattr(m, 'weight') and (m.weight is not None):
                        nn.init.normal_(m.weight, 0.0, 0.001)
                    if hasattr(m, 'bias') and (m.bias is not None):
                        nn.init.constant_(m.bias, 0.0)
     


    def _get_predictions(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        #[8, 64, 96, 312]
        # (1) HeatMap
        heat_min, heat_max = 1e-4, (1. - 1e-4)
        center_heatmap_pred = torch.clamp(torch.sigmoid(self.heatmap_head(feat)), heat_min, heat_max) #[8, 3, 96, 312]

        # (2) Offset
        wh_pred = self.wh_head(feat) #[8, 2, 96, 312]
        offset_pred = self.offset_head(feat) #[8, 2, 96, 312]
        center2kpt_offset_pred = self.center2kpt_offset_head(feat) #[8, 20, 96, 312]
        
        # (3) Dimension
        dim_pred = self.dim_head(feat) #[8, 3, 96, 312]

        # (5) Direction
        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat) #[8, 12, 96, 312]
        alpha_offset_pred = self.dir_reg(alpha_feat) #[8, 12, 96, 312]

        
        # (4) Depth

        direct_depth_pred = self.direct_depth_head(feat) #[8, 1, 96, 312]

        direct_depth_pred[:, 0, :, :] = (1. / (torch.sigmoid(direct_depth_pred[:, 0, :, :]) + EPS)) - 1

        depth_uncertainty = self.depth_uncertainty(feat) #[8, 4, 96, 312]

        return {
            'center_heatmap_pred': center_heatmap_pred,
            'wh_pred': wh_pred,
            'offset_pred': offset_pred,
            'center2kpt_offset_pred':center2kpt_offset_pred,
            'dim_pred': dim_pred,
            'direct_depth_pred': direct_depth_pred,
            'depth_uncertainty': depth_uncertainty,
            'alpha_cls_pred': alpha_cls_pred,
            'alpha_offset_pred': alpha_offset_pred}


    def decode_heatmap(self, 
                       data_dict: Dict[str, Any], 
                       pred_dict: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor]]:
        
        img_h, img_w = data_dict['img_metas']['pad_shape'][0]
        
        center_heatmap_pred = pred_dict['center_heatmap_pred']
        batch, _, feat_h, feat_w = center_heatmap_pred.shape
        down_ratio = img_h / feat_h
        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, 
            kernel=self.local_maximum_kernel)
        
        # (B, K)
        scores, indices, topk_labels, ys, xs = \
            get_topk_from_heatmap(center_heatmap_pred, k=self.topk)
        
        # offset and dimension
        offset = transpose_and_gather_feat(pred_dict['offset_pred'], indices)       # (B, K, 2)
        dim_pred = transpose_and_gather_feat(pred_dict['dim_pred'], indices)
        
        # Convert bin class and offset to alpha.
        alpha_cls = transpose_and_gather_feat(pred_dict['alpha_cls_pred'], indices)         # (B, K, 12)
        alpha_offset = transpose_and_gather_feat(pred_dict['alpha_offset_pred'], indices)   # (B, K, 12)
        alpha = self.decode_alpha(alpha_cls, alpha_offset)                                  # (B, K, 1)
        
        # indirect depth
        cam2imgs = torch.cat([torch.Tensor(calib.P2).unsqueeze(0) for calib in data_dict['calib']], dim=0).cuda()
        cam2imgs = cam2imgs.unsqueeze(1).repeat(1, self.topk, 1, 1).view(batch* self.topk, 3, 4)

        center2kpt_offset_pred = transpose_and_gather_feat(pred_dict['center2kpt_offset_pred'], indices) 

        indirect_depth_pred = self.target_generator.keypoints2depth(center2kpt_offset_pred.view(-1, self.num_kpts,2),
                                                    dim_pred.view(-1, 3), cam2imgs) # N x 3

        # direct depth
        direct_depth_pred = transpose_and_gather_feat(pred_dict['direct_depth_pred'], indices).view(batch*self.topk, 1) 

        depth_uncertainty = transpose_and_gather_feat(pred_dict['depth_uncertainty'], indices).view(batch*self.topk, 4) 

        depth_uncertainty = torch.clamp(
            depth_uncertainty, self.uncertainty_range[0],
            self.uncertainty_range[1])

        # combined depth
        pred_depth_uncertainty = depth_uncertainty.exp()
        pred_depth = torch.cat((direct_depth_pred, indirect_depth_pred), dim=1)
        pred_combined_depth = \
            self.target_generator.combine_depths(pred_depth, pred_depth_uncertainty)
        
        # calculate the center point of 3d box
        points = torch.cat([xs.view(-1, 1),
                            ys.view(-1, 1).float()],
                           dim=1)

        points = (points + offset.view(batch * self.topk, -1)) * down_ratio 
        center2d = points.reshape(batch, self.topk, -1)
        depth = pred_combined_depth.view(batch, self.topk, 1)    # (B, K, 1)
        center3d = torch.cat([center2d, depth], dim=-1)                                         # (B, K, 3)
        center3d = self.convert_pts2D_to_pts3D(center3d, batched_calib=data_dict['calib'])      # (B, K, 3)

        
        # calculate rot_y
        rot_y = self.calculate_roty(center2d, alpha, batched_calib=data_dict['calib'])      # (B, K, 1)
        

        dim = dim_pred.view(batch, self.topk, 3)

        sigma, _ = torch.exp(- depth_uncertainty).max(1)                # (B, K)
        scores = scores[..., None]

        scores[..., -1] = (scores[..., -1] * sigma.view(batch, self.topk))

        bboxes_3d = torch.cat([center3d, dim, rot_y, scores], dim=-1)
        
        box_mask = (scores[..., -1] > self.test_thres)                                   # (B, K)
        
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
        
        #
        # Heatmap Losses
        #    
        ct_heat_loss = self.crit_center_heatmap(pred_dict['center_heatmap_pred'], target_dict['center_heatmap_target'])    
      
        # 2D Loss
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

        #
        # Keypoint Losses
        #
        
        center2kpt_offset_pred = extract_input(pred_dict['center2kpt_offset_pred'], indices, mask_target)
        center2kpt_offset_target = extract_target(target_dict['center2kpt_offset_target'], mask_target)
        mask_center2kpt_offset = extract_target(target_dict['mask_center2kpt_offset'], mask_target)


        center2kpt_offset_loss = self.crit_center2kpt_offset(center2kpt_offset_pred * mask_center2kpt_offset,
                                                             center2kpt_offset_target * mask_center2kpt_offset,
                                                             avg_factor=(mask_center2kpt_offset.sum() + EPS))

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
        
        cam2imgs = extract_target(target_dict['cam2img'], mask_target)
 

        pred_indirect_depth = self.target_generator.keypoints2depth(center2kpt_offset_pred.view(-1, self.num_kpts,2),
                                                    dim_pred, cam2imgs) # N x 3

        direct_depth_pred = extract_input(pred_dict['direct_depth_pred'], indices, mask_target)

        depth_uncertainty = extract_input(pred_dict['depth_uncertainty'], indices, mask_target)
        depth_uncertainty = torch.clamp(
            depth_uncertainty, self.uncertainty_range[0],
            self.uncertainty_range[1])

        pred_depth_uncertainty = depth_uncertainty.exp()

        pred_depth = torch.cat((direct_depth_pred, pred_indirect_depth), dim=1)
        pred_combined_depth = \
            self.target_generator.combine_depths(pred_depth, pred_depth_uncertainty)

        depth_target = extract_target(target_dict['depth_target'], mask_target)

        direct_depth_weights = torch.exp(-depth_uncertainty[:, 0])
        loss_direct_depth = self.crit_direct_depth(
            direct_depth_pred, depth_target,
            direct_depth_weights)

        loss_uncertainty_1 =\
            depth_uncertainty[:, 0] * 0.1
        loss_direct_depth = loss_direct_depth + loss_uncertainty_1.mean()

        mask_group_depth = extract_target(target_dict['mask_group_depth'], mask_target)
        valid_indirect_uncertainty = depth_uncertainty[:, 1:][mask_group_depth]
        valid_indirect_depth_weights = torch.exp(
            -valid_indirect_uncertainty)

        indirect_depth_target = depth_target.repeat(1, 3)
        loss_indirect_depth = self.crit_indirect_depth(
            pred_indirect_depth[mask_group_depth], indirect_depth_target[mask_group_depth],
            valid_indirect_depth_weights)

        loss_uncertainty_2 =\
            valid_indirect_uncertainty * 0.1

        loss_indirect_depth = loss_indirect_depth + loss_uncertainty_2.mean()
        

        loss_combined_depth = self.crit_combine_depth(
            pred_combined_depth.squeeze(), depth_target.squeeze())

        return {
            'loss_center_heatmap': ct_heat_loss,
            'loss_offset': offset_loss,
            'loss_wh': wh_loss,
            'loss_dim': dim_loss,
            'center2kpt_offset_loss': center2kpt_offset_loss,
            'loss_alpha_cls': loss_alpha_cls,
            'loss_alpha_reg': loss_alpha_reg,
            'loss_combined_depth': loss_combined_depth,
            'loss_indirect_depth': loss_indirect_depth,
            'loss_direct_depth': loss_direct_depth,
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
