import os
import sys
import torch
import numpy as np

from typing import Tuple, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.tensor_ops import gaussian_radius, generate_gaussian_target, get_ellip_gaussian_2D


# Constants
PI = np.pi


# Target Generator
class GUPNetTargetGenerator:
    def __init__(self, 
                 num_classes: int = 3,
                 max_objs: int = 30, 
                 num_kpt: int = 9,
                 num_alpha_bins: int = 12):
        
        self.num_classes = num_classes
        self.max_objs = max_objs
        self.num_kpt = num_kpt
        self.num_alpha_bins = num_alpha_bins
        

    def __call__(self, 
                 input_dict: Dict[str, Any],
                 feat_shape: Tuple[int]) -> Dict[str, torch.Tensor]:
    
        device = input_dict['img'].device
        
        metas = input_dict['img_metas']
        label = input_dict['label']
        
        ori_h, ori_w = metas['pad_shape'][0]
        batch_size, _, feat_h, feat_w = feat_shape
        h_ratio, w_ratio = (feat_h / ori_h), (feat_w / ori_w)

        target = self._create_empty_target(feat_shape=feat_shape, device=device)

        for b_idx in range(batch_size):
            
            # Mask
            mask = label['mask'][b_idx].type(torch.BoolTensor)

            
            # Valid 2D Bboxes
            bboxes = label['gt_bboxes'][b_idx][mask]
            bbox_labels = label['gt_labels'][b_idx][mask].type(torch.LongTensor)
            if len(bboxes) < 1:
                continue
            
            bbox_ctx = (bboxes[:, 0] + bboxes[:, 2]) * w_ratio / 2.
            bbox_cty = (bboxes[:, 1] + bboxes[:, 3]) * h_ratio / 2.
            bbox_ct = torch.cat([bbox_ctx.unsqueeze(1), bbox_cty.unsqueeze(1)], dim=1)
            
            centers = label['centers2d'][b_idx][mask] * w_ratio

            calib_fs = label['calib_f'][b_idx][mask]

            # Valid 3D Bboxes and Depth
            bboxes_3d = label['gt_bboxes_3d'][b_idx][mask]
            depth = label['depths'][b_idx][mask]
            
            for o_idx, ct in enumerate(centers):
                
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                is_in_feature = (0 <= ctx_int <= feat_w) and (0 <= cty_int <= feat_h)
                if not is_in_feature: continue

                bbox_ctx, bbox_cty = bbox_ct[o_idx]
                
                feat_box_h = (bboxes[o_idx, 3] - bboxes[o_idx, 1]) * h_ratio
                feat_box_w = (bboxes[o_idx, 2] - bboxes[o_idx, 0]) * w_ratio
                
                dim = bboxes_3d[o_idx][3:6]
                alpha = bboxes_3d[o_idx][6]
                                
                target_radius = gaussian_radius((feat_box_h, feat_box_w))
                target_radius = max(0, int(target_radius))
                c_idx = bbox_labels[o_idx]
                target['cls_ids'][b_idx, o_idx] = c_idx
                generate_gaussian_target(target['center_heatmap_target'][b_idx, c_idx],
                                         center=[ctx_int, cty_int],
                                         radius=target_radius)
                
                target['indices'][b_idx, o_idx] = (cty_int * feat_w + ctx_int)
                
                target['wh_target'][b_idx, o_idx] = torch.Tensor([feat_box_w, feat_box_h])
                target['offset_3d'][b_idx, o_idx] = torch.Tensor([(ctx - ctx_int), (cty - cty_int)])
                target['offset_2d'][b_idx, o_idx] = torch.Tensor([(bbox_ctx - ctx_int), (bbox_cty - cty_int)])
                
                target['dim_target'][b_idx, o_idx] = dim
                target['depth_target'][b_idx, o_idx] = depth[o_idx]

                target['calib_f'][b_idx, o_idx] = calib_fs[o_idx]
                
                alpha_cls, alpha_offset = self._convert_angle_to_class(alpha)
                target['alpha_cls_target'][b_idx, o_idx] = alpha_cls
                target['alpha_offset_target'][b_idx, o_idx] = alpha_offset
                
                target['mask_target'][b_idx, o_idx] = 1
                                   
        target['mask_target'] = (target['mask_target']).type(torch.BoolTensor)
        return target
    
    
    def _convert_angle_to_class(self, angle: float):
        angle = angle % (2 * PI)
        assert (angle >= 0 and angle <= 2 * PI)
        
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * PI)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle
    
        
    def _create_empty_target(self, feat_shape: Tuple[int], device: str = None) -> Dict[str, torch.Tensor]:
        batch_size, _, feat_h, feat_w = feat_shape
        container = {
            'center_heatmap_target': torch.zeros((batch_size, self.num_classes, feat_h, feat_w)),
            'wh_target': torch.zeros((batch_size, self.max_objs, 2)),
            'offset_2d': torch.zeros((batch_size, self.max_objs, 2)),
            'offset_3d': torch.zeros((batch_size, self.max_objs, 2)),
            'dim_target': torch.zeros((batch_size, self.max_objs, 3)),
            'alpha_cls_target': torch.zeros((batch_size, self.max_objs, 1)),
            'alpha_offset_target': torch.zeros((batch_size, self.max_objs, 1)),
            'depth_target': torch.zeros((batch_size, self.max_objs, 1)),

            'cls_ids': torch.zeros((batch_size, self.max_objs)),
            'indices': torch.zeros((batch_size, self.max_objs)).type(torch.LongTensor),
            
            'calib_f': torch.zeros((batch_size, self.max_objs)),
            'mask_target': torch.zeros((batch_size, self.max_objs))}
        
        if device is None:
            device = 'cpu'
        for k in container.keys():
            container[k] = container[k].to(device)
        return container
