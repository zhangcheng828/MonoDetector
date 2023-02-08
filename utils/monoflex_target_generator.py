import os
import sys
import torch
from torch.nn import functional as F
import numpy as np

from typing import Tuple, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.tensor_ops import gaussian_radius, generate_gaussian_target, get_ellip_gaussian_2D
from utils.geometry_ops import approx_proj_center

# Constants
PI = np.pi


# Target Generator
class MonoFlexTargetGenerator:
    def __init__(self, 
                 num_classes: int = 3,
                 max_objs: int = 30, 
                 num_kpt: int = 10,
                 num_alpha_bins: int = 12):
        
        self.num_classes = num_classes
        self.max_objs = max_objs
        self.num_kpt = num_kpt
        self.num_alpha_bins = num_alpha_bins
        self.eps=1e-3

    def __call__(self, 
                 input_dict: Dict[str, Any],
                 feat_shape: Tuple[int]) -> Dict[str, torch.Tensor]:
    
        device = input_dict['img'].device
        
        metas = input_dict['img_metas']
        label = input_dict['label']
        calibs = input_dict['calib']

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

            bbox_ctx = (bboxes[:, 0] + bboxes[:, 2]) / 2.
            bbox_cty = (bboxes[:, 1] + bboxes[:, 3]) / 2.
            bbox_ct = torch.cat([bbox_ctx.unsqueeze(1), bbox_cty.unsqueeze(1)], dim=1)
            
            # Valid 2D projected centers on image
            centers = label['centers2d'][b_idx][mask]
            
            # Valid 2D Keypoints
            kpts_2d = label['gt_kpts_2d'][b_idx][mask]
            kpts_2d = kpts_2d.reshape(-1, self.num_kpt, 2)
            
            kpts_2d[:, :, 0] = (kpts_2d[:, :, 0] * w_ratio)
            kpts_2d[:, :, 1] = (kpts_2d[:, :, 1] * h_ratio)
            kpts_mask = label['gt_kpts_valid_mask'][b_idx][mask]
            
            # Valid 3D Bboxes and Depth
            bboxes_3d = label['gt_bboxes_3d'][b_idx][mask]
            depth = label['depths'][b_idx][mask]
            
            for o_idx, ct in enumerate(centers):
                ctx, cty = ct
                is_in_image = (0 <= ctx <= ori_w) and (0 <= cty <= ori_h)
                if not is_in_image: #把边缘交点当作中心点
                    continue
                    # ct = approx_proj_center(ct, bbox_ct[o_idx].reshape(1, 2), (ori_w, ori_h))

                ct = ct * h_ratio
                ctx, cty = ct
                ctx_int, cty_int = ct.int()
                
                feat_box_h = (bboxes[o_idx, 3] - bboxes[o_idx, 1]) * h_ratio
                feat_box_w = (bboxes[o_idx, 2] - bboxes[o_idx, 0]) * w_ratio

                if not is_in_image:
                    scale_box_w = min(ctx_int - bboxes[o_idx][0],
                                      bboxes[o_idx][2] - ctx_int)
                    scale_box_h = min(cty_int - bboxes[o_idx][1],
                                      bboxes[o_idx][3] - cty_int)
                    radius_x = scale_box_w  * 0.5
                    radius_y = scale_box_h * 0.5
                    radius_x, radius_y = max(0, int(radius_x)), max(
                        0, int(radius_y))
                    assert min(radius_x, radius_y) == 0
                    c_idx = bbox_labels[o_idx]
                    get_ellip_gaussian_2D(
                        target['center_heatmap_target'][b_idx, c_idx],
                        [ctx_int, cty_int], radius_x,
                        radius_y)
                else:
                    target_radius = gaussian_radius((feat_box_h, feat_box_w))
                    target_radius = max(0, int(target_radius))
                    c_idx = bbox_labels[o_idx]
                    
                    generate_gaussian_target(target['center_heatmap_target'][b_idx, c_idx],
                                            center=[ctx_int, cty_int],
                                            radius=target_radius)
                
                  
                target['indices'][b_idx, o_idx] = (cty_int * feat_w + ctx_int)

                target['wh_target'][b_idx, o_idx] = torch.Tensor([feat_box_w, feat_box_h])
                target['offset_target'][b_idx, o_idx] = torch.Tensor([(ctx - ctx_int), (cty - cty_int)])
                
                target['base_center_2d'][b_idx, o_idx] = torch.Tensor([(ctx_int), (cty_int)])
                dim = bboxes_3d[o_idx][3:6]
                target['dim_target'][b_idx, o_idx] = dim
                target['depth_target'][b_idx, o_idx] = depth[o_idx]
                target['cam2img'][b_idx, o_idx] = torch.Tensor(calibs[b_idx].P2)
                alpha = bboxes_3d[o_idx][6]
                alpha_cls, alpha_offset = self._convert_angle_to_class(alpha)
                target['alpha_cls_target'][b_idx, o_idx] = alpha_cls
                target['alpha_offset_target'][b_idx, o_idx] = alpha_offset
                
                target['mask_target'][b_idx, o_idx] = 1


                kpt_2d = kpts_2d[o_idx]
                kpt_mask = kpts_mask[o_idx]

                # Keypoints
                for k_idx in range(self.num_kpt):
                    kpt = kpt_2d[k_idx]
                    kptx_int, kpty_int = kpt.int()
                    kptx, kpty = kpt
                    
                    vis_level = kpt_mask[k_idx]
                    if vis_level < 1:
                        continue
                    
                    target['center2kpt_offset_target'][b_idx, o_idx, (k_idx * 2)] = (kptx - ctx_int)
                    target['center2kpt_offset_target'][b_idx, o_idx, (k_idx * 2) + 1] = (kpty - cty_int)                    
                    
                    is_kpt_inside_feat = (0 <= kptx_int < feat_w) and (0 <= kpty_int < feat_h)
                    if is_kpt_inside_feat:
                        target['mask_center2kpt_offset'][b_idx, o_idx, (k_idx * 2): ((k_idx + 1) * 2)] = 1

                mask_o_kpt = target['mask_center2kpt_offset'][b_idx, o_idx, ::2]
                target['mask_group_depth'][b_idx, o_idx] = torch.stack((mask_o_kpt[[1, 2, 4, 7]].all(),
                                                                    mask_o_kpt[[0, 3, 5, 6]].all(),
                                                                    mask_o_kpt[[8, 9]].all())
                                                                    ).float()

        target['mask_target'] = (target['mask_target']).type(torch.BoolTensor)
        target['mask_group_depth'] = (target['mask_group_depth']).type(torch.BoolTensor)
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
            'offset_target': torch.zeros((batch_size, self.max_objs, 2)),
            'dim_target': torch.zeros((batch_size, self.max_objs, 3)),
            'alpha_cls_target': torch.zeros((batch_size, self.max_objs, 1)),
            'alpha_offset_target': torch.zeros((batch_size, self.max_objs, 1)),
            'depth_target': torch.zeros((batch_size, self.max_objs, 1)),
            'base_center_2d': torch.zeros((batch_size, self.max_objs, 2)),
            'center2kpt_offset_target': torch.zeros((batch_size, self.max_objs, self.num_kpt * 2)),
            
            'indices': torch.zeros((batch_size, self.max_objs)).type(torch.LongTensor),
            
            'cam2img': torch.zeros((batch_size, self.max_objs, 3, 4)),
            'mask_group_depth': torch.zeros((batch_size, self.max_objs, 3)),
            'mask_target': torch.zeros((batch_size, self.max_objs)),
            'mask_center2kpt_offset': torch.zeros((batch_size, self.max_objs, self.num_kpt * 2))}
        
        if device is None:
            device = 'cpu'
        for k in container.keys():
            container[k] = container[k].to(device)
        return container


    def keypoints2depth(self,
                        keypoints2d,
                        dimensions,
                        cam2imgs,
                        downsample_ratio=4,
                        group0_index=[(2, 7), (1, 4)],
                        group1_index=[(6, 3), (5, 0)]):
        """Decode depth form three groups of keypoints and geometry projection
        model. 2D keypoints inlucding 8 coreners and top/bottom centers will be
        divided into three groups which will be used to calculate three depths
        of object.
        Args:
            keypoints2d (torch.Tensor): Keypoints of objects.
                8 vertices + top/bottom center.
                shape: (N, 10, 2)
            dimensions (torch.Tensor): Dimensions of objetcts.
                shape: (N, 3)
            cam2imgs (torch.Tensor): Batch images' camera intrinsic matrix.
                shape: kitti (N, 4, 4)  nuscenes (N, 3, 3)
            downsample_ratio (int, opitonal): The stride of feature map.
                Defaults: 4.
            group0_index(list[tuple[int]], optional): Keypoints group 0
                of index to calculate the depth.
                Defaults: [0, 3, 4, 7].
            group1_index(list[tuple[int]], optional): Keypoints group 1
                of index to calculate the depth.
                Defaults: [1, 2, 5, 6]
        Return:
            tuple(torch.Tensor): Depth computed from three groups of
                keypoints (top/bottom, group0, group1)
                shape: (N, 3)
        """

        pred_height_3d = dimensions[:, 1].clone()
        f_u = cam2imgs[:, 0, 0]
        center_height = keypoints2d[:, -1, 1] - keypoints2d[:, -2, 1]
        corner_group0_height = keypoints2d[:, [2, 7], 1] \
            - keypoints2d[:, [1, 4], 1]
        corner_group1_height = keypoints2d[:, [6, 3], 1] \
            - keypoints2d[:, [5, 0], 1]
        center_depth = f_u * pred_height_3d / (
            F.relu(center_height) * downsample_ratio + self.eps)
        corner_group0_depth = (f_u * pred_height_3d).unsqueeze(-1) / (
            F.relu(corner_group0_height) * downsample_ratio + self.eps)
        corner_group1_depth = (f_u * pred_height_3d).unsqueeze(-1) / (
            F.relu(corner_group1_height) * downsample_ratio + self.eps)

        corner_group0_depth = corner_group0_depth.mean(dim=1)
        corner_group1_depth = corner_group1_depth.mean(dim=1)

        keypoints_depth = torch.stack(
            (corner_group0_depth, corner_group1_depth, center_depth), dim=1)
        keypoints_depth = torch.clamp(
            keypoints_depth, min=0.1, max=100)

        return keypoints_depth
    

    def combine_depths(self, depth, depth_uncertainty):
        """Combine all the prediced depths with depth uncertainty.
        Args:
            depth (torch.Tensor): Predicted depths of each object.
                2D bboxes.
                shape: (N, 4)
            depth_uncertainty (torch.Tensor): Depth uncertainty for
                each depth of each object.
                shape: (N, 4)
        Returns:
            torch.Tenosr: combined depth.
        """
        uncertainty_weights = 1 / depth_uncertainty
        uncertainty_weights = \
            uncertainty_weights / \
            uncertainty_weights.sum(dim=1, keepdim=True)
        combined_depth = torch.sum(depth * uncertainty_weights, dim=1)

        return combined_depth
