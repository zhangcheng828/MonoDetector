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
from model import AttnBatchNorm2d
from utils.tensor_ops import (extract_input, 
                              extract_target, 
                              transpose_and_gather_feat,
                              get_local_maximum, 
                              get_topk_from_heatmap)

from utils.data_classes import KITTICalibration
from utils.kitti_convert_utils import convert_to_kitti_2d, convert_to_kitti_3d


# Constants
EPS = 1e-12
PI = np.pi


# Default Test Config
DEFAULT_TEST_CFG = {
    'topk': 30,
    'local_maximum_kernel': 3,
    'max_per_img': 30,
    'test_thres': 0.4,
}


class BaseDenseHeads(nn.Module):
    def __init__(self,
                 in_ch: int = 64,
                 feat_ch: int = 64,
                 num_kpts: int = 9,
                 num_alpha_bins: int = 12,
                 num_classes: int = 3,
                 max_objs: int = 30,
                 test_config: Dict[str, Any] = None):
        
        super().__init__()
        
        self.max_objs = max_objs
        self.num_kpts = num_kpts
        self.num_alpha_bins = num_alpha_bins
        self.num_classes = num_classes
        
        # Configuration for Test
        if test_config is None:
            test_config = DEFAULT_TEST_CFG
        self.test_config = test_config
        
        for k, v in test_config.items():
            setattr(self, k, v)


    def _build_head(self, in_ch: int, feat_ch: int, out_channel: int) -> nn.Module:
        layers = [
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1),
            AttnBatchNorm2d(feat_ch, 10, momentum=0.03, eps=0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_ch, out_channel, kernel_size=1)]
        return nn.Sequential(*layers)


    def _build_roi_head(self, in_ch: int, feat_ch: int, out_channel: int) -> nn.Module:
        layers = [
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1, bias=True),
            AttnBatchNorm2d(feat_ch, 10, momentum=0.03, eps=0.001),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_ch, out_channel, kernel_size=1)]
        return nn.Sequential(*layers)


    def _build_dir_head(self, in_ch: int, feat_ch: int) -> Tuple[nn.Module]:
        dir_feat = nn.Sequential(
            nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1),
            AttnBatchNorm2d(feat_ch, 10, momentum=0.03, eps=0.001),
            nn.ReLU(inplace=True))
        dir_cls = nn.Sequential(nn.Conv2d(feat_ch, self.num_alpha_bins, kernel_size=1))
        dir_reg = nn.Sequential(nn.Conv2d(feat_ch, self.num_alpha_bins, kernel_size=1))
        
        return dir_feat, dir_cls, dir_reg


    def init_weights(self, prior_prob: float = 0.1) -> None:
        raise NotImplementedError

    
    
    # Get predictions and losses for train
    def forward_train(self, 
                      feat: torch.Tensor, 
                      data_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        
        target_dict = self.target_generator(data_dict, feat_shape=tuple(feat.shape))
        pred_dict = self._get_predictions(feat)
        loss_dict = self._get_losses(pred_dict, target_dict)
        return (pred_dict, loss_dict)
    
    
    # Get predictions for test
    def forward_test(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._get_predictions(feat)


    def _get_predictions(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
        
    def _get_losses(self, 
                    pred_dict: Dict[str, torch.Tensor], 
                    target_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
        
        
    def _get_bboxes(self, 
                    data_dict: Dict[str, Any],
                    pred_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        
        raise NotImplementedError
    
    def get_two_stage_pred(self, pred_dict, indices, 
                    mask_target, labels, calibs) -> None:

        pass

    # Data used for kitti evaluation
    def _get_eval_formats(self,
                          data_dict: Dict[str, Any],
                          pred_dict: Dict[str, torch.Tensor],
                          get_vis_format: bool = False) -> Dict[str, Any]:
        
        raise NotImplementedError
        

    def decode_alpha(self, 
                     alpha_cls: torch.Tensor, 
                     alpha_offset: torch.Tensor) -> torch.Tensor:
        
        # Bin Class and Offset
        _, cls = alpha_cls.max(dim=-1)
        cls = cls.unsqueeze(2)
        alpha_offset = alpha_offset.gather(2, cls)
        
        # Convert to Angle
        angle_per_class = (2 * PI) / float(self.num_alpha_bins)
        angle_center = (cls * angle_per_class)
        alpha = (angle_center + alpha_offset)

        # Refine Angle
        alpha[alpha > PI] = alpha[alpha > PI] - (2 * PI)
        alpha[alpha < -PI] = alpha[alpha < -PI] + (2 * PI)
        return alpha
    
    
    def decode_heatmap(self, 
                       data_dict: Dict[str, Any], 
                       pred_dict: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor]]:
        
        raise NotImplementedError
        
    def calculate_roty(self, 
                       kpts: torch.Tensor, 
                       alpha: torch.Tensor,
                       batched_calib: List[KITTICalibration]) -> torch.Tensor:
        
        """
        * Args:
            - 'kpts'
                torch.Tensor / (B, K, 2)
            - 'alpha'
                torch.Tensor / (B, K, 1)
            - 'batched_calib'
                List[KITTICalibration] / Length: B
        """
        
        device = kpts.device
        proj_matrices = torch.cat([torch.from_numpy(calib.P2).unsqueeze(0) for calib in batched_calib], dim=0)
        proj_matrices = proj_matrices.type(torch.FloatTensor).to(device)                        # (B, 3, 4)
        
        # kpts[:, :, 0:1]       -> (B, K, 1)
        # calib[:, 0:1, 0:1]    -> (B, 1, 1)

        si = torch.zeros_like(kpts[:, :, 0:1]) + proj_matrices[:, 0:1, 0:1]
        rot_y = alpha + torch.atan2(kpts[:, :, 0:1] - proj_matrices[:, 0:1, 2:3], si)

        while (rot_y > PI).any():
            rot_y[rot_y > PI] = rot_y[rot_y > PI] - 2 * PI
        while (rot_y < -PI).any():
            rot_y[rot_y < -PI] = rot_y[rot_y < -PI] + 2 * PI

        return rot_y


    def convert_pts2D_to_pts3D(self, 
                               points_2d: torch.Tensor, 
                               batched_calib: List[KITTICalibration]) -> torch.Tensor:
        
        """
        * Args:
            - 'points_2d'
                torch.Tensor / (B, K, 3)
            - 'batched_calib'
                List[KITTICalibration] / Length: B
        """
        
        if isinstance(batched_calib, KITTICalibration):
            batched_calib = [batched_calib, ] * len(points_2d)
                
        # 'points_2d': (B, K, 3)
        centers = points_2d[:, :, :2]                                       # (B, K, 2)
        depths = points_2d[:, :, 2:3]                                       # (B, K, 1)
        unnorm_points = torch.cat([(centers * depths), depths], dim=-1)     # (B, K, 3)
        
        points_3d_result = []
        
        # 'unnorm_point': (K, 3)
        for b_idx, unnorm_point in enumerate(unnorm_points):
            
            proj_mat = batched_calib[b_idx].P2
            viewpad = torch.eye(4)
            viewpad[:proj_mat.shape[0], :proj_mat.shape[1]] = points_2d.new_tensor(proj_mat)
            inv_viewpad = torch.inverse(viewpad).transpose(0, 1).to(points_2d.device)

            # Do operation in homogenous coordinates.
            nbr_points = unnorm_point.shape[0]                              # K
            homo_points = torch.cat(
                [unnorm_point,
                points_2d.new_ones((nbr_points, 1))], dim=1)                # (K, 4)
            points_3d = torch.mm(homo_points, inv_viewpad)[:, :3]           # (K, 4) * (4, 4) = (K, 4) -> (K, 3)
            
            points_3d_result.append(points_3d.unsqueeze(0))                 # (1, K, 4)
            
        points_3d = torch.cat(points_3d_result, dim=0)                      # (B, K, 3)
        return points_3d


    def bbox_2d_to_result(self, 
                          bboxes_2d: torch.Tensor, 
                          labels: torch.Tensor, 
                          num_classes: int) -> List[np.ndarray]:
        
        if bboxes_2d.shape[0] == 0:
            return [np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)]
        
        assert isinstance(bboxes_2d, torch.Tensor)
        
        bboxes_2d = bboxes_2d.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        return [bboxes_2d[labels == c_i, :] for c_i in range(num_classes)]
    
    
    def bbox_3d_to_result(self,
                          bboxes_3d: torch.Tensor,
                          scores: torch.Tensor,
                          labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        result_dict = dict(
            boxes_3d=bboxes_3d.cpu(),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu())
        
        return result_dict
