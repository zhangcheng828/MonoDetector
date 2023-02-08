import torch
import torch.nn.functional as F

from math import sqrt
from typing import Tuple, List


def extract_input(input: torch.Tensor, ind: torch.Tensor, mask: torch.Tensor):
    input = transpose_and_gather_feat(input, ind)
    return input[mask]


def extract_target(target: torch.Tensor, mask: torch.Tensor):
    return target[mask]


def get_local_maximum(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return (heatmap * keep)


def get_topk_from_heatmap(scores: torch.Tensor, k: int = 20) -> Tuple[torch.Tensor]:
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(feat, ind, mask=None):
    """Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.
        mask (Tensor | None): Mask of feature map. Default: None.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def gaussian2D(radius: int, sigma: int = 1, device: str = None) -> torch.Tensor:
    
    if device is None:
        device = 'cpu'
    
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device).view(1, -1)
    y = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device).view(-1, 1)

    h = (-(x * x + y * y) / (2 * sigma * sigma)).exp()

    h[h < torch.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size: Tuple[int, int], min_overlap: float = 0.3) -> float:
    
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = sqrt((b1 ** 2) - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = sqrt((b2 ** 2) - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = sqrt((b3 ** 2) - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    
    return min(r1, r2, r3)

def generate_gaussian_target(heatmap_canvas: torch.Tensor,
                             center: List[int],
                             radius: int,
                             k: int = 1) -> torch.Tensor:
    
    device = heatmap_canvas.device
    
    diameter = (2 * radius) + 1
    gaussian_kernel = gaussian2D(radius, sigma=(diameter / 6), device=device)

    x, y = center

    height, width = heatmap_canvas.shape[:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap_canvas[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius - top:radius + bottom,
                                    radius - left:radius + right]
    out_heatmap = heatmap_canvas

    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap

def ellip_gaussian2D(radius,
                     sigma_x,
                     sigma_y,
                     dtype=torch.float32,
                     device='cpu'):
    """Generate 2D ellipse gaussian kernel.
    Args:
        radius (tuple(int)): Ellipse radius (radius_x, radius_y) of gaussian
            kernel.
        sigma_x (int): X-axis sigma of gaussian function.
        sigma_y (int): Y-axis sigma of gaussian function.
        dtype (torch.dtype, optional): Dtype of gaussian tensor.
            Default: torch.float32.
        device (str, optional): Device of gaussian tensor.
            Default: 'cpu'.
    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius_y + 1) * (2 * radius_x + 1)`` shape.
    """
    x = torch.arange(
        -radius[0], radius[0] + 1, dtype=dtype, device=device).view(1, -1)
    y = torch.arange(
        -radius[1], radius[1] + 1, dtype=dtype, device=device).view(-1, 1)

    h = (-(x * x) / (2 * sigma_x * sigma_x) - (y * y) /
         (2 * sigma_y * sigma_y)).exp()
    h[h < torch.finfo(h.dtype).eps * h.max()] = 0

    return 

def get_ellip_gaussian_2D(heatmap, center, radius_x, radius_y, k=1):
    """Generate 2D ellipse gaussian heatmap.
    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius_x (int): X-axis radius of gaussian kernel.
        radius_y (int): Y-axis radius of gaussian kernel.
        k (int, optional): Coefficient of gaussian kernel. Default: 1.
    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    diameter_x, diameter_y = 2 * radius_x + 1, 2 * radius_y + 1
    gaussian_kernel = ellip_gaussian2D((radius_x, radius_y),
                                       sigma_x=diameter_x / 6,
                                       sigma_y=diameter_y / 6,
                                       dtype=heatmap.dtype,
                                       device=heatmap.device)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_x), min(width - x, radius_x + 1)
    top, bottom = min(y, radius_y), min(height - y, radius_y + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian_kernel[radius_y - top:radius_y + bottom,
                                      radius_x - left:radius_x + right]
    out_heatmap = heatmap
    torch.max(
        masked_heatmap,
        masked_gaussian * k,
        out=out_heatmap[y - top:y + bottom, x - left:x + right])

    return out_heatmap