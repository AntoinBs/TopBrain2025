import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss, SoftclDiceLoss

def _soft_erode3d(x):
    return -F.max_pool3d(-x, kernel_size=3, stride=1, padding=1)

def _soft_dilate3d(x):
    return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)

def _soft_open3d(x):
    return _soft_dilate3d(_soft_erode3d(x))

def _soft_skel3d(x, iter_=5):
    x = torch.clamp(x, 0.0, 1.0)
    skel = F.relu(x - _soft_open3d(x))
    xp = x.clone()
    for _ in range(iter_):
        xp = _soft_erode3d(xp)
        opened = _soft_open3d(xp)
        delta = F.relu(xp - opened)
        skel = torch.clamp(skel + F.relu(delta * (1.0 - skel)), 0.0, 1.0)
    return skel

def soft_cldice_loss_3d(pred_fg, target_fg, iter_=5, smooth=1e-6):
    """
    pred_fg, target_fg: (B,1,H,W,D) ∈ [0,1]
    Return loss (0 = perfect, ~1 = very bad).
    """
    pred_fg = torch.clamp(pred_fg, 0.0, 1.0)
    target_fg = torch.clamp(target_fg, 0.0, 1.0)

    SkelP = _soft_skel3d(pred_fg, iter_=iter_)
    SkelL = _soft_skel3d(target_fg, iter_=iter_)

    # Topology precision & recall (from original clDice paper)
    tprec = (torch.sum(SkelP * target_fg) + smooth) / (torch.sum(SkelP) + smooth)
    trec  = (torch.sum(SkelL * pred_fg)   + smooth) / (torch.sum(SkelL) + smooth)

    # clDice score and loss
    cldice_score = (2.0 * tprec * trec) / (tprec + trec + smooth)
    cldice_loss = 1.0 - cldice_score
    return cldice_loss

class HybridDiceCLDiceLoss(torch.nn.Module):
    def __init__(self, iter_=15, smooth=1e-5, include_background=False, weight_dice=1.0, weight_cldice=0.3, class_weights=None):
        super().__init__()
        self.dice_ce_loss = DiceCELoss(
            include_background=include_background,
            softmax=True,   # multiclass → softmax
            to_onehot_y=True,
            reduction="mean",
            weight=class_weights
        )
        self.weight_dice = weight_dice
        self.weight_cldice = weight_cldice
        self.iter_ = iter_
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred   : (B, C, H, W, D) logits of model
            target : (B, 1, H, W, D) integer labels [0..C-1]
        """
        # Dice multiclasses
        dice_ce = self.dice_ce_loss(pred, target)

        # Foreground binarization for clDice
        # (all labels >0 become 1)
        target_fg = (target > 0).float()
        pred_fg = torch.softmax(pred, dim=1)[:, 1:, ...].sum(dim=1, keepdim=True)

        cldice = soft_cldice_loss_3d(pred_fg, target_fg, iter_=self.iter_, smooth=self.smooth)

        return self.weight_dice * dice_ce + self.weight_cldice * cldice