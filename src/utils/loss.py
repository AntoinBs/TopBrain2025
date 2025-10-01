import torch
from monai.losses import DiceLoss, SoftclDiceLoss

class HybridDiceCLDiceLoss(torch.nn.Module):
    def __init__(self, iter_=15, smooth=1e-5, include_background=False, weight_dice=1.0, weight_cldice=1.0, class_weights=None):
        super().__init__()
        self.dice_loss = DiceLoss(
            include_background=include_background,
            softmax=True,   # multiclass → softmax
            to_onehot_y=True,
            reduction="mean",
            weight=class_weights
        )
        self.cldice_loss = SoftclDiceLoss(
            iter_=iter_,
            smooth=smooth
        )
        self.weight_dice = weight_dice
        self.weight_cldice = weight_cldice

    def forward(self, pred, target):
        """
        Args:
            pred   : (B, C, H, W, D) logits of model
            target : (B, 1, H, W, D) integer labels [0..C-1]
        """
        # Dice multiclasses
        dice = self.dice_loss(pred, target)

        # Foreground binarization for clDice
        # (all labels >0 become 1)
        target_fg = (target > 0).float()
        pred_fg = torch.softmax(pred, dim=1)[:, 1:, ...].sum(dim=1, keepdim=True)

        cldice = self.cldice_loss(pred_fg, target_fg)

        return self.weight_dice * dice + self.weight_cldice * cldice