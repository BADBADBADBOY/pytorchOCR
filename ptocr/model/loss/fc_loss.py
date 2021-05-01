import torch
import torch.nn as nn
from .basical_loss import CrossEntropyLoss

class FCLoss(nn.Module):
    def __init__(self,ignore_index = -1):
        super(FCLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss(ignore_index = ignore_index)
       

    def forward(self, pred_bach, gt_batch):
        loss = self.cross_entropy_loss(pred_bach['pred'],gt_batch['gt'])
        metrics = dict(loss_fc=loss)
        return loss, metrics