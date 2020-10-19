# from torch.nn import CTCLoss as PytorchCTCLoss
from warpctc_pytorch import CTCLoss as PytorchCTCLoss
import torch.nn as nn
from .basical_loss import focal_ctc_loss

class CTCLoss(nn.Module):
    def __init__(self,config):
        super(CTCLoss,self).__init__()
#         self.criterion = PytorchCTCLoss(reduction = config['loss']['reduction'])
        self.criterion = PytorchCTCLoss()
        self.config = config
        
    def forward(self,pre_batch,gt_batch):
        preds,preds_size = pre_batch['preds'],pre_batch['preds_size']
        labels,labels_len = gt_batch['labels'],gt_batch['labels_len']
        loss = self.criterion(preds, labels, preds_size, labels_len)
        if self.config['loss']['reduction']=='none':
            loss = focal_ctc_loss(loss)
        return loss/self.config['trainload']['batch_size']