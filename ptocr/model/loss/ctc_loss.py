from torch.nn import CTCLoss as PytorchCTCLoss
import torch.nn as nn

class CTCLoss(nn.Module):
    def __init__(self,config):
        super(CTCLoss,self).__init__()
        self.criterion = PytorchCTCLoss(reduction = config['loss']['reduction'])
        
    def forward(self,pre_batch,gt_batch):
        preds,preds_size = pre_batch['preds'],pre_batch['preds_size']
        labels,labels_len = gt_batch['labels'],gt_batch['labels_len']
        loss = self.criterion(preds, labels, preds_size, labels_len)
        return loss