import torch.nn as nn
class CTCLoss(nn.Module):
    def __init__(self,config):
        super(CTCLoss,self).__init__()
        self.config = config
        if config['loss']['ctc_type'] == 'warpctc':
            from warpctc_pytorch import CTCLoss as PytorchCTCLoss
            self.criterion = PytorchCTCLoss()
        else:
            from torch.nn import CTCLoss as PytorchCTCLoss
            self.criterion = PytorchCTCLoss(reduction = 'none')
        
    def forward(self,pre_batch,gt_batch):
        preds,preds_size = pre_batch['preds'],pre_batch['preds_size']
        labels,labels_len = gt_batch['labels'],gt_batch['labels_len']
        if self.config['loss']['ctc_type'] != 'warpctc':
            preds = preds.log_softmax(2).requires_grad_() # torch.ctcloss
        loss = self.criterion(preds, labels, preds_size, labels_len)
        if self.config['loss']['use_ctc_weight']:
            loss = gt_batch['ctc_loss_weight']*loss.cuda()
        loss = loss.sum()
        return loss/self.config['trainload']['batch_size']