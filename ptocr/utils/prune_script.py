import torch
import torch.nn as nn

def updateBN(model,args):
    for indedx,m in enumerate(model.modules()):
        if(indedx>187):
            break
        if isinstance(m, nn.BatchNorm2d):
            if hasattr(m.weight, 'data'):
                m.weight.grad.data.add_(args.sr_lr*torch.sign(m.weight.data)) #L1正则

