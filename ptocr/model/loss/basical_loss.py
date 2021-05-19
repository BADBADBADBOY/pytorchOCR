#-*- coding:utf-8 _*-
"""
@author:fxw
@file: basical_loss.py
@time: 2020/08/10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MulClassLoss(nn.Module):
    def __init__(self, ):
        super(MulClassLoss, self).__init__()
        
    def forward(self,pre_score,gt_score,n_class):
        gt_score = gt_score.reshape(-1)
        index = gt_score>0
        if index.sum()>0:
            pre_score = pre_score.permute(0,2,3,1).reshape(-1,n_class)
            gt_score = gt_score[index]
            gt_score = gt_score - 1
            pre_score = pre_score[index]
            class_loss = F.cross_entropy(pre_score,gt_score.long(), ignore_index=-1) 
        else:
            class_loss = torch.tensor(0.0).cuda()
        return class_loss


class CrossEntropyLoss(nn.Module):

    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.criteron = nn.CrossEntropyLoss(weight=weight,
                                size_average=size_average,
                                ignore_index=ignore_index,
                                reduce=reduce,
                                reduction=reduction)

    def forward(self, pred, target, *args):
        return self.criteron(pred.contiguous().view(-1, pred.shape[-1]), target.to(pred.device).contiguous().view(-1))


class DiceLoss(nn.Module):
    def __init__(self,eps=1e-6):
        super(DiceLoss,self).__init__()
        self.eps = eps
    def forward(self,pre_score,gt_score,train_mask):
        pre_score = pre_score.contiguous().view(pre_score.size()[0], -1)
        gt_score = gt_score.contiguous().view(gt_score.size()[0], -1)
        train_mask = train_mask.contiguous().view(train_mask.size()[0], -1)

        pre_score = pre_score * train_mask
        gt_score = gt_score * train_mask

        a = torch.sum(pre_score * gt_score, 1)
        b = torch.sum(pre_score * pre_score, 1) + self.eps
        c = torch.sum(gt_score * gt_score, 1) + self.eps
        d = (2 * a) / (b + c)
        dice_loss = torch.mean(d)
        return 1 - dice_loss


class Agg_loss(nn.Module):
    def __init__(self, Agg_Value=0.5):
        super(Agg_loss, self).__init__()
        self.agg_value = Agg_Value

    def get_tag(self, gt_kernel_key):
        gt_kernel_key = gt_kernel_key.cpu().numpy()
        return sorted(set(gt_kernel_key[gt_kernel_key != 0]))

    def cal_agg_batch(self, similarity_vector, gt_kernel_key, gt_text_key, training_mask):
        similarity_vector = similarity_vector.permute((0, 2, 3, 1))
        Lagg_loss = []
        batch = similarity_vector.shape[0]
        for i in range(batch):
            tags1 = self.get_tag(gt_kernel_key[i] * training_mask[i])
            tags2 = self.get_tag(gt_text_key[i] * training_mask[i])
            if (len(tags1) < 1 or len(tags2) < 1 or len(tags1) != len(tags2)):
                continue
            loss_single = self.cal_agg_single(similarity_vector[i], tags1, tags2, gt_text_key[i], gt_kernel_key[i])
            Lagg_loss.append(loss_single)
        if (len(Lagg_loss) == 0):
            Lagg_loss = torch.tensor(0.0)
        else:
            Lagg_loss = torch.mean(torch.stack(Lagg_loss))
        if torch.cuda.is_available():
            Lagg_loss = Lagg_loss.cuda()
        return Lagg_loss

    def cal_agg_single(self, similarity_vector, tags1, tags2, gt_text, gt_kernel):
        sum_agg = []
        loss_base = torch.tensor(0.0)
        if torch.cuda.is_available():
            loss_base = loss_base.cuda()
        for item in tags1:
            if (item not in tags2):
                continue
            index_k = (gt_kernel == item)
            index_t = (gt_text == item)

            similarity_vector_t = similarity_vector[index_t]
            similarity_vector_k = torch.sum(similarity_vector[index_k], 0) / similarity_vector[index_k].shape[0]
            out = torch.norm((similarity_vector_t - similarity_vector_k), 2, 1) - self.agg_value
            out = torch.max(out, loss_base).pow(2)
            ev_ = torch.log(out + 1).mean()
            sum_agg.append(ev_)
        if (len(sum_agg) == 0):
            loss_single = torch.tensor(0.0)
        else:
            loss_single = torch.mean(torch.stack(sum_agg))
        if torch.cuda.is_available():
            loss_single = loss_single.cuda()
        return loss_single

    def forward(self, gt_text_key, gt_kernel_key, training_mask, similarity_vector):
        loss_agg = self.cal_agg_batch(similarity_vector, gt_kernel_key, gt_text_key, training_mask)
        return loss_agg


class Dis_loss(nn.Module):
    def __init__(self,Lgg_Value=3):
        super(Dis_loss, self).__init__()
        self.lgg_value = Lgg_Value
    def get_kernel_compose(self, tag):
        get_i = 0
        out = []
        while (get_i < (len(tag) - 1)):
            for get_j in range(get_i + 1, len(tag)):
                out.append([tag[get_i], tag[get_j]])
                out.append([tag[get_j], tag[get_i]])
            get_i += 1
        return out

    def get_tag(self, gt_kernel_key):
        gt_kernel_key = gt_kernel_key.cpu().numpy()
        return sorted(set(gt_kernel_key[gt_kernel_key != 0]))

    def cal_Ldis_single(self, similarity_vector, gt_compose, gt_kernel):
        loss_sum = []
        loss_base = torch.tensor(0.0)
        if torch.cuda.is_available():
            loss_base = loss_base.cuda()
        for tag_s in gt_compose:
            index_k_i = (gt_kernel == tag_s[0])
            similarity_vector_k_i = torch.sum(similarity_vector[index_k_i], 0) / similarity_vector[index_k_i].shape[0]
            index_k_j = (gt_kernel == tag_s[1])
            similarity_vector_k_j = torch.sum(similarity_vector[index_k_j], 0) / similarity_vector[index_k_j].shape[0]
            out = torch.max(self.lgg_value - torch.norm(similarity_vector_k_i - similarity_vector_k_j),
                            loss_base).pow(2)
            out = torch.log(out + 1)
            loss_sum.append(out)
        if (len(loss_sum) == 0):
            loss_single = torch.tensor(0.0).float()
        else:
            loss_single = torch.mean(torch.stack(loss_sum))
        if torch.cuda.is_available():
            loss_single = loss_single.cuda()
        return loss_single

    def cal_Ldis_batch(self, similarity_vector, gt_kernel_key, training_mask):
        similarity_vector = similarity_vector.permute((0, 2, 3, 1))
        Ldis_loss = []
        batch = similarity_vector.shape[0]
        for i in range(batch):
            tags = self.get_tag(gt_kernel_key[i] * training_mask[i])
            if (len(tags) < 2):
                continue
            gt_compose = self.get_kernel_compose(tags)
            loss_single = self.cal_Ldis_single(similarity_vector[i], gt_compose, gt_kernel_key[i])
            Ldis_loss.append(loss_single)

        if (len(Ldis_loss) == 0):
            Ldis_loss = torch.tensor(0.0)
        else:
            Ldis_loss = torch.mean(torch.stack(Ldis_loss))
        if torch.cuda.is_available():
            Ldis_loss = Ldis_loss.cuda()
        return Ldis_loss

    def forward(self, gt_kernel_key, training_mask, similarity_vector):
        loss_dis = self.cal_Ldis_batch(similarity_vector, gt_kernel_key, training_mask)
        return loss_dis

class BalanceCrossEntropyLoss(nn.Module):

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor):
        '''
        Args:
            pred: shape :math:`(N, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()),
                            int(positive_count * self.negative_ratio))
        loss = nn.functional.binary_cross_entropy(
            pred, gt, reduction='none')
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) /\
            (positive_count + negative_count + self.eps)
        return balance_loss
    
    
def focal_ctc_loss(ctc_loss,alpha=0.95,gamma=1): # 0.99,1
#     import pdb
#     pdb.set_trace()
    prob = torch.exp(-ctc_loss)
    focal_loss = alpha*(1-prob).pow(gamma)*ctc_loss
    return focal_loss.sum()


class focal_bin_cross_entropy(nn.Module):
    def __init__(self,alpha=0.25,gamma=2,eps=1e-6):
        super(focal_bin_cross_entropy,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    def forward(self,pred,gt):
        loss = -self.alpha*(1-pred).pow(self.gamma)*gt*torch.log(pred+self.eps)-\
               (1-self.alpha)*pred.pow(self.gamma)*(1-gt)*torch.log(1-pred+self.eps)
        return loss

class FocalCrossEntropyLoss(nn.Module):

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(FocalCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.focal_bin_loss = focal_bin_cross_entropy()

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor):
        '''
        Args:
            pred: shape :math:`(N, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        loss = self.focal_bin_loss(pred*mask,gt*mask)
        return loss.mean()

class MaskL1Loss(nn.Module):
    def __init__(self):
        super(MaskL1Loss, self).__init__()

    def forward(self, pred: torch.Tensor,
                        gt: torch.Tensor,
                        mask: torch.Tensor):
        mask_sum = mask.sum()
        if mask_sum.item() == 0:
            return mask_sum, dict(l1_loss=mask_sum)
        else:
            loss = (torch.abs(pred - gt) * mask).sum() / mask_sum
            return loss, dict(loss_l1=loss)

def ohem_single(score, gt_text, training_mask):
    pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))

    if pos_num == 0:
        # selected_mask = gt_text.copy() * 0 # may be not good
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_num = (int)(np.sum(gt_text <= 0.5))
    neg_num = (int)(min(pos_num * 3, neg_num))

    if neg_num == 0:
        selected_mask = training_mask
        selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
        return selected_mask

    neg_score = score[gt_text <= 0.5]
    neg_score_sorted = np.sort(-neg_score)
    threshold = -neg_score_sorted[neg_num - 1]

    selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
    selected_mask = selected_mask.reshape(1, selected_mask.shape[0], selected_mask.shape[1]).astype('float32')
    return selected_mask

def ohem_batch(scores, gt_texts, training_masks):
    scores = scores.data.cpu().numpy()
    gt_texts = gt_texts.data.cpu().numpy()
    training_masks = training_masks.data.cpu().numpy()

    selected_masks = []
    for i in range(scores.shape[0]):
        selected_masks.append(ohem_single(scores[i, :, :], gt_texts[i, :, :], training_masks[i, :, :]))

    selected_masks = np.concatenate(selected_masks, 0)
    selected_masks = torch.from_numpy(selected_masks).float()

    return selected_masks