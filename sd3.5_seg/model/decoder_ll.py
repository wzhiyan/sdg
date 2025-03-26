import torch.nn as nn
import torch.nn.functional as F


class seg_decorder_ll(nn.Module):
      def __init__(self, num_class):
          super(seg_decorder_ll, self).__init__()
          fpn_dim = 16    
          norm_layer=nn.BatchNorm2d     

          self.conv_last = nn.Sequential(
            nn.Conv2d(4 * fpn_dim, fpn_dim, kernel_size=1, padding=0, bias=False),
            norm_layer( fpn_dim),
            nn.ReLU(inplace=True),
           # nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1, bias=False))

      def forward(self, attfeature):
          seg = self.conv_last(attfeature)
          return seg

      
         


class onece(nn.Module):
    def __init__(self, ignore_label=255):
        super(onece, self).__init__()

        #self.configer = configer

        self.ignore_index = ignore_label
 
        self.seg_criterion = OhemCrossEntropy2dTensor(ignore_index=self.ignore_index).cuda()
 

    def forward(self, preds, target, with_embed=True):
        h, w = target.size(1), target.size(2)

        assert "seg" in preds
        #assert "embed" in preds

        seg = preds['seg']
        #embedding = preds['embed']

        pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)
 
        return loss 


class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
 
        self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean", ignore_index=ignore_index)
 
                                                       
    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
 
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)

