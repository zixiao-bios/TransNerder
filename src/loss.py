from torch import nn
import torch


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self, pred, label, valid_len): # type: ignore
        assert pred.ndim == 3 and label.ndim == 2 and valid_len.ndim == 2, f'pred.ndim = {pred.ndim}, label.ndim = {label.ndim}, valid_len.ndim = {valid_len.ndim}'
        assert pred.shape[0] == label.shape[0] == valid_len.shape[0], f'pred.shape[0] = {pred.shape[0]}, label.shape[0] = {label.shape[0]}, valid_len.shape[0] = {valid_len.shape[0]}'
        assert pred.shape[1] == label.shape[1], f'pred.shape[1] = {pred.shape[1]} != label.shape[1] = {label.shape[1]}'
        
        # pred.shape = (batch_size, seq_len, vocab_size)
        # label.shape = (batch_size, seq_len)
        # valid_len.shape = (batch_size, 1)
        
        seq_range = torch.arange(pred.shape[1], device=pred.device).repeat(pred.shape[0], 1)
        # seq_range.shape = (batch_size, seq_len)
        
        mask = (seq_range < valid_len).float()
        # mask.shape = (batch_size, seq_len)
        
        loss = super(MaskedSoftmaxCELoss, self).forward(pred.permute(0, 2, 1), label)
        loss = torch.sum(loss * mask) / mask.sum()
        return loss
