import torch.nn as nn

class MultitaskLoss(nn.Module):
    def __init__(self, lambda_cnt):
        super().__init__()
        self.lambda_cnt = lambda_cnt
        self.cls_loss_fn = nn.NLLLoss()
        self.cnt_loss_fn = nn.SmoothL1Loss()

    def forward(self, prediciton, target):
        cls_loss = self.cls_loss_fn(prediciton[0], target[0])
        cnt_loss = self.cnt_loss_fn(prediciton[1], target[1])
        return (cls_loss + self.lambda_cnt * cnt_loss, cls_loss, cnt_loss)


class RegressionLoss(nn.Module):
    def __init__(self, lambda_cnt):
        super().__init__()
        self.lambda_cnt = lambda_cnt
        self.cnt_loss_fn = nn.SmoothL1Loss()

    def forward(self, prediciton, target):
        cnt_loss = self.cnt_loss_fn(prediciton[1], target[1])
        return (self.lambda_cnt * cnt_loss, 0.0 * cnt_loss, cnt_loss)