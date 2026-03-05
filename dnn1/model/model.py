import torch
import torch.nn as nn

N_PAIR_CLASSES = 15
N_CLS_CLASSES = 135
N_CNT_CLASSES = 6


class GeGLU(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, 2 * dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        h = self.layernorm(x)
        h, gate = self.linear(h).chunk(2, dim=-1)
        h = self.dropout(h)
        h *= self.gelu(gate)
        return x + h


class Model(nn.Module):
    def __init__(
        self,
        n_cls_classes=N_CLS_CLASSES,
        n_cnt_classes=N_CNT_CLASSES,
        dropout=0.4
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(64 * 28 * 28, 256), nn.ReLU()
        )
        self.n_cls_classes = n_cls_classes
        self.n_cnt_classes = n_cnt_classes
        
        self.head_cls = nn.Sequential(
            GeGLU(256, dropout),
            GeGLU(256, dropout),
            nn.Dropout(dropout),
            nn.Linear(256, n_cls_classes),
            nn.LogSoftmax(dim=1)
        )
        self.head_cnt = nn.Sequential(
            GeGLU(256, dropout),
            GeGLU(256, dropout),
            nn.Dropout(dropout),
            nn.Linear(256, n_cnt_classes)
        )

    def forward(self, x):
        x = x / 128.0 - 1.0
        x = self.backbone(x)
        out_cls = self.head_cls(x)
        out_cnt = self.head_cnt(x)
        return out_cls, out_cnt
    

    def forward_cls(self, x):
        x = x / 128.0 - 1.0
        x = self.backbone(x)
        out_cls = self.head_cls(x)
        return (
            out_cls,
            torch.zeros(
                (x.shape[0], self.n_cnt_classes),
                device=x.device,
                dtype=x.dtype
            )
        )

    def forward_cnt(self, x):
        x = x / 128.0 - 1.0
        x = self.backbone(x)
        out_cnt = self.head_cnt(x)
        return torch.zeros(
            (x.shape[0], self.n_cls_classes),
            device=x.device,
            dtype=x.dtype
        ), out_cnt
