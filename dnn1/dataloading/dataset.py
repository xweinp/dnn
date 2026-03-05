import torch
import pandas as pd
import numpy as np
import torchvision.io as io
from torch.utils.data import Dataset

from model import amounts_to_class
from dataloading import HorizontalFlipAugmentor, VerticalFlipAugmentor, Rotation90Augmentor

class ImageDataset(Dataset):
    def __init__(self, r, augment, device, dtype):
        data = pd.read_csv('data/labels.csv')
        self.data = data.iloc[r]
        self.images = [
            io.read_image(
                'data/' + name,
                mode=io.ImageReadMode.GRAY
            ) for name in self.data['name']
        ]
        self.augment = augment
        self.augmentations = [
            HorizontalFlipAugmentor(0.5),
            VerticalFlipAugmentor(0.5),
            Rotation90Augmentor(2/3),
        ]
        self.device = device
        self.dtype = dtype

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = self.images[idx]
        amounts = row.drop('name').values.astype(np.float32)
        amounts = torch.from_numpy(amounts)

        if self.augment:
            for augmentor in self.augmentations:
                img, amounts = augmentor(img, amounts)

        cls = amounts_to_class(amounts)
        return (
            img.to(self.device, self.dtype),
            cls.to(self.device),
            amounts.to(self.device, self.dtype)
        )

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        imgs, masks, amounts = zip(*batch)
        imgs = torch.stack(imgs)
        cls = torch.cat(masks)
        amounts = torch.stack(amounts)
        return imgs, cls, amounts