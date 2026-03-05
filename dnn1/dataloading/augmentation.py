import torch
import torch.nn as nn

def horizontal_flip_augment(self, x, y):
    x = torch.flip(x, dims=[2])
    # swap triangle right/left
    y = y.clone()
    y[[3, 5]] = y[[5, 3]]
    return x, y

def vertical_flip_augment(self, x, y):
    x = torch.flip(x, dims=[1])
    # swap triangle up/down
    y = y.clone()
    y[[2, 4]] = y[[4, 2]]
    return x, y

def rotation90_clockwise_augment(self, x, y):
    x = torch.rot90(x, -1, (1, 2))
    y = y.clone()
    y[[2, 3, 4, 5]] = y[[5, 2, 3, 4]]
    return x, y

def rotation90_counterclockwise_augment(self, x, y):
    x = torch.rot90(x, 1, (1, 2))
    y = y.clone()
    y[[2, 3, 4, 5]] = y[[3, 4, 5, 2]]
    return x, y

class Augmentor(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def augment(self, x, y):
        raise NotImplementedError()

    def forward(self, x, y):
        if torch.rand(1).item() >= self.p:
            return x, y
        return self.augment(x, y.clone())


class HorizontalFlipAugmentor(Augmentor):
    def augment(self, x, y):
        x = torch.flip(x, dims=[2])
        # swap triangle right/left
        y[[3, 5]] = y[[5, 3]]
        return x, y


class VerticalFlipAugmentor(Augmentor):
    def augment(self, x, y):
        x = torch.flip(x, dims=[1])
        # swap triangle up/down
        y[[2, 4]] = y[[4, 2]]
        return x, y


class Rotation90Augmentor(Augmentor):
    def augment(self, x, y):
        side = torch.randint(0, 2, (1,)).item()
        if side == 0:  # clockwise
            x = torch.rot90(x, -1, (1, 2))
            y[[2, 3, 4, 5]] = y[[5, 2, 3, 4]]
        else:  # counterclockwise
            x = torch.rot90(x, 1, (1, 2))
            y[[2, 3, 4, 5]] = y[[3, 4, 5, 2]]
        return x, y