import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

from model import class_to_pair_encoding, N_CLS_CLASSES, N_PAIR_CLASSES, N_CNT_CLASSES
from plot import get_metric_fig, get_bar_fig

class Metric(ABC):
    def __init__(self):
        self.values = []
    name: str
    values: list

    @abstractmethod
    def __call__(self, preds, targets):
        pass

    @abstractmethod
    def plot(self, metric_values):
        pass


class Top1Accuracy(Metric):
    name: str = "Top-1 accuracy"

    def __call__(self, preds, targets):
        preds = preds[0]  # classification
        targets = targets[0]
        argmax = torch.argmax(preds, dim=-1)
        correct = argmax == targets
        accuracy = torch.sum(correct) / targets.numel()
        self.values.append(accuracy.item())
        return accuracy.item()

    def plot(self, metric_values):
        return get_metric_fig(
            metric_values,
            title=self.name
        )


class PerPairAccuracy(Metric):
    name: str = "Per-pair accuracy"

    def __call__(self, preds, targets):
        preds = preds[0]  # classification
        targets = targets[0]
        argmax = torch.argmax(preds, dim=-1)
        correct = argmax == targets

        pair_classes = class_to_pair_encoding(targets)
        class_counts = pair_classes.bincount(minlength=N_PAIR_CLASSES)
        correct_counts = pair_classes.bincount(
            correct, minlength=N_PAIR_CLASSES)

        accuracies = correct_counts / class_counts
        self.values = accuracies.cpu().tolist()
        return accuracies.cpu().tolist()

    def plot(self, metric_values):
        return get_bar_fig(
            metric_values,
            title=self.name,
            xlabel="Pair class",
            ylabel="Accuracy"
        )


class MacroF1Score(Metric):
    name: str = "Macro F1 Score"

    def __call__(self, preds, targets):
        preds = preds[0]  # classification
        targets = targets[0]
        argmax = torch.argmax(preds, dim=-1)
        correct = argmax == targets
        incorrect = argmax != targets

        TP = targets.bincount(correct, minlength=N_CLS_CLASSES)
        FP = argmax.bincount(minlength=N_CLS_CLASSES) - TP
        FN = targets.bincount(incorrect, minlength=N_CLS_CLASSES)

        eps = 1e-9
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)

        f1 = 2 * precision * recall / (precision + recall + eps)
        macro_f1 = torch.mean(f1)
        self.values.append(macro_f1.item())
        return macro_f1

    def plot(self, metric_values):
        return get_metric_fig(
            metric_values,
            title=self.name
        )


class RMSEPerClass(Metric):
    name: str = "RMSE per class"

    def __call__(self, preds, targets):
        preds = preds[1]  # counts
        targets = targets[1]
        loss = F.mse_loss(preds, targets, reduction="none")
        mse = torch.mean(loss, dim=0)  # mean along batch
        self.values = torch.sqrt(mse).cpu().tolist()
        return self.values

    def plot(self, metric_values):
        return get_bar_fig(
            metric_values,
            title=self.name,
            xlabel="Count class",
            ylabel="RMSE"
        )


class RMSE(Metric):
    name: str = "RMSE"

    def __call__(self, preds, targets):
        preds = preds[1]  # counts
        targets = targets[1]
        loss = F.mse_loss(preds, targets, reduction='none')
        mse = torch.mean(loss)
        rmse = torch.sqrt(mse).item()
        self.values.append(rmse)
        return rmse

    def plot(self, metric_values):
        return get_metric_fig(
            metric_values,
            title=self.name
        )


class MAEPerClass(Metric):
    name: str = "MAE per class"

    def __call__(self, preds, targets):
        preds = preds[1]  # counts
        targets = targets[1]
        mae = F.l1_loss(preds, targets, reduction='none')
        self.values = torch.mean(mae, dim=0).cpu().tolist()
        return self.values

    def plot(self, metric_values):
        return get_bar_fig(
            metric_values,
            title=self.name,
            xlabel="Count class",
            ylabel="MAE"
        )


class MAE(Metric):
    name: str = "MAE"

    def __call__(self, preds, targets):
        preds = preds[1]  # counts
        targets = targets[1]
        mae = F.l1_loss(preds, targets).item()
        self.values.append(mae)
        return mae

    def plot(self, metric_values):
        return get_metric_fig(
            metric_values,
            title=self.name
        )