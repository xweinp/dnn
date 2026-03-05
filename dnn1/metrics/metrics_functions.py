import torch
import torch.nn.functional as F
from model import class_to_pair_encoding, N_CLS_CLASSES, N_PAIR_CLASSES

# Classification
def top_1_accuracy(preds, targets):
    argmax = torch.argmax(preds, dim=-1)
    correct = argmax == targets
    accuracy = torch.sum(correct) / targets.numel()
    return accuracy.item()

def per_pair_accuracy(preds, targets):
    argmax = torch.argmax(preds, dim=-1)
    correct = argmax == targets

    pair_classes = class_to_pair_encoding(targets)
    class_counts = pair_classes.bincount(minlength=N_PAIR_CLASSES)
    correct_counts = pair_classes.bincount(correct, minlength=N_PAIR_CLASSES)

    accuracies = correct_counts / class_counts
    return accuracies.cpu().tolist()

def macro_f1_score(preds, targets):
    argmax = torch.argmax(preds, dim=-1)
    correct = argmax == targets
    incorrect = argmax != targets
    
    TP = targets.bincount(correct, minlength=N_CLS_CLASSES)
    FP = argmax.bincount(minlength=N_CLS_CLASSES) - TP
    FN = targets.bincount(incorrect, minlength=N_CLS_CLASSES)

    eps = 1e-9
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    
    f1 = 2 * precision * recall  / (precision + recall + eps)
    macro_f1 = torch.mean(f1)

    return macro_f1

# Regression
def rmse_per_class(preds, targets):
    loss = F.mse_loss(preds, targets, reduction="none")
    mse = torch.mean(loss, dim=0) # mean along batch
    return torch.sqrt(mse)

def rmse(preds, targets):
    mse = F.mse_loss(preds, targets)
    return torch.sqrt(mse)

def mae_per_class(preds, targets):
    loss = F.l1_loss(preds, targets, reduction="none")
    mae = torch.mean(loss, dim=0) # mean along batch
    return mae


def mae(preds, targets):
    return F.l1_loss(preds, targets).item()
