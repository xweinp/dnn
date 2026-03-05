import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        loss_fn: callable,
        eval_metrics: list,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        patience: int
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.loss_fn = loss_fn
        self.eval_metrics = eval_metrics
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.patience = patience

    def print_losses(
        self,
        avg_whole_loss,
        avg_cls_loss,
        avg_cnt_loss,
    ):
        print(f"    Loss: {avg_whole_loss:.4f}")
        print(f"    Classification : {avg_cls_loss:.4f}")
        print(f"    Regression: {avg_cnt_loss:.4f}")

    @torch.no_grad()
    def validation(self):
        all_preds = []
        all_targets = []

        avg_whole_loss = 0.0
        avg_cls_loss = 0.0
        avg_cnt_loss = 0.0

        self.model.eval()
        for batch in self.eval_dataloader:
            x, y_cls, y_cnt = batch
            pred_cls, pred_cnt = self.model(x)

            all_preds.append((pred_cls, pred_cnt))
            all_targets.append((y_cls, y_cnt))

            whole_loss, cls_loss, cnt_loss = self.loss_fn(
                (pred_cls, pred_cnt), (y_cls, y_cnt))

            avg_whole_loss += whole_loss.item() * x.shape[0]
            avg_cls_loss += cls_loss.item() * x.shape[0]
            avg_cnt_loss += cnt_loss.item() * x.shape[0]

        avg_whole_loss /= len(self.eval_dataloader.dataset)
        avg_cls_loss /= len(self.eval_dataloader.dataset)
        avg_cnt_loss /= len(self.eval_dataloader.dataset)

        all_preds = (
            torch.cat([p[0] for p in all_preds]),
            torch.cat([p[1] for p in all_preds])
        )
        all_targets = (
            torch.cat([t[0] for t in all_targets]),
            torch.cat([t[1] for t in all_targets])
        )
        print("Validation:")
        self.print_losses(
            avg_whole_loss,
            avg_cls_loss,
            avg_cnt_loss
        )

        for metric in self.eval_metrics:
            result = metric(all_preds, all_targets)
            self.eval_metrics_values[metric.name].append(result)
            print(f"    {metric.name}: {result}")

        self.eval_losses.append(avg_whole_loss)
        self.eval_cls_losses.append(avg_cls_loss)
        self.eval_cnt_losses.append(avg_cnt_loss)
        return avg_whole_loss

    def train(self):
        start_time = time.time()

        best_eval_loss = float('inf')
        best_eval_loss_epoch = -1

        self.train_losses = []
        self.train_cls_losses = []
        self.train_cnt_losses = []

        self.eval_losses = []
        self.eval_cls_losses = []
        self.eval_cnt_losses = []

        self.eval_metrics_values = {
            metric.name: [] for metric in self.eval_metrics
        }

        for epoch in range(self.n_epochs):
            print(f'Epoch {epoch+1}/{self.n_epochs}')
            avg_whole_loss = 0.0
            avg_cls_loss = 0.0
            avg_cnt_loss = 0.0

            self.model.train()

            for batch in self.train_dataloader:
                self.optimizer.zero_grad()

                x, y_cls, y_cnt = batch
                pred_cls, pred_cnt = self.model(x)

                whole_loss, cls_loss, cnt_loss = self.loss_fn(
                    (pred_cls, pred_cnt), (y_cls, y_cnt))

                avg_whole_loss += whole_loss.item() * x.shape[0]
                avg_cls_loss += cls_loss.item() * x.shape[0]
                avg_cnt_loss += cnt_loss.item() * x.shape[0]

                whole_loss.backward()
                self.optimizer.step()

            avg_whole_loss /= len(self.train_dataloader.dataset)
            avg_cls_loss /= len(self.train_dataloader.dataset)
            avg_cnt_loss /= len(self.train_dataloader.dataset)

            self.train_losses.append(avg_whole_loss)
            self.train_cls_losses.append(avg_cls_loss)
            self.train_cnt_losses.append(avg_cnt_loss)

            print("Train loss:")
            self.print_losses(
                avg_whole_loss,
                avg_cls_loss,
                avg_cnt_loss
            )

            eval_loss = self.validation()
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_eval_loss_epoch = epoch
            elif epoch - best_eval_loss_epoch >= self.patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")