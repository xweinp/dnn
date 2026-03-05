import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots


FIGSIZE = (12, 8)
PX_SIZE = (1280, 720)


def show_image(
    image,
    title=None,
    figsize=FIGSIZE
):
    plt.figure(figsize=figsize)
    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()


def get_metrics_fig(
    metrics_dict,
    title="Metrics"
):
    df = pd.DataFrame(metrics_dict)
    df['Epoch'] = df.index

    df_long = df.melt(id_vars='Epoch', var_name='Metric', value_name='Value')

    fig = px.line(
        df_long,
        x='Epoch',
        y='Value',
        color='Metric',
        title=title,
        labels={'Value': 'Metric Value'}
    )
    return fig


def get_metric_fig(
    metric_values,
    title,
):
    fig = px.line(
        y=metric_values,
        title=title,
        labels={'x': 'Epoch', 'y': title}
    )
    return fig


def get_bar_fig(
    metric_values,
    title,
    xlabel,
    ylabel,
):
    x = np.arange(len(metric_values))
    y = metric_values
    fig = px.bar(
        x=x,
        y=y,
        title=title,
        labels={'x': xlabel, 'y': ylabel}
    )
    return fig


def plot_figs(
    figs,
    grid_shape,
    title="Metrics",
    px_size=PX_SIZE
):
    fig = make_subplots(
        rows=grid_shape[0],
        cols=grid_shape[1],
        subplot_titles=[f.layout.title.text for f in figs]
    )
    for r in range(grid_shape[0]):
        for c in range(grid_shape[1]):
            if r * grid_shape[1] + c >= len(figs):
                break
            sub_fig = figs[r * grid_shape[1] + c]
            for trace in sub_fig.data:
                fig.add_trace(
                    trace,
                    row=r + 1,
                    col=c + 1
                )
    fig.update_layout(title_text=title)
    fig.update_layout(width=px_size[0], height=px_size[1])
    fig.show()


def plot_metric(
    metric_values,
    title,
    px_size=PX_SIZE
):
    fig = get_metric_fig(
        metric_values,
        title
    )
    fig.update_layout(width=px_size[0], height=px_size[1])
    fig.show()


def plot_bar(
    metric_values,
    title,
    xlabel,
    ylabel,
    px_size=PX_SIZE
):
    fig = get_bar_fig(
        metric_values,
        title,
        xlabel,
        ylabel
    )
    fig.update_layout(width=px_size[0], height=px_size[1])
    fig.show()


def make_metric_plots(
    trainer,
    eval_metrics: list,
    title="Multitask training",
    use_cls_metrics=True,
    use_cnt_metrics=True,
    grid_shape=(2, 2)
):
    cnt_losses = get_metrics_fig({
        "train": trainer.train_cnt_losses,
        "eval": trainer.eval_cnt_losses
    },
        title="Regression Losses"
    )
    cls_losses = get_metrics_fig({
        "train": trainer.train_cls_losses,
        "eval": trainer.eval_cls_losses
    },
        title="Classification Losses"
    )
    figs = [
        metric.plot(metric.values) for metric in eval_metrics
    ]
    if use_cls_metrics:
        figs = [cls_losses] + figs
    if use_cnt_metrics:
        figs = [cnt_losses] + figs

    plot_figs(
        figs,
        grid_shape=grid_shape,
        title=title
    )

def confusion_matrix_fig(cm, n_classes, title="Confusion Matrix"):
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=np.arange(n_classes),
        y=np.arange(n_classes),
        title=title
    )
    return fig