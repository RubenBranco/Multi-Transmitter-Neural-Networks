import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_transmitter_networks.layers import MultiTransmitterFeedForwardLayer
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


class MNISTModel(pl.LightningModule):
    def __init__(self, learning_rate: int, layers_config: list):
        super().__init__()
        self.lr = learning_rate
        self.layers = self.build_layers(layers_config)
        metrics = MetricCollection(
            Accuracy(num_classes=10),
            Precision(num_classes=10, average="macro"),
            Recall(num_classes=10, average="macro"),
            F1Score(num_classes=10, average="macro"),
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def build_layers(self, layers_config: list) -> nn.ModuleList:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)

        for l in self.layers:
            x = F.relu(l(x))

        return x

    def training_step(self, batch: tuple, _: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        self.log_dict(self.train_metrics(logits.softmax(dim=-1), y), prog_bar=True)
        return F.cross_entropy(logits, y)

    def validation_step(self, batch: tuple, _: int) -> None:
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)
        self.log_dict(self.val_metrics(logits.softmax(dim=-1), y), prog_bar=True)

    def test_step(self, batch: tuple, _: int) -> None:
        x, y = batch
        logits = self(x)
        self.log_dict(self.test_metrics(logits.softmax(dim=-1), y), prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class PTDenseModel(MNISTModel):
    def build_layers(self, layers_config: list) -> nn.ModuleList:
        # layers_config is a list of ints, indicating output size
        # first layer size
        fl_size = layers_config[0]
        layers = nn.ModuleList(
            [nn.Linear(28 * 28, fl_size)]
            + [
                nn.Linear(
                    layers_config[i - 1],  # input size from previous layer output size
                    out_size,
                )
                for i, out_size in enumerate(layers_config[1:], start=1)
            ]
        )
        return layers


class MultiTransmitterDenseModel(MNISTModel):
    def build_layers(self, layers_config: list) -> nn.ModuleList:
        # layers_config is a list of tuples, each tuple indicating the
        # output size and number of neurotransmitters
        # first layer config
        fl_size, fl_neurotrans = layers_config[0]
        layers = nn.ModuleList(
            [MultiTransmitterFeedForwardLayer(28 * 28, fl_size, fl_neurotrans)]
            + [
                MultiTransmitterFeedForwardLayer(
                    layers_config[i - 1][
                        0
                    ],  # input size from previous layer output size
                    out_size,
                    num_neurotransmitters,
                )
                for i, (out_size, num_neurotransmitters) in enumerate(
                    layers_config[1:], start=1
                )
            ]
        )
        return layers
