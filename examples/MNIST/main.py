import click
import pytorch_lightning as pl

from data import MNISTDataModule
from models import MultiTransmitterDenseModel, PTDenseModel


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--accelerator",
    default="auto",
    type=click.Choice(["cpu", "gpu", "auto"], case_sensitive=True),
    help="Accelerator to run on. Auto will choose gpu if available, else cpu.",
)
@click.option("--epochs", default=10, help="Maximum number of epochs to train.")
@click.option("--learning_rate", default=1e-3, help="Learning rate.")
@click.option("--batch_size", default=32, help="Batch size.")
@click.option("--early_stopping", is_flag=True, help="Whether to do early stopping.")
@click.option("--patience", default=3, help="Early stopping patience.")
@click.option("--monitor_metric", default="val_loss", help="Early stopping metric.")
@click.option(
    "--deterministic", is_flag=True, help="Whether to make everything deterministic."
)
@click.option("--seed", default=42, help="Random seed.")
@click.option(
    "--layer_config",
    default=[100, 10],
    type=int,
    multiple=True,
    help="Sequence of output layer sizes.",
)
def train_and_test_pytorch_model(
    accelerator,
    epochs,
    learning_rate,
    batch_size,
    early_stopping,
    patience,
    monitor_metric,
    deterministic,
    seed,
    layer_config,
):
    pl.seed_everything(seed)
    data_module = MNISTDataModule(seed, batch_size)
    model = PTDenseModel(learning_rate, list(layer_config))
    callbacks = []
    if early_stopping:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=monitor_metric, mode="min", patience=patience
            )
        )
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
        deterministic=deterministic,
        max_epochs=epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


@click.command()
@click.option(
    "--accelerator",
    default="auto",
    type=click.Choice(["cpu", "gpu", "auto"], case_sensitive=True),
    help="Accelerator to run on. Auto will choose gpu if available, else cpu.",
)
@click.option("--epochs", default=10, help="Maximum number of epochs to train.")
@click.option("--learning_rate", default=1e-3, help="Learning rate.")
@click.option("--batch_size", default=32, help="Batch size.")
@click.option("--early_stopping", is_flag=True, help="Whether to do early stopping.")
@click.option("--patience", default=3, help="Early stopping patience.")
@click.option("--monitor_metric", default="val_loss", help="Early stopping metric.")
@click.option(
    "--deterministic", is_flag=True, help="Whether to make everything deterministic."
)
@click.option("--seed", default=42, help="Random seed.")
@click.option(
    "--layer_config",
    default=[(17, 3), (10, 3)],
    type=(int, int),
    multiple=True,
    help="Sequence of output layer sizes and number of neurotransmitters.",
)
def train_and_test_multi_transmitter_model(
    accelerator,
    epochs,
    learning_rate,
    batch_size,
    early_stopping,
    patience,
    monitor_metric,
    deterministic,
    seed,
    layer_config,
):
    print(layer_config)
    pl.seed_everything(seed)
    data_module = MNISTDataModule(seed, batch_size)
    model = MultiTransmitterDenseModel(learning_rate, list(layer_config))
    callbacks = []
    if early_stopping:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=monitor_metric, mode="min", patience=patience
            )
        )
    trainer = pl.Trainer(
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
        deterministic=deterministic,
        max_epochs=epochs,
        callbacks=callbacks,
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    cli.add_command(train_and_test_pytorch_model)
    cli.add_command(train_and_test_multi_transmitter_model)
    cli()
