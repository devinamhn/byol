import pytorch_lightning as pl
import pytorch_lightning as pl
import torch
import torchmetrics as tm
import torch.nn as nn

from einops import rearrange
from torch import Tensor

from byol.ftheads import LogisticRegression, Matryoshka_CE_Loss
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context


class FineTuneMRL(pl.LightningModule):
    """
    Parent class for self-supervised LightningModules to perform MRL evaluation with multiple
    data-sets.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head,
        dim: int,
        n_classes,
        n_epochs=100,
        n_layers=0,
        batch_size=1024,
        lr_decay=0.75,
        seed=69,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["encoder", "head"])

        self.n_layers = n_layers
        self.batch_size = batch_size
        self.encoder = encoder
        self.lr_decay = lr_decay
        self.n_epochs = n_epochs
        self.seed = seed
        self.head = head
        self.n_classes = n_classes
        self.layers = []
        self.nesting_start = 1
        self.nesting_list = [2**i for i in range(self.nesting_start, 10)] #if self.nesting else None

        # Set head
        if head == "linear":
            self.head = LogisticRegression(input_dim=dim, output_dim=n_classes)
            self.head_type = "linear"
        elif isinstance(head, nn.Module):
            self.head = head
            self.head_type = "custom"
        else:
            raise ValueError("Head must be either 'linear' or a PyTorch Module")

        # Set finetuning layers for easy access
        if self.n_layers:
            layers = self.encoder.finetuning_layers
            assert self.n_layers <= len(
                layers
            ), f"Network only has {len(layers)} layers, {self.n_layers} specified for finetuning"

            self.layers = layers[::-1][:n_layers]

        self.train_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.val_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.test_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

    # this will return a tuple of logits of length = len(nesting_list)
    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        x = self.head(x)
        return x

    def on_fit_start(self):
        # Log size of data-sets #

        self.train_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)
        self.val_acc = tm.Accuracy(
            task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
        ).to(self.device)

        self.test_acc = nn.ModuleList(
            [
                tm.Accuracy(
                    task="multiclass", average="micro", threshold=0, num_classes=self.n_classes
                ).to(self.device)
            ]
            * len(self.trainer.datamodule.data["test"])
        )

        logging_params = {f"n_{key}": len(value) for key, value in self.trainer.datamodule.data.items()}
        self.logger.log_hyperparams(logging_params)

        # Make sure network that isn't being finetuned is frozen
        # probably unnecessary but best to be sure
        set_grads(self.encoder, False)
        if self.n_layers:
            for layer in self.layers:
                set_grads(layer, True)

    def training_step(self, batch, batch_idx):
        # Load data and targets
        x, y = batch
        #this returns a tuple of logits [len(nesting_list), batch, nclasses]
        logits = self.forward(x)
        
        # y_pred = torch.stack(logits, dim = 0).softmax(dim=-1)
        
        # loss = F.cross_entropy(y_pred, y, label_smoothing=0.1 if self.n_layers else 0)
        
        loss_fn = Matryoshka_CE_Loss(label_smoothing = 0.1 if self.n_layers else 0)
        loss = loss_fn.forward(logits, y)

        self.log("finetuning/train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        preds = self.forward(x)

        preds = torch.stack(preds, dim = 0).softmax(dim=-1)
        #this is calculated using torch metrics accuracy -- need to check if it can handle 
        #multiple outputs?
        #can log multiple train/test accuracies

        for i in range(len(preds)):
            self.val_acc(preds[i], y)
            self.log(f"finetuning/val_acc{i}", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        name = list(self.trainer.datamodule.data["test"].keys())[dataloader_idx]

        preds = self.forward(x)
        preds = torch.stack(preds, dim = 0).softmax(dim=-1)
        
        for i in range(len(preds)):

            self.test_acc[dataloader_idx](preds[i], y)

            self.log(
                f"finetuning/test/{name}_acc{i}",
                self.test_acc[dataloader_idx],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )

    def configure_optimizers(self):
        if not self.n_layers and self.head_type == "linear":
            # Scale base lr=0.1
            lr = 0.1 * self.batch_size / 256
            params = self.head.parameters()
            return torch.optim.SGD(params, momentum=0.9, lr=lr)
        else:
            lr = 0.001 * self.batch_size / 256
            params = [{"params": self.head.parameters(), "lr": lr}]
            # layers.reverse()

            # Append parameters of layers for finetuning along with decayed learning rate
            for i, layer in enumerate(self.layers):
                params.append({"params": layer.parameters(), "lr": lr * (self.lr_decay**i)})

            # Initialize AdamW optimizer with cosine decay learning rate
            opt = torch.optim.AdamW(params, weight_decay=0.05, betas=(0.9, 0.999))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.n_epochs)
            return [opt], [scheduler]


def set_grads(module, value: bool):
    for params in module.parameters():
        params.requires_grad = value