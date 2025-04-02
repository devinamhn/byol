import wandb
import pytorch_lightning as pl
import logging
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from byol.paths import Path_Handler
from byol.config import load_config_finetune_mrl
from byol.models import BYOL
from byol.datamodules import RGZ_DataModule_Finetune
from byol.ftheads import MLPHead, MRL_Linear_Layer
from byol.ftclass import FineTuneMRL

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def run_finetuning(config, encoder, datamodule, logger):
    checkpoint = ModelCheckpoint(
        monitor=None,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        # dirpath=config["files"] / config["run_id"] / "finetuning",
        # e.g. byol/files/(run_id)/checkpoints/12-344-18.134.ckpt.
        filename="{epoch}",  # filename may not work here TODO
        save_weights_only=True,
        # save_top_k=3,
    )

    callbacks = []

    early_stop_callback = EarlyStopping(
        monitor="finetuning/train_loss", min_delta=0.00, patience=3, verbose=True, mode="min"
    )

    if config["finetune"]["early_stopping"]:
        callbacks.append(early_stop_callback)

    ## Initialise pytorch lightning trainer ##
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config["finetune"]["n_epochs"],
        **config["trainer"],
    )

    # Initialize head
    if config["finetune"]["head"] == "linear":
        # head = LogisticRegression(input_dim=encoder.dim, output_dim=config["finetune"]["n_classes"])
        head = "linear"

    elif config["finetune"]["head"] == "mlp":
        head = MLPHead(
            input_dim=encoder.dim,
            depth=config["finetune"]["depth"],
            width=config["finetune"]["width"],
            output_dim=config["finetune"]["n_classes"],
        )

    elif config["finetune"]["head"] == 'mrl':
        nesting_start = 1
        nesting_list = [2**i for i in range(nesting_start, 10)]
        head = MRL_Linear_Layer(nesting_list, num_classes=config["finetune"]["n_classes"])
   
    else:
        raise ValueError("Head must be either linear or mlp or mrl")

    model = FineTuneMRL(
        encoder,
        head,
        dim=encoder.dim,
        n_classes=config["finetune"]["n_classes"],
        n_epochs=config["finetune"]["n_epochs"],
        n_layers=config["finetune"]["n_layers"],
        batch_size=config["finetune"]["batch_size"],
        lr_decay=config["finetune"]["lr_decay"],
        seed=config["seed"],
        head_type=config["finetune"]["head"],
    )

    trainer.fit(model, datamodule)

    trainer.test(model, dataloaders=datamodule)

    return checkpoint, model


def set_grads(module, value: bool):
    for params in module.parameters():
        params.requires_grad = value


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Load paths
    paths = Path_Handler()._dict()

    # Load up finetuning config
    config_finetune = load_config_finetune_mrl()

    ## Run finetuning ##
    for seed in range(config_finetune["finetune"]["iterations"]):
        # for seed in range(1, 10):

        if config_finetune["finetune"]["run_id"].lower() != "none":
            experiment_dir = paths["files"] / config_finetune["finetune"]["run_id"] / "checkpoints"
            model = BYOL.load_from_checkpoint(experiment_dir / "last.ckpt")
        else:
            model = BYOL.load_from_checkpoint("byol.ckpt")

        ## Load up config from model to save correct hparams for easy logging ##
        config = model.config
        config.update(config_finetune)
        config["finetune"]["dim"] = model.encoder.dim

        # Compatibility with old style config
        if config["augmentations"]["center_crop"] is True:
            config["augmentations"]["center_crop"] = config["augmentations"]["center_crop_size"]

        project_name = "BYOL_finetune_MRL"

        config["finetune"]["seed"] = seed
        pl.seed_everything(seed)

        # Initiate wandb logging
        wandb.init(project=project_name, config=config)

        logger = pl.loggers.WandbLogger(
            project=project_name,
            save_dir=paths["files"] / "finetune" / str(wandb.run.id),
            reinit=True,
            config=config,
        )

        finetune_datamodule = RGZ_DataModule_Finetune(
            paths["mb"],
            batch_size=config["finetune"]["batch_size"],
            center_crop=config["augmentations"]["center_crop"],
            val_size=config["finetune"]["val_size"],
            num_workers=config["dataloading"]["num_workers"],
            prefetch_factor=config["dataloading"]["prefetch_factor"],
            pin_memory=config["dataloading"]["pin_memory"],
            seed=config["finetune"]["seed"],
        )
        run_finetuning(config, model.encoder, finetune_datamodule, logger)
        logger.experiment.finish()
        wandb.finish()


if __name__ == "__main__":
    main()
