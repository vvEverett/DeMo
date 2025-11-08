#!/usr/bin/env python3
"""
Training script for the GMM predictor model.
Use GMM (Gaussian Mixture Model) as the predictor head as a baseline.
"""
import os
import sys
import hydra
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


@hydra.main(version_base=None, config_path="conf", config_name="config_gmm")
def main(conf):
    """Main training function for GMM baseline model.
    
    Args:
        conf: Hydra configuration object containing all training parameters
    """
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    # Setup loggers
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    
    # Configure wandb logger
    wandb_logger = WandbLogger(
        project=conf.get('wandb_project', 'DeMo-GMM'),
        name=conf.get('wandb_run_name', None),
        save_dir=output_dir,
        log_model=False,
    )
    
    # Use both loggers
    logger = [tb_logger, wandb_logger]

    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="{epoch}",
            monitor=f"{conf.monitor}",
            mode="min",
            save_top_k=conf.save_top_k,
            save_last=True,
        ),
        RichModelSummary(max_depth=1),
        RichProgressBar(),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Use auto strategy for single GPU or ddp_find_unused_parameters_true for multi-GPU
    strategy = "auto" if conf.gpus == 1 else "ddp_find_unused_parameters_true"
    
    trainer = pl.Trainer(
        logger=logger,
        gradient_clip_val=conf.gradient_clip_val,
        gradient_clip_algorithm=conf.gradient_clip_algorithm,
        max_epochs=conf.epochs,
        accelerator="gpu",
        devices=conf.gpus,
        strategy=strategy,
        callbacks=callbacks,
        limit_train_batches=conf.limit_train_batches,
        limit_val_batches=conf.limit_val_batches,
        sync_batchnorm=conf.sync_bn,
    )

    model = instantiate(conf.model.target)
    os.system('cp -a %s %s' % ('conf', output_dir))
    os.system('cp -a %s %s' % ('src', output_dir))
    
    with open(f'{output_dir}/model.txt', 'w') as f:
        original_stdout = sys.stdout  
        sys.stdout = f  
        print(model)  
        sys.stdout = original_stdout
    
    print("="*80)
    print("TRAINING GMM PREDICTOR MODEL")
    print("="*80)
    print(f"Model: Mode Localization Module with GMM Predictor")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {conf.epochs}")
    print(f"Learning rate: {conf.lr}")
    print(f"Batch size: {conf.batch_size}")
    print(f"GPUs: {conf.gpus}")
    print("="*80)
    
    datamodule = instantiate(conf.datamodule.target)
    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)
    trainer.validate(model, datamodule.val_dataloader())


if __name__ == "__main__":
    main()
