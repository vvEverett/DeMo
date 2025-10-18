#!/usr/bin/env python3
"""
Training script for the MoE (Mixture of Experts) predictor model.
Replace GMM with MoE in the original script.
"""
import os
import sys
import torch
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
from pytorch_lightning.strategies import DDPStrategy


@hydra.main(version_base=None, config_path="conf", config_name="config_moe_supervised")
def main(conf):
    """
    Main training script for DeMo with Supervised Mixture of Experts (MoE).
    
    Architecture:
    - 1 Shared Expert: learns general patterns from ALL data (30% weight)
    - 5 Unshared Experts: specialize in specific behaviors (70% weight, top-2 activation)
      * Expert 1: Lane Keeping (Straight/Lane Change)
      * Expert 2: Turn Left
      * Expert 3: Turn Right
      * Expert 4: Constraint-Driven Deceleration (Stop/Yield/Junction)
      * Expert 5: Others (Long-tail behaviors)
    
    Training:
    - Shared expert trained on all samples
    - Unshared experts trained on their assigned samples via supervised routing
    - Router supervised by ground truth expert labels from classification CSV
    """
    pl.seed_everything(conf.seed, workers=True)
    output_dir = HydraConfig.get().runtime.output_dir

    # Setup loggers
    tb_logger = TensorBoardLogger(save_dir=output_dir, name="logs")
    
    # Configure wandb logger only on rank 0 to avoid port conflicts in distributed training
    loggers = [tb_logger]
    
    # Only initialize WandB on the main process (rank 0) in distributed training
    # This prevents multiple processes from trying to start WandB service simultaneously
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        try:
            wandb_logger = WandbLogger(
                project=conf.get('wandb_project', 'DeMo'),
                name=conf.get('wandb_run_name', None),
                save_dir=output_dir,
                log_model=False,  # Set to True if you want to save models to wandb
            )
            loggers.append(wandb_logger)
            print("WandB logger initialized successfully on rank 0")
        except Exception as e:
            print(f"Warning: Failed to initialize WandB logger: {e}")
            print("Continuing with TensorBoard logger only")
    
    logger = loggers

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

    # For MoE models, we need to set find_unused_parameters=True in DDPStrategy
    # because not all experts are used in every forward pass (only top-k are activated)
    strategy = "auto" if conf.gpus == 1 else DDPStrategy(find_unused_parameters=True)
    
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
    
    # Backup configuration and source code
    os.system('cp -a %s %s' % ('conf', output_dir))
    os.system('cp -a %s %s' % ('src', output_dir))
    
    with open(f'{output_dir}/model.txt', 'w') as f:
        original_stdout = sys.stdout  
        sys.stdout = f  
        print(model)  
        sys.stdout = original_stdout
    
    # Print MoE-specific information
    print("="*80)
    print("TRAINING SUPERVISED MOE PREDICTOR MODEL")
    print("="*80)
    print(f"Model: Mode Localization Module with Supervised MoE Predictor")
    print(f"Architecture: 1 Shared Expert (30%) + 5 Unshared Experts (70%, Top-2)")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {conf.epochs}")
    print(f"Learning rate: {conf.lr}")
    print(f"Batch size: {conf.batch_size}")
    print(f"GPUs: {conf.gpus}")
    
    # Print MoE architecture details
    try:
        print(f"\n--- Supervised MoE Architecture ---")
        print(f"Shared expert weight: {model.net.time_decoder.predictor.shared_weight * 100:.0f}%")
        print(f"Unshared expert weight: {model.net.time_decoder.predictor.unshared_weight * 100:.0f}%")
        print(f"Number of unshared experts: {model.net.time_decoder.predictor.num_unshared_experts}")
        print(f"Top-K activation: {model.net.time_decoder.predictor.top_k}")
        print(f"Embedding dimension: {conf.model.target.model.embed_dim}")
        print(f"Future steps: {conf.model.target.model.future_steps}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        shared_expert_params = sum(p.numel() for p in model.net.time_decoder.predictor.shared_expert.parameters())
        unshared_expert_params = sum(p.numel() for p in model.net.time_decoder.predictor.unshared_experts.parameters())
        router_params = sum(p.numel() for p in model.net.time_decoder.predictor.router.parameters())
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Shared expert parameters: {shared_expert_params:,}")
        print(f"Unshared experts parameters: {unshared_expert_params:,} ({unshared_expert_params // 5:,} per expert)")
        print(f"Router parameters: {router_params:,}")
        
        print(f"\n--- Regularization Techniques ---")
        print(f"Router Gumbel noise std: {model.net.time_decoder.predictor.router.noise_std:.2f}")
        print(f"Load balancing weight: {model.net.time_decoder.predictor.load_balance_weight:.3f}")
        print(f"Router loss weight: {conf.router_loss_weight:.2f}")
        print(f"Auxiliary loss weight: {conf.aux_loss_weight:.3f}")
        print(f"Expert-specific loss weight: {conf.expert_loss_weight:.2f}")
        
        print(f"\n--- Expert Definitions ---")
        print(f"Expert 1: Lane Keeping (Straight/Lane Change)")
        print(f"Expert 2: Turn Left")
        print(f"Expert 3: Turn Right")
        print(f"Expert 4: Constraint-Driven Deceleration")
        print(f"Expert 5: Others (Long-tail behaviors)")
    except Exception as e:
        print(f"Could not extract MoE architecture details: {e}")
    
    print("="*80)
    
    datamodule = instantiate(conf.datamodule.target)
    trainer.fit(model, datamodule, ckpt_path=conf.checkpoint)
    trainer.validate(model, datamodule.val_dataloader())


if __name__ == "__main__":
    main()
