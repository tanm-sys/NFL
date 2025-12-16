#!/usr/bin/env python3
"""
Fine-tune NFL Trajectory Model from Best Checkpoint
====================================================

This script fine-tunes the best model checkpoint (epoch=04, ADE=0.5403)
with optimized hyperparameters to achieve better performance.

Improvements:
- 5x lower learning rate (0.0003)
- Increased weight decay (0.08)
- All 18 weeks of data
- Gradient accumulation for effective batch size 128
- Early stopping with patience=3
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    StochasticWeightAveraging,
)
from torch_geometric.loader import DataLoader as PyGDataLoader
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from src.train import NFLGraphPredictor
from src.data_loader import DataLoader, build_play_metadata, expand_play_tuples, GraphDataset

console = Console()

# Configuration
BEST_CHECKPOINT = "checkpoints/nfl_worldclass-epoch=04-val_ade=0.545.ckpt"
OUTPUT_DIR = Path("checkpoints_finetuned")
CACHE_DIR = Path("cache/finetune")

# Fine-tuning hyperparameters
FINETUNE_CONFIG = {
    "lr": 0.0003,              # 5x lower than original (0.0015)
    "weight_decay": 0.08,       # Increased from 0.05
    "max_epochs": 10,           # Short fine-tuning
    "batch_size": 32,           # Keep same
    "accumulate_grad_batches": 4,  # Effective batch = 128
    "early_stopping_patience": 3,
    "weeks": list(range(1, 19)),  # All 18 weeks
}


def create_dataloaders(weeks, batch_size, history_len=5, future_seq_len=10):
    """Create train/val dataloaders using all specified weeks."""
    loader = DataLoader(".")
    
    console.print(f"[cyan]Loading data for weeks: {weeks}[/cyan]")
    
    # Build play metadata
    play_meta = build_play_metadata(loader, weeks, history_len, future_seq_len)
    console.print(f"[green]Found {len(play_meta)} plays[/green]")
    
    # Split 80/20 by play
    split_idx = int(len(play_meta) * 0.8)
    train_meta = play_meta[:split_idx]
    val_meta = play_meta[split_idx:]
    
    # Expand to frame-level
    train_tuples = expand_play_tuples(train_meta)
    val_tuples = expand_play_tuples(val_meta)
    
    console.print(f"[green]Train samples: {len(train_tuples)}, Val samples: {len(val_tuples)}[/green]")
    
    # Create datasets
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    train_dataset = GraphDataset(
        loader=loader,
        play_tuples=train_tuples,
        radius=20.0,
        future_seq_len=future_seq_len,
        history_len=history_len,
        cache_dir=CACHE_DIR / "train",
        persist_cache=True,
        in_memory_cache_size=16,
    )
    
    val_dataset = GraphDataset(
        loader=loader,
        play_tuples=val_tuples,
        radius=20.0,
        future_seq_len=future_seq_len,
        history_len=history_len,
        cache_dir=CACHE_DIR / "val",
        persist_cache=True,
        in_memory_cache_size=16,
    )
    
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,  # Async prefetch 4 batches ahead
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
    )
    
    return train_loader, val_loader


def finetune():
    """Main fine-tuning function."""
    console.print(Panel.fit(
        "[bold green]🚀 Fine-tuning NFL Model[/bold green]\n"
        f"Base checkpoint: {BEST_CHECKPOINT}",
        border_style="green"
    ))
    
    # Print config
    console.print("\n[cyan]Fine-tuning Configuration:[/cyan]")
    for k, v in FINETUNE_CONFIG.items():
        console.print(f"  {k}: {v}")
    
    # Setup output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    console.print("\n[yellow]Loading base checkpoint...[/yellow]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = NFLGraphPredictor.load_from_checkpoint(
        BEST_CHECKPOINT,
        map_location=device,
        lr=FINETUNE_CONFIG["lr"],
        weight_decay=FINETUNE_CONFIG["weight_decay"],
    )
    
    console.print(f"[green]✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters[/green]")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        weeks=FINETUNE_CONFIG["weeks"],
        batch_size=FINETUNE_CONFIG["batch_size"],
    )
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_ade",
            mode="min",
            patience=FINETUNE_CONFIG["early_stopping_patience"],
            verbose=True,
        ),
        ModelCheckpoint(
            dirpath=OUTPUT_DIR,
            filename="finetuned-{epoch:02d}-{val_ade:.4f}",
            monitor="val_ade",
            mode="min",
            save_top_k=3,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        StochasticWeightAveraging(
            swa_lrs=1e-5,
            swa_epoch_start=int(FINETUNE_CONFIG["max_epochs"] * 0.6),
            annealing_epochs=2,
        ),
    ]
    
    # Setup trainer with full GPU optimization
    trainer = pl.Trainer(
        max_epochs=FINETUNE_CONFIG["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=callbacks,
        accumulate_grad_batches=FINETUNE_CONFIG["accumulate_grad_batches"],
        gradient_clip_val=1.0,
        precision="16-mixed",  # Mixed precision for Tensor Cores
        log_every_n_steps=50,
        enable_progress_bar=True,
        benchmark=True if torch.cuda.is_available() else False,  # cuDNN autotuner
        deterministic=False,  # Non-deterministic for max speed
    )
    
    # torch.compile for 2x speedup (PyTorch 2.0+)
    if torch.cuda.is_available() and hasattr(torch, 'compile'):
        try:
            console.print("[yellow]⚡ Compiling model with torch.compile...[/yellow]")
            model.model = torch.compile(model.model, mode="reduce-overhead")
            console.print("[green]✅ Model compiled for maximum performance![/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ torch.compile failed: {e}[/yellow]")
    
    console.print("\n[bold green]Starting fine-tuning...[/bold green]\n")
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Get best result
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_val_ade = trainer.checkpoint_callback.best_model_score
    
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        f"[bold green]✅ Fine-tuning Complete![/bold green]\n\n"
        f"Best checkpoint: {best_model_path}\n"
        f"Best val_ade: {best_val_ade:.4f}\n"
        f"Baseline was: 0.5403",
        border_style="green"
    ))
    
    # Calculate improvement
    if best_val_ade:
        improvement = ((0.5403 - best_val_ade) / 0.5403) * 100
        if improvement > 0:
            console.print(f"[bold green]📈 Improvement: {improvement:.1f}%[/bold green]")
        else:
            console.print(f"[yellow]No improvement achieved (Δ = {improvement:.1f}%)[/yellow]")
    
    # Save config
    config_path = OUTPUT_DIR / "finetune_config.json"
    with open(config_path, "w") as f:
        json.dump(FINETUNE_CONFIG, f, indent=2)
    console.print(f"\nConfig saved to: {config_path}")
    
    return best_model_path, best_val_ade


if __name__ == "__main__":
    try:
        best_path, best_ade = finetune()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
