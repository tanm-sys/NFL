#!/usr/bin/env python3
"""
Production-Ready NFL Analytics Engine Training Script
=====================================================

Features:
- Comprehensive experiment tracking (MLflow + W&B)
- Automatic model versioning and checkpointing
- Data validation and quality checks
- Robust error handling and recovery
- Performance monitoring and profiling
- Multi-GPU support with DDP
- Mixed precision training (FP16/BF16)
- Gradient accumulation for large batches
- Learning rate scheduling with warmup
- Model export (ONNX, TorchScript)
- Automated hyperparameter tuning
- Production-ready logging and metrics

Usage:
    python train_production.py --config configs/production.yaml
    python train_production.py --mode tune --trials 20
    python train_production.py --resume-from checkpoint.ckpt
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger, WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    LearningRateMonitor,
    ModelSummary,
    RichProgressBar,
    RichModelSummary,
    DeviceStatsMonitor,
    GradientAccumulationScheduler,
    StochasticWeightAveraging  # SWA for better generalization
)
from pytorch_lightning.strategies import DDPStrategy
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Batch
import numpy as np
import polars as pl_df
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from loguru import logger
import mlflow
import wandb
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Project imports
from src.data_loader import (
    DataLoader,
    GraphDataset,
    build_play_metadata,
    expand_play_tuples,
)
from src.train import NFLGraphPredictor, AttentionVisualizationCallback
from src.models.gnn import NFLGraphTransformer

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# HARDWARE OPTIMIZATIONS FOR RTX 4050 (Ada Lovelace Architecture)
# ============================================================================
# Enable Tensor Cores - 'high' for best quality/speed balance on RTX 40 series
torch.set_float32_matmul_precision('high')

# Enable cuDNN optimizations for maximum GPU utilization
torch.backends.cudnn.benchmark = True   # Auto-tune convolution algorithms
torch.backends.cudnn.enabled = True     # Ensure cuDNN is enabled
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matmuls
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 in cuDNN

# Flag for torch.compile (PyTorch 2.0+ JIT compilation)
# DISABLED: torch.compile is incompatible with PyTorch Geometric
# PyG dynamically clones tensors during batching which breaks CUDA graph capture
# Even with cudagraphs=False, the inductor can still cause issues with PyG
# Re-enable when PyG adds native torch.compile support
USE_TORCH_COMPILE = False

# Note: The following optimizations are still active and provide good speedup:
# - TF32 matmul precision (line 77)
# - cuDNN benchmark mode (line 80)
# - Mixed precision training (bf16-mixed via Trainer)

# Rich console for beautiful output
console = Console()


def detect_optimal_precision() -> str:
    """
    Automatically detect the optimal precision based on GPU capabilities.
    
    Returns:
        str: Optimal precision string ('bf16-mixed', '16-mixed', or '32')
    
    Precision selection logic:
    - RTX 40 series (Ada Lovelace) / A100 / H100: bf16-mixed (best performance)
    - RTX 30 series (Ampere): bf16-mixed (good performance)
    - RTX 20 series (Turing) and older: 16-mixed (fp16) or 32 (fp32)
    - CPU: 32 (fp32)
    """
    if not torch.cuda.is_available():
        return "32"  # CPU: use fp32
    
    try:
        # Get GPU compute capability
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        gpu_name = props.name
        
        # Check compute capability (major, minor)
        # Ampere (RTX 30/40, A100): 8.0+
        # Ada Lovelace (RTX 40): 8.9
        # Turing (RTX 20): 7.5
        # Volta (V100): 7.0
        compute_capability = props.major + props.minor / 10.0
        
        # Check if bf16 is supported (Ampere+ with compute capability >= 8.0)
        if compute_capability >= 8.0:
            # Verify bf16 support by checking if torch supports it
            if hasattr(torch.cuda, 'is_bf16_supported') and torch.cuda.is_bf16_supported():
                return "bf16-mixed"
            # Fallback: try to create a bf16 tensor to test support
            try:
                test_tensor = torch.tensor([1.0], dtype=torch.bfloat16, device='cuda')
                del test_tensor
                return "bf16-mixed"
            except (RuntimeError, TypeError):
                pass
        
        # For older GPUs (Turing, Pascal, etc.), use fp16 if available
        if compute_capability >= 7.0:
            return "16-mixed"  # fp16 mixed precision
        
        # Very old GPUs or fallback
        return "32"  # fp32 full precision
        
    except Exception as e:
        logger.warning(f"Could not detect optimal precision: {e}. Defaulting to 16-mixed")
        return "16-mixed"


class ProductionConfig:
    """Production training configuration with validation."""
    
    def __init__(self, **kwargs):
        # Data Configuration
        self.data_dir = kwargs.get('data_dir', '.')
        self.weeks = kwargs.get('weeks', list(range(1, 19)))
        self.train_split = kwargs.get('train_split', 0.8)
        self.val_split = kwargs.get('val_split', 0.1)
        self.test_split = kwargs.get('test_split', 0.1)
        
        # Model Architecture
        self.input_dim = kwargs.get('input_dim', 9)
        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.num_gnn_layers = kwargs.get('num_gnn_layers', 4)
        self.heads = kwargs.get('heads', 4)
        self.future_seq_len = kwargs.get('future_seq_len', 10)
        self.probabilistic = kwargs.get('probabilistic', False)
        self.num_modes = kwargs.get('num_modes', 6)
        self.dropout = kwargs.get('dropout', 0.1)
        
        # Training Configuration
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.max_epochs = kwargs.get('max_epochs', 100)
        self.min_epochs = kwargs.get('min_epochs', 10)
        self.num_workers = kwargs.get('num_workers', 4)
        self.accumulate_grad_batches = kwargs.get('accumulate_grad_batches', 1)
        self.limit_train_batches = kwargs.get('limit_train_batches', 1.0)
        self.limit_val_batches = kwargs.get('limit_val_batches', 1.0)
        self.limit_test_batches = kwargs.get('limit_test_batches', 1.0)
        
        # Optimization
        self.optimizer = kwargs.get('optimizer', 'adamw')
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.gradient_clip_val = kwargs.get('gradient_clip_val', 1.0)
        self.lr_scheduler = kwargs.get('lr_scheduler', 'cosine_warmup')
        self.warmup_epochs = kwargs.get('warmup_epochs', 5)
        
        # Loss Weights
        self.velocity_weight = kwargs.get('velocity_weight', 0.3)
        self.acceleration_weight = kwargs.get('acceleration_weight', 0.1)
        self.collision_weight = kwargs.get('collision_weight', 0.05)
        self.coverage_weight = kwargs.get('coverage_weight', 0.5)
        
        # Regularization
        self.use_augmentation = kwargs.get('use_augmentation', True)
        self.use_huber_loss = kwargs.get('use_huber_loss', False)
        self.huber_delta = kwargs.get('huber_delta', 1.0)
        self.label_smoothing = kwargs.get('label_smoothing', 0.0)
        
        # Hardware
        precision_input = kwargs.get('precision', 'auto')
        if precision_input == 'auto' or precision_input is None:
            self.precision = detect_optimal_precision()
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Auto-detected precision: {self.precision} for GPU: {gpu_name}")
        else:
            self.precision = precision_input
        self.accelerator = kwargs.get('accelerator', 'auto')
        self.devices = kwargs.get('devices', 'auto')
        self.strategy = kwargs.get('strategy', 'auto')
        
        # Callbacks
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.save_top_k = kwargs.get('save_top_k', 3)
        self.monitor_metric = kwargs.get('monitor_metric', 'val_ade')
        
        # Experiment Tracking
        self.experiment_name = kwargs.get('experiment_name', f'nfl_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.use_mlflow = kwargs.get('use_mlflow', True)
        self.use_wandb = kwargs.get('use_wandb', True)
        self.use_tensorboard = kwargs.get('use_tensorboard', True)
        
        # Paths
        self.checkpoint_dir = kwargs.get('checkpoint_dir', './checkpoints')
        self.log_dir = kwargs.get('log_dir', './logs')
        self.output_dir = kwargs.get('output_dir', './outputs')
        
        # Reproducibility
        self.seed = kwargs.get('seed', 42)
        self.deterministic = kwargs.get('deterministic', True)
        
        # Production Features
        self.enable_profiling = kwargs.get('enable_profiling', False)
        self.export_onnx = kwargs.get('export_onnx', True)
        self.export_torchscript = kwargs.get('export_torchscript', True)
        self.validate_data = kwargs.get('validate_data', True)
        # Skip expensive sample-batch warmup unless explicitly enabled.
        self.enable_sample_batch_warmup = kwargs.get('enable_sample_batch_warmup', False)
        
    def validate(self):
        """Validate configuration parameters."""
        assert self.train_split + self.val_split + self.test_split == 1.0, "Splits must sum to 1.0"
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.max_epochs > self.min_epochs, "Max epochs must be greater than min epochs"
        
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def save(self, path: str):
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class DataValidator:
    """Validate data quality and integrity."""
    
    @staticmethod
    def validate_dataframe(df: pl_df.DataFrame, week: int) -> Tuple[bool, List[str]]:
        """Validate loaded dataframe."""
        issues = []
        
        # Check required columns (dis is optional - can be derived from speed)
        required_cols = ['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y', 's', 'a', 'o', 'dir']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        # Check for nulls in critical columns
        for col in ['x', 'y', 's', 'a']:
            if col in df.columns:
                null_count = df[col].null_count()
                if null_count > 0:
                    issues.append(f"Column '{col}' has {null_count} null values")
        
        # Check data ranges
        if 'x' in df.columns:
            x_min, x_max = df['x'].min(), df['x'].max()
            if x_min < 0 or x_max > 120:
                issues.append(f"X coordinates out of field bounds: [{x_min}, {x_max}]")
        
        if 'y' in df.columns:
            y_min, y_max = df['y'].min(), df['y'].max()
            if y_min < 0 or y_max > 53.3:
                issues.append(f"Y coordinates out of field bounds: [{y_min}, {y_max}]")
        
        # Check for duplicate frames
        if all(col in df.columns for col in ['game_id', 'play_id', 'nfl_id', 'frame_id']):
            duplicates = df.group_by(['game_id', 'play_id', 'nfl_id', 'frame_id']).len()
            duplicates = duplicates.filter(pl_df.col('len') > 1)
            if len(duplicates) > 0:
                issues.append(f"Found {len(duplicates)} duplicate frames")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class EpochSummaryCallback(pl.Callback):
    """
    Custom callback for detailed, clean epoch summaries.
    Displays training progress, metrics, GPU stats, and time estimates.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epoch_start_time = None
        self.training_start_time = None
        self.best_val_ade = float('inf')
        self.best_epoch = 0
        self.epoch_times = []
        
    def on_train_start(self, trainer, pl_module):
        """Called when training begins."""
        self.training_start_time = datetime.now()
        
        # Print training header
        console.print("\n" + "="*70, style="bold cyan")
        console.print("🏈 NFL ANALYTICS ENGINE - TRAINING STARTED", style="bold white", justify="center")
        console.print("="*70, style="bold cyan")
        
        # GPU Info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(f"\n🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)", style="green")
        
        # Model info
        total_params = sum(p.numel() for p in pl_module.parameters())
        console.print(f"🧠 Model: {total_params:,} parameters", style="green")
        console.print(f"📊 Batch Size: {self.config.batch_size} | LR: {self.config.learning_rate}", style="green")
        console.print(f"🎯 Target: val_ade < {self.config.monitor_metric} with patience={self.config.early_stopping_patience}", style="green")
        console.print("")
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each training epoch."""
        self.epoch_start_time = datetime.now()
        
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch - before validation."""
        pass  # We'll log after validation
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation - display epoch summary."""
        if trainer.sanity_checking:
            return
            
        epoch = trainer.current_epoch
        epoch_time = (datetime.now() - self.epoch_start_time).total_seconds()
        self.epoch_times.append(epoch_time)
        
        # Get metrics
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss', 0)
        val_ade = metrics.get('val_ade', 0)
        val_fde = metrics.get('val_fde', 0)
        val_miss = metrics.get('val_miss_rate_2yd', 0)
        val_cov = metrics.get('val_cov_acc', 0)
        lr = trainer.optimizers[0].param_groups[0]['lr']
        
        # Track best
        is_best = False
        if val_ade < self.best_val_ade and val_ade > 0:
            self.best_val_ade = val_ade
            self.best_epoch = epoch
            is_best = True
        
        # Calculate progress
        progress_pct = (epoch + 1) / trainer.max_epochs * 100
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = trainer.max_epochs - epoch - 1
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        
        # GPU Memory
        gpu_mem_used = 0
        gpu_util = 0
        if torch.cuda.is_available():
            gpu_mem_used = torch.cuda.memory_allocated(0) / 1e9
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_util = (gpu_mem_used / gpu_mem_total) * 100
        
        # Create summary table
        console.print("")
        console.print("─" * 70, style="dim")
        
        # Epoch header with progress bar
        progress_bar = "█" * int(progress_pct / 5) + "░" * (20 - int(progress_pct / 5))
        best_marker = "⭐ NEW BEST!" if is_best else ""
        console.print(f"📈 Epoch {epoch:03d}/{trainer.max_epochs} [{progress_bar}] {progress_pct:.1f}% {best_marker}", 
                     style="bold green" if is_best else "bold white")
        
        # Metrics table
        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        
        table.add_row(
            "Train Loss", f"{float(train_loss):.4f}",
            "Val ADE", f"[{'green' if is_best else 'white'}]{float(val_ade):.4f} yds[/]"
        )
        table.add_row(
            "Val FDE", f"{float(val_fde):.4f} yds",
            "Miss Rate", f"{float(val_miss)*100:.1f}%"
        )
        table.add_row(
            "Coverage Acc", f"{float(val_cov)*100:.1f}%",
            "Learning Rate", f"{lr:.2e}"
        )
        table.add_row(
            "GPU Memory", f"{gpu_mem_used:.1f}/{gpu_mem_total:.1f} GB ({gpu_util:.0f}%)" if torch.cuda.is_available() else "N/A",
            "Epoch Time", f"{epoch_time:.1f}s"
        )
        table.add_row(
            "Best ADE", f"[green]{self.best_val_ade:.4f}[/] (ep {self.best_epoch})",
            "ETA", f"{eta_str}"
        )
        
        console.print(table)
        console.print("─" * 70, style="dim")
        
    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        total_time = datetime.now() - self.training_start_time
        
        console.print("\n" + "="*70, style="bold green")
        console.print("✅ TRAINING COMPLETE", style="bold white", justify="center")
        console.print("="*70, style="bold green")
        
        # Final summary table
        table = Table(title="📊 Final Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("Best val_ade", f"{self.best_val_ade:.4f} yards")
        table.add_row("Best Epoch", str(self.best_epoch))
        table.add_row("Total Epochs", str(trainer.current_epoch + 1))
        table.add_row("Total Time", str(total_time).split('.')[0])
        table.add_row("Avg Epoch Time", f"{sum(self.epoch_times)/len(self.epoch_times):.1f}s")
        
        if torch.cuda.is_available():
            table.add_row("Peak GPU Memory", f"{torch.cuda.max_memory_allocated(0)/1e9:.2f} GB")
        
        console.print(table)
        console.print("")


class ProductionTrainer:
    """Production-ready training orchestrator."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.config.validate()
        
        # Set random seeds for reproducibility
        pl.seed_everything(config.seed, workers=True)
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize trackers
        self.loggers = self._setup_loggers()
        
        logger.info("Production Trainer initialized")
        logger.info(f"Experiment: {config.experiment_name}")
        
    def _setup_logging(self):
        """Configure production logging."""
        log_file = Path(self.config.log_dir) / f"{self.config.experiment_name}.log"
        logger.add(
            log_file,
            rotation="500 MB",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
    def _setup_loggers(self) -> List:
        """Setup experiment tracking loggers."""
        loggers = []
        
        if self.config.use_tensorboard:
            tb_logger = TensorBoardLogger(
                save_dir=self.config.log_dir,
                name=self.config.experiment_name,
                version="tensorboard"
            )
            loggers.append(tb_logger)
            logger.info("TensorBoard logging enabled")
        
        if self.config.use_mlflow:
            mlflow_logger = MLFlowLogger(
                experiment_name=self.config.experiment_name,
                tracking_uri="file:./mlruns",
                save_dir=self.config.log_dir
            )
            loggers.append(mlflow_logger)
            logger.info("MLflow logging enabled")
        
        if self.config.use_wandb:
            wandb_logger = WandbLogger(
                project="nfl-analytics-production",
                name=self.config.experiment_name,
                save_dir=self.config.log_dir,
                log_model=True
            )
            loggers.append(wandb_logger)
            logger.info("W&B logging enabled")
        
        return loggers
    
    def load_and_validate_data(self):
        """Load metadata and prepare streaming datasets with deterministic splits."""
        console.print("\n[bold cyan]Loading and Validating Data...[/bold cyan]")
        
        loader = DataLoader(self.config.data_dir)
        data_stats = []
        play_meta = []

        for week in self.config.weeks:
            try:
                logger.info(f"Loading week {week}...")
                df = loader.load_week_data(week)
                df = loader.standardize_tracking_directions(df)

                if self.config.validate_data:
                    is_valid, issues = DataValidator.validate_dataframe(df, week)
                    if not is_valid:
                        logger.warning(f"Week {week} validation issues: {issues}")
                        console.print(f"[yellow]⚠️  Week {week} has validation issues[/yellow]")

                counts = (
                    df.group_by(["game_id", "play_id"])
                    .agg(pl_df.col("frame_id").n_unique().alias("n_frames"))
                    .sort(["game_id", "play_id"])
                )

                for game_id, play_id, n_frames in counts.iter_rows():
                    num_graphs = max(0, n_frames - 5 - self.config.future_seq_len)
                    play_meta.append((week, int(game_id), int(play_id), int(num_graphs)))

                data_stats.append({
                    'week': week,
                    'rows': df.shape[0],
                    'graphs': int(counts["n_frames"].sum()),
                    'games': df['game_id'].n_unique(),
                    'plays': df['play_id'].n_unique()
                })

                logger.info(f"Week {week}: {counts['n_frames'].sum()} frames from {df.shape[0]} rows")

            except FileNotFoundError:
                logger.warning(f"Week {week} data not found, skipping...")
                console.print(f"[yellow]⚠️  Week {week} not found[/yellow]")
                continue
            except Exception as e:
                logger.error(f"Error loading week {week}: {e}")
                console.print(f"[red]❌ Error loading week {week}: {e}[/red]")
                continue

        self._display_data_summary(data_stats, sum([m[3] for m in play_meta]))

        train_ds, val_ds, test_ds = self._split_data(loader, play_meta)
        logger.info(f"Data split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
        return train_ds, val_ds, test_ds
    
    def _display_data_summary(self, stats: List[Dict], total_graphs: int):
        """Display beautiful data summary table."""
        table = Table(title="📊 Data Loading Summary", show_header=True, header_style="bold magenta")
        table.add_column("Week", style="cyan", justify="center")
        table.add_column("Rows", justify="right")
        table.add_column("Graphs", justify="right")
        table.add_column("Games", justify="right")
        table.add_column("Plays", justify="right")
        
        for stat in stats:
            table.add_row(
                str(stat['week']),
                f"{stat['rows']:,}",
                f"{stat['graphs']:,}",
                str(stat['games']),
                str(stat['plays'])
            )
        
        table.add_row(
            "[bold]TOTAL[/bold]",
            "-",
            f"[bold]{total_graphs:,}[/bold]",
            "-",
            "-",
            style="bold green"
        )
        
        console.print(table)
    
    def _split_data(self, loader: DataLoader, play_meta: List[Tuple[int, int, int, int]]):
        """Deterministic split by play_id with streaming datasets."""
        unique_plays = [(w, g, p) for (w, g, p, n) in play_meta if n > 0]
        unique_plays = sorted(unique_plays)

        split_file = Path(self.config.output_dir) / "splits_production.json"
        if split_file.exists():
            with open(split_file, "r") as f:
                split_data = json.load(f)
            train_keys = {tuple(x) for x in split_data.get("train", [])}
            val_keys = {tuple(x) for x in split_data.get("val", [])}
            test_keys = {tuple(x) for x in split_data.get("test", [])}
            logger.info(f"Loaded splits from {split_file}")
        else:
            rng = np.random.default_rng(self.config.seed)
            play_keys_shuffled = unique_plays.copy()
            rng.shuffle(play_keys_shuffled)
            n_total = len(play_keys_shuffled)
            train_idx = int(self.config.train_split * n_total)
            val_idx = int((self.config.train_split + self.config.val_split) * n_total)
            train_keys = set(play_keys_shuffled[:train_idx])
            val_keys = set(play_keys_shuffled[train_idx:val_idx])
            test_keys = set(play_keys_shuffled[val_idx:])
            split_file.parent.mkdir(parents=True, exist_ok=True)
            with open(split_file, "w") as f:
                json.dump(
                    {
                        "train": [list(x) for x in train_keys],
                        "val": [list(x) for x in val_keys],
                        "test": [list(x) for x in test_keys],
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved splits to {split_file}")

        train_tuples = expand_play_tuples(play_meta, allowed_plays=train_keys)
        val_tuples = expand_play_tuples(play_meta, allowed_plays=val_keys)
        test_tuples = expand_play_tuples(play_meta, allowed_plays=test_keys)

        cache_dir = Path("./cache/graphs")
        train_ds = GraphDataset(
            loader,
            train_tuples,
            radius=20.0,
            future_seq_len=self.config.future_seq_len,
            history_len=5,
            cache_dir=cache_dir,
            persist_cache=True,
            in_memory_cache_size=6,
        )
        val_ds = GraphDataset(
            loader,
            val_tuples,
            radius=20.0,
            future_seq_len=self.config.future_seq_len,
            history_len=5,
            cache_dir=cache_dir,
            persist_cache=True,
            in_memory_cache_size=3,
        )
        test_ds = GraphDataset(
            loader,
            test_tuples,
            radius=20.0,
            future_seq_len=self.config.future_seq_len,
            history_len=5,
            cache_dir=cache_dir,
            persist_cache=True,
            in_memory_cache_size=2,
        )

        return train_ds, val_ds, test_ds
    
    def create_dataloaders(self, train_data: List, val_data: List, test_data: List) -> Tuple:
        """Create production dataloaders with optimizations."""
        train_loader = PyGDataLoader(
            train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False,
            prefetch_factor=2 if self.config.num_workers > 0 else None
        )
        
        val_loader = PyGDataLoader(
            val_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        
        test_loader = PyGDataLoader(
            test_data,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def create_model(self) -> NFLGraphPredictor:
        """Create model with production configuration."""
        model = NFLGraphPredictor(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            lr=self.config.learning_rate,
            future_seq_len=self.config.future_seq_len,
            probabilistic=self.config.probabilistic,
            num_modes=self.config.num_modes,
            velocity_weight=self.config.velocity_weight,
            acceleration_weight=self.config.acceleration_weight,
            collision_weight=self.config.collision_weight,
            coverage_weight=self.config.coverage_weight,
            use_augmentation=self.config.use_augmentation,
            use_huber_loss=self.config.use_huber_loss,
            huber_delta=self.config.huber_delta
        )
        
        # Apply torch.compile for 2x speedup on RTX 4050 (PyTorch 2.0+)
        if USE_TORCH_COMPILE:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("torch.compile enabled - expect 2x speedup after warmup")
            except Exception as e:
                logger.warning(f"torch.compile failed, using eager mode: {e}")
        
        # Log model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {total_params:,} total params, {trainable_params:,} trainable")
        
        return model
    
    def create_callbacks(self, sample_batch=None) -> List:
        """Create production callbacks."""
        callbacks = [
            # Model checkpointing
            ModelCheckpoint(
                dirpath=self.config.checkpoint_dir,
                filename=f'{self.config.experiment_name}-{{epoch:02d}}-{{val_ade:.3f}}',
                monitor=self.config.monitor_metric,
                mode='min',
                save_top_k=self.config.save_top_k,
                save_last=True,
                verbose=True
            ),
            
            # Early stopping
            EarlyStopping(
                monitor=self.config.monitor_metric,
                patience=self.config.early_stopping_patience,
                mode='min',
                verbose=True,
                min_delta=0.001
            ),
            
            # Learning rate monitoring
            LearningRateMonitor(logging_interval='epoch'),
            
            # Rich progress bar
            RichProgressBar(),
            
            # Model summary
            RichModelSummary(max_depth=2),
            
            # Custom epoch summary with rich logging
            EpochSummaryCallback(self.config),
        ]
        
        # Add Stochastic Weight Averaging (SWA) for better generalization
        # SWA averages weights from multiple points in training for smoother minima
        swa_callback = StochasticWeightAveraging(
            swa_lrs=1e-5,           # Lower LR during SWA phase
            swa_epoch_start=0.75,   # Start SWA at 75% of training
            annealing_epochs=5,     # Anneal LR over 5 epochs
            annealing_strategy="cos"
        )
        callbacks.append(swa_callback)
        console.print("[green]✓ SWA (Stochastic Weight Averaging) enabled[/green]")
        
        if sample_batch is not None:
            callbacks.append(AttentionVisualizationCallback(sample_batch, log_every_n_epochs=10))
        
        # Note: DeviceStatsMonitor removed - replaced by EpochSummaryCallback for cleaner output
        
        # Note: Gradient accumulation is handled via Trainer's accumulate_grad_batches parameter
        # Do NOT use GradientAccumulationScheduler callback as it conflicts with the Trainer arg
        
        return callbacks
    
    def create_trainer(self, callbacks: List) -> pl.Trainer:
        """Create production trainer."""
        # Strategy for multi-GPU
        strategy = self.config.strategy
        if strategy == 'auto' and torch.cuda.device_count() > 1:
            strategy = DDPStrategy(find_unused_parameters=False)
        
        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            min_epochs=self.config.min_epochs,
            accelerator=self.config.accelerator,
            devices=self.config.devices,
            strategy=strategy,
            precision=self.config.precision,
            gradient_clip_val=self.config.gradient_clip_val,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            callbacks=callbacks,
            logger=self.loggers,
            log_every_n_steps=50,
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=self.config.deterministic,
            benchmark=not self.config.deterministic,
            profiler='simple' if self.config.enable_profiling else None,
            detect_anomaly=False,  # Disable for production
            enable_checkpointing=True,
            limit_train_batches=self.config.limit_train_batches,
            limit_val_batches=self.config.limit_val_batches,
            limit_test_batches=self.config.limit_test_batches,
        )
        
        return trainer
    
    def train(self, resume_from: Optional[str] = None):
        """Execute full production training pipeline."""
        console.print("\n[bold green]🚀 Starting Production Training Pipeline[/bold green]\n")
        
        # Save configuration
        config_path = Path(self.config.output_dir) / f"{self.config.experiment_name}_config.json"
        self.config.save(str(config_path))
        logger.info(f"Configuration saved to {config_path}")
        
        # Load data
        train_data, val_data, test_data = self.load_and_validate_data()
        
        if len(train_data) == 0:
            logger.error("No training data loaded. Exiting.")
            return
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders(train_data, val_data, test_data)
        
        # Create model
        model = self.create_model()
        
        # Sample batch for attention visualization (optional warmup)
        sample_batch = None
        if self.config.enable_sample_batch_warmup:
            try:
                sample_batch = next(iter(val_loader))
                logger.info("Sample batch warmup enabled for callbacks")
            except Exception as exc:
                logger.warning(f"Sample batch warmup failed; continuing without it: {exc}")
                sample_batch = None
        else:
            logger.info("Skipping sample batch warmup (enable_sample_batch_warmup=False)")
        
        # Create callbacks
        callbacks = self.create_callbacks(sample_batch=sample_batch)
        
        # Create trainer
        trainer = self.create_trainer(callbacks)
        
        # Log configuration to experiment trackers
        for logger_instance in self.loggers:
            if hasattr(logger_instance, 'log_hyperparams'):
                logger_instance.log_hyperparams(self.config.to_dict())
        
        # Train
        console.print("\n[bold cyan]🏋️  Training Started...[/bold cyan]\n")
        logger.info("Training started")
        
        try:
            trainer.fit(
                model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader,
                ckpt_path=resume_from
            )
            
            logger.info("Training completed successfully")
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            console.print("\n[yellow]⚠️  Training interrupted[/yellow]")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            console.print(f"\n[red]❌ Training failed: {e}[/red]")
            raise
        
        # Final validation
        console.print("\n[bold cyan]📊 Running Final Validation...[/bold cyan]\n")
        val_metrics = trainer.validate(model, val_loader)
        
        # Test evaluation
        if len(test_data) > 0:
            console.print("\n[bold cyan]🧪 Running Test Evaluation...[/bold cyan]\n")
            test_metrics = trainer.test(model, test_loader)
            self._display_final_metrics(val_metrics[0], test_metrics[0])
        else:
            self._display_final_metrics(val_metrics[0], None)
        
        # Export models
        if self.config.export_onnx or self.config.export_torchscript:
            self._export_models(model)
        
        console.print("\n[bold green]✅ Training Pipeline Complete![/bold green]\n")
        
    def _display_final_metrics(self, val_metrics: Dict, test_metrics: Optional[Dict]):
        """Display beautiful final metrics table."""
        table = Table(title="🎯 Final Model Performance", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Validation", justify="right", style="green")
        if test_metrics:
            table.add_column("Test", justify="right", style="blue")
        
        for key in sorted(val_metrics.keys()):
            if key.startswith('val_'):
                metric_name = key.replace('val_', '').upper()
                val_value = f"{val_metrics[key]:.4f}"
                
                if test_metrics:
                    test_key = key.replace('val_', 'test_')
                    test_value = f"{test_metrics.get(test_key, 'N/A'):.4f}" if test_key in test_metrics else "N/A"
                    table.add_row(metric_name, val_value, test_value)
                else:
                    table.add_row(metric_name, val_value)
        
        console.print(table)
    
    def _export_models(self, model: NFLGraphPredictor):
        """Export model to ONNX and TorchScript."""
        console.print("\n[bold cyan]📦 Exporting Models...[/bold cyan]")
        
        export_dir = Path(self.config.output_dir) / "exported_models"
        export_dir.mkdir(exist_ok=True)
        
        model.eval()
        
        # Export to TorchScript
        if self.config.export_torchscript:
            try:
                script_path = export_dir / f"{self.config.experiment_name}_torchscript.pt"
                scripted_model = torch.jit.script(model.model)
                torch.jit.save(scripted_model, str(script_path))
                logger.info(f"TorchScript model saved to {script_path}")
                console.print(f"[green]✓[/green] TorchScript: {script_path}")
            except Exception as e:
                logger.warning(f"TorchScript export failed: {e}")
                console.print(f"[yellow]⚠️  TorchScript export failed[/yellow]")
        
        logger.info("Model export complete")
        
        # Export to ONNX
        if self.config.export_onnx:
            try:
                onnx_path = export_dir / f"{self.config.experiment_name}.onnx"

                class OnnxWrapper(torch.nn.Module):
                    def __init__(self, inner, probabilistic=False):
                        super().__init__()
                        self.inner = inner
                        self.probabilistic = probabilistic

                    def forward(self, x, edge_index, edge_attr, batch_idx, valid_mask):
                        data = Batch(batch=batch_idx, x=x, edge_index=edge_index, edge_attr=edge_attr)
                        data.valid_mask = valid_mask
                        out = self.inner(data, return_attention_weights=False, return_distribution=self.probabilistic)
                        if self.probabilistic:
                            preds = out[0]
                        else:
                            preds = out[0]
                        return preds

                wrapper = OnnxWrapper(model.model, probabilistic=model.probabilistic)
                in_features = model.model.encoder.embedding.in_features
                dummy_x = torch.zeros((8, in_features), dtype=torch.float)
                dummy_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                dummy_edge_attr = torch.zeros((2, 5), dtype=torch.float)
                dummy_batch = torch.zeros(8, dtype=torch.long)
                dummy_mask = torch.ones(8, dtype=torch.bool)

                dynamic_axes = {
                    "x": {0: "num_nodes"},
                    "edge_index": {1: "num_edges"},
                    "edge_attr": {0: "num_edges"},
                    "batch_idx": {0: "num_nodes"},
                    "valid_mask": {0: "num_nodes"},
                    "output": {0: "num_nodes"}
                }

                torch.onnx.export(
                    wrapper,
                    (dummy_x, dummy_edge_index, dummy_edge_attr, dummy_batch, dummy_mask),
                    onnx_path,
                    input_names=["x", "edge_index", "edge_attr", "batch_idx", "valid_mask"],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    opset_version=12,
                )
                logger.info(f"ONNX model saved to {onnx_path}")
                console.print(f"[green]✓[/green] ONNX: {onnx_path}")
            except Exception as e:
                logger.warning(f"ONNX export failed: {e}")
                console.print(f"[yellow]⚠️  ONNX export failed[/yellow]")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Production NFL Analytics Training")
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'tune', 'test'],
                       help='Training mode')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='.',
                       help='Data directory')
    parser.add_argument('--weeks', type=int, nargs='+', default=None,
                       help='Weeks to train on (default: all 18)')
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--probabilistic', action='store_true',
                       help='Use probabilistic GMM decoder')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='Maximum epochs')
    parser.add_argument('--precision', type=str, default='auto',
                       choices=['auto', '32', '16-mixed', 'bf16-mixed'],
                       help='Training precision (auto=detect based on GPU, 32=fp32, 16-mixed=fp16, bf16-mixed=bfloat16)')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Dataloader worker count (overrides config)')
    parser.add_argument('--limit-train-batches', type=float, default=None,
                       help='Limit training batches (float fraction or int count)')
    parser.add_argument('--limit-val-batches', type=float, default=None,
                       help='Limit validation batches (float fraction or int count)')
    parser.add_argument('--limit-test-batches', type=float, default=None,
                       help='Limit test batches (float fraction or int count)')
    
    # Resume
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Experiment
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    
    # Tuning
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of Optuna trials')
    
    # Production
    parser.add_argument('--no-validation', action='store_true',
                       help='Disable data validation')
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling')
    parser.add_argument('--enable-sample-batch-warmup', action='store_true',
                       help='Warm up callbacks using a validation sample batch')
    
    args = parser.parse_args()
    
    # Load configuration from YAML if provided
    config_dict = {}
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Flatten nested YAML structure into config_dict
        for section, values in yaml_config.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    config_dict[key] = val
            else:
                config_dict[section] = values
        
        console.print(f"[green]✓ Loaded configuration from {args.config}[/green]")
    
    # Override with CLI arguments (CLI takes precedence)
    if args.data_dir != '.':
        config_dict['data_dir'] = args.data_dir
    if args.weeks:
        config_dict['weeks'] = args.weeks
    if args.hidden_dim != 64:
        config_dict['hidden_dim'] = args.hidden_dim
    if args.probabilistic:
        config_dict['probabilistic'] = True
    if args.batch_size != 32:
        config_dict['batch_size'] = args.batch_size
    if args.learning_rate != 1e-3:
        config_dict['learning_rate'] = args.learning_rate
    if args.max_epochs != 100:
        config_dict['max_epochs'] = args.max_epochs
    if args.precision != '16-mixed':
        config_dict['precision'] = args.precision
    if args.num_workers is not None:
        config_dict['num_workers'] = args.num_workers
    if args.limit_train_batches is not None:
        config_dict['limit_train_batches'] = args.limit_train_batches
    if args.limit_val_batches is not None:
        config_dict['limit_val_batches'] = args.limit_val_batches
    if args.limit_test_batches is not None:
        config_dict['limit_test_batches'] = args.limit_test_batches
    if args.experiment_name:
        config_dict['experiment_name'] = args.experiment_name
    if args.no_validation:
        config_dict['validate_data'] = False
    if args.profile:
        config_dict['enable_profiling'] = True
    if args.enable_sample_batch_warmup:
        config_dict['enable_sample_batch_warmup'] = True
    
    # Set defaults for required fields if not in config
    if 'weeks' not in config_dict:
        config_dict['weeks'] = list(range(1, 19))
    if 'experiment_name' not in config_dict:
        config_dict['experiment_name'] = f'nfl_prod_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    config = ProductionConfig(**config_dict)
    
    # Create trainer
    trainer = ProductionTrainer(config)
    
    # Execute
    if args.mode == 'train':
        trainer.train(resume_from=args.resume_from)
    elif args.mode == 'tune':
        console.print("[yellow]Hyperparameter tuning not yet implemented in this script[/yellow]")
    elif args.mode == 'test':
        console.print("[yellow]Test mode not yet implemented in this script[/yellow]")


if __name__ == '__main__':
    main()
