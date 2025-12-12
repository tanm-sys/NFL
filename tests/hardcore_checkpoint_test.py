#!/usr/bin/env python3
"""
🔥 HARDCORE CHECKPOINT TESTING 🔥
================================
Comprehensive stress testing of all model checkpoints on real NFL 2023 data.

Tests:
1. All 8 checkpoints across multiple weeks of real data
2. ADE/FDE metrics computation
3. Prediction reasonableness (bounds, NaN, Inf)
4. Edge case handling
5. Batch size stress testing
6. GPU memory stress
7. Numerical stability
8. Speed benchmarking
"""

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import os
import time
import json
import torch
import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

from src.train import NFLGraphPredictor
from src.data_loader import DataLoader as TrackingLoader
from src.features import create_graph_data

console = Console()

# Configuration
PROJECT_ROOT = Path("/home/tanmay/Desktop/NFL")
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
TRAIN_DATA_DIR = PROJECT_ROOT / "train"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "hardcore_test_results"

# Test weeks to evaluate
TEST_WEEKS = [1, 5, 9, 13, 17, 18]  # Spread across season


@dataclass
class CheckpointMetrics:
    """Metrics for a single checkpoint."""
    checkpoint_name: str
    epoch: int
    val_ade_reported: float
    
    # Core metrics
    ade_mean: float = 0.0
    ade_std: float = 0.0
    fde_mean: float = 0.0
    fde_std: float = 0.0
    
    # Per-timestep metrics
    ade_per_step: List[float] = None
    
    # Edge case metrics
    nan_predictions: int = 0
    inf_predictions: int = 0
    out_of_bounds: int = 0
    extreme_predictions: int = 0  # > 50 yards per step
    
    # Performance metrics
    inference_time_ms: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # Stability metrics
    numerical_issues: int = 0
    
    # Sample counts
    total_samples: int = 0
    total_predictions: int = 0
    
    def __post_init__(self):
        if self.ade_per_step is None:
            self.ade_per_step = []


@dataclass
class WeekMetrics:
    """Metrics for a single week."""
    week: int
    num_plays: int
    num_graphs: int
    checkpoint_metrics: Dict[str, CheckpointMetrics] = None
    
    def __post_init__(self):
        if self.checkpoint_metrics is None:
            self.checkpoint_metrics = {}


def get_checkpoints() -> List[Path]:
    """Get all checkpoint files sorted by epoch."""
    checkpoints = list(CHECKPOINTS_DIR.glob("*.ckpt"))
    # Sort by epoch number
    def get_epoch(path):
        name = path.stem
        if "epoch=" in name:
            try:
                return int(name.split("epoch=")[1].split("-")[0])
            except:
                return 999
        return 999
    return sorted(checkpoints, key=get_epoch)


def parse_checkpoint_info(ckpt_path: Path) -> Tuple[int, float]:
    """Parse epoch and val_ade from checkpoint filename."""
    name = ckpt_path.stem
    epoch = 0
    val_ade = 0.0
    
    if "epoch=" in name:
        try:
            epoch = int(name.split("epoch=")[1].split("-")[0])
        except:
            pass
    
    if "val_ade=" in name:
        try:
            val_ade = float(name.split("val_ade=")[1].split(".ckpt")[0])
        except:
            pass
    
    return epoch, val_ade


def load_week_data(week: int, max_plays: int = 100) -> Tuple[pl.DataFrame, int]:
    """Load and preprocess week data using proper DataLoader to add required columns."""
    csv_path = TRAIN_DATA_DIR / f"input_2023_w{week:02d}.csv"
    if not csv_path.exists():
        console.print(f"[yellow]Week {week} data not found[/yellow]")
        return None, 0
    
    loader = TrackingLoader(str(PROJECT_ROOT))
    
    try:
        # Use load_week_data which properly adds role_id, weight_norm, side_id
        df = loader.load_week_data(week)
        df = loader.standardize_tracking_directions(df)
    except Exception as e:
        console.print(f"[red]Failed to load week {week}: {e}[/red]")
        return None, 0
    
    # Get unique plays
    unique_plays = df.select(["game_id", "play_id"]).unique()
    num_plays = len(unique_plays)
    
    # Limit plays for testing speed
    if max_plays and len(unique_plays) > max_plays:
        unique_plays = unique_plays.head(max_plays)
    
    # Filter to selected plays
    df = df.join(unique_plays, on=["game_id", "play_id"])
    
    return df, num_plays


def compute_metrics(
    model: NFLGraphPredictor,
    graphs: List,
    device: torch.device,
    batch_size: int = 32
) -> CheckpointMetrics:
    """Compute comprehensive metrics for a model on given graphs."""
    metrics = CheckpointMetrics(
        checkpoint_name="",
        epoch=0,
        val_ade_reported=0.0
    )
    
    if len(graphs) == 0:
        return metrics
    
    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
    
    all_ade = []
    all_fde = []
    step_ade = {i: [] for i in range(10)}  # Per-timestep
    
    start_time = time.time()
    max_gpu_mem = 0
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            try:
                # Forward pass
                if hasattr(model, 'probabilistic') and model.probabilistic:
                    params, mode_probs, _, _ = model.model(batch, return_distribution=True)
                    mu = params[..., :2]
                    best_mode = mode_probs.argmax(dim=-1)
                    idx = torch.arange(mu.size(0), device=device)
                    preds = mu[idx, :, best_mode, :]
                else:
                    preds, cov_pred, _ = model(batch)
                
                targets = batch.y
                
                # Validate shapes
                if preds.shape != targets.shape:
                    # Handle shape mismatch
                    min_t = min(preds.shape[1], targets.shape[1])
                    preds = preds[:, :min_t, :]
                    targets = targets[:, :min_t, :]
                
                future_len = preds.shape[1]
                
                # Check for NaN/Inf
                nan_count = torch.isnan(preds).sum().item()
                inf_count = torch.isinf(preds).sum().item()
                metrics.nan_predictions += nan_count
                metrics.inf_predictions += inf_count
                
                # Skip if bad predictions
                if nan_count > 0 or inf_count > 0:
                    metrics.numerical_issues += 1
                    continue
                
                # Compute errors
                errors = torch.norm(preds - targets, dim=2)  # [N, T]
                
                # ADE per sample
                ade_per_sample = errors.mean(dim=1)  # [N]
                all_ade.extend(ade_per_sample.cpu().numpy().tolist())
                
                # FDE per sample
                fde_per_sample = errors[:, -1]  # [N]
                all_fde.extend(fde_per_sample.cpu().numpy().tolist())
                
                # Per-timestep ADE
                for t in range(min(future_len, 10)):
                    step_ade[t].extend(errors[:, t].cpu().numpy().tolist())
                
                # Check for extreme predictions
                extreme_mask = preds.abs() > 50
                metrics.extreme_predictions += extreme_mask.sum().item()
                
                # Out of bounds (field is ~120 yards x 53.3 yards)
                # But predictions are relative, so check magnitude
                out_mask = preds.abs() > 120
                metrics.out_of_bounds += out_mask.sum().item()
                
                metrics.total_predictions += preds.numel()
                metrics.total_samples += preds.shape[0]
                
                # Track GPU memory
                if device.type == 'cuda':
                    gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                    max_gpu_mem = max(max_gpu_mem, gpu_mem)
                
            except Exception as e:
                console.print(f"[red]Batch error: {e}[/red]")
                metrics.numerical_issues += 1
                continue
    
    elapsed = time.time() - start_time
    
    # Compute final metrics
    if len(all_ade) > 0:
        metrics.ade_mean = float(np.mean(all_ade))
        metrics.ade_std = float(np.std(all_ade))
        metrics.fde_mean = float(np.mean(all_fde))
        metrics.fde_std = float(np.std(all_fde))
        
        # Per-step ADE
        metrics.ade_per_step = [float(np.mean(step_ade[i])) if len(step_ade[i]) > 0 else 0.0 
                                for i in range(10)]
    
    metrics.inference_time_ms = elapsed * 1000 / max(metrics.total_samples, 1)
    metrics.gpu_memory_mb = max_gpu_mem
    
    return metrics


def test_batch_sizes(
    model: NFLGraphPredictor,
    graphs: List,
    device: torch.device,
    batch_sizes: List[int] = [1, 8, 16, 32, 64, 128]
) -> Dict[int, float]:
    """Stress test with different batch sizes."""
    results = {}
    
    for bs in batch_sizes:
        if len(graphs) < bs:
            continue
        
        subset = graphs[:bs]
        batch = Batch.from_data_list(subset).to(device)
        
        try:
            start = time.time()
            with torch.no_grad():
                preds, _, _ = model(batch)
            elapsed = time.time() - start
            
            results[bs] = {
                "time_ms": elapsed * 1000,
                "success": True,
                "error": None
            }
        except Exception as e:
            results[bs] = {
                "time_ms": 0,
                "success": False,
                "error": str(e)
            }
    
    return results


def test_numerical_stability(
    model: NFLGraphPredictor,
    graphs: List,
    device: torch.device
) -> Dict:
    """Test numerical stability with edge cases."""
    results = {
        "gradient_overflow": False,
        "activation_explosion": False,
        "weight_issues": 0,
        "deterministic": True
    }
    
    if len(graphs) < 2:
        return results
    
    # Test determinism
    batch = Batch.from_data_list(graphs[:5]).to(device)
    model.eval()
    
    with torch.no_grad():
        preds1, _, _ = model(batch)
        preds2, _, _ = model(batch)
    
    if not torch.allclose(preds1, preds2, atol=1e-5):
        results["deterministic"] = False
    
    # Check weight statistics
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            results["weight_issues"] += 1
        if torch.isinf(param).any():
            results["weight_issues"] += 1
        if param.abs().max() > 1e6:
            results["activation_explosion"] = True
    
    return results


def run_hardcore_tests():
    """Run all hardcore tests."""
    console.print(Panel.fit(
        "[bold red]🔥 HARDCORE CHECKPOINT TESTING 🔥[/bold red]\n"
        "Testing all checkpoints on real 2023 NFL data",
        border_style="red"
    ))
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"\n[cyan]Device:[/cyan] {device}")
    
    if device.type == 'cuda':
        console.print(f"[cyan]GPU:[/cyan] {torch.cuda.get_device_name()}")
        console.print(f"[cyan]VRAM:[/cyan] {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Get checkpoints
    checkpoints = get_checkpoints()
    console.print(f"\n[green]Found {len(checkpoints)} checkpoints[/green]")
    for ckpt in checkpoints:
        epoch, val_ade = parse_checkpoint_info(ckpt)
        console.print(f"  • {ckpt.name} (epoch={epoch}, val_ade={val_ade:.3f})")
    
    # Results storage
    results = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name() if device.type == 'cuda' else "N/A",
        "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoints": {},
        "week_summaries": {},
        "overall_best": None,
        "issues_found": []
    }
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Test each checkpoint
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        main_task = progress.add_task(
            "[cyan]Testing checkpoints...",
            total=len(checkpoints) * len(TEST_WEEKS)
        )
        
        for ckpt_path in checkpoints:
            ckpt_name = ckpt_path.name
            epoch, val_ade = parse_checkpoint_info(ckpt_path)
            
            console.print(f"\n[bold blue]{'='*60}[/bold blue]")
            console.print(f"[bold]Testing: {ckpt_name}[/bold]")
            console.print(f"[bold blue]{'='*60}[/bold blue]")
            
            # Load model
            try:
                model = NFLGraphPredictor.load_from_checkpoint(
                    str(ckpt_path),
                    map_location=device
                )
                model.eval()
                model.to(device)
                console.print("[green]✓ Model loaded successfully[/green]")
            except Exception as e:
                console.print(f"[red]✗ Failed to load model: {e}[/red]")
                results["issues_found"].append({
                    "checkpoint": ckpt_name,
                    "issue": "load_failed",
                    "error": str(e)
                })
                for _ in range(len(TEST_WEEKS)):
                    progress.advance(main_task)
                continue
            
            ckpt_results = {
                "epoch": epoch,
                "val_ade_reported": val_ade,
                "weeks": {},
                "aggregate_ade": [],
                "aggregate_fde": [],
                "batch_size_test": None,
                "stability_test": None
            }
            
            # Get future_seq_len from model
            try:
                future_seq_len = model.hparams.get("future_seq_len", 10)
            except:
                future_seq_len = 10
            
            # Test on each week
            for week in TEST_WEEKS:
                progress.update(main_task, description=f"[cyan]Checkpoint {epoch}, Week {week}...")
                
                # Load week data
                df, num_plays = load_week_data(week, max_plays=50)
                if df is None:
                    progress.advance(main_task)
                    continue
                
                console.print(f"\n[yellow]Week {week}:[/yellow] {num_plays} total plays, testing sample...")
                
                # Create graphs
                try:
                    graphs = create_graph_data(
                        df,
                        radius=20.0,
                        future_seq_len=future_seq_len,
                        history_len=5
                    )
                except Exception as e:
                    console.print(f"[red]  Graph creation failed: {e}[/red]")
                    results["issues_found"].append({
                        "checkpoint": ckpt_name,
                        "week": week,
                        "issue": "graph_creation_failed",
                        "error": str(e)
                    })
                    progress.advance(main_task)
                    continue
                
                if len(graphs) == 0:
                    console.print("[yellow]  No graphs created[/yellow]")
                    progress.advance(main_task)
                    continue
                
                console.print(f"  Created {len(graphs)} graphs")
                
                # Compute metrics
                metrics = compute_metrics(model, graphs, device, batch_size=32)
                metrics.checkpoint_name = ckpt_name
                metrics.epoch = epoch
                metrics.val_ade_reported = val_ade
                
                ckpt_results["weeks"][week] = asdict(metrics)
                ckpt_results["aggregate_ade"].append(metrics.ade_mean)
                ckpt_results["aggregate_fde"].append(metrics.fde_mean)
                
                # Display week results
                console.print(f"  [green]ADE: {metrics.ade_mean:.4f} ± {metrics.ade_std:.4f}[/green]")
                console.print(f"  [green]FDE: {metrics.fde_mean:.4f} ± {metrics.fde_std:.4f}[/green]")
                
                if metrics.nan_predictions > 0:
                    console.print(f"  [red]⚠ NaN predictions: {metrics.nan_predictions}[/red]")
                if metrics.inf_predictions > 0:
                    console.print(f"  [red]⚠ Inf predictions: {metrics.inf_predictions}[/red]")
                if metrics.extreme_predictions > 0:
                    console.print(f"  [yellow]⚠ Extreme predictions: {metrics.extreme_predictions}[/yellow]")
                
                progress.advance(main_task)
            
            # Batch size stress test (on last week's data)
            if 'graphs' in dir() and len(graphs) > 10:
                console.print("\n[cyan]Batch size stress test...[/cyan]")
                batch_results = test_batch_sizes(model, graphs, device)
                ckpt_results["batch_size_test"] = batch_results
                for bs, res in batch_results.items():
                    status = "✓" if res["success"] else "✗"
                    console.print(f"  Batch {bs}: {status} ({res['time_ms']:.1f}ms)")
            
            # Stability test
            if 'graphs' in dir() and len(graphs) > 2:
                console.print("\n[cyan]Numerical stability test...[/cyan]")
                stability = test_numerical_stability(model, graphs, device)
                ckpt_results["stability_test"] = stability
                if stability["deterministic"]:
                    console.print("  [green]✓ Deterministic[/green]")
                else:
                    console.print("  [red]✗ Non-deterministic[/red]")
                if stability["weight_issues"] > 0:
                    console.print(f"  [red]✗ Weight issues: {stability['weight_issues']}[/red]")
            
            # Compute aggregate metrics
            if ckpt_results["aggregate_ade"]:
                ckpt_results["overall_ade"] = float(np.mean(ckpt_results["aggregate_ade"]))
                ckpt_results["overall_fde"] = float(np.mean(ckpt_results["aggregate_fde"]))
            
            results["checkpoints"][ckpt_name] = ckpt_results
            
            # Clear GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Find best checkpoint
    best_ckpt = None
    best_ade = float('inf')
    for ckpt_name, ckpt_data in results["checkpoints"].items():
        if "overall_ade" in ckpt_data and ckpt_data["overall_ade"] < best_ade:
            best_ade = ckpt_data["overall_ade"]
            best_ckpt = ckpt_name
    
    results["overall_best"] = {
        "checkpoint": best_ckpt,
        "ade": best_ade
    }
    
    # Summary table
    console.print("\n")
    console.print(Panel.fit("[bold green]📊 SUMMARY RESULTS[/bold green]", border_style="green"))
    
    table = Table(title="Checkpoint Performance Summary")
    table.add_column("Checkpoint", style="cyan")
    table.add_column("Epoch", justify="right")
    table.add_column("Reported ADE", justify="right", style="yellow")
    table.add_column("Actual ADE", justify="right", style="green")
    table.add_column("Actual FDE", justify="right", style="green")
    table.add_column("Issues", justify="right", style="red")
    
    for ckpt_name, ckpt_data in results["checkpoints"].items():
        issues = sum(1 for i in results["issues_found"] if i.get("checkpoint") == ckpt_name)
        table.add_row(
            ckpt_name[:40] + "..." if len(ckpt_name) > 40 else ckpt_name,
            str(ckpt_data.get("epoch", "?")),
            f"{ckpt_data.get('val_ade_reported', 0):.3f}",
            f"{ckpt_data.get('overall_ade', 'N/A'):.4f}" if isinstance(ckpt_data.get('overall_ade'), float) else "N/A",
            f"{ckpt_data.get('overall_fde', 'N/A'):.4f}" if isinstance(ckpt_data.get('overall_fde'), float) else "N/A",
            str(issues)
        )
    
    console.print(table)
    
    # Best checkpoint
    console.print(f"\n[bold green]🏆 Best Checkpoint: {best_ckpt}[/bold green]")
    console.print(f"[bold green]   Overall ADE: {best_ade:.4f}[/bold green]")
    
    # Issues summary
    if results["issues_found"]:
        console.print(f"\n[bold red]⚠ Issues Found: {len(results['issues_found'])}[/bold red]")
        for issue in results["issues_found"][:5]:
            console.print(f"  • {issue.get('checkpoint', 'unknown')}: {issue.get('issue', 'unknown')}")
    
    # Save results
    results_file = RESULTS_DIR / f"hardcore_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    console.print(f"\n[green]Results saved to: {results_file}[/green]")
    
    return results


if __name__ == "__main__":
    try:
        results = run_hardcore_tests()
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        import traceback
        traceback.print_exc()
