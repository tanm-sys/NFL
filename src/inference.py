import argparse
from pathlib import Path
from typing import List

import torch
import polars as pl
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.train import NFLGraphPredictor
from src.data_loader import DataLoader as TrackingLoader
from src.features import create_graph_data
from src.visualization import create_football_field


def load_graphs_from_csv(csv_path: Path, radius: float, future_seq_len: int) -> List:
    loader = TrackingLoader(csv_path.parent)
    df = pl.read_csv(csv_path)
    df = loader.standardize_tracking_directions(df)
    graphs = create_graph_data(df, radius=radius, future_seq_len=future_seq_len)
    return graphs


def run_inference(
    checkpoint: Path,
    input_csv: Path,
    output_csv: Path,
    radius: float = 20.0,
    batch_size: int = 32,
    visualize: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NFLGraphPredictor.load_from_checkpoint(str(checkpoint), map_location=device)
    model.eval()
    model.to(device)

    future_seq_len = model.hparams.get("future_seq_len", 10)
    graphs = load_graphs_from_csv(input_csv, radius=radius, future_seq_len=future_seq_len)

    if len(graphs) == 0:
        print("No graphs generated from input.")
        return

    loader = PyGDataLoader(graphs, batch_size=batch_size, shuffle=False)
    outputs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if model.probabilistic:
                params, mode_probs, _, _ = model.model(batch, return_distribution=True)
                mu = params[..., :2]
                best_mode = mode_probs.argmax(dim=-1)
                idx = torch.arange(mu.size(0), device=device)
                preds = mu[idx, :, best_mode, :]
            else:
                preds, _, _ = model(batch)
            current_pos = batch.current_pos.unsqueeze(1)
            pred_abs = preds + current_pos

            for i in range(pred_abs.size(0)):
                game_id = int(batch.game_id[i]) if hasattr(batch, "game_id") else -1
                play_id = int(batch.play_id[i]) if hasattr(batch, "play_id") else -1
                for t in range(pred_abs.size(1)):
                    outputs.append(
                        {
                            "game_id": game_id,
                            "play_id": play_id,
                            "node_idx": int(i),
                            "step": int(t),
                            "x": float(pred_abs[i, t, 0].cpu()),
                            "y": float(pred_abs[i, t, 1].cpu()),
                        }
                    )

    out_df = pl.DataFrame(outputs)
    out_df.write_csv(output_csv)
    print(f"Predictions saved to {output_csv}")

    if visualize:
        try:
            fig, ax = create_football_field()
            first_play = out_df.filter(pl.col("play_id") == out_df["play_id"][0])
            for node_id in first_play["node_idx"].unique():
                sub = first_play.filter(pl.col("node_idx") == node_id)
                ax.plot(sub["x"], sub["y"], label=f"agent {node_id}", alpha=0.7)
            ax.legend()
            fig.savefig(output_csv.with_suffix(".png"))
            print(f"Visualization saved to {output_csv.with_suffix('.png')}")
        except Exception as e:
            print(f"Visualization failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone inference for NFL trajectories")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.ckpt)")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to tracking CSV for inference")
    parser.add_argument("--output-csv", type=str, default="predictions.csv", help="Where to save predictions")
    parser.add_argument("--radius", type=float, default=20.0, help="Graph radius for edge building")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--visualize", action="store_true", help="Save a field plot for the first play")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        checkpoint=Path(args.checkpoint),
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        radius=args.radius,
        batch_size=args.batch_size,
        visualize=args.visualize,
    )
