import sys
sys.path.append('src')

import unittest
import torch
import polars as pl
from features import create_graph_data
from train import NFLGraphPredictor, tune_model
from models.gnn import NFLGraphTransformer
from torch_geometric.data import Batch

class TestSOTA(unittest.TestCase):
    def test_create_graph_data(self):
        # Dummy DF: 2 players, enough frames for sequence
        # With history_len=5 and future_seq_len=2, need at least 5+2=7 frames
        # Loop iterates range(history_len, num_frames - future_seq_len)
        
        frames = []
        for i in range(10):  # 10 frames for history_len=5, future_seq_len=2
            frames.append(pl.DataFrame({
                "game_id": [1, 1],
                "play_id": [1, 1],
                "frame_id": [i, i],
                "nfl_id": [1, 2],
                "std_x": [float(i), float(i)+5.0],
                "std_y": [0.0, 0.0],
                "s": [0., 0.],
                "a": [0., 0.],
                "std_dir": [0., 0.],
                "std_o": [0., 0.],
                "weight_norm": [0., 0.],
                "role_id": [0, 1],
                "side_id": [0, 1]
            }))
        df = pl.concat(frames)
        
        # With history_len=5, future_seq_len=2: range(5, 10-2) = range(5, 8) = 3 graphs
        graphs = create_graph_data(df, radius=10.0, future_seq_len=2, history_len=5)
        self.assertEqual(len(graphs), 3)
        data = graphs[0]
        
        print(f"Graph Data: {data}")
        self.assertTrue(hasattr(data, 'edge_index'))
        self.assertTrue(hasattr(data, 'edge_attr'))
        self.assertTrue(hasattr(data, 'y'))
        
        # Check Shapes
        # 2 nodes, edge_attr should be 5D
        self.assertEqual(data.edge_attr.shape[1], 5)  # 5D edge features
        self.assertEqual(data.y.shape, (2, 2, 2))  # [Nodes, Future=2, Coords=2]
        
        # Verify relative trajectory support (P0)
        self.assertTrue(hasattr(data, 'current_pos'))  # Current position for absolute conversion
        self.assertEqual(data.current_pos.shape, (2, 2))  # [Nodes, 2]
        
        # Verify temporal history (P1)
        self.assertTrue(hasattr(data, 'history'))  # Motion history for LSTM
        self.assertEqual(data.history.shape, (2, 4, 4))  # [Nodes, History-1, 4]
        
    def test_gnn_forward(self):
        # Create dummy batch with correct dimensions
        # Need enough frames for history_len=5 + future_seq_len=2
        frames = []
        for i in range(10):  # 10 frames
            frames.append(pl.DataFrame({
                "game_id": [1, 1],
                "play_id": [1, 1],
                "frame_id": [i, i],
                "nfl_id": [1, 2],
                "std_x": [0.0, 5.0],
                "std_y": [0.0, 0.0],
                "s": [0., 0.],
                "a": [0., 0.],
                "std_dir": [0., 0.],
                "std_o": [0., 0.],
                "weight_norm": [0., 0.],
                "role_id": [0, 1],
                "side_id": [0, 1]
            }))
        df = pl.concat(frames)
        
        # 10 frames, history_len=5, future_seq_len=2: range(5, 8) = 3 graphs
        graphs = create_graph_data(df, future_seq_len=2, history_len=5)
        batch = Batch.from_data_list(graphs)
        
        # Model with correct dimensions (P3 - use_scene_encoder)
        model = NFLGraphTransformer(input_dim=7, hidden_dim=16, future_seq_len=2, edge_dim=5, use_scene_encoder=True)
        pred, cov, attn = model(batch)
        print(f"Model Output Shape: {pred.shape}")
        
        # Expect [Num_Nodes_Total, Future_Seq, 2]
        # 3 graphs * 2 nodes = 6 nodes total in batch
        self.assertEqual(pred.shape[0], 6) 
        self.assertEqual(pred.shape[1], 2)  # Future Seq
        self.assertEqual(pred.shape[2], 2)  # X, Y
        
    def test_optuna_loop(self):
        # Run 1 trial to verify no crash
        best_params = tune_model(num_trials=1)
        self.assertIn('lr', best_params)
        self.assertIn('hidden_dim', best_params)

if __name__ == '__main__':
    unittest.main()
