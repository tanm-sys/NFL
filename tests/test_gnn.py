import sys
sys.path.append('src')

import unittest
import torch
import polars as pl
from features import create_graph_data
from train import NFLGraphPredictor, tune_model, NFLGraphTransformer
from torch_geometric.data import Batch

class TestSOTA(unittest.TestCase):
    def test_create_graph_data(self):
        # Dummy DF: 2 players, 11 frames (needed for sequence)
        # 10 frames future + 1 current = 11?
        # create_graph_data loops `range(num_frames - future_seq_len)`
        # If future=2, need 3 frames to get 1 graph.
        
        frames = []
        for i in range(5):
            frames.append(pl.DataFrame({
                "frame_id": [i, i],
                "nfl_id": [1, 2],
                "std_x": [float(i), float(i)+5.0], # moving
                "std_y": [0.0, 0.0],
                "s": [0., 0.],
                "a": [0., 0.],
                "std_dir": [0., 0.],
                "std_o": [0., 0.]
            }))
        df = pl.concat(frames)
        
        graphs = create_graph_data(df, radius=10.0, future_seq_len=2)
        # 5 frames total. future=2. range(5-2) = 3 graphs (t=0,1,2).
        self.assertEqual(len(graphs), 3)
        data = graphs[0]
        
        print(f"Graph Data: {data}")
        self.assertTrue(hasattr(data, 'edge_index'))
        self.assertTrue(hasattr(data, 'edge_attr'))
        self.assertTrue(hasattr(data, 'y'))
        
        # Check Shapes
        # 2 nodes, 2 edges (fully connected)
        self.assertEqual(data.edge_attr.shape, (2, 2)) # [Edges, Edge_Dim=2]
        self.assertEqual(data.y.shape, (2, 2, 2)) # [Nodes, Future=2, Coords=2]
        
    def test_gnn_forward(self):
        # Create dummy batch
        # Need enough data for graph creation
        frames = []
        for i in range(5):
            frames.append(pl.DataFrame({
                "frame_id": [i, i],
                "nfl_id": [1, 2],
                "std_x": [0.0, 5.0],
                "std_y": [0.0, 0.0],
                "s": [0., 0.],
                "a": [0., 0.],
                "std_dir": [0., 0.],
                "std_o": [0., 0.]
            }))
        df = pl.concat(frames)
        
        graphs = create_graph_data(df, future_seq_len=2)
        batch = Batch.from_data_list(graphs)
        
        # Model
        model = NFLGraphTransformer(input_dim=6, hidden_dim=16, future_seq_len=2)
        out = model(batch)
        print(f"Model Output Shape: {out.shape}")
        
        # Expect [Num_Nodes_Total, Future_Seq, 2]
        # 3 graphs * 2 nodes = 6 nodes total in batch
        self.assertEqual(out.shape[0], 6) 
        self.assertEqual(out.shape[1], 2) # Future Seq
        self.assertEqual(out.shape[2], 2) # X, Y
        
    def test_optuna_loop(self):
        # Run 1 trial to verify no crash
        best_params = tune_model(num_trials=1)
        self.assertIn('lr', best_params)
        self.assertIn('hidden_dim', best_params)

if __name__ == '__main__':
    unittest.main()
