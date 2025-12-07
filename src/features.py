import polars as pl
import numpy as np
import h3
import torch
from typing import List, Tuple
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

def calculate_distance_to_ball(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculates distance from each player to the football.
    Requires 'std_x', 'std_y' for players and ball.
    """
    # This usually requires joining the ball position back to the player rows
    # Filter for ball
    ball_df = df.filter(pl.col("player_name") == "football").select(
        ["game_id", "play_id", "frame_id", "std_x", "std_y"]
    ).rename({"std_x": "ball_x", "std_y": "ball_y"})
    
    # Join back
    df = df.join(ball_df, on=["game_id", "play_id", "frame_id"], how="left")
    
    # Calc distance
    df = df.with_columns(
        (((pl.col("std_x") - pl.col("ball_x")) ** 2 + 
          (pl.col("std_y") - pl.col("ball_y")) ** 2).sqrt())
        .alias("dist_to_ball")
    )
    
    return df

def create_baseline_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates simple features for baseline model:
    - Speed (s)
    - Acceleration (a)
    - Distance to ball
    - Relative orientation to ball
    """
    df = calculate_distance_to_ball(df)
    
    # Relative angle to ball
    # angle = arctan2(ball_y - y, ball_x - x)
    # rel_angle = angle - o (orientation)
    
    df = df.with_columns(
        pl.map_batches(["std_x", "std_y", "ball_x", "ball_y"], 
                       lambda s: np.arctan2(s[3] - s[1], s[2] - s[0]) * 180 / np.pi)
        .alias("angle_to_ball")
    )
    
    df = df.with_columns(
        ((pl.col("angle_to_ball") - pl.col("std_o")).mod(360))
        .alias("rel_angle_to_ball")
    )
    
    return df

def add_h3_indices(df: pl.DataFrame, resolution: int = 10) -> pl.DataFrame:
    """
    Converts std_x, std_y to H3 indices.
    Note: H3 expects Lat/Lon. We must project field coordinates to dummy Lat/Lon.
    Field center (60, 26.65) -> (0, 0) Lat/Lon approx.
    1 degree lat ~ 111km. Field is 100 yds ~ 91m. 
    Scale is extremely small for standard H3, but we can map 1 yard = 0.00001 deg.
    Actually, H3 works on sphere.
    Alternative: Just map x,y to lat,lon with small factor.
    x_lat = (x - 60) * 0.0001
    y_lon = (y - 26.65) * 0.0001
    """
    # Simple projection for hex binning
    
    def get_h3(x, y):
        # Dummy projection: 1 unit ~ 0.0001 deg
        lat = (x - 60) * 0.0001
        lon = (y - 26.66) * 0.0001
        return h3.geo_to_h3(lat, lon, resolution)

    # Polars map_elements (slower but works for strings/objects)
    # Better to use vectorized if possible, but h3 is C-binding
    # For speed, usually apply on unique x,y or pandas
    
    # We'll use map_rows or map_elements
    # To optimize: convert to pandas, apply, back to polars 
    # OR use map_batches if we can vectorize
    
    # Lets stick to basics first
    return df.with_columns(
        pl.struct(["std_x", "std_y"]).map_elements(
            lambda row: get_h3(row["std_x"], row["std_y"]),
            return_dtype=pl.Utf8
        ).alias("h3_index")
    )

def prepare_tensor_data(df: pl.DataFrame, seq_len: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepares data for Transformer:
    Input: [Batch, Seq_Len, Num_Agents, Features]
    Num_Agents usually fixed to 23 (22 players + ball)
    Features: [x, y, s, a, dir, o]
    """
    # Filter relevant columns
    feature_cols = ["std_x", "std_y", "s", "a", "std_dir", "std_o"]
    
    # Need to pivot to get all players per frame
    # This is complex in Polars without specific player sorting
    # Assumption: data is sorted by frame_id, then player_id?
    
    # Strategy:
    # 1. Sort by game, play, frame, nfl_id (football last/first)
    # 2. Pivot/Reshape
    
    # For simplicity of this sprint, let's assume we handle Batching in DataModule
    # This function processes a SINGLE play DataFrame into a tensor
    
    # Sort
    df = df.sort(["frame_id", "nfl_id"])
    
    frames = df["frame_id"].unique().sort()
    if len(frames) < seq_len:
        return None, None
        
    # Valid sequences
    # stride 1
    sequences = []
    targets = []
    
    # Group by frame to verify we have consistent agent count
    # agent_counts = df.group_by("frame_id").count()
    # agents = df.filter(pl.col("frame_id") == frames[0])["nfl_id"].sort()
    
    # Convert to numpy for fast slicing
    # We need a matrix: [Total_Frames, Agents, Feats]
    
    # Pivot: Index=Frame, Columns=Agent, Values=Features
    # Helper: Just get list of agents
    agents = df["nfl_id"].unique().sort()
    num_agents = len(agents)
    
    # Map features to tensor
    # Shape: [Frames, Agents, Feats]
    # We iterate frames and fill
    
    # Optimization: Use pivot
    # But pivot only supports 1 value column usually.
    # We can perform multiple pivots and stack.
    
    # Let's do Pandas for the pivot logic as its robust
    pdf = df.to_pandas()
    
    # Ensure football is included (nfl_id might be NaN for football? check data)
    # Football usually has NA nfl_id. Fill with 999999 or similar
    pdf['nfl_id'] = pdf['nfl_id'].fillna(999999)
    
    # Matrix list
    feature_matrices = []
    for col in feature_cols:
        pivot_table = pdf.pivot(index='frame_id', columns='nfl_id', values=col)
        # Fill missing? padding?
        pivot_table = pivot_table.fillna(0) # Pad
        feature_matrices.append(pivot_table.values) # [Frames, Agents]
        
    # Stack features: [Frames, Agents, Features]
    data_all_frames = np.stack(feature_matrices, axis=-1)
    
    # Create sequences
    # X: [t, t+seq_len], Y: [t+seq_len, t+2*seq_len] (future prediction)
    
    num_frames = data_all_frames.shape[0]
    
    X_list = []
    Y_list = []
    
    for i in range(num_frames - 2 * seq_len):
        X_list.append(data_all_frames[i : i+seq_len])
        Y_list.append(data_all_frames[i+seq_len : i+2*seq_len, :, :2]) # Predict X,Y only for simplicity?
        
    if not X_list:
        return None, None
        
    return torch.FloatTensor(np.array(X_list)), torch.FloatTensor(np.array(Y_list))

def create_graph_data(df: pl.DataFrame, radius: float = 20.0, future_seq_len: int = 10, history_len: int = 5) -> List[Data]:
    """
    Converts a dataframe into a list of PyG Data objects (graphs).
    Optimized: Pivots to Tensor first, then slices.
    Adds:
    - y: Future trajectory [Num_Agents, Future_Seq, 2] (relative)
    - history: Past motion [Num_Agents, History_Len, 4] (vel_x, vel_y, acc_x, acc_y)
    - edge_attr: [Num_Edges, 5] rich edge features
    """
    graph_list = []
    
    # Process per play to avoid cross-play contamination
    play_groups = df.partition_by(["game_id", "play_id"], maintain_order=True)
    
    for play_df in play_groups:
        # Sort
        play_df = play_df.sort(["frame_id", "nfl_id"])
        pdf = play_df.to_pandas()
        # Handle Football NaN nfl_id
        pdf['nfl_id'] = pdf['nfl_id'].fillna(999999)
        
        # Determine Context (Single row per play)
        # Context Features (Global for this play)
        context_tensor = None
        formation_tensor = None
        alignment_tensor = None
        
        # Extract game_id and play_id for this play (for train/val splitting)
        game_id = play_df["game_id"][0]
        play_id = play_df["play_id"][0]
        
        if "down" in play_df.columns and "yards_to_go" in play_df.columns:
            cols = ["down", "yards_to_go"]
            if "defenders_box_norm" in play_df.columns:
                cols.append("defenders_box_norm")
                
            row = play_df.select(cols).head(1).to_dict(as_series=False)
            down = row["down"][0] if row["down"][0] is not None else 1
            ytg = row["yards_to_go"][0] if row["yards_to_go"][0] is not None else 10
            box = row["defenders_box_norm"][0] if "defenders_box_norm" in row else 0.0
            
            context_tensor = torch.tensor([[down, ytg, box]], dtype=torch.float)
            
        if "formation_id" in play_df.columns:
            fid = play_df["formation_id"][0]
            formation_tensor = torch.tensor([fid], dtype=torch.long)
            
        if "alignment_id" in play_df.columns:
            aid = play_df["alignment_id"][0]
            alignment_tensor = torch.tensor([aid], dtype=torch.long)
            
        # Coverage
        coverage_tensor = None
        if "coverage_label" in play_df.columns:
            row = play_df.select(["coverage_label"]).head(1).to_dict(as_series=False)
            cov_label = row["coverage_label"][0]
            if cov_label is not None:
                 coverage_tensor = torch.tensor([cov_label], dtype=torch.float)
        
        # Pivot Features - now including side_id for team awareness
        feature_cols = ["std_x", "std_y", "s", "a", "std_dir", "std_o", "weight_norm", "role_id", "side_id"]
        feature_matrices = []
        for col in feature_cols:
            val_fill = 0
            if col == "role_id": val_fill = 4
            if col == "side_id": val_fill = 2  # Unknown side
            if col not in pdf.columns:
                # Create placeholder if column doesn't exist
                pdf[col] = 0
            pivot = pdf.pivot(index='frame_id', columns='nfl_id', values=col).fillna(val_fill)
            feature_matrices.append(pivot.values)
            
        # [Frames, Agents, Feats] - now 9 features
        data_tensor = np.stack(feature_matrices, axis=-1)
        data_tensor = torch.FloatTensor(data_tensor)
        
        num_frames = data_tensor.shape[0]
        
        # radius graph index helpers
        # Iterate Frames - need history_len before and future_seq_len after
        min_start = history_len
        max_end = num_frames - future_seq_len
        
        for t in range(min_start, max_end):
            x_t = data_tensor[t]  # [Agents, 9]
            
            # Extract Role and Side (Last 2 Cols)
            role_t = x_t[:, 7].long()
            side_t = x_t[:, 8].long()
            
            # Keep features up to weight (first 7)
            # x, y, s, a, dir, o, weight
            input_x = x_t[:, :7] 
            pos_t = input_x[:, :2]  # x, y
            speed_t = input_x[:, 2]  # speed
            dir_t = input_x[:, 4]    # direction
            
            # Compute Motion History (P1) - velocity and acceleration from past positions
            # history_positions: [history_len, Agents, 2]
            if history_len > 1:
                history_positions = data_tensor[t-history_len:t, :, :2]  # [H, Agents, 2]
                history_positions = history_positions.permute(1, 0, 2)   # [Agents, H, 2]
                
                # Compute velocity (position differences)
                velocity = history_positions[:, 1:, :] - history_positions[:, :-1, :]  # [Agents, H-1, 2]
                
                # Compute acceleration (velocity differences)  
                if velocity.shape[1] > 1:
                    acceleration = velocity[:, 1:, :] - velocity[:, :-1, :]  # [Agents, H-2, 2]
                    # Pad acceleration to match velocity length
                    acc_pad = torch.zeros(acceleration.shape[0], 1, 2)
                    acceleration = torch.cat([acc_pad, acceleration], dim=1)
                else:
                    acceleration = torch.zeros_like(velocity)
                
                # Combine: [vel_x, vel_y, acc_x, acc_y] -> [Agents, H-1, 4]
                history_t = torch.cat([velocity, acceleration], dim=-1)
            else:
                history_t = None
            
            # Future Targets - RELATIVE to current position (critical for accuracy)
            future_abs = data_tensor[t+1 : t+1+future_seq_len, :, :2]  # [T, Agents, 2]
            future_abs = future_abs.permute(1, 0, 2)  # [Agents, T, 2]
            
            # Convert to relative displacements from current position
            current_pos = pos_t.unsqueeze(1)  # [Agents, 1, 2]
            y_t = future_abs - current_pos  # [Agents, T, 2] - relative displacements 
            
            # Edges with RICHER FEATURES
            dist_matrix = torch.cdist(pos_t, pos_t)
            mask = (dist_matrix < radius) & (dist_matrix > 1e-5)
            edge_index = mask.nonzero().t()
            
            if edge_index.shape[1] > 0:
                row_idx, col_idx = edge_index
                
                # 1. Distance (normalized by radius)
                diff = pos_t[row_idx] - pos_t[col_idx]
                dist = torch.norm(diff, p=2, dim=-1).view(-1, 1) / radius
                
                # 2. Relative Angle
                angle = torch.atan2(diff[:, 1], diff[:, 0]).view(-1, 1)
                
                # 3. Relative Velocity (speed difference)
                rel_speed = (speed_t[row_idx] - speed_t[col_idx]).view(-1, 1)
                
                # 4. Relative Direction (direction difference, normalized to [-1, 1])
                dir_diff = ((dir_t[row_idx] - dir_t[col_idx] + 180) % 360 - 180) / 180.0
                rel_dir = dir_diff.view(-1, 1)
                
                # 5. Same-Team Indicator (1 if same team, 0 otherwise)
                same_team = (side_t[row_idx] == side_t[col_idx]).float().view(-1, 1)
                
                # Combine: [dist, angle, rel_speed, rel_dir, same_team] = 5D edge features
                edge_attr = torch.cat([dist, angle, rel_speed, rel_dir, same_team], dim=-1)
            else:
                edge_attr = torch.zeros((0, 5))
            
            data = Data(x=input_x, edge_index=edge_index, edge_attr=edge_attr, y=y_t, pos=pos_t)
            
            # Store current position for converting predictions back to absolute
            data.current_pos = pos_t  # [Agents, 2]
            
            # Add temporal encoding (normalized frame position)
            data.frame_t = torch.tensor([t / num_frames], dtype=torch.float)
            
            if role_t is not None: data.role = role_t
            if side_t is not None: data.side = side_t
            if formation_tensor is not None: data.formation = formation_tensor
            if alignment_tensor is not None: data.alignment = alignment_tensor
            if context_tensor is not None: data.context = context_tensor
            # Always set y_coverage for consistent batching (-1.0 = unknown/missing)
            data.y_coverage = coverage_tensor if coverage_tensor is not None else torch.tensor([-1.0], dtype=torch.float)
            if history_t is not None: data.history = history_t  # [Agents, H-1, 4]
            
            # Store play identifiers for train/val splitting
            data.game_id = torch.tensor([game_id], dtype=torch.long)
            data.play_id = torch.tensor([play_id], dtype=torch.long)
            
            graph_list.append(data)
            
    return graph_list

