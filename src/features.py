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

def create_graph_data(df: pl.DataFrame, radius: float = 20.0, future_seq_len: int = 10) -> List[Data]:
    """
    Converts a dataframe into a list of PyG Data objects (graphs).
    Optimized: Pivots to Tensor first, then slices.
    Adds:
    - y: Future trajectory [Num_Agents, Future_Seq, 2]
    - edge_attr: [Num_Edges, 2] (Distance, Relative Angle)
    """
    # 1. Pivot to [Frames, Agents, Features]
    feature_cols = ["std_x", "std_y", "s", "a", "std_dir", "std_o"]
    
    # Sort
    df = df.sort(["frame_id", "nfl_id"])
    pdf = df.to_pandas()
    # Handle Football NaN nfl_id
    pdf['nfl_id'] = pdf['nfl_id'].fillna(999999)
    
    # helper
    unique_frames = pdf['frame_id'].unique() # sorted by df sort
    # verifying timestamps? simpler to assume 10Hz contiguous
    
    # Pivot
    # Multi-index pivot is tricky, stick to list of pivots
    feature_matrices = []
    for col in feature_cols:
        # fillna(0) for missing agents
        pivot = pdf.pivot(index='frame_id', columns='nfl_id', values=col).fillna(0)
        feature_matrices.append(pivot.values)
        
    # Stack: [Frames, Agents, Features]
    # Features order: x, y, s, a, dir, o
    data_tensor = np.stack(feature_matrices, axis=-1)
    data_tensor = torch.FloatTensor(data_tensor)
    
    num_frames, num_agents, num_feats = data_tensor.shape
    
    graph_list = []
    
    # Feature indices
    IDX_X, IDX_Y = 0, 1
    IDX_DIR = 4 
    
    # Context Features (Global for this play)
    # Check if context cols exist (from updated loader)
    context_tensor = None
    if "down" in df.columns and "yards_to_go" in df.columns:
        # Take first row (constant per play)
        # down is 1-4, ytg is distance. Normalize?
        # Down: 1-4. YTG: 1-99.
        # Let's keep raw for embedding or normalize vaguely.
        # Normalize simple: Down/4, YTG/100
        row = df.select(["down", "yards_to_go"]).head(1).to_dict(as_series=False)
        down = row["down"][0]
        ytg = row["yards_to_go"][0]
        
        # Handle NA or casts
        if down is None: down = 1
        if ytg is None: ytg = 10
        
        context_tensor = torch.tensor([[down, ytg]], dtype=torch.float) # [1, 2]
        
    # Multi-Task Label: Coverage Type
    coverage_tensor = None
    if "coverage_label" in df.columns:
        row = df.select(["coverage_label"]).head(1).to_dict(as_series=False)
        cov_label = row["coverage_label"][0]
        
        if cov_label is not None:
             coverage_tensor = torch.tensor([cov_label], dtype=torch.float) # [1] float (since BCEWithLogits takes float)
             # Or Long if CrossEntropy. Binary is simpler with BCE.
             # Let's use Float [1] for BCE.
    
    for t in range(num_frames - future_seq_len):
        # Current Frame Features
        # [Agents, Feats]
        x_t = data_tensor[t] 
        pos_t = x_t[:, :2] # x, y
        
        # Future Targets
        # [Future, Agents, 2] -> Transpose to [Agents, Future, 2]
        # range t+1 to t+1+future
        y_t = data_tensor[t+1 : t+1+future_seq_len, :, :2]
        y_t = y_t.permute(1, 0, 2) 
        
        # Absolute diff targets? Or absolute positions?
        # Model usually predicts relative to current pos or absolute.
        # Let's attach Absolute Y.
        
        # Edge Creation
        # Dist Matrix
        dist_matrix = torch.cdist(pos_t, pos_t) # [Agents, Agents]
        mask = (dist_matrix < radius) & (dist_matrix > 1e-5)
        edge_index = mask.nonzero().t() # [2, Num_Edges]
        
        row, col = edge_index
        
        # Edge Attributes: [Distance, Relative Angle]
        # Dist
        dist = dist_matrix[row, col].unsqueeze(1) # [Edges, 1]
        
        # Relative Angle
        # atan2(y_j - y_i, x_j - x_i) - dir_i
        diff_x = pos_t[col, 0] - pos_t[row, 0]
        diff_y = pos_t[col, 1] - pos_t[row, 1]
        angle_abs = torch.atan2(diff_y, diff_x) * 180 / np.pi
        
        # dir is degrees, 0..360. vector is usually 0=East? Need to check semantics.
        # Assuming standard math angle for simplicity or standardized direction.
        # If standardized: 0=Right (East).
        dir_i = x_t[row, IDX_DIR]
        rel_angle = (angle_abs - dir_i) % 360
        # Normalize to -180..180 or 0..1?
        rel_angle = rel_angle.unsqueeze(1) / 360.0
        
        edge_attr = torch.cat([dist, rel_angle], dim=1) # [Edges, 2]
        
        # Create Data
        # Flatten y? Or keep as tensor? PyG handles any tensor in Data
        data = Data(x=x_t, edge_index=edge_index, edge_attr=edge_attr, y=y_t, pos=pos_t)
        
        # Attach Context if available
        if context_tensor is not None:
            data.context = context_tensor # [1, 2]
            
        # Attach Coverage Label if available
        if coverage_tensor is not None:
            data.y_coverage = coverage_tensor # [1]
            
        graph_list.append(data)
        
    return graph_list
