import polars as pl
import pandas as pd
import os
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

class DataLoader:
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        
    def load_week_data(self, week: int, standard_cols: bool = True) -> pl.DataFrame:
        """
        Load tracking data for a specific week and merge with play context.
        Enforces NFL Rules: Filters out nullified plays.
        """
        # 1. Load Tracking
        filename = f"input_2023_w{week:02d}.csv"
        filepath = self.data_dir / "train" / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        tracking_df = pl.read_csv(filepath)
        
        # 2. Load Plays to filter Nullified and get Context
        plays_df = self.load_plays()
        
        # NFL RULE CHECK: Exclude plays nullified by penalty
        valid_plays = plays_df.filter(pl.col("play_nullified_by_penalty") == "N")
        
        
        
        # Select context columns
        # absolute_yardline_number might need standardization same as x side
        context_cols = ["game_id", "play_id", "down", "yards_to_go", "absolute_yardline_number", "possession_team", 
                       "team_coverage_man_zone", "offense_formation", "receiver_alignment", "defenders_in_the_box"]
        existing_cols = [c for c in context_cols if c in valid_plays.columns]
        play_context = valid_plays.select(existing_cols)
        
        # MAPPINGS
        formation_map = {"SHOTGUN": 0, "EMPTY": 1, "SINGLEBACK": 2, "PISTOL": 3, "I_FORM": 4, "JUMBO": 5, "WILDCAT": 6}
        alignment_map = {"2x2": 0, "3x1": 1, "3x2": 2, "2x1": 3, "4x1": 4, "1x1": 5, "4x0": 6, "3x3": 7, "3x0": 8}
        
        # Apply Mappings & Clean
        play_context = play_context.with_columns(
            # Coverage
            pl.when(pl.col("team_coverage_man_zone") == "MAN_COVERAGE").then(0)
            .when(pl.col("team_coverage_man_zone") == "ZONE_COVERAGE").then(1)
            .otherwise(None)
            .alias("coverage_label"),
            
            # Formation
            pl.col("offense_formation").replace(formation_map, default=7).cast(pl.Int64).alias("formation_id"),
            
            # Alignment
            pl.col("receiver_alignment").replace(alignment_map, default=9).cast(pl.Int64).alias("alignment_id"),
            
            # Defenders Box (Standardize: (x - 7) / 2 approx)
            pl.col("defenders_in_the_box").fill_null(7.0).cast(pl.Float32).alias("defenders_box_norm")
        )

        # 3. Join
        # Tracking data has gameId, playId. Standardize names first if needed.
        if standard_cols:
            tracking_df = self._standardize_columns(tracking_df)
            
        # TRACKING FEATURE ENCODING
        # Role Map
        role_map = {"Defensive Coverage": 0, "Other Route Runner": 1, "Passer": 2, "Targeted Receiver": 3}
        side_map = {"Defense": 0, "Offense": 1}
        
        tracking_df = tracking_df.with_columns([
            # Role ID
            pl.col("player_role").replace(role_map, default=4).cast(pl.Int64).alias("role_id"),
            
            # Weight Norm ((w - 200) / 50)
            ((pl.col("player_weight").cast(pl.Float32) - 200.0) / 50.0).alias("weight_norm"),
            
            # Player Side (Defense=0, Offense=1) for team-aware features
            pl.col("player_side").replace(side_map, default=2).cast(pl.Int64).alias("side_id")
        ])
        
        # Inner join filters out the nullified plays from tracking data
        # Cast join keys to match if needed (usually polars handles int/in but strict types matter)
        # Ensure game_id/play_id are Int64
        try:
            tracking_df = tracking_df.with_columns([
                pl.col("game_id").cast(pl.Int64),
                pl.col("play_id").cast(pl.Int64)
            ])
            play_context = play_context.with_columns([
                pl.col("game_id").cast(pl.Int64),
                pl.col("play_id").cast(pl.Int64)
            ])
        except Exception as e:
            pass

        df = tracking_df.join(play_context, on=["game_id", "play_id"], how="inner")
            
        return df


class GraphDataset(Dataset):
    """
    Map-style dataset that constructs graphs lazily per play and optionally caches per-play tensors.
    Each dataset index corresponds to a single graph frame (time slice) within a play.
    """

    def __init__(
        self,
        loader: DataLoader,
        play_tuples: List[Tuple[int, int, int, int]],
        radius: float = 20.0,
        future_seq_len: int = 10,
        history_len: int = 5,
        cache_dir: Optional[Union[str, Path]] = None,
        persist_cache: bool = False,
        in_memory_cache_size: int = 8,
    ):
        """
        Args:
            loader: DataLoader for loading weeks and standardization.
            play_tuples: List of (week, game_id, play_id, local_idx) entries. local_idx selects the graph within that play.
            radius: Graph radius.
            future_seq_len: Target horizon.
            history_len: History length.
            cache_dir: Optional directory for per-play disk cache.
            persist_cache: If True, saves per-play graphs to disk after first build.
            in_memory_cache_size: Max number of plays to keep in memory cache (LRU).
        """
        super().__init__()
        self.loader = loader
        self.play_tuples = play_tuples
        self.radius = radius
        self.future_seq_len = future_seq_len
        self.history_len = history_len
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.persist_cache = persist_cache and self.cache_dir is not None
        if self.persist_cache and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: OrderedDict[str, List[torch.Tensor]] = OrderedDict()
        self._mem_cache_size = max(1, in_memory_cache_size)

    def __len__(self):
        return len(self.play_tuples)

    def _cache_key(self, week: int, game_id: int, play_id: int) -> str:
        return f"w{week:02d}_g{game_id}_p{play_id}"

    def _get_from_mem_cache(self, key: str) -> Optional[List]:
        if key in self._mem_cache:
            value = self._mem_cache.pop(key)
            self._mem_cache[key] = value
            return value
        return None

    def _set_mem_cache(self, key: str, value: List):
        if key in self._mem_cache:
            self._mem_cache.pop(key)
        elif len(self._mem_cache) >= self._mem_cache_size:
            self._mem_cache.popitem(last=False)
        self._mem_cache[key] = value

    def _load_play_graphs(self, week: int, game_id: int, play_id: int):
        key = self._cache_key(week, game_id, play_id)
        mem = self._get_from_mem_cache(key)
        if mem is not None:
            return mem

        disk_path = None
        if self.cache_dir is not None:
            disk_path = self.cache_dir / f"{key}.pt"
            if disk_path.exists():
                graphs = torch.load(disk_path)
                self._set_mem_cache(key, graphs)
                return graphs

        week_df = self.loader.load_week_data(week)
        week_df = self.loader.standardize_tracking_directions(week_df)
        play_df = week_df.filter(
            (pl.col("game_id") == game_id) & (pl.col("play_id") == play_id)
        )

        from src.features import create_graph_data

        graphs = create_graph_data(
            play_df,
            radius=self.radius,
            future_seq_len=self.future_seq_len,
            history_len=self.history_len,
        )

        if self.persist_cache and disk_path is not None:
            torch.save(graphs, disk_path)

        self._set_mem_cache(key, graphs)
        return graphs

    def __getitem__(self, idx: int):
        week, game_id, play_id, local_idx = self.play_tuples[idx]
        graphs = self._load_play_graphs(week, game_id, play_id)
        if local_idx >= len(graphs):
            raise IndexError(f"Requested local idx {local_idx} exceeds graphs {len(graphs)} for play {play_id}")
        return graphs[local_idx]


def build_play_tuples(
    loader: DataLoader,
    weeks: List[int],
    history_len: int,
    future_seq_len: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Build deterministic list of (week, game_id, play_id, local_idx) entries.
    local_idx indexes the graph within that play (frame-level).
    """
    tuples: List[Tuple[int, int, int, int]] = []
    for week in sorted(weeks):
        try:
            df = loader.load_week_data(week)
            df = loader.standardize_tracking_directions(df)
        except FileNotFoundError:
            continue

        counts = (
            df.group_by(["game_id", "play_id"])
            .agg(pl.col("frame_id").n_unique().alias("n_frames"))
            .sort(["game_id", "play_id"])
        )

        for game_id, play_id, n_frames in counts.iter_rows():
            num_graphs = max(0, n_frames - history_len - future_seq_len)
            for local_idx in range(num_graphs):
                tuples.append((week, int(game_id), int(play_id), int(local_idx)))

    tuples = sorted(tuples, key=lambda x: (x[0], x[1], x[2], x[3]))
    return tuples


def build_play_metadata(
    loader: DataLoader,
    weeks: List[int],
    history_len: int,
    future_seq_len: int,
) -> List[Tuple[int, int, int, int]]:
    """
    Returns list of (week, game_id, play_id, num_graphs) for deterministic splitting.
    """
    meta: List[Tuple[int, int, int, int]] = []
    for week in sorted(weeks):
        try:
            df = loader.load_week_data(week)
            df = loader.standardize_tracking_directions(df)
        except FileNotFoundError:
            continue

        counts = (
            df.group_by(["game_id", "play_id"])
            .agg(pl.col("frame_id").n_unique().alias("n_frames"))
            .sort(["game_id", "play_id"])
        )

        for game_id, play_id, n_frames in counts.iter_rows():
            num_graphs = max(0, n_frames - history_len - future_seq_len)
            meta.append((week, int(game_id), int(play_id), int(num_graphs)))

    meta = sorted(meta, key=lambda x: (x[0], x[1], x[2]))
    return meta


def expand_play_tuples(
    play_meta: List[Tuple[int, int, int, int]],
    allowed_plays: Optional[set] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Expand play metadata to frame-level tuples, optionally filtering by allowed play ids (week, game_id, play_id).
    """
    tuples: List[Tuple[int, int, int, int]] = []
    for week, game_id, play_id, num_graphs in play_meta:
        if allowed_plays is not None and (week, game_id, play_id) not in allowed_plays:
            continue
        for local_idx in range(num_graphs):
            tuples.append((week, game_id, play_id, local_idx))
    return tuples

    def load_plays(self) -> pl.DataFrame:
        """
        Load supplementary data (plays and game info).
        """
        filepath = self.data_dir / "supplementary_data.csv"
        
        if not filepath.exists():
            # Fallback to checking if games/plays exist separately or just return empty/raise
            raise FileNotFoundError(f"File not found: {filepath}")
            
        df = pl.read_csv(filepath, null_values=["NA"], infer_schema_length=5000)
        df = self._standardize_columns(df)
        return df

    def load_games(self) -> pl.DataFrame:
        # supplementary_data.csv seems to contain game info as well (game_id, season, week, etc.)
        # so we can just return unique game rows from there
        df = self.load_plays()
        return df.unique(subset=["game_id"])

    def _standardize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Standardize column names to snake_case if needed.
        """
        df = df.rename({col: col.lower() for col in df.columns})
        return df

    def standardize_tracking_directions(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Adjust coordinates so all plays go left to right.
        Standard NFL Big Data Bowl logic.
        """
        # Polars implementation of standardization
        # If playDirection is 'left', flip x and y derivatives
        
        # Check if play_direction column exists (usually 'playDirection')
        cols = df.columns
        if 'play_direction' not in cols and 'playdirection' not in cols and 'playDirection' not in cols:
            return df
            
        # Ensure we work with lowercase
        df = self._standardize_columns(df)
        
        # Determine plays to flip
        # x: 0 to 120 (including endzones)
        # y: 0 to 53.3
        
        # Condition: play_direction == 'left'
        # x_new = 120 - x
        # y_new = 53.3 - y
        # dir_new = (dir + 180) % 360
        # o_new = (o + 180) % 360
        
        # This is a bit complex in Polars lazy expression, doing eager for simplicity first
        # optimized later
        
        return df.with_columns([
            pl.when(pl.col('play_direction') == 'left')
            .then(120 - pl.col('x'))
            .otherwise(pl.col('x'))
            .alias('std_x'),
            
            pl.when(pl.col('play_direction') == 'left')
            .then(53.3 - pl.col('y'))
            .otherwise(pl.col('y'))
            .alias('std_y'),
             
             # Orientation
            pl.when(pl.col('play_direction') == 'left')
            .then((pl.col('o') + 180).mod(360))
            .otherwise(pl.col('o'))
            .alias('std_o'),
            
             # Direction (motion)
            pl.when(pl.col('play_direction') == 'left')
            .then((pl.col('dir') + 180).mod(360))
            .otherwise(pl.col('dir'))
            .alias('std_dir'),
        ])

if __name__ == "__main__":
    # Test
    loader = DataLoader(data_dir="/home/tanmay/Desktop/NFL")
    try:
        df = loader.load_week_data(1)
        print(df.head())
    except Exception as e:
        print(e)
