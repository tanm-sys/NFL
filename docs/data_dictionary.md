# Data Dictionary

> Complete reference for all features, encodings, and tensor shapes.

## 📋 Input Features

### Node Features (9D)

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 0 | `x` | float | [0, 120] | X position on field (yards) |
| 1 | `y` | float | [0, 53.33] | Y position on field (yards) |
| 2 | `speed` | float | [0, 12] | Player speed (yards/sec) |
| 3 | `acceleration` | float | [-5, 5] | Acceleration (yards/sec²) |
| 4 | `direction` | float | [0, 360] | Movement direction (degrees) |
| 5 | `velocity_x` | float | [-12, 12] | X velocity component |
| 6 | `velocity_y` | float | [-12, 12] | Y velocity component |
| 7 | `is_ball_carrier` | int | {0, 1} | Ball carrier indicator |
| 8 | `motion_intensity` | float | [0, 15] | Combined motion metric |

### Edge Features (5D)

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 0 | `distance` | float | [0, radius] | Euclidean distance (yards) |
| 1 | `angle` | float | [0, 2π] | Relative angle (radians) |
| 2 | `rel_speed` | float | [-15, 15] | Speed difference |
| 3 | `rel_direction` | float | [0, 360] | Direction difference |
| 4 | `same_team` | int | {0, 1} | Same team indicator |

### Context Features (3D)

| Index | Name | Type | Range | Description |
|-------|------|------|-------|-------------|
| 0 | `down` | float | [1, 4] | Normalized down number |
| 1 | `distance` | float | [0, 20] | Yards to first down |
| 2 | `box_count` | float | [0, 11] | Defenders in box |

---

## 🏷️ Strategic Embeddings

### Role Embedding (5 classes)

| ID | Role | Description |
|----|------|-------------|
| 0 | QB | Quarterback |
| 1 | RB/FB | Running Back / Fullback |
| 2 | WR/TE | Wide Receiver / Tight End |
| 3 | OL | Offensive Line |
| 4 | DEF | Defensive players |

### Side Embedding (3 classes)

| ID | Side | Description |
|----|------|-------------|
| 0 | Defense | Defensive team |
| 1 | Offense | Offensive team |
| 2 | Ball | Football |

### Formation Embedding (8 classes)

| ID | Formation |
|----|-----------|
| 0 | I_FORM |
| 1 | SHOTGUN |
| 2 | PISTOL |
| 3 | SINGLEBACK |
| 4 | JUMBO |
| 5 | WILDCAT |
| 6 | EMPTY |
| 7 | UNKNOWN |

### Alignment Embedding (10 classes)

| ID | Alignment |
|----|-----------|
| 0-3 | Offensive line positions |
| 4-6 | Receiver positions |
| 7-8 | Backfield positions |
| 9 | UNKNOWN |

---

## 📐 Tensor Shapes

### Input Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| `x` | `[N, 9]` | Node features |
| `edge_index` | `[2, E]` | Edge connectivity |
| `edge_attr` | `[E, 5]` | Edge features |
| `batch` | `[N]` | Batch assignment |
| `context` | `[B, 3]` | Play context |
| `current_pos` | `[N, 2]` | Current positions |
| `target` | `[N, 10, 2]` | Future positions |
| `history` | `[N, 5, 4]` | Motion history |
| `role` | `[N]` | Player roles |
| `side` | `[N]` | Team side |

### Output Tensors

| Tensor | Shape | Description |
|--------|-------|-------------|
| `predictions` | `[N, 10, 2]` | Trajectory predictions |
| `mu` | `[N, 10, K, 2]` | GMM means (K modes) |
| `sigma` | `[N, 10, K, 2]` | GMM std devs |
| `rho` | `[N, 10, K]` | GMM correlations |
| `mode_probs` | `[N, K]` | Mode probabilities |
| `coverage` | `[B, 1]` | Coverage classification |

---

## 📊 Ground Truth Labels

### Trajectory Target

```python
# Future positions [N, T, 2]
# T = 10 frames (1 second @ 10 Hz)

target[:, :, 0]  # X positions
target[:, :, 1]  # Y positions
```

### Coverage Label

```python
# Binary classification [B, 1]
# 0 = Zone coverage
# 1 = Man coverage
```

---

## 🗃️ CSV Schema

### tracking_week_*.csv

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | int | Game identifier |
| `play_id` | int | Play identifier |
| `nfl_id` | int | Player identifier |
| `display_name` | str | Player name |
| `frame_id` | int | Frame number (10 Hz) |
| `time` | str | Timestamp |
| `jersey_number` | int | Jersey number |
| `club` | str | Team abbreviation |
| `play_direction` | str | "left" or "right" |
| `x` | float | X position (yards) |
| `y` | float | Y position (yards) |
| `s` | float | Speed (yards/sec) |
| `a` | float | Acceleration |
| `dis` | float | Distance traveled |
| `o` | float | Orientation (degrees) |
| `dir` | float | Direction (degrees) |
| `event` | str | Play event |

### plays.csv

| Column | Type | Description |
|--------|------|-------------|
| `game_id` | int | Game identifier |
| `play_id` | int | Play identifier |
| `quarter` | int | Quarter (1-4) |
| `down` | int | Down (1-4) |
| `yards_to_go` | int | Yards to first down |
| `possession_team` | str | Offensive team |
| `defensive_team` | str | Defensive team |
| `offense_formation` | str | Offensive formation |
| `defenders_in_box` | int | Box count |
| `pass_result` | str | Pass outcome |
