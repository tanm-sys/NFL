# Data Dictionary

This document details the features used by the model, their sources, and their encodings.

## 1. Node Features (Tracking Data)
These features exist for **every player** (and the ball) at **every frame**.

| Feature | Type | Source | Description | Normalization |
| :--- | :--- | :--- | :--- | :--- |
| `std_x` | Float | Tracking | X coordinate (Length of field). | Standardized 0-120. |
| `std_y` | Float | Tracking | Y coordinate (Width of field). | Standardized 0-53.3. |
| `s` | Float | Tracking | Speed (yards/sec). | Raw. |
| `a` | Float | Tracking | Acceleration (yards/sec²). | Raw. |
| `std_dir` | Float | Tracking | Motion direction (degrees). | Standardized. |
| `std_o` | Float | Tracking | Player orientation (degrees). | Standardized. |
| `weight_norm` | Float | Roster | Player weight. | `(w - 200) / 50` |
| `role_id` | Int | Roster | Player's tactical role. | **Embedding (Dim 5)** |

### Role ID Mapping
*   `0`: Defensive Coverage (CB, S, LB)
*   `1`: Other Route Runner (WR, TE, RB)
*   `2`: Passer (QB)
*   `3`: Targeted Receiver
*   `4`: Unknown / Ball

## 2. Context Features (Play Level)
These features are constant for the entire play and are fused into the graph.

| Feature | Type | Source | Description | Encoding |
| :--- | :--- | :--- | :--- | :--- |
| `down` | Int | Plays | Current down (1-4). | Raw (1-4). |
| `yards_to_go` | Int | Plays | Distance to first down. | Raw. |
| `defenders_box`| Float| Plays | Defenders in the box. | `(N - 7) / 2` |

## 3. Strategic Features (Play Level)
High-level strategy signals fused via Embeddings.

### Offense Formation (`formation_id`)
Source: `offense_formation` -> `nn.Embedding(8)`
*   `0`: SHOTGUN
*   `1`: EMPTY
*   `2`: SINGLEBACK
*   `3`: PISTOL
*   `4`: I_FORM
*   `5`: JUMBO
*   `6`: WILDCAT
*   `7`: Unknown

### Receiver Alignment (`alignment_id`)
Source: `receiver_alignment` -> `nn.Embedding(10)`
*   `0`: 2x2
*   `1`: 3x1
*   `2`: 3x2
*   `3`: 2x1
*   `4`: 4x1
*   `5`: 1x1
*   `6`: 4x0
*   `7`: 3x3
*   `8`: 3x0
*   `9`: Unknown

## 4. Targets (Outputs)

### Trajectory (`y`)
*   **Shape**: `[Num_Nodes, Future_Frames, 2]`
*   **Content**: Future `(std_x, std_y)` coordinates for the next 10 frames (1.0 second).

### Coverage (`y_coverage`)
*   **Type**: Binary Classification
*   **0**: Man Coverage
*   **1**: Zone Coverage
