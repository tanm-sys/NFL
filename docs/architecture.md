# System Architecture

## Overview

The NFL Analytics Engine is a **Hybrid Graph-Transformer** model designed to solve the multi-agent trajectory prediction problem in American Football. It combines spatial reasoning (Graph Neural Networks) with temporal sequence modeling (Transformers) and strategic context awareness through learned embeddings.

### Design Philosophy

1. **Spatial Reasoning**: GATv2 layers model player-to-player interactions within each frame
2. **Temporal Modeling**: Transformer decoder predicts future trajectories
3. **Strategic Context**: Embeddings capture formation, alignment, and role information
4. **Multi-Task Learning**: Simultaneous trajectory prediction and coverage classification

### v2.0 Enhancements (P0-P3)

5. **Relative Predictions (P0)**: Predict displacements instead of absolute positions
6. **Motion History (P1)**: LSTM encodes past 5 frames of velocity/acceleration
7. **Scene Understanding (P3)**: Set Transformer captures global play dynamics
8. **Hierarchical Decoding (P3)**: Coarse-to-fine trajectory refinement

### v3.0 SOTA Edition (NEW)

9. **🦁 Lion Optimizer**: 15% faster convergence, less memory than AdamW
10. **📉 DropPath (Stochastic Depth)**: SOTA regularization from ViT, linearly increasing drop rates
11. **🔄 SWA Integration**: Stochastic Weight Averaging at 75% of training
12. **⚡ RTX 40 Optimizations**: bf16-mixed, TF32, Tensor Cores, cuDNN benchmark (torch.compile disabled for PyG compatibility)
13. **📦 New Libraries**: `lion-pytorch`, `einops`, `safetensors`, `timm`

**Production v2 defaults (configs/production.yaml):**
- 6× GATv2 layers, hidden_dim=128, heads=8, dropout=0.15, droppath_rate=0.08
- Probabilistic 8-mode decoder enabled by default
- bf16 mixed precision + SWA and cosine warmup scheduler
- Deterministic play splits persisted to `outputs/splits_production.json`
- Config snapshot written to `outputs/nfl_production_v2_config.json`

## High-Level Architecture

```mermaid
graph TB
    subgraph Input_Layer[Input Layer]
        Track[Tracking Data - x, y, s, a, dir, o, w]
        Context[Context - down, dist, box]
        Strategy[Strategic - role, formation, alignment]
    end
    
    subgraph Embedding_Layer[Embedding Layer]
        Track --> NodeEmb[Node Embedding - Linear 7→64]
        Context --> CtxEmb[Context Embedding - Linear 3→64]
        Strategy --> StratEmb[Strategic Embeddings - role, formation, alignment]
    end
    
    subgraph Fusion[Feature Fusion]
        NodeEmb --> Fuse[Residual Fusion]
        CtxEmb --> Fuse
        StratEmb --> Fuse
        Fuse --> FusedNode[Fused Node Features - 64-dim]
    end
    
    subgraph GNN[Multi-Layer GNN Encoder - 4 to 8 layers]
        FusedNode --> GAT1[GATv2 Layer 1 - + Residual + LayerNorm + DropPath]
        GAT1 --> GAT2[GATv2 Layer 2 - + Residual + LayerNorm + DropPath]
        GAT2 --> GATN[... More GATv2 Layers ...]
        GATN --> GATF[GATv2 Layer N - + Residual + LayerNorm + DropPath]
        GATF --> NodeRep[Node Representations - hidden_dim]
    end
    
    subgraph Decoder[Multi-Task Heads]
        NodeRep --> TrajDec[Trajectory Decoder - Transformer]
        TrajDec --> TrajOut[Trajectory Output - Nx10x2]
        
        NodeRep --> Pool[Global Mean Pool]
        Pool --> CovHead[Coverage Classifier - Linear 64→1]
        CovHead --> CovOut[Coverage Logits - Bx1]
    end
```

## Detailed Tensor Flow

### Input Processing

```mermaid
graph LR
    subgraph Node_Features[Node Features - Per Player]
        X["x, y, s, a, dir, o, w - [Total_Nodes, 7]"]
        Role["role_id - [Total_Nodes]"]
        Side["side_id - [Total_Nodes]"]
    end
    
    subgraph Graph_Features[Graph Features - Per Play]
        Ctx["down, dist, box - [Batch, 3]"]
        Form["formation_id - [Batch]"]
        Align["alignment_id - [Batch]"]
    end
    
    subgraph Edge_Features[Edge Features]
        EdgeIdx["edge_index - [2, Num_Edges]"]
        EdgeAttr["distance, angle, rel_speed, rel_dir, same_team - [Num_Edges, 5]"]
    end
    
    X --> EmbX["Linear(7, 64) - [Total_Nodes, 64]"]
    Role --> EmbRole["Embedding(5, 64) - [Total_Nodes, 64]"]
    Side --> EmbSide["Embedding(3, 32) - [Total_Nodes, 32]"]
    
    Ctx --> EmbCtx["Linear(3, 64) - [Batch, 64]"]
    Form --> EmbForm["Embedding(8, 64) - [Batch, 64]"]
    Align --> EmbAlign["Embedding(10, 64) - [Batch, 64]"]
```

### Embedding Fusion

The model uses **residual-style fusion** to combine node-level and graph-level features:

**Node Stream:**
$$h_{node} = \text{Linear}(x) + \text{Embedding}(role) + \text{Pad}(\text{Embedding}(side))$$

**Graph Stream:**
$$h_{graph} = \text{Linear}(context) + \text{Embedding}(formation) + \text{Embedding}(alignment)$$

**Broadcast and Fuse:**
$$h_{input} = h_{node} + \text{Broadcast}(h_{graph})$$

Where `Broadcast` expands graph-level embeddings to each node using the batch index.

### Graph Attention Network

The model uses **configurable stacked GATv2 layers** (default 4, up to 8) with residual connections and **DropPath (Stochastic Depth)** regularization:

```python
for i in range(num_gnn_layers):
    h_residual = h
    h = GATv2Conv(h, edge_index, edge_attr)  # edge_attr is 5D
    h = ReLU(h)
    h = Dropout(h, p=0.1)
    h = DropPath(h, drop_prob=i * droppath_rate / (num_layers - 1))  # SOTA regularization
    h = LayerNorm(h + h_residual)  # Residual connection
```

**DropPath** (Stochastic Depth):
- Randomly drops entire residual paths during training
- Drop probability increases linearly through layers
- Used in Vision Transformers, ConvNeXt, and modern architectures

**GATv2 Attention Mechanism:**

For each edge $(i, j)$:

$$\alpha_{ij} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}h_i \| \mathbf{W}h_j \| \mathbf{e}_{ij}]))}{\sum_{k \in \mathcal{N}(i)} \exp(\text{LeakyReLU}(\mathbf{a}^T [\mathbf{W}h_i \| \mathbf{W}h_k \| \mathbf{e}_{ik}]))}$$

Where:
- $h_i, h_j$ are node features
- $\mathbf{e}_{ij}$ are edge attributes (distance, angle, rel_speed, rel_dir, same_team - 5D)
- $\mathbf{W}$ is learnable weight matrix
- $\mathbf{a}$ is attention parameter vector

**Node Update:**
$$h_i' = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} h_j$$

### Trajectory Decoder

The trajectory decoder uses a **Transformer architecture** to predict future positions:

```mermaid
graph TD
    NodeEmb["Node Embeddings - [Total_Nodes, 64]"] --> Expand["Repeat 10 times - [Total_Nodes, 10, 64]"]
    QueryPos["Query Position Embeddings - [1, 10, 64]"] --> Expand
    Expand --> Add["Element-wise Add - [Total_Nodes, 10, 64]"]
    Add --> Trans["Transformer Encoder - 2 layers, 4 heads"]
    Trans --> Head["Linear(64, 2) - [Total_Nodes, 10, 2]"]
    Head --> Output["Trajectory Predictions - Δx, Δy for 10 frames"]
```

**Key Features:**
- **Non-autoregressive**: Predicts all 10 future frames simultaneously
- **Query embeddings**: Learned positional encodings distinguish t+1 from t+10
- **Per-agent prediction**: Each player gets independent trajectory

### Multi-Task Learning

The model optimizes two objectives simultaneously:

```mermaid
graph TD
    subgraph Forward_Pass
        Input[Batch Data] --> Model[NFLGraphTransformer]
        Model --> TrajPred["Trajectory Predictions - [Total_Nodes, 10, 2]"]
        Model --> CovPred["Coverage Logits - [Batch, 1]"]
    end
    
    subgraph Ground_Truth
        GTTraj["GT Trajectories - [Total_Nodes, 10, 2]"]
        GTCov["GT Coverage Labels - [Batch]"]
    end
    
    subgraph Loss_Computation
        TrajPred --> MSE["MSE Loss"]
        GTTraj --> MSE
        
        CovPred --> BCE["BCE with Logits Loss"]
        GTCov --> BCE
        
        MSE --> Weighted["Weighted Sum - L = L_traj + 0.5 * L_cov"]
        BCE --> Weighted
        Weighted --> TotalLoss["Total Loss"]
    end
```

**Loss Function:**
$$\mathcal{L} = \mathcal{L}_{trajectory} + \lambda \mathcal{L}_{coverage}$$

Where:
- $\mathcal{L}_{trajectory} = \text{MSE}(\hat{y}, y)$ - Mean Squared Error for trajectory
- $\mathcal{L}_{coverage} = \text{BCE}(\hat{c}, c)$ - Binary Cross-Entropy for coverage
- $\lambda = 0.5$ - Coverage loss weight

## Component Specifications

### 1. GraphPlayerEncoder

**Purpose**: Encode spatial state and strategic context into node embeddings

**Input Dimensions:**
- `x`: `[Total_Nodes, 7]` - Node features (x, y, s, a, dir, o, w)
- `edge_index`: `[2, Num_Edges]` - Graph connectivity
- `edge_attr`: `[Num_Edges, 5]` - Edge features (distance, angle, rel_speed, rel_dir, same_team)
- `context`: `[Batch, 3]` - Game context (down, dist, box)
- `role`: `[Total_Nodes]` - Player role IDs (0-4)
- `side`: `[Total_Nodes]` - Team side IDs (0-2)
- `formation`: `[Batch]` - Formation IDs (0-7)
- `alignment`: `[Batch]` - Alignment IDs (0-9)
- `frame_t`: `[Batch]` - Normalized frame position in play
- `history`: `[Total_Nodes, T, 4]` - Motion history (vel_x, vel_y, acc_x, acc_y)
- `batch`: `[Total_Nodes]` - Batch assignment vector

**Architecture:**
```python
GraphPlayerEncoder(
    input_dim=7,
    hidden_dim=64,
    heads=4,
    context_dim=3,
    edge_dim=5,           # 5D edge features
    num_layers=4,         # Configurable: 4-8 layers
    dropout=0.1,
    droppath_rate=0.1     # SOTA: Stochastic Depth
)
```

**Output:**
- `node_embeddings`: `[Total_Nodes, 64]`
- `attention_weights`: `(edge_index, alpha)` - Optional for visualization

### 2. TrajectoryDecoder

**Purpose**: Predict future trajectory from spatial embeddings

**Input:**
- `context_emb`: `[Total_Nodes, 64]` - Node embeddings from GNN

**Architecture:**
```python
TrajectoryDecoder(
    hidden_dim=64,
    num_heads=4,
    future_seq_len=10
)
```

**Output:**
- `predictions`: `[Total_Nodes, 10, 2]` - Future (Δx, Δy) for 10 frames

### 3. NFLGraphTransformer

**Purpose**: Complete end-to-end model

**Forward Pass:**
```python
def forward(data, return_attention_weights=False):
    # 1. Encode spatial + strategic features
    node_embs, attn = encoder(
        x=data.x,
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        context=data.context,
        role=data.role,
        side=data.side,
        formation=data.formation,
        alignment=data.alignment,
        batch=data.batch
    )
    
    # 2. Decode trajectories
    trajectories = decoder(node_embs)
    
    # 3. Classify coverage
    global_emb = global_mean_pool(node_embs, data.batch)
    coverage = classifier(global_emb)
    
    return trajectories, coverage, attn
```

## Edge Construction

Edges are created using a **radius graph** approach:

1. **Compute pairwise distances** between all players in a frame
2. **Create edges** where distance < 20 yards
3. **Compute edge attributes** (5D):
   - **Distance**: Euclidean distance in yards
   - **Relative angle**: Angle from player i to player j
   - **Relative speed**: Speed difference between players
   - **Relative direction**: Direction difference
   - **Same team**: Binary indicator (1 if same team, 0 otherwise)

```python
# Radius graph construction
edge_index = radius_graph(
    x=positions,  # [N, 2] (x, y coordinates)
    r=20.0,       # 20 yard radius
    batch=batch,  # Batch assignment
    loop=False    # No self-loops
)

# Edge attributes (5D)
edge_attr = compute_edge_features(positions, velocities, team_ids, edge_index)
# Returns [Num_Edges, 5]: [distance, angle, rel_speed, rel_dir, same_team]
```

## Model Capacity

**Parameter Count (v3.0 SOTA Edition):**

| Component | Parameters | Description |
|-----------|-----------|-------------|
| Node Embedding | 448 | Linear(7, 64) |
| Role Embedding | 320 | Embedding(5, 64) |
| Side Embedding | 96 | Embedding(3, 32) |
| Temporal Embedding | 6,400 | Embedding(100, 64) |
| Context Encoder | 256 | Linear(3, 64) |
| Formation Embedding | 512 | Embedding(8, 64) |
| Alignment Embedding | 640 | Embedding(10, 64) |
| GATv2 Layers (×4-8) | ~65K-130K | Configurable depth with DropPath |
| Layer Norms | 512-1024 | LayerNorm per layer |
| DropPath layers | 0 | Only modifies forward pass |
| SocialPoolingLayer | ~24K | Gated pairwise interactions |
| **TemporalHistoryEncoder (P1)** | ~33K | 2-layer LSTM for motion history |
| **SceneFlowEncoder (P3)** | ~25K | Set Transformer with inducing points |
| GoalConditionedDecoder (P3) | ~45K | Goal proposals + trajectory decoder |
| HierarchicalDecoder (P3) | ~20K | Coarse-to-fine prediction |
| Trajectory Decoder | ~33K | Transformer + head |
| Coverage Classifier | 65 | Linear(64, 1) |
| **Total** | **~810K - 1.2M** | Depends on configuration |

## Training Configuration

**Optimizer (v3.0 SOTA):** Lion (preferred) or AdamW
- Lion: ~3x lower LR, ~3x higher weight decay
- Learning rate: 1e-3 (AdamW) or 3e-4 (Lion)
- Weight decay: 1e-4 (AdamW) or 0.01 (Lion)

**Regularization:**
- Dropout: 0.1-0.15 (in GNN layers)
- **DropPath (Stochastic Depth)**: 0.1 rate, linearly increasing through layers
- Layer normalization after each GATv2 layer
- **SWA (Stochastic Weight Averaging)**: Enabled at 75% of training

**Batch Size:** 32-160 graphs (adjustable based on GPU memory, 160 optimal for RTX 4050 with bf16)

**Training Strategy:**
1. Load data with Polars DataLoader
2. Construct graphs with radius=20.0
3. Forward pass through model
4. Compute multi-task loss
5. Backpropagate and update weights
6. Log metrics (ADE, FDE, Coverage Accuracy)

## Inference Pipeline

```mermaid
graph LR
    Raw[Raw Tracking CSV] --> Load[DataLoader]
    Load --> Standard[Standardize Directions]
    Standard --> Features[Feature Engineering]
    Features --> Graph[Graph Construction]
    Graph --> Model[NFLGraphTransformer]
    Model --> Traj[Trajectory Predictions]
    Model --> Cov[Coverage Predictions]
    Traj --> Viz[Visualization]
    Cov --> Analysis[Strategic Analysis]
```

**Steps:**
1. Load tracking data for a play
2. Standardize coordinates (left-to-right)
3. Engineer features (normalize weights, encode roles)
4. Construct graph with edges
5. Run model inference
6. Extract predictions and attention weights
7. Visualize results

## Key Design Decisions

### Why GATv2 over GAT?

GATv2 uses **dynamic attention** where the attention mechanism can learn different patterns:
- GAT: $\alpha_{ij} = \text{softmax}(\mathbf{a}^T [\mathbf{W}h_i \| \mathbf{W}h_j])$
- GATv2: $\alpha_{ij} = \text{softmax}(\mathbf{a}^T \text{LeakyReLU}(\mathbf{W}[h_i \| h_j]))$

The LeakyReLU before the attention parameter allows more expressive attention patterns.

### Why 4 Layers?

Empirical testing showed:
- 2 layers: Underfitting, limited receptive field
- 4 layers: Optimal balance of capacity and training stability
- 6+ layers: Diminishing returns, harder to train

### Why Residual Connections?

Residual connections enable:
- **Gradient flow**: Prevents vanishing gradients in deep networks
- **Feature preservation**: Maintains low-level features alongside high-level abstractions
- **Training stability**: Easier optimization landscape

### Why Multi-Task Learning?

Training on both trajectory and coverage:
- **Shared representations**: Forces model to learn strategic patterns
- **Regularization**: Coverage task acts as auxiliary objective
- **Practical utility**: Both predictions useful for analysis

## Extensibility

The architecture is designed for easy extension:

1. **Add new strategic features**: Simply add new embedding layers
2. **Modify GNN depth**: Change `num_layers` parameter
3. **Alternative decoders**: Replace TrajectoryDecoder with RNN/LSTM
4. **Additional tasks**: Add new heads (e.g., play outcome prediction)
5. **Attention visualization**: Extract and plot attention weights

## References

- **GATv2**: Brody et al., "How Attentive are Graph Attention Networks?" (2021)
- **Transformers**: Vaswani et al., "Attention is All You Need" (2017)
- **Multi-Agent Prediction**: Alahi et al., "Social LSTM" (2016)
- **Graph Neural Networks**: Kipf & Welling, "Semi-Supervised Classification with GCNs" (2017)
