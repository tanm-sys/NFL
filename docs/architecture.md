# System Architecture

## Overview
The NFL Analytics Engine is a **Hybrid Graph-Transformer** model designed to solve the "Multi-Agent Trajectory Prediction" problem in American Football. It combines spatial reasoning (Graph Neural Networks) with temporal sequence modeling (Transformers) and strategic context awareness.

## Tensor Flow Diagram

```mermaid
graph TD
    subgraph Inputs
    X["Node Features<br/>(x, y, s, a, dir, o, w)"] -- "[B*N, 7]" --> EmbX["Linear -> [B*N, 64]"]
    Role["Player Role<br/>(Int)"] -- "[B*N]" --> EmbRole["Embedding -> [B*N, 64]"]
    
    Context["Context<br/>(Down, Dist, Box)"] -- "[B, 3]" --> EmbCtx["Linear -> [B, 64]"]
    Form["Formation<br/>(Int)"] -- "[B]" --> EmbForm["Embedding -> [B, 64]"]
    Align["Alignment<br/>(Int)"] -- "[B]" --> EmbAlign["Embedding -> [B, 64]"]
    end

    subgraph Fusion_Layer
    EmbX --> SumNodes(("Sum"))
    EmbRole --> SumNodes
    SumNodes -- "Node Emb [B*N, 64]" --> GNN_Input
    
    EmbCtx --> SumCtx(("Sum"))
    EmbForm --> SumCtx
    EmbAlign --> SumCtx
    SumCtx -- "Graph Emb [B, 64]" --> Broadcast
    Broadcast -- "Expand to Nodes" --> GNN_Input(("Sum"))
    end

    subgraph Encoder_GNN
    GNN_Input -- "[B*N, 64]" --> GAT1["GATv2Conv 1<br/>(Cross-Attention)"]
    GAT1 --> GAT2["GATv2Conv 2<br/>(Extract Attention)"]
    GAT2 -- "Final Node Emb [B*N, 64]" --> Pool[Global Mean Pool]
    end

    subgraph Heads
    GAT2 --> Decoder["Trajectory Decoder<br/>(Transformer)"]
    Decoder -- "Query Expansion" --> Forecast["Forecast Output<br/>[B*N, 10, 2]"]
    
    Pool -- "Global Emb [B, 64]" --> Classifier["Coverage Head<br/>(MLP)"]
    Classifier --> CovPred["Coverage Logits<br/>[B, 1]"]
    end
```

## Component Specifications

### 1. GraphPlayerEncoder
**Input Shapes**:
*   `x`: `[Total_Nodes, 7]` (Batch * ~22 Players)
*   `context`: `[Batch_Size, 3]`
*   `role`: `[Total_Nodes]` (Long)
*   `formation/alignment`: `[Batch_Size]` (Long)

**Fusion Logic**:
The model uses residual-style fusion to combine specific logic (Nodes) with global logic (Context).
1.  **Node Stream**: $h_{node} = Linear(x) + Embedding(role)$
2.  **Context Stream**: $h_{ctx} = Linear(c) + Embedding(formation) + Embedding(alignment)$
3.  **Combined**: $h_{in} = h_{node} + Broadcast(h_{ctx})$

### 2. GATv2 Layers
**Mechanism**:
*   Uses `edge_attr` (Distance, Angle) to modulate attention scores.
*   **Layer 1**: Aggregates neighborhood info.
*   **Layer 2**: Refines embeddings and extracts *Attention Weights* ($\alpha_{ij}$) for Explainability.

### 3. Trajectory Decoder
**Temporal Expansion**:
*   Takes spatial embeddings `[Total_Nodes, 64]`.
*   Repeats them `Future_Seq_Len` (10) times.
*   Adds learned `Query Position Embeddings` to distinguish $t+1$ from $t+10$.
*   Transformer Encoder processes the sequence.
*   Output Head projects to $\Delta x, \Delta y$.

## Training & Loss Flow

The model optimizes two objectives simultaneously (Multi-Task Learning).

```mermaid
graph TD
    subgraph Forward_Pass
    Batch[Input Batch] --> Model[NFLGraphTransformer]
    Model --> PredTraj["Trajectory Pred<br/>(x,y)"]
    Model --> PredCov["Coverage Logits<br/>(Man/Zone)"]
    end

    subgraph Targets
    GT_Traj[Ground Truth Path]
    GT_Cov[Ground Truth Label]
    end

    subgraph Loss_Calculation
    PredTraj & GT_Traj --> MSE[MSE Loss]
    PredCov & GT_Cov --> BCE[BCEWithLogits Loss]
    
    MSE --> WeightedSum(("Sum"))
    BCE -- "x 0.5" --> WeightedSum
    WeightedSum --> TotalLoss["Total Backward Loss"]
    end
```
3.  **Feature Engineering**:
    *   Standardizes play direction (Left-to-Right).
    *   Encodes categorical features (Formation, Role) into IDs.
    *   Normalizes continuous features (Weight, Down, Distance).
4.  **Graph Construction**: Converts frames into PyG `Data` objects with `edge_index` and `edge_attr`.
5.  **Inference**: The model processes the graph batch and outputs predictions.
