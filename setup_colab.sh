#!/bin/bash
# ============================================================================
# NFL Analytics Engine - Production Training Setup Script for Google Colab
# ============================================================================
# This script automates the complete setup process on Google Colab

set -e  # Exit on error

echo "🏈 NFL Analytics Engine - Production Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check GPU
echo -e "${YELLOW}[1/8] Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo -e "${GREEN}✓ GPU detected${NC}"
else
    echo -e "${RED}✗ No GPU detected. Please enable GPU in Colab settings.${NC}"
    echo "Runtime → Change runtime type → GPU"
    exit 1
fi
echo ""

# Step 2: Check Python version
echo -e "${YELLOW}[2/8] Checking Python version...${NC}"
python_version=$(python --version)
echo "$python_version"
echo -e "${GREEN}✓ Python OK${NC}"
echo ""

# Step 3: Install PyTorch Geometric
echo -e "${YELLOW}[3/8] Installing PyTorch Geometric...${NC}"
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', ''))")

echo "PyTorch: $TORCH_VERSION, CUDA: $CUDA_VERSION"

# Install torch-geometric first (works without scatter/sparse in 2.7.0+)
pip install -q torch-geometric

# Try to install optional optimized ops (may fail for newest PyTorch versions)
echo "Attempting to install optional PyG extensions..."
WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION}.html"

# Check if wheels exist before attempting install
if curl -s --head "$WHEEL_URL" | grep -q "200 OK"; then
    pip install -q torch-scatter torch-sparse -f "$WHEEL_URL" 2>/dev/null || \
        echo -e "${YELLOW}⚠ Optional PyG extensions not available for this PyTorch version. Core functionality will still work.${NC}"
else
    echo -e "${YELLOW}⚠ Pre-built wheels not found for PyTorch ${TORCH_VERSION}. Using torch-geometric without optional extensions.${NC}"
fi

echo -e "${GREEN}✓ PyTorch Geometric installed${NC}"
echo ""

# Step 4: Install dependencies
echo -e "${YELLOW}[4/8] Installing production dependencies...${NC}"
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 5: Install project
echo -e "${YELLOW}[5/8] Installing NFL Analytics Engine...${NC}"
pip install -q -e .

# Install SOTA training libraries (v3.0)
echo "Installing SOTA libraries (Lion, einops, safetensors, timm)..."
pip install -q lion-pytorch einops safetensors timm

echo -e "${GREEN}✓ Project installed${NC}"
echo ""

# Step 6: Verify installation
echo -e "${YELLOW}[6/8] Verifying installation...${NC}"
python -c "
import torch
import pytorch_lightning as pl
import torch_geometric
import polars
import rich
from src.models.gnn import NFLGraphTransformer
from src.train import NFLGraphPredictor
print('All imports successful!')
"
echo -e "${GREEN}✓ Installation verified${NC}"
echo ""

# Step 7: Create directories
echo -e "${YELLOW}[7/8] Creating directories...${NC}"
mkdir -p checkpoints logs outputs mlruns
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Step 8: Check data
echo -e "${YELLOW}[8/8] Checking training data...${NC}"
if [ -d "train" ]; then
    week_count=$(ls train/input_2023_w*.csv 2>/dev/null | wc -l)
    echo "Found $week_count week files"
    if [ $week_count -gt 0 ]; then
        echo -e "${GREEN}✓ Training data found${NC}"
    else
        echo -e "${YELLOW}⚠ No training data found in train/ directory${NC}"
        echo "Please upload input_2023_w*.csv files"
    fi
else
    echo -e "${YELLOW}⚠ train/ directory not found${NC}"
    echo "Please create train/ directory and upload data"
fi
echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run sanity check:"
echo "   python train_production.py --config configs/sanity.yaml"
echo ""
echo "2. Run full production training:"
echo "   python train_production.py --config configs/production.yaml"
echo ""
echo "3. Monitor with TensorBoard:"
echo "   %load_ext tensorboard"
echo "   %tensorboard --logdir logs/"
echo ""
echo "🏈 Ready to train!"
