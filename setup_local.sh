#!/bin/bash
# ============================================================================
# NFL Analytics Engine - Local Laptop Setup Script
# ============================================================================
# This script sets up a virtual environment and installs all dependencies
# for training on your local machine with NVIDIA GPU.

set -e  # Exit on error

echo "🏈 NFL Analytics Engine - Local Setup"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

# Step 1: Check GPU
echo -e "${YELLOW}[1/8] Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}✓ GPU detected${NC}"
else
    echo -e "${YELLOW}⚠ nvidia-smi not found. Training will use CPU.${NC}"
fi
echo ""

# Step 2: Check Python version
echo -e "${YELLOW}[2/8] Checking Python version...${NC}"
python_version=$(python3 --version)
echo "$python_version"
echo -e "${GREEN}✓ Python OK${NC}"
echo ""

# Step 3: Create virtual environment
echo -e "${YELLOW}[3/8] Creating virtual environment...${NC}"
if [ -d "$VENV_DIR" ]; then
    echo -e "${CYAN}   Virtual environment already exists at $VENV_DIR${NC}"
    echo -e "${CYAN}   Reusing existing environment...${NC}"
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created at $VENV_DIR${NC}"
fi
echo ""

# Step 4: Activate virtual environment
echo -e "${YELLOW}[4/8] Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Step 5: Upgrade pip
echo -e "${YELLOW}[5/8] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Step 6: Install PyTorch with CUDA
echo -e "${YELLOW}[6/8] Installing PyTorch with CUDA support...${NC}"
# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "   NVIDIA Driver: $CUDA_VERSION"
    
    # Install PyTorch with CUDA 12.8 (latest stable)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
    # CPU only
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install PyTorch Geometric
pip install torch-geometric

# Try optional extensions (may fail for newest PyTorch)
echo "   Attempting to install optional PyG extensions..."
pip install torch-scatter torch-sparse 2>/dev/null || \
    echo -e "${YELLOW}   ⚠ Optional PyG extensions not available. Core functionality will still work.${NC}"

echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Step 7: Install project dependencies
echo -e "${YELLOW}[7/8] Installing project dependencies...${NC}"
pip install -r requirements.txt
pip install -e .

# Install SOTA training libraries (v3.0)
echo "   Installing SOTA libraries (Lion, einops, safetensors, timm)..."
pip install lion-pytorch einops safetensors timm

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 8: Verify installation
echo -e "${YELLOW}[8/8] Verifying installation...${NC}"
python -c "
import torch
print(f'   PyTorch: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

import pytorch_lightning as pl
print(f'   PyTorch Lightning: {pl.__version__}')

import torch_geometric
print(f'   PyTorch Geometric: {torch_geometric.__version__}')

from src.models.gnn import NFLGraphTransformer
from src.train import NFLGraphPredictor
print('   ✓ All project imports successful!')
"
echo -e "${GREEN}✓ Installation verified${NC}"
echo ""

# Create directories
mkdir -p checkpoints logs outputs mlruns

# Check data
echo -e "${YELLOW}Checking training data...${NC}"
if [ -d "train" ]; then
    week_count=$(ls train/input_2023_w*.csv 2>/dev/null | wc -l)
    echo "   Found $week_count week files"
    if [ $week_count -gt 0 ]; then
        echo -e "${GREEN}✓ Training data found${NC}"
    else
        echo -e "${YELLOW}⚠ No training data found in train/ directory${NC}"
    fi
else
    echo -e "${YELLOW}⚠ train/ directory not found${NC}"
fi
echo ""

# Summary
echo "======================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "======================================"
echo ""
echo -e "${CYAN}IMPORTANT: Always activate the virtual environment before training:${NC}"
echo ""
echo -e "   ${GREEN}source .venv/bin/activate${NC}"
echo ""
echo "Then run:"
echo ""
echo "1. Quick sanity check (2 min):"
echo "   python train_production.py --config configs/sanity.yaml"
echo ""
echo "2. High-accuracy training:"
echo "   python train_production.py --config configs/high_accuracy.yaml"
echo ""
echo "3. Monitor with TensorBoard:"
echo "   tensorboard --logdir logs/"
echo ""
echo "🏈 Ready to train!"
