#!/bin/bash
# chmod +x setup.sh

# Setup script for semantic-ids repository on RunPod
# Optimized for performance by using local storage for caches and venv

set -e  # Exit on error

# ============================
# PERFORMANCE OPTIMIZATIONS
# ============================

echo "=== Setting up optimized environment for fast startup ==="

# Set all caches to local fast storage (/root)
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export TORCH_HOME=/root/.cache/torch
export TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
export UV_CACHE_DIR=/root/.cache/uv
export PIP_CACHE_DIR=/root/.cache/pip
export UV_LINK_MODE=copy

# Tell uv to use local storage for the project environment
export UV_PROJECT_ENVIRONMENT=/root/.venv

# Also set for future sessions
ENV_EXPORTS="
export HF_HOME=/root/.cache/huggingface
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export TORCH_HOME=/root/.cache/torch
export TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
export UV_CACHE_DIR=/root/.cache/uv
export PIP_CACHE_DIR=/root/.cache/pip
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT=/root/.venv
export TERM=xterm-256color
"

# Create cache directories
mkdir -p /root/.cache/{huggingface,torch,torch_extensions,uv,pip}

# ============================
# PYTHON ENVIRONMENT SETUP
# ============================

echo "Setting up Python environment with uv..."

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install --upgrade pip uv
else
    echo "uv already installed"
fi

# Navigate to project directory
cd /workspace/semantic-ids

# Handle existing .venv in workspace
if [ -e "/workspace/semantic-ids/.venv" ]; then
    if [ -L "/workspace/semantic-ids/.venv" ]; then
        # It's a symlink, remove it
        echo "Removing existing .venv symlink..."
        rm /workspace/semantic-ids/.venv
    elif [ -d "/workspace/semantic-ids/.venv" ]; then
        # It's a directory, move it out of the way
        echo "Found existing .venv directory, backing it up..."
        rm -rf /workspace/semantic-ids/.venv.backup
        mv /workspace/semantic-ids/.venv /workspace/semantic-ids/.venv.backup
        echo "Moved old .venv to .venv.backup"
    fi
fi

# Run uv sync - it will use UV_PROJECT_ENVIRONMENT location
echo "Running uv sync with venv on local storage (/root/.venv)..."
uv sync --link-mode=copy

# Create symlink for convenience (so you can use .venv in workspace)
if [ ! -e "/workspace/semantic-ids/.venv" ]; then
    ln -s /root/.venv /workspace/semantic-ids/.venv
    echo "Created symlink from /workspace/semantic-ids/.venv to /root/.venv"
fi

# Activate the environment
source /root/.venv/bin/activate

# ============================
# DEPENDENCIES & FIXES
# ============================

echo "Handling special dependencies..."

# Handle flash-attn dependency issue
echo "Installing flash-attn with no build isolation..."
uv pip install flash-attn --no-build-isolation

# If you have any other special dependencies that need special handling, add them here
# For example, if unsloth needs special treatment:
# uv pip install unsloth --no-deps
# uv sync  # Re-sync to get other dependencies

# ============================
# SYSTEM PACKAGES
# ============================

echo "Installing system packages..."
apt update
apt install -y tmux htop nvtop

# ============================
# SHELL CONFIGURATION
# ============================

echo "Configuring shell environment..."

# Add to bashrc
if [ -f ~/.bashrc ]; then
    # Remove old exports if they exist (to avoid duplicates)
    sed -i '/export HF_HOME=/d' ~/.bashrc
    sed -i '/export HF_DATASETS_CACHE=/d' ~/.bashrc
    sed -i '/export TORCH_HOME=/d' ~/.bashrc
    sed -i '/export TORCH_EXTENSIONS_DIR=/d' ~/.bashrc
    sed -i '/export UV_CACHE_DIR=/d' ~/.bashrc
    sed -i '/export PIP_CACHE_DIR=/d' ~/.bashrc
    sed -i '/export UV_LINK_MODE=/d' ~/.bashrc
    sed -i '/export UV_PROJECT_ENVIRONMENT=/d' ~/.bashrc
    sed -i '/export TERM=xterm-256color/d' ~/.bashrc
    
    # Add all environment exports
    echo "$ENV_EXPORTS" >> ~/.bashrc
    echo "Added environment exports to ~/.bashrc"
fi

# Add to bash_profile
if [ -f ~/.bash_profile ]; then
    sed -i '/export HF_HOME=/d' ~/.bash_profile
    sed -i '/export HF_DATASETS_CACHE=/d' ~/.bash_profile
    sed -i '/export TORCH_HOME=/d' ~/.bash_profile
    sed -i '/export TORCH_EXTENSIONS_DIR=/d' ~/.bash_profile
    sed -i '/export UV_CACHE_DIR=/d' ~/.bash_profile
    sed -i '/export PIP_CACHE_DIR=/d' ~/.bash_profile
    sed -i '/export UV_LINK_MODE=/d' ~/.bash_profile
    sed -i '/export UV_PROJECT_ENVIRONMENT=/d' ~/.bash_profile
    sed -i '/export TERM=xterm-256color/d' ~/.bash_profile
    
    echo "$ENV_EXPORTS" >> ~/.bash_profile
    echo "Added environment exports to ~/.bash_profile"
fi

# Add to zshrc if using zsh
if [ -f ~/.zshrc ]; then
    sed -i '/export HF_HOME=/d' ~/.zshrc
    sed -i '/export HF_DATASETS_CACHE=/d' ~/.zshrc
    sed -i '/export TORCH_HOME=/d' ~/.zshrc
    sed -i '/export TORCH_EXTENSIONS_DIR=/d' ~/.zshrc
    sed -i '/export UV_CACHE_DIR=/d' ~/.zshrc
    sed -i '/export PIP_CACHE_DIR=/d' ~/.zshrc
    sed -i '/export UV_LINK_MODE=/d' ~/.zshrc
    sed -i '/export UV_PROJECT_ENVIRONMENT=/d' ~/.zshrc
    sed -i '/export TERM=xterm-256color/d' ~/.zshrc
    
    echo "$ENV_EXPORTS" >> ~/.zshrc
    echo "Added environment exports to ~/.zshrc"
fi

# ============================
# SSH SETUP
# ============================

echo "Setting up SSH..."
if [ -d "/workspace/.ssh" ]; then
    mkdir -p ~/.ssh
    cp -r /workspace/.ssh/* ~/.ssh/
    chmod 600 ~/.ssh/id_ed25519
    chmod 700 ~/.ssh
    echo "SSH keys configured"
fi

# ============================
# VERIFY INSTALLATION
# ============================

echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# ============================
# FINAL STATUS
# ============================

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo ""
echo "Performance optimizations applied:"
echo "  • Virtual environment on local storage: /root/.venv"
echo "  • All caches set to /root/.cache/ (fast local storage)"
echo "  • UV link mode set to 'copy' to avoid hardlink issues"
echo "  • UV project environment set to /root/.venv"
echo ""
echo "Quick commands:"
echo "  • Activate environment: source /root/activate.sh"
echo "  • Update dependencies: cd /workspace/semantic-ids && uv sync"
echo "  • Pre-download models: python /root/download_models.py"
echo ""
echo "Tips for faster training:"
echo "  1. Pre-download models to local cache"
echo "  2. Copy training data to local: cp -r /workspace/semantic-ids/data /root/data"
echo "  3. Use local paths in your scripts when possible"
echo ""