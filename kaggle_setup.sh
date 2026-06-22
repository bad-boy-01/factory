#!/bin/bash
# Novel Video Factory v4 — Kaggle Setup Script
# Run once at the start of each Kaggle session
set -e

echo "=========================================="
echo " Novel Video Factory v4 — Kaggle Setup"
echo "=========================================="

# 1. System packages
echo "[1/5] Installing system packages..."
apt-get update -qq
apt-get install -y -qq zstd espeak-ng imagemagick libgl1-mesa-glx ffmpeg libsm6 libxext6 cmake g++

# Fix ImageMagick policy that blocks PDF/video operations
if [ -f /etc/ImageMagick-6/policy.xml ]; then
    sed -i 's/rights="none" pattern="PDF"/rights="read|write" pattern="PDF"/' /etc/ImageMagick-6/policy.xml
fi

# 2. Install and Start Ollama
echo "[2/5] Installing and starting Ollama..."
curl -fsSL https://ollama.com/install.sh | sh
nohup ollama serve > ollama.log 2>&1 &
sleep 5
ollama pull qwen2.5:7b

# 3. Python packages
echo "[3/5] Installing Python packages..."
# We use --no-warn-conflicts because Kaggle has pre-installed packages (like MNE and RAPIDS)
# that have conflicting requirements with MoviePy and Torch. These can be safely ignored.
pip install --quiet --no-cache-dir -r requirements.txt || echo "  (Some dependency conflicts occurred, which is normal on Kaggle. Continuing...)"

# Install xformers for VRAM savings
pip install --quiet xformers || echo "  xformers not available on this GPU — continuing without it"

echo "  Python packages installed ✓"

# 4. Verify Groq key (optional but recommended)
echo "[4/5] Checking Groq API key..."
if [ -n "$GROQ_API_KEY" ]; then
    echo "  GROQ_API_KEY found in environment ✓"
else
    echo "  GROQ_API_KEY not set — will use Ollama as LLM"
    echo "  TIP: Add GROQ_API_KEY to Kaggle Secrets for better LLM quality (free at console.groq.com)"
fi

# 5. Verify GPU
echo "[5/5] GPU check..."
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')" 2>/dev/null || echo "  torch not yet imported"

echo ""
echo "=========================================="
echo " Setup complete! Ready to run the pipeline."
echo "=========================================="
