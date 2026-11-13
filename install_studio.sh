#!/usr/bin/env bash
# CheckurAI: user-space install of Physical AI Studio (backend xpu + UI). Does not touch arms/cameras.
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH="$HOME/.local/bin:$PATH"
uv --version
cd ~/physical-ai-studio/application/backend
uv python install 3.13
uv sync --extra xpu
uv run --no-sync python -c "import physicalai, lerobot, pyrealsense2, torch; print(\"BACKEND OK torch\", torch.__version__, \"xpu\", torch.xpu.is_available())"
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] || (curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash)
set +u; . "$NVM_DIR/nvm.sh"; set -u
cd ~/physical-ai-studio/application/ui
set +u; nvm install; nvm use; set -u
node -v; npm -v
npm ci
echo "ALL DONE"
