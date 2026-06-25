#!/usr/bin/env bash
set -euo pipefail
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv libsdl2-dev
python -m venv venv 2>/dev/null || { echo "venv fout"; exit; }
source venv/bin/activate
pip install moderngl pygame requests websockets langdetect
