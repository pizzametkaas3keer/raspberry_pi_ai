#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-all}"
VENV_DIR=".venv"

ensure_venv() {
  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    rm -rf "${VENV_DIR}"
    if ! python3 -m venv --copies "${VENV_DIR}"; then
      echo "Venv aanmaken mislukt in $(pwd)."
      echo "Tip: gebruik een Linux-ext4 map (bijv. /home/admin/raspberry-ai) i.p.v. /media."
      exit 1
    fi
  fi

  if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Kon ${VENV_DIR}/bin/activate niet vinden."
    echo "Verplaats dit project naar /home/admin en probeer opnieuw."
    exit 1
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip
}

echo "Installatie gestart voor target: ${TARGET}"

if [[ "${TARGET}" == "pi" || "${TARGET}" == "all" ]]; then
  echo "Pi dependencies installeren..."
  sudo apt-get update
  sudo apt-get install -y python3 python3-pip python3-venv libsdl2-dev
  ensure_venv
  pip install requests websockets pygame moderngl
fi

if [[ "${TARGET}" == "laptop" || "${TARGET}" == "all" ]]; then
  echo "Laptop dependencies installeren..."
  ensure_venv
  pip install websockets faster-whisper sounddevice pyttsx3 numpy SpeechRecognition
fi

echo "Installatie klaar."
