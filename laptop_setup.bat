@echo off
title Jarvis Environment Setup
echo 🚀 Starten met het opzetten van de omgeving...

:: 1. Controleer of Python aanwezig is
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ FOUT: Python is niet gevonden. Installeer Python (3.10 of 3.11 aanbevolen).
    pause
    exit /b
)

:: 2. Maak een virtuele omgeving aan als deze nog niet bestaat
if not exist venv (
    echo 📦 Virtuele omgeving aan het maken...
    python -m venv venv
)

:: 3. Activeer de omgeving
call venv\Scripts\activate.bat

:: 4. Update pip
python -m pip install --upgrade pip

:: 5. Installeer vereiste dependencies
echo 🛠️ Packages aan het installeren (dit kan even duren)...
pip install numpy torch sounddevice pyttsx3 websockets scipy whisper openai-whisper faster-whisper speechbrain

:: 6. Controleer op PyTorch (belangrijk voor modellen)
echo ⚙️ Controleer PyTorch installatie...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ✅ Setup voltooid!
echo ----------------------------------------------------
echo 🤖 Jarvis wordt gestart...
python JOUW_SCRIPT_NAAM.py
pause
