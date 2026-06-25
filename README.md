# Raspberry Pi 5 Voice Bot (2 apps + wave visual)

### Current project functionality:
- **Pi app:** Ollama bot + fullscreen energy-core visual that reacts to the AI voice.
- **Laptop app:** Uses the connected device's microphone and speakers, handles Whisper transcription, sends text to the Pi, and plays the response.
- **Live Sync:** While speaking the response, the laptop sends live wave information back to the Pi, so the screen moves in sync with the AI's voice.
- **Test file on Pi:** `pi_test_3d.py` contains the standalone shader test that the live visual is based on.

## Installation

### Raspberry Pi
`bash install.sh pi`
`source .venv/bin/activate`

### Laptop
Preferably use Python 3.11 on Windows.
`py -3.11 -m venv .venv311`
`.\.venv311\Scripts\activate`
`python -m pip install --upgrade pip`
`pip install faster-whisper websockets sounddevice pyttsx3 numpy`
(If available in the `laptop_setup.bat` script)

## Testing the visual on Pi
`source .venv/bin/activate`
`python pi_test_3d.py`

**Controls:**
- `SPACE` = Change emotion
- `ESC` = Exit

## Start Pi server
`source .venv/bin/activate`
`python pi_app.py --host 0.0.0.0 --port 8765 --model phi3:mini`

## Start Laptop client
`python -u .\laptop_app.py --host 192.168.1.73 --port 8765 --ollama-model phi3:mini --whisper-model base`

**Live listening as in `test.py`:**
`python -u .\laptop_app.py --host 192.168.1.73 --port 8765 --ollama-model phi3:mini --whisper-model base --mode live`

**Live listening with wake word:**
`python -u .\laptop_app.py --host 192.168.1.73 --port 8765 --ollama-model phi3:mini --whisper-model base --mode live --wake-word hey`

**Optional: select a different microphone:**
`python -u .\laptop_app.py --host 192.168.1.73 --input-device 1`

## Customizing Emotions
In `pi_app.py`:
- `EMOTION_KEYWORDS` determines which words map to which emotion.
- `EMOTION_LEVELS` determines the intensity of the wave movement per emotion.
- `EMOTION_COLORS` determines the color per emotion.

## Pi tools via AI
The Pi app can now perform simple actions. For security, this operates primarily within your home directory and project folder.

**Examples:**
- create folder test
- list folder test
- create file note.txt with text "hello"
- read file note.txt
- delete file note.txt
- search online raspberry pi 5 temperature
- what is the cpu temperature
- set volume to 50 percent
- what is the pi status
- what is the ip address
