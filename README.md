# Raspberry Pi 5 Voice Bot (2 apps + golf visual)

Wat dit project nu doet:
- Pi app: Ollama bot + fullscreen energy-core visual die reageert op de AI-stem.
- Laptop app: gebruikt de microfoon en speakers van het verbonden apparaat, doet Whisper transcriptie, stuurt tekst naar de Pi en speelt het antwoord af.
- Tijdens het uitspreken van het antwoord stuurt de laptop live golf-info terug naar de Pi, zodat het scherm meebeweegt met de stem van de AI.
- Testfile op Pi: `pi_test_3d.py` bevat de losse shader-test waarop de live visual is gebaseerd.

## Installeren

### Raspberry Pi
```bash
bash install.sh pi
source .venv/bin/activate
```

### Laptop
Gebruik bij voorkeur Python 3.11 op Windows.
```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\activate
python -m pip install --upgrade pip
pip install faster-whisper websockets sounddevice pyttsx3 numpy
```

## Visual testen op Pi
```bash
source .venv/bin/activate
python pi_test_3d.py
```

Bediening:
- `SPATIE` = emotie wisselen
- `ESC` = afsluiten

## Pi server starten
```bash
source .venv/bin/activate
python pi_app.py --host 0.0.0.0 --port 8765 --model phi3:mini
```

## Laptop client starten
```powershell
python -u .\laptop_app.py --host 192.168.1.73 --port 8765 --ollama-model phi3:mini --whisper-model base
```

Live luisteren zoals in `test.py`:
```powershell
python -u .\laptop_app.py --host 192.168.1.73 --port 8765 --ollama-model phi3:mini --whisper-model base --mode live
```

Live luisteren met wake word:
```powershell
python -u .\laptop_app.py --host 192.168.1.73 --port 8765 --ollama-model phi3:mini --whisper-model base --mode live --wake-word hey
```

Optioneel kun je een andere microfoon kiezen:
```powershell
python -u .\laptop_app.py --host 192.168.1.73 --input-device 1
```

## Emoties aanpassen
In `pi_app.py`:
- `EMOTION_KEYWORDS` bepaalt welke woorden naar welke emotie gaan.
- `EMOTION_LEVELS` bepaalt hoe sterk de golf beweegt per emotie.
- `EMOTION_COLORS` bepaalt de kleur per emotie.

## Pi tools via de AI
De Pi-app kan nu ook simpele acties uitvoeren. Voor veiligheid werkt dit vooral binnen je home-map en projectmap.

Voorbeelden:
```text
maak een map test
lijst map test
maak bestand notitie.txt met tekst "hallo"
lees bestand notitie.txt
verwijder bestand notitie.txt
zoek online raspberry pi 5 temperatuur
wat is de cpu temperatuur
zet volume naar 50 procent
wat is de pi status
wat is het ip adres
```
