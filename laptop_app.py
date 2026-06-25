# ═══════════════════════════════════════════════════════════════════════════════
# ⚙️  SETTINGS - PAS DEZE INSTELLINGEN AAN NAAR WENS
# ═══════════════════════════════════════════════════════════════════════════════

# Fix Windows console encoding voor emoji support
import sys
import io
import os
import logging
from datetime import datetime
import argparse
import asyncio
import json
import math
import platform
import queue
import subprocess
import tempfile
import threading
import time
import wave
import numpy as np
import pyttsx3
import sounddevice as sd
import torch
import websockets
import atexit

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Voorkom SpeechBrain symlink warning op Windows
import warnings
if sys.platform == 'win32':
    warnings.filterwarnings('ignore', category=UserWarning, 
                          message='.*symlink.*')

# ═══════════════════════════════════════════════════════════════════════════════
# 🛠️  MEMORY MANAGEMENT - OPTIMALISATIE VOOR GROTE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

import gc

# Model singletons om memory te besparen
_speechbrain_model = None
_whisper_models = {}  # Cache voor Whisper models per model type
_speechbrain_model_loaded = False

def get_speechbrain_model():
    """Haal SpeechBrain model op (singleton pattern)."""
    global _speechbrain_model, _speechbrain_model_loaded
    if not _speechbrain_model_loaded:
        try:
            from speechbrain.inference.speaker import SpeakerRecognition as SBSpeakerRecognition
            debug_log("SpeechBrain model laden...", "MEMORY")
            _speechbrain_model = SBSpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb"
            )
            _speechbrain_model_loaded = True
            debug_log("SpeechBrain model geladen (singleton)", "MEMORY")
            # Forceer garbage collection na laden
            gc.collect()
        except Exception as e:
            debug_log(f"SpeechBrain model load failed: {e}", "ERROR")
            _speechbrain_model = None
            _speechbrain_model_loaded = False
    return _speechbrain_model

def get_whisper_model(model_name: str = "base"):
    """Haal Whisper model op met caching."""
    if model_name not in _whisper_models:
        try:
            import whisper
            debug_log(f"Whisper model laden: {model_name}", "MEMORY")
            _whisper_models[model_name] = whisper.load_model(model_name)
            # Forceer garbage collection na laden
            gc.collect()
        except Exception as e:
            debug_log(f"Whisper model load failed: {e}", "ERROR")
            return None
    return _whisper_models[model_name]

def cleanup_models():
    """Cleanup grote models om geheugen vrij te maken."""
    global _speechbrain_model, _whisper_models, _speechbrain_model_loaded
    if _speechbrain_model is not None:
        del _speechbrain_model
        _speechbrain_model = None
        _speechbrain_model_loaded = False
        debug_log("SpeechBrain model verwijderd", "MEMORY")
    for model in list(_whisper_models.keys()):
        del _whisper_models[model]
        debug_log(f"Whisper model verwijderd: {model}", "MEMORY")
    gc.collect()
    debug_log("Garbage collection uitgevoerd", "MEMORY")

# ═══════════════════════════════════════════════════════════════════════════════

# 🌐 CLIENT INSTELLINGEN
DEFAULT_HOST = "192.x.x.x"        # Default server host
DEFAULT_PORT = 8765              # Default server poort
DEFAULT_MODEL = "phi3:mini"     # Default AI model
DEFAULT_WHISPER_MODEL = "base"  # Default speech recognition model

# 🔒 TAILSCALE COMPATIBILITEIT
TAILSCALE_MODE = True            # Tailscale compatibiliteit - ondersteunt Tailscale IP's (100.x.x.x)
AUTO_DETECT_TAILSCALE = True      # Automatisch Tailscale IP's detecteren in netwerk
USE_WSS_FOR_TAILSCALE = True     # Gebruik WSS voor Tailscale verbindingen indien beschikbaar
TAILSCALE_IP_RANGE = "100."      # Tailscale IP prefix (standaard 100.x.x.x)
# Tailscale: Draai `tailscale up` op beide apparaten om ze te verbinden, of gebruik ingebouwde Tailscale bridge op server

# 🔊 AUDIO INSTELLINGEN
DEFAULT_RECORD_SECONDS = 8      # Standaard opnameduur in seconden

# 🎛️  VOICE INSTELLINGEN
DEFAULT_LANGUAGE = "nl"          # Standaard taalcode (nl, en, de, etc)
DEFAULT_WAKE_WORD = "jarvis"     # Wake word voor live modus
DEFAULT_MODE = "press"           # Chat modus (press, live, hold)
DEFAULT_INPUT_DEVICE = None      # Audio input device (None voor auto)

# 🔐 SSH INSTELLINGEN
SSH_ENABLED = True               # SSH commando functionaliteit enabled
SSH_TIMEOUT = 10                 # SSH commando timeout in seconden
SSH_PREFIXES = ["ssh:", "cmd:", "exec:"]  # SSH commando prefixes

# 🎨 VISUAL SETTINGS
EMOTION_DISPLAY = True           # Toon emotionen op scherm
AUDIO_LEVEL_DISPLAY = True       # Toon audio niveau
TTS_ENABLED = True              # Text-to-Speech aan

# 🚀 PERFORMANCE INSTELLINGEN
CONNECTION_TIMEOUT = 30          # Seconden voor connectie timeout
MESSAGE_TIMEOUT = 30            # Seconden voor message response
MAX_RETRIES = 3                 # Max aantal verbindings pogingen
AUDIO_BUFFER_SIZE = 512         # Audio buffer grootte voor performance
ASYNC_POOL_SIZE = 5             # Max async tasks concurrent

# ═══════════════════════════════════════════════════════════════════════════════
# 🛠️  DEBUG MODE - VOLLEDIGE LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

DEBUG_VERBOSE = False # Zet op True voor volledige logging
DEBUG_LOG_FILE = "jarvis_debug.log"

# Setup logging
if DEBUG_VERBOSE:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(DEBUG_LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    debug_logger = logging.getLogger('JarvisDebug')
    debug_logger.info("=" * 60)
    debug_logger.info("JARVIS DEBUG MODE GESTART")
    debug_logger.info(f"Python version: {sys.version}")
    debug_logger.info(f"Working directory: {os.getcwd()}")
    debug_logger.info("=" * 60)
else:
    debug_logger = None

def debug_log(message: str, category: str = "INFO"):
    """Debug logging helper"""
    if DEBUG_VERBOSE and debug_logger:
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] [{category}] {message}"
        debug_logger.info(log_msg)
        print(log_msg)

# ═══════════════════════════════════════════════════════════════════════════════

# 🎯 VOICE TRAINING ZINNEN - Voor betere spraakherkenning
VOICE_TRAINING_SENTENCES = [
    # Algemene zinnen met diverse klanken
    "Hallo, dit is een test van mijn stem.",
    "Ik spreek nu duidelijk en rustig.",
    "De honden blaffen in de buurt.",
    "Het is vandaag een mooie dag om te werken.",
    "Mijn naam is spraakherkenning training.",

    # Lange zinnen voor betere continue spraak
    "Dit is een langere zin om te testen hoe goed de spraakherkenning werkt met continue spraak.",
    "We gaan nu testen of het systeem mijn stem kan herkennen bij verschillende zinnen en uitspraken.",

    # Zinnen met cijfers en specifieke woorden
    "Ik heb drie honden en twee katten.",
    "De tijd is nu kwart over drie in de middag.",
    "Dit is nummer één, twee, drie, vier, vijf.",

    # Korte zinnen met sterke klanken
    "Goedemorgen allemaal.",
    "Tot ziens en bedankt.",
    "Ja dat klopt helemaal.",

    # Zinnen met Nederlandse klanken
    "De koeien in de wei staan te grazen.",
    "Mijn fiets heeft een lege band.",
    "De lucht is vandaag erg mooi en helder.",

    # Zinnen met variatie in intonatie
    "Wow! Dit is echt geweldig nieuws.",
    "Helaas, dat werkt niet zo goed.",
    "Natuurlijk! Dat is een goed idee."
]

def get_training_sentence(index: int, total: int) -> str:
    """Haal een training zin op basis van index."""
    # Gebruik modulo om door de lijst te cyclen
    sentence = VOICE_TRAINING_SENTENCES[index % len(VOICE_TRAINING_SENTENCES)]
    return sentence

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 WHISPER TRANSCRIPTIE FUNCTIES
# ═══════════════════════════════════════════════════════════════════════════════

async def transcribe_audio(audio: np.ndarray, sample_rate: int, language: str = "nl", model: str = "base") -> str:
    """Transcribeer audio met Whisper voor spraakherkenning training."""
    try:
        debug_log(f"Whisper transcript aanvraag: model={model}, audio shape={audio.shape}", "STT")

        # Gebruik singleton pattern voor Whisper model
        whisper_model = get_whisper_model(model)

        if whisper_model is None:
            debug_log("Whisper model kon niet worden geladen", "ERROR")
            return ""

        debug_log(f"Whisper singleton model geladen: {model}", "STT")

        # Transcribeer audio
        result = whisper_model.transcribe(
            audio,
            language=language,
            fp16=False  # Gebruik fp32 voor compatibiliteit
        )

        transcript = result["text"].strip()
        debug_log(f"Whisper transcript: '{transcript}'", "STT")

        return transcript
    except Exception as e:
        import traceback
        debug_log(f"Whisper transcribe failed: {e}", "ERROR")
        traceback.print_exc()
        print(f"⚠️  Whisper transcribe error: {e}")
        return ""

# ═══════════════════════════════════════════════════════════════════════════════

# Debug module import
try:
    from debug_module import (logger)
    DEBUG_ENABLED = True
except ImportError:
    DEBUG_ENABLED = False
    print("⚠️  Debug module niet gevonden, debug features uitgeschakeld")


def normalize_language_tag(language: str | None, engine: str) -> str | None:
    if not language:
        return None

    lowered = language.lower()
    if engine == "google":
        if lowered == "nl":
            return "nl-NL"
        if lowered == "en":
            return "en-US"
    return lowered


class SpeechSyncState:
    def __init__(self):
        self.speaking = False
        self.pulse_until = 0.0
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            self.speaking = True
            self.pulse_until = time.monotonic() + 0.18

    def pulse(self):
        with self.lock:
            self.pulse_until = time.monotonic() + 0.18

    def stop(self):
        with self.lock:
            self.speaking = False
            self.pulse_until = 0.0

    def level(self) -> float:
        with self.lock:
            if not self.speaking:
                return 0.0
            if time.monotonic() < self.pulse_until:
                return 0.95
            return 0.28


class FasterWhisperTranscriber:
    def __init__(self, model_name: str):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")

    def transcribe_wav(self, wav_path: str, language: str = "nl") -> str:
        segments, _ = self.model.transcribe(
            wav_path,
            language=normalize_language_tag(language, "whisper"),
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt="Dit gesprek is in het Nederlands.",
        )
        return " ".join(segment.text for segment in segments).strip()

    def transcribe_array(self, audio: np.ndarray, language: str | None = None) -> str:
        segments, _ = self.model.transcribe(
            audio,
            language=normalize_language_tag(language, "whisper"),
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt="Dit gesprek is in het Nederlands.",
        )
        return " ".join(segment.text for segment in segments).strip()


class SpeechRecognitionTranscriber:
    def __init__(self):
        import speech_recognition as sr

        self.sr = sr
        self.recognizer = sr.Recognizer()

    def transcribe_wav(self, wav_path: str, language: str = "nl-NL") -> str:
        with self.sr.AudioFile(wav_path) as source:
            audio = self.recognizer.record(source)
        return self.recognizer.recognize_google(
            audio,
            language=normalize_language_tag(language, "google"),
        ).strip()

    def transcribe_array(self, audio: np.ndarray, language: str | None = None) -> str:
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp.close()
        int_audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(temp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(int_audio.tobytes())
        google_language = normalize_language_tag(language or "nl", "google")
        return self.transcribe_wav(temp.name, google_language)


def create_transcriber(model_name: str):
    try:
        transcriber = FasterWhisperTranscriber(model_name)
        print(f"Transcriptie: faster-whisper ({model_name})")
        return transcriber
    except Exception as exc:
        print(f"faster-whisper niet bruikbaar op deze pc: {exc}")

    try:
        transcriber = SpeechRecognitionTranscriber()
        print("Transcriptie fallback actief: SpeechRecognition via internet")
        return transcriber
    except Exception as exc:
        raise RuntimeError(
            "Geen werkende spraak-naar-tekst gevonden. Installeer SpeechRecognition als fallback met: "
            "pip install SpeechRecognition"
        ) from exc


def get_audio_device_summary() -> tuple[str, str]:
    devices = sd.query_devices()
    default_input, default_output = sd.default.device

    input_name = "Onbekend"
    output_name = "Onbekend"

    if default_input is not None and default_input >= 0:
        input_name = devices[default_input]["name"]
    if default_output is not None and default_output >= 0:
        output_name = devices[default_output]["name"]

    return input_name, output_name


class SettingsManager:
    """Centrale settings manager voor alle configuraties."""

    def __init__(self, settings_file: str = "settings.json"):
        from pathlib import Path

        self.settings_file = Path(settings_file)
        self.settings = self._load_settings()

    def _load_settings(self) -> dict:
        """Laad settings uit JSON of gebruik defaults."""
        default_settings = {
            "connection": {
                "host": "localhost",
                "port": 8765,
                "auto_connect": False
            },
            "ai": {
                "ollama_model": "phi3:mini",
                "whisper_model": "base",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "audio": {
                "record_seconds": 8,
                "language": "nl",
                "mode": "press",
                "wake_word": "jarvis",
                "input_device": None,
                "sample_rate": 16000,
                "noise_suppression": True
            },
            "speaker_recognition": {
                "voice_profiles_file": "voice_profiles.json",
                "threshold": 0.7,  # SpeechBrain threshold (distance < 0.7 = match)
                "enabled": True
            },
            "permissions": {
                "speaker_permissions_file": "speaker_permissions.json",
                "roles": {
                    "admin": {
                        "description": "Volledige toegang tot alle functies",
                        "full_chat": True,
                        "ssh_commands": True,
                        "system_commands": True,
                        "file_operations": True,
                        "max_audio_level": 1.0,
                        "request_timeout": 60
                    },
                    "user": {
                        "description": "Standaard gebruiker met basis rechten",
                        "full_chat": True,
                        "ssh_commands": True,
                        "system_commands": False,
                        "file_operations": True,
                        "max_audio_level": 1.0,
                        "request_timeout": 30
                    },
                    "guest": {
                        "description": "Gast met beperkte rechten",
                        "full_chat": True,
                        "ssh_commands": False,
                        "system_commands": False,
                        "file_operations": False,
                        "max_audio_level": 0.7,
                        "request_timeout": 20
                    },
                    "restricted": {
                        "description": "Zeer beperkte toegang (alleen chat)",
                        "full_chat": True,
                        "ssh_commands": False,
                        "system_commands": False,
                        "file_operations": False,
                        "max_audio_level": 0.5,
                        "request_timeout": 15
                    }
                },
                "known": {
                    "full_chat": True,
                    "ssh_commands": True,
                    "system_commands": True,
                    "file_operations": True,
                    "max_audio_level": 1.0,
                    "request_timeout": 30
                },
                "unknown": {
                    "full_chat": True,
                    "ssh_commands": False,
                    "system_commands": False,
                    "file_operations": False,
                    "max_audio_level": 0.5,
                    "request_timeout": 15
                }
            },
            "interface": {
                "show_welcome": True,
                "show_timestamps": False,
                "audio_feedback": True,
                "confirm_dangerous": True
            },
            "performance": {
                "audio_chunk_size": 1024,
                "cache_models": True,
                "lazy_loading": True
            },
            "advanced": {
                "debug_mode": False,
                "verbose_logging": False,
                "save_recordings": False,
                "recordings_path": "recordings/"
            }
        }

        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                # Merge met defaults
                return self._deep_merge(default_settings, loaded)
            except Exception as e:
                print(f"⚠️  Kon settings niet laden: {e}")
                return default_settings
        return default_settings

    def _deep_merge(self, base: dict, override: dict) -> dict:
        """Merge twee dictionaries diep."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _save_settings(self):
        """Sla settings op naar JSON."""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)
            print(f"💾 Settings opgeslagen in {self.settings_file}")
        except Exception as e:
            print(f"⚠️  Kon settings niet opslaan: {e}")

    def get_connection_settings(self) -> dict:
        """Haal verbindingsinstellingen op."""
        return self.settings.get("connection", {})

    def get_ai_settings(self) -> dict:
        """Haal AI instellingen op."""
        return self.settings.get("ai", {})

    def get_audio_settings(self) -> dict:
        """Haal audio instellingen op."""
        return self.settings.get("audio", {})

    def get_speaker_recognition_settings(self) -> dict:
        """Haal speaker recognition instellingen op."""
        return self.settings.get("speaker_recognition", {})

    def get_permissions_settings(self) -> dict:
        """Haal rechten instellingen op."""
        return self.settings.get("permissions", {})

    def set_connection_settings(self, settings: dict):
        """Stel verbindingsinstellingen in."""
        self.settings["connection"] = {**self.settings["connection"], **settings}
        self._save_settings()

    def set_ai_settings(self, settings: dict):
        """Stel AI instellingen in."""
        self.settings["ai"] = {**self.settings["ai"], **settings}
        self._save_settings()

    def set_audio_settings(self, settings: dict):
        """Stel audio instellingen in."""
        self.settings["audio"] = {**self.settings["audio"], **settings}
        self._save_settings()

    def set_permissions(self, permissions: dict):
        """Stel rechten in."""
        self.settings["permissions"] = {**self.settings["permissions"], **permissions}
        self._save_settings()

    def get_default_arguments(self) -> dict:
        """Genereer command line arguments uit settings."""
        args = {}

        conn = self.get_connection_settings()
        args["host"] = conn.get("host", "localhost")
        args["port"] = conn.get("port", 8765)

        ai = self.get_ai_settings()
        args["ollama_model"] = ai.get("ollama_model", "phi3:mini")
        args["whisper_model"] = ai.get("whisper_model", "base")

        audio = self.get_audio_settings()
        args["record_seconds"] = audio.get("record_seconds", 8)
        args["language"] = audio.get("language", "nl")
        args["mode"] = audio.get("mode", "press")
        args["wake_word"] = audio.get("wake_word", "jarvis")
        args["input_device"] = audio.get("input_device")

        return args

    def get_interface_settings(self) -> dict:
        """Haal interface instellingen op."""
        return self.settings.get("interface", {})

    def get_performance_settings(self) -> dict:
        """Haal performance instellingen op."""
        return self.settings.get("performance", {})

    def get_advanced_settings(self) -> dict:
        """Haal advanced instellingen op."""
        return self.settings.get("advanced", {})

    def set_interface_settings(self, settings: dict):
        """Stel interface instellingen in."""
        self.settings["interface"] = {**self.settings["interface"], **settings}
        self._save_settings()

    def set_performance_settings(self, settings: dict):
        """Stel performance instellingen in."""
        self.settings["performance"] = {**self.settings["performance"], **settings}
        self._save_settings()

    def set_advanced_settings(self, settings: dict):
        """Stel advanced instellingen in."""
        self.settings["advanced"] = {**self.settings["advanced"], **settings}
        self._save_settings()


class SpeakerRecognition:
    """Speaker recognition systeem met training functionaliteit."""

    def __init__(self, voice_profiles_path: str = None, settings_manager: SettingsManager = None):
        debug_log("SpeakerRecognition.__init__ gestart", "INIT")

        try:
            # Gebruik singleton pattern voor SpeechBrain model
            self.speaker_recognition = get_speechbrain_model()
            debug_log("SpeechBrain singleton model opgehaald", "INIT")

            if self.speaker_recognition is not None:
                self.use_speechbrain = True
                if DEBUG_ENABLED and logger:
                    logger.info("SpeechBrain speaker recognition loaded successfully")
                print("✅ SpeechBrain speaker recognition geladen met ECAPA-VOXCELEB model (singleton)")
                debug_log("SpeechBrain actief - ECAPA-VOXCELEB model (singleton)", "SUCCESS")
            else:
                self.use_speechbrain = False
                print("⚠️  SpeechBrain model kon niet worden geladen, using fallback mode")
                print("⚠️  Fallback mode heeft beperkingen: kan niet perfect discrimineren tussen geluiden")
                debug_log("Fallback mode geactiveerd (model load fail)", "FALLBACK")
        except ImportError as e:
            debug_log(f"SpeechBrain import failed: {e}", "ERROR")

            if DEBUG_ENABLED and logger:
                logger.warning("SpeechBrain not available, using fallback speaker recognition",
                            solution="Install speechbrain with: pip install speechbrain",
                            original_error=e)
            self.speaker_recognition = None
            self.use_speechbrain = False
            print("⚠️  SpeechBrain not available - using fallback mode")
            print("⚠️  Fallback mode heeft beperkingen: kan niet perfect discrimineren tussen geluiden")
            print("💡 Voor perfecte speaker recognition: installeer speechbrain en draai PowerShell als Administrator")
            debug_log("Fallback mode geactiveerd (import fail)", "FALLBACK")

        from pathlib import Path

        # Gebruik settings manager of fallback naar directe path
        if settings_manager:
            speaker_settings = settings_manager.get_speaker_recognition_settings()
            self.voice_profiles_path = Path(speaker_settings.get("voice_profiles_file", "voice_profiles.json"))
            # SpeechBrain threshold: veel hoger dan fallback omdat embeddings discriminerender zijn
            # Fallback: 0.25 (feature-based, minder discriminerend)
            # SpeechBrain: 0.7 (professioneel, veel discriminerender)
            default_threshold = 0.7 if self.use_speechbrain else 0.25
            self.threshold = speaker_settings.get("threshold", default_threshold)
            debug_log(f"Threshold ingesteld op {self.threshold} (SpeechBrain: {self.use_speechbrain})", "INIT")
        else:
            self.voice_profiles_path = Path(voice_profiles_path or "voice_profiles.json")
            default_threshold = 0.7 if self.use_speechbrain else 0.25
            self.threshold = default_threshold
            debug_log(f"Threshold ingesteld op {self.threshold} (SpeechBrain: {self.use_speechbrain})", "INIT")

        self.voice_profiles = self._load_voice_profiles()
        # Eén SpeechBrain-instantie via de singleton (get_speechbrain_model).
        # Voormalig werd hier een tweede, ongebruikt model in _load_model geladen (~1 GB RAM).

    def _load_voice_profiles(self) -> dict:
        """Laad opgeslagen voice profiles uit JSON."""
        if self.voice_profiles_path.exists():
            try:
                with open(self.voice_profiles_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)

                # Backward compatibility: als het oude format is, converteer naar nieuw
                if isinstance(loaded, dict):
                    return loaded
                else:
                    print("⚠️  Oud voice_profiles formaat gevonden, converteren naar nieuw formaat")
                    return {}
            except Exception as e:
                print(f"⚠️  Kon voice profiles niet laden: {e}")
                return {}
        return {}

    def _save_voice_profiles(self):
        """Sla voice profiles op naar JSON."""
        try:
            with open(self.voice_profiles_path, 'w', encoding='utf-8') as f:
                json.dump(self.voice_profiles, f, indent=2)
            print(f"💾 Voice profiles opgeslagen: {len(self.voice_profiles)} stem(men)")
        except Exception as e:
            print(f"⚠️  Kon voice profiles niet opslaan: {e}")

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        """Extraheer stem embedding van audio."""
        debug_log(f"extract_embedding gestart (audio shape: {audio.shape}, sample_rate: {sample_rate})", "EMBED")

        if not self.use_speechbrain:
            debug_log("Using fallback embedding (audio features)", "EMBED")
            if DEBUG_ENABLED and logger:
                logger.debug("Using fallback embedding (audio features)")
            return self._extract_fallback_embedding(audio, sample_rate)

        # Gebruik de singleton via self.speaker_recognition (zie __init__).
        if self.speaker_recognition is None:
            debug_log("SpeechBrain singleton niet beschikbaar, using fallback embedding", "EMBED")
            if DEBUG_ENABLED and logger:
                logger.warning("SpeechBrain singleton niet beschikbaar, using fallback embedding")
            return self._extract_fallback_embedding(audio, sample_rate)

        try:
            debug_log("SpeechBrain embedding extractie gestart", "EMBED")
            start_time = time.time()

            # Resample als nodig
            if sample_rate != 16000:
                from scipy import signal
                num_samples = int(len(audio) * 16000 / sample_rate)
                audio = signal.resample(audio, num_samples)
                debug_log(f"Audio resampled: {sample_rate} -> 16000 Hz", "EMBED")

            # SpeechBrain verwacht 1D audio array
            # Converteer naar tensor met correcte shape (batch_size, audio_length)
            audio_tensor = torch.tensor(audio).unsqueeze(0)  # (1, audio_length)

            debug_log(f"Audio tensor shape: {audio_tensor.shape}", "EMBED")

            # Extract embedding - SpeechBrain speaker recognition gebruikt encode_batch
            with torch.no_grad():
                # encode_batch verwacht shape (batch, audio_samples)
                embedding = self.speaker_recognition.encode_batch(audio_tensor)

            extract_time = time.time() - start_time
            debug_log(f"SpeechBrain embedding geëxtraheerd in {extract_time:.3f}s", "EMBED")
            debug_log(f"Embedding shape: {embedding.shape}", "EMBED")

            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            debug_log(f"SpeechBrain embedding extractie failed: {e}", "ERROR")
            if DEBUG_ENABLED and logger:
                logger.error("Fout bij embedding extractie met SpeechBrain",
                            solution="Using fallback embedding",
                            original_error=e)
            print(f"⚠️  Fout bij embedding extractie: {e}, using fallback")
            return self._extract_fallback_embedding(audio, sample_rate)

    def _extract_fallback_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Fallback embedding op basis van audio features (geen SpeechBrain nodig)."""
        try:
            # Resample naar 16kHz indien nodig
            if sample_rate != 16000:
                from scipy import signal
                num_samples = int(len(audio) * 16000 / sample_rate)
                audio = signal.resample(audio, num_samples)

            # Extract basis features (wordt discriminator gebruikt voor sprekeridentificatie)
            import scipy.signal as signal

            # 1. MFCC-like features (Mel-frequency cepstral coefficients - approximatie)
            # Gebruik meerdere spectral features
            f, t, Sxx = signal.spectrogram(audio, fs=16000, nperseg=512, noverlap=256)
            spectral_features = np.array([
                np.mean(Sxx),      # Gemiddelde energie
                np.std(Sxx),       # Variatie in spectrum
                np.median(Sxx),     # Median spectrum
                np.max(Sxx),       # Maximum energie
                np.percentile(Sxx, 25),
                np.percentile(Sxx, 50),
                np.percentile(Sxx, 75),
                np.percentile(Sxx, 90),
            ])

            # 2. Temporal features (audio amplitude statistiek)
            temporal_features = np.array([
                np.mean(np.abs(audio)),      # RMS amplitude
                np.std(np.abs(audio)),       # Amplitude variatie
                np.max(np.abs(audio)),       # Peak amplitude
                np.min(np.abs(audio)),       # Min amplitude
                np.percentile(np.abs(audio), 90),  # 90e percentiele
                np.percentile(np.abs(audio), 50),  # Median amplitude
                np.percentile(np.abs(audio), 10),  # 10e percentiele
            ])

            # 3. Zero crossing rate (temporaal kenmerk)
            zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
            zcr_feature = np.array([zero_crossings / len(audio)])

            # 4. Spectral centroid (frequentie zwaartpunt)
            spectral_centroids = []
            for t in range(Sxx.shape[1]):
                spectral_slice = Sxx[:, t]
                if np.sum(spectral_slice) > 0:
                    centroid = np.sum(f * spectral_slice) / np.sum(spectral_slice)
                    spectral_centroids.append(centroid)
            centroid_feature = np.array([np.mean(spectral_centroids) if spectral_centroids else 0])

            # 5. Spectral rolloff (energy verdeling over frequentie)
            spectral_rolloff = []
            for t in range(Sxx.shape[1]):
                spectral_slice = Sxx[:, t]
                # Energie in onderste 50% vs bovenste 50%
                half_idx = len(spectral_slice) // 2
                lower_energy = np.sum(spectral_slice[:half_idx])
                upper_energy = np.sum(spectral_slice[half_idx:])
                if lower_energy + upper_energy > 0:
                    rolloff = upper_energy / (lower_energy + upper_energy)
                    spectral_rolloff.append(rolloff)
            rolloff_feature = np.array([np.mean(spectral_rolloff) if spectral_rolloff else 0.5])

            # Combine alle features
            features = np.concatenate([
                spectral_features,
                temporal_features,
                zcr_feature,
                centroid_feature,
                rolloff_feature
            ])

            # Normalize features
            features = features / (np.linalg.norm(features) + 1e-8)

            # Padding naar vaste grootte (512 dimensions voor compatibiliteit met opgeslagen profiles)
            target_dim = 512
            if len(features) < target_dim:
                features = np.pad(features, (0, target_dim - len(features)))
            else:
                features = features[:target_dim]

            return features

        except Exception as e:
            if DEBUG_ENABLED and logger:
                logger.error("Fout bij fallback embedding extractie",
                            solution="Using zero embedding",
                            original_error=e)
            print(f"⚠️  Fout bij fallback embedding: {e}")
            # Return zero embedding als laatste fallback
            return np.zeros(512)

    def train_voice(self, name: str, audio: np.ndarray, sample_rate: int = 16000, role: str = "user", transcript: str = "") -> bool:
        """Train een nieuwe stem of voeg sample toe aan bestaande stem met naam en rol."""
        embedding = self.extract_embedding(audio, sample_rate)
        if embedding is None:
            print("❌ Kon geen embedding extracten voor training")
            if DEBUG_ENABLED and logger:
                logger.error("Embedding extraction failed for voice training",
                            solution="Check audio device and try again")
            return False

        # Als profile al bestaat, voeg embedding toe aan lijst
        if name in self.voice_profiles:
            existing_profile = self.voice_profiles[name]
            if isinstance(existing_profile, dict):
                embeddings = existing_profile.get("embeddings", [])
                transcripts = existing_profile.get("transcripts", [])
                current_role = existing_profile.get("role", role)
            else:
                # Handle oudere format
                if isinstance(existing_profile, list):
                    embeddings = existing_profile
                else:
                    embeddings = [existing_profile]
                transcripts = []
                current_role = role

            # Voeg nieuwe embedding toe
            embeddings.append(embedding.tolist())
            # Voeg transcript toe als beschikbaar
            if transcript:
                transcripts.append(transcript)
            total_samples = len(embeddings)

            # Update profile
            self.voice_profiles[name] = {
                "embeddings": embeddings,
                "transcripts": transcripts,
                "role": current_role,
                "sample_count": total_samples,
                "created": existing_profile.get("created", time.time()) if isinstance(existing_profile, dict) else time.time(),
                "updated": time.time()
            }

            debug_log(f"Extra embedding toegevoegd aan {name}, totaal: {total_samples} samples, {len(transcripts)} transcripts", "TRAINING")
            print(f"✅ Extra sample toegevoegd voor: {name} (Totaal: {total_samples} opnames)")
            if transcript:
                print(f"   📝 Transcript: '{transcript}'")
        else:
            # Nieuw profile met lijst van embeddings
            transcripts = [transcript] if transcript else []
            self.voice_profiles[name] = {
                "embeddings": [embedding.tolist()],
                "transcripts": transcripts,
                "role": role,
                "sample_count": 1,
                "created": time.time(),
                "updated": time.time()
            }
            debug_log(f"Nieuw profile aangemaakt voor {name} met transcript", "TRAINING")
            print(f"✅ Stem getraind voor: {name} (Rol: {role})")
            if transcript:
                print(f"   📝 Transcript: '{transcript}'")

        self._save_voice_profiles()

        if DEBUG_ENABLED and logger:
            sample_count = self.voice_profiles[name].get("sample_count", 1)
            transcript_count = len(self.voice_profiles[name].get("transcripts", []))
            logger.log_resource("VoiceProfile", "TRAINED", f"Name: {name}, Role: {role}, Samples: {sample_count}, Transcripts: {transcript_count}")
        return True

    def identify_speaker(self, audio: np.ndarray, sample_rate: int = 16000, threshold: float = None) -> str | None:
        """Identificeer spreker van audio."""
        debug_log(f"identify_speaker gestart (audio shape: {audio.shape})", "RECOGNITION")


        if not self.voice_profiles:
            debug_log("Geen voice profiles beschikbaar", "RECOGNITION")
            return None

        # Threshold hangt af van de embedding-modus: SpeechBrain-embeddings zijn
        # genormaliseerd (afstand ~0.0-1.5, drempel 0.7), fallback-features hebben
        # een heel andere schaal (drempel 0.25). Kies per modus als niets is meegegeven.
        if threshold is None:
            threshold = self.threshold
            if not self.use_speechbrain and threshold > 0.4:
                # Ongeldige fallback-drempel (vermoedelijk uit settings voor SpeechBrain)
                # corrigeren zodat herkenning überhaupt kan slagen.
                threshold = 0.25
        debug_log(f"Threshold: {threshold} (SpeechBrain: {self.use_speechbrain})", "RECOGNITION")

        debug_log("Embedding extractie starten...", "RECOGNITION")
        embedding = self.extract_embedding(audio, sample_rate)

        if embedding is None:
            debug_log("Embedding extractie failed", "RECOGNITION")
            return None

        debug_log(f"Embedding geëxtraheerd, shape: {embedding.shape}", "RECOGNITION")

        # Vergelijk met alle opgeslagen profiles
        debug_log(f"Vergelijken met {len(self.voice_profiles)} opgeslagen profiles", "RECOGNITION")
        best_match = None
        best_score = float('inf')
        best_match_details = {}

        for name, profile in self.voice_profiles.items():
            # Handle nieuwe dictionary format
            if isinstance(profile, dict):
                embeddings = profile.get("embeddings", [])
                # Als embeddings een lijst is, gebruik de eerste of average
                if isinstance(embeddings, list) and len(embeddings) > 0:
                    # Gebruik alle embeddings voor betere nauwkeurigheid
                    stored_embeddings = [np.array(emb) for emb in embeddings if isinstance(emb, (list, np.ndarray)) and len(emb) > 0]
                    if not stored_embeddings:
                        continue
                else:
                    continue
            else:
                # Handle oude format (direct embedding array of lijst)
                if isinstance(profile, list):
                    stored_embeddings = [np.array(emb) for emb in profile if isinstance(emb, (list, np.ndarray)) and len(emb) > 0]
                else:
                    stored_embeddings = [np.array(profile)]

            if len(stored_embeddings) == 0:
                debug_log(f"Lege embedding voor {name}, overslaan", "RECOGNITION")
                continue

            debug_log(f"{name}: {len(stored_embeddings)} embeddings beschikbaar", "RECOGNITION")

            # Bereken gemiddelde similarity met alle embeddings
            similarities = []
            for stored_emb in stored_embeddings:
                similarity = np.dot(embedding, stored_emb) / (np.linalg.norm(embedding) * np.linalg.norm(stored_emb) + 1e-8)
                similarities.append(similarity)

            avg_similarity = np.mean(similarities)
            distance = 1 - avg_similarity

            debug_log(f"{name}: avg_similarity={avg_similarity:.4f}, avg_distance={distance:.4f} (van {len(similarities)} samples)", "RECOGNITION")

            best_match_details[name] = {
                "avg_similarity": avg_similarity,
                "avg_distance": distance,
                "sample_count": len(similarities)
            }

            if distance < best_score:
                best_score = distance
                best_match = name

        debug_log(f"Best match: {best_match} (distance: {best_score:.4f})", "RECOGNITION")

        # Return alleen als score goed genoeg is (fallback heeft beperkingen)
        if best_score < threshold:
            details = best_match_details.get(best_match, {})
            debug_log(f"Match accepted: {best_match} (distance {best_score:.4f} < threshold {threshold}, {details.get('sample_count', 1)} samples)", "SUCCESS")
            return best_match
        # Alles boven threshold = onbekend
        debug_log(f"Match rejected: distance {best_score:.4f} >= threshold {threshold}", "REJECTION")
        return None

    def list_voices(self) -> list[str]:
        """Geef lijst van getrainde stemmen."""
        return list(self.voice_profiles.keys())

    def add_training_samples(self, name: str, audio_samples: list[np.ndarray], sample_rate: int = 16000, role: str = "user") -> bool:
        """Voeg extra training samples toe aan een bestaande voice profile."""
        if name not in self.voice_profiles:
            return False

        # Haal bestaande profile op
        profile = self.voice_profiles[name]

        # Handle dictionary format
        if isinstance(profile, dict):
            existing_embeddings = profile.get("embeddings", [])
            current_role = profile.get("role", role)
        else:
            # Handle oudere format
            if isinstance(profile, list):
                existing_embeddings = profile
            else:
                existing_embeddings = [profile]
            current_role = role

        # Extract nieuwe embeddings
        new_embeddings = []
        for audio in audio_samples:
            embedding = self.extract_embedding(audio, sample_rate)
            if embedding is not None:
                new_embeddings.append(embedding.tolist())

        if not new_embeddings:
            return False

        # Combineer met bestaande embeddings
        all_embeddings = existing_embeddings + new_embeddings

        # Update profile
        self.voice_profiles[name] = {
            "embeddings": all_embeddings,
            "role": current_role,
            "sample_count": len(all_embeddings),
            "updated": time.time()
        }

        # Save naar file
        self._save_voice_profiles()

        print(f"💾 {len(new_embeddings)} extra opnames toegevoegd aan {name} (totaal: {len(all_embeddings)})")
        debug_log(f"{len(new_embeddings)} extra opnames toegevoegd aan {name}, totaal: {len(all_embeddings)}", "TRAINING")

        return True


class SpeakerPermissions:
    """Rechten systeem voor bekende en onbekende sprekers."""

    def __init__(self, settings_manager: SettingsManager = None):
        if settings_manager:
            self.settings_manager = settings_manager
            perms = settings_manager.get_permissions_settings()
            self.permissions = perms
        else:
            # Fallback naar direct JSON voor backward compatibility
            from pathlib import Path
            self.permissions_file = Path("speaker_permissions.json")
            self.permissions = self._load_permissions_direct()
            self.settings_manager = None

    def get_role_permissions(self, role: str) -> dict:
        """Haal permissions op voor een specifieke rol."""
        roles = self.permissions.get("roles", {})
        return roles.get(role, {})

    def get_speaker_role(self, speaker_name: str) -> str:
        """Haal rol op voor een specifieke spreker."""
        if not speaker_name:
            return "unknown"

        # Check of spreker specifieke rechten heeft (legacy)
        if speaker_name in self.permissions.get("speakers", {}):
            return self.permissions["speakers"][speaker_name].get("role", "user")

        # Return default user rol voor bekende sprekers
        return "user"

    def _load_permissions_direct(self) -> dict:
        """Laad per-spreker rechten uit JSON (direct fallback)."""
        if self.permissions_file.exists():
            try:
                with open(self.permissions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Kon permissions niet laden: {e}")
                return {"known": {}, "unknown": {"full_chat": True, "ssh_commands": False}}
        return {"known": {}, "unknown": {"full_chat": True, "ssh_commands": False}}

    def _load_permissions(self) -> dict:
        """Laad per-spreker rechten uit JSON."""
        if self.settings_manager:
            return self.settings_manager.get_permissions_settings()

        if self.permissions_file.exists():
            try:
                with open(self.permissions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Kon permissions niet laden: {e}")
        return {"known": {}, "unknown": {"full_chat": True, "ssh_commands": False}}

    def _save_permissions(self):
        """Sla per-spreker rechten op naar JSON."""
        if self.settings_manager:
            self.settings_manager.set_permissions(self.permissions)
        else:
            try:
                with open(self.permissions_file, 'w', encoding='utf-8') as f:
                    json.dump(self.permissions, f, indent=2)
                print("💾 Permissions opgeslagen")
            except Exception as e:
                print(f"⚠️  Kon permissions niet opslaan: {e}")

    def get_permissions(self, speaker_name: str | None, speaker_rec: SpeakerRecognition = None) -> dict:
        """Haal rechten op voor een spreker (met rol support)."""
        if not speaker_name:
            # Onbekende spreker - gebruik unknown permissions
            return self.permissions.get("unknown", {
                "full_chat": True,
                "ssh_commands": False,
                "system_commands": False,
                "file_operations": False,
                "max_audio_level": 0.5,
                "request_timeout": 15
            })

        # Check of spreker specifieke rechten heeft (legacy format)
        if speaker_name in self.permissions.get("known", {}):
            return self.permissions["known"][speaker_name]

        # Haal rol op van spreker (via voice profiles)
        role = "user"  # default
        if speaker_rec:
            # Check voice profiles voor rol
            voice_data = speaker_rec.voice_profiles.get(speaker_name, {})
            if isinstance(voice_data, dict):
                role = voice_data.get("role", "user")

        # Haal rol-specifieke permissions op
        role_perms = self.get_role_permissions(role)
        if role_perms:
            return role_perms

        # Fallback naar known permissions
        return self.permissions.get("known", {
            "full_chat": True,
            "ssh_commands": True,
            "system_commands": True,
            "file_operations": True,
            "max_audio_level": 1.0,
            "request_timeout": 30
        })

    def set_speaker_permissions(self, speaker_name: str, permissions: dict):
        """Stel rechten in voor een specifieke spreker."""
        if "known" not in self.permissions:
            self.permissions["known"] = {}

        self.permissions["known"][speaker_name] = permissions
        self._save_permissions()

    def set_unknown_permissions(self, permissions: dict):
        """Stel rechten in voor onbekende sprekers."""
        self.permissions["unknown"] = permissions
        self._save_permissions()

    def list_speakers(self) -> list[str]:
        """Geef lijst van sprekers met specifieke rechten."""
        return list(self.permissions.get("known", {}).keys())

    def check_permission(self, speaker_name: str | None, action: str, speaker_rec: SpeakerRecognition = None) -> bool:
        """Check of een spreker een specifieke actie mag uitvoeren."""
        perms = self.get_permissions(speaker_name, speaker_rec)
        return perms.get(action, False)


def configure_input_device(input_device: int | None):
    if input_device is not None:
        _, current_output = sd.default.device
        sd.default.device = (input_device, current_output)


def trim_and_normalize_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = audio.reshape(-1).astype(np.float32)
    if len(mono) == 0:
        return mono

    abs_audio = np.abs(mono)
    threshold = max(0.015, float(np.max(abs_audio)) * 0.12)
    active_indices = np.where(abs_audio >= threshold)[0]

    if len(active_indices) == 0:
        return mono

    pad = int(sample_rate * 0.15)
    start = max(0, int(active_indices[0]) - pad)
    end = min(len(mono), int(active_indices[-1]) + pad)
    clipped = mono[start:end]

    peak = float(np.max(np.abs(clipped))) if len(clipped) else 0.0
    if peak > 0:
        clipped = np.clip(clipped / peak * 0.92, -1.0, 1.0)

    return clipped


def write_wave_file(audio_array: np.ndarray, wav_path: str, sample_rate: int = 16000):
    """Schrijf audio array naar WAV bestand."""
    int_audio = np.clip(audio_array * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_audio.tobytes())


def record_audio(seconds: int, sample_rate: int = 16000) -> tuple[str, float]:
    print(f"Opnemen voor {seconds} seconden...")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    audio_float = audio.astype(np.float32) / 32768.0
    cleaned_audio = trim_and_normalize_audio(audio_float, sample_rate)
    rms = float(np.sqrt(np.mean(np.square(cleaned_audio)))) if len(cleaned_audio) else 0.0
    level = max(0.0, min(1.0, rms * 12.0))
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp.close()
    int_audio = np.clip(cleaned_audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(temp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_audio.tobytes())

    # Cleanup audio arrays
    del audio
    del audio_float
    del cleaned_audio
    del int_audio
    gc.collect()

    return temp.name, level


def configure_emotional_voice(tts, emotion: str = "neutral"):
    """Configureer pyttsx3 met emotionele instellingen."""
    voices = tts.getProperty("voices")
    
    # Stem voorkeuren per emotie voor meer variatie
    emotion_voice_preferences = {
        "neutral": ["dutch", "netherlands", "nederlands", "female"],
        "happy": ["dutch", "netherlands", "nederlands", "female"],
        "sad": ["dutch", "netherlands", "nederlands", "male"],
        "angry": ["dutch", "netherlands", "nederlands", "male"],
    }
    
    preferred_markers = emotion_voice_preferences.get(emotion, emotion_voice_preferences["neutral"])
    
    # Probeer verschillende stemmen per emotie
    chosen_voice_id = None
    for voice in voices:
        haystack = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')} {getattr(voice, 'languages', [])}".lower()
        if any(marker in haystack for marker in preferred_markers):
            chosen_voice_id = voice.id
            break
    
    # Fallback naar andere stemmen als geen nederlandse
    if not chosen_voice_id:
        fallback_markers = ["english", "en-gb", "en-us", "female", "male"]
        for voice in voices:
            haystack = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')}".lower()
            if any(marker in haystack for marker in fallback_markers):
                chosen_voice_id = voice.id
                break
    
    if chosen_voice_id:
        tts.setProperty("voice", chosen_voice_id)
    
    # Emotionele instellingen via rate en volume
    emotion_settings = {
        "neutral": {"rate": 150, "volume": 0.9},
        "happy": {"rate": 175, "volume": 1.0},   # Sneller, luider
        "sad": {"rate": 120, "volume": 0.7},     # Langzamer, stiller
        "angry": {"rate": 180, "volume": 1.0},  # Snel, luider
    }
    
    settings = emotion_settings.get(emotion, emotion_settings["neutral"])
    tts.setProperty("rate", settings["rate"])
    tts.setProperty("volume", settings["volume"])


def configure_jarvis_voice(tts):
    """Standaard configuratie voor initialisatie."""
    configure_emotional_voice(tts, "neutral")


def speak(engine, text: str, emotion: str = "neutral"):
    """Speak met emotionele instellingen."""
    # Configureer stem voor emotie
    configure_emotional_voice(engine, emotion)
    engine.say(text)
    engine.runAndWait()


def estimate_speech_duration(text: str) -> float:
    """Duur schatting voor pyttsx3 met emotionele variatie."""
    word_count = max(1, len(text.split()))
    # Pyttsx3 is wat trager (0.55 seconden per woord)
    return max(2.0, min(45.0, 1.0 + (word_count * 0.55)))


def speech_level_at(elapsed: float, duration: float) -> float:
    progress = max(0.0, min(1.0, elapsed / max(duration, 0.1)))
    attack = min(1.0, elapsed / 0.35)
    release = min(1.0, (duration - elapsed) / 0.65)
    envelope = max(0.0, min(attack, release))
    word_pulse = 0.5 + (0.5 * math.sin(elapsed * 7.0))
    slow_breath = 0.5 + (0.5 * math.sin((progress * math.tau) - 1.2))
    return max(0.0, min(1.0, envelope * (0.34 + word_pulse * 0.28 + slow_breath * 0.18)))


async def send_visual_state(ws, emotion: str, text: str, audio_level: float):
    payload = {
        "type": "visual",
        "emotion": emotion,
        "text": text,
        "audio_level": max(0.0, min(1.0, audio_level)),
    }
    await ws.send(json.dumps(payload))


async def open_ws(uri: str):
    return await websockets.connect(
        uri,
        max_size=5_000_000,
        ping_interval=20,
        ping_timeout=120,
        close_timeout=3,
        open_timeout=120,
    )


async def wait_for_acceptance(ws, timeout: float = 30.0) -> bool:
    """Wacht op de per-verbinding goedkeuring van de server.

    De server stuurt nu eerst {'status':'pending', ...} en pas na goedkeuring in
    het Pi-GUI (J/N) een {'status':'accepted'} of {'status':'rejected'}.
    Eerdere code ging ervan uit dat de server direct 'accepted' stuurde, waardoor
    elke chat-poging vastliep op een 30s timeout. Deze loop lost dat op.
    Geeft True bij accepted, False bij rejected/timeout/fout.
    """
    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            data = json.loads(raw)
            if data.get("status") == "accepted":
                print("✅ Verbinding geaccepteerd door server.")
                return True
            if data.get("status") == "rejected":
                print("❌ Verbinding geweigerd door server (gebruikte J/N op de Pi).")
                return False
            if data.get("status") == "pending":
                print(f"⏳ Wachten op goedkeuring in het Pi-GUI (J=ja / N=nee)... peer={data.get('peer')}")
                continue
            if "error" in data:
                print(f"Fout: {data['error']}")
                return False
            # Onbekend bericht: negeer en blijf wachten op de status.
    except asyncio.TimeoutError:
        print("Timeout: geen goedkeuring ontvangen (gebruik J in het Pi-GUI of verhoog de timeout).")
        return False
    except (json.JSONDecodeError, websockets.exceptions.ConnectionClosed) as exc:
        print(f"Verbindingscontrole mislukt: {exc}")
        return False


async def upload_voice_profile_to_server(uri: str, voice_name: str, voice_data: dict) -> bool:
    """Upload voice profile naar server voor server-side opslag."""
    try:
        ws = await open_ws(uri)
        payload = {
            "type": "upload_voice_profile",
            "voice_name": voice_name,
            "voice_data": voice_data
        }
        await ws.send(json.dumps(payload))
        
        # Wacht op response
        response = json.loads(await ws.recv())
        # Close websocket met timeout om hanging te voorkomen
        try:
            await asyncio.wait_for(ws.close(), timeout=3.0)
        except asyncio.TimeoutError:
            debug_log("WebSocket close timeout, connection might be stale", "SYNC")
        except Exception as e:
            debug_log(f"WebSocket close error: {e}", "SYNC")
        
        if response.get("type") == "voice_profile_upload_success":
            debug_log(f"Voice profile '{voice_name}' succesvol geupload naar server", "SYNC")
            return True
        else:
            debug_log(f"Voice profile upload failed: {response.get('message', 'Unknown error')}", "SYNC")
            return False
            
    except Exception as exc:
        debug_log(f"Voice profile upload exception: {exc}", "ERROR")
        return False


async def download_voice_profile_from_server(uri: str, voice_name: str) -> dict | None:
    """Download voice profile van server."""
    try:
        ws = await open_ws(uri)
        payload = {
            "type": "download_voice_profile",
            "voice_name": voice_name
        }
        await ws.send(json.dumps(payload))
        
        # Wacht op response
        response = json.loads(await ws.recv())
        # Close websocket met timeout om hanging te voorkomen
        try:
            await asyncio.wait_for(ws.close(), timeout=3.0)
        except asyncio.TimeoutError:
            debug_log("WebSocket close timeout, connection might be stale", "SYNC")
        except Exception as e:
            debug_log(f"WebSocket close error: {e}", "SYNC")
        
        if response.get("type") == "voice_profile_data":
            debug_log(f"Voice profile '{voice_name}' succesvol gedownload van server", "SYNC")
            return response.get("voice_data")
        else:
            debug_log(f"Voice profile download failed: {response.get('message', 'Unknown error')}", "SYNC")
            return None
            
    except Exception as exc:
        debug_log(f"Voice profile download exception: {exc}", "ERROR")
        return None


async def list_server_voice_profiles(uri: str) -> list:
    """Haal lijst van voice profiles op van server."""
    try:
        ws = await open_ws(uri)
        payload = {
            "type": "list_voice_profiles"
        }
        await ws.send(json.dumps(payload))
        
        # Wacht op response
        response = json.loads(await ws.recv())
        # Close websocket met timeout om hanging te voorkomen
        try:
            await asyncio.wait_for(ws.close(), timeout=3.0)
        except asyncio.TimeoutError:
            debug_log("WebSocket close timeout, connection might be stale", "SYNC")
        except Exception as e:
            debug_log(f"WebSocket close error: {e}", "SYNC")
        
        if response.get("type") == "voice_profiles_list":
            voice_profiles = response.get("voice_profiles", [])
            debug_log(f"Server voice profiles: {voice_profiles}", "SYNC")
            return voice_profiles
        else:
            debug_log(f"List voice profiles failed: {response.get('message', 'Unknown error')}", "SYNC")
            return []
            
    except Exception as exc:
        debug_log(f"List voice profiles exception: {exc}", "ERROR")
        return []


async def sync_voice_profile_to_server(uri: str, voice_name: str, speaker_rec: SpeakerRecognition) -> bool:
    """Sync een lokaal voice profile naar de server."""
    if voice_name not in speaker_rec.voice_profiles:
        debug_log(f"Voice profile '{voice_name}' niet lokaal gevonden", "SYNC")
        return False

    voice_data = speaker_rec.voice_profiles[voice_name]
    return await upload_voice_profile_to_server(uri, voice_name, voice_data)


async def upload_settings_to_server(uri: str, settings_data: dict) -> bool:
    """Upload settings naar server voor server-side opslag."""
    try:
        ws = await open_ws(uri)
        payload = {
            "type": "upload_settings",
            "settings_data": settings_data
        }
        await ws.send(json.dumps(payload))

        # Wacht op response
        response = json.loads(await ws.recv())
        # Close websocket met timeout om hanging te voorkomen
        try:
            await asyncio.wait_for(ws.close(), timeout=3.0)
        except asyncio.TimeoutError:
            debug_log("WebSocket close timeout, connection might be stale", "SYNC")
        except Exception as e:
            debug_log(f"WebSocket close error: {e}", "SYNC")

        if response.get("type") == "settings_upload_success":
            debug_log("Settings succesvol geupload naar server", "SYNC")
            return True
        else:
            debug_log(f"Settings upload failed: {response.get('message', 'Unknown error')}", "SYNC")
            return False

    except Exception as exc:
        debug_log(f"Settings upload exception: {exc}", "ERROR")
        return False


async def download_settings_from_server(uri: str) -> dict | None:
    """Download settings van server."""
    try:
        ws = await open_ws(uri)
        payload = {
            "type": "download_settings"
        }
        await ws.send(json.dumps(payload))

        # Wacht op response
        response = json.loads(await ws.recv())
        # Close websocket met timeout om hanging te voorkomen
        try:
            await asyncio.wait_for(ws.close(), timeout=3.0)
        except asyncio.TimeoutError:
            debug_log("WebSocket close timeout, connection might be stale", "SYNC")
        except Exception as e:
            debug_log(f"WebSocket close error: {e}", "SYNC")

        if response.get("type") == "settings_data":
            debug_log("Settings succesvol gedownload van server", "SYNC")
            return response.get("settings_data")
        else:
            debug_log(f"Settings download failed: {response.get('message', 'Unknown error')}", "SYNC")
            return None

    except Exception as exc:
        debug_log(f"Settings download exception: {exc}", "ERROR")
        return None


async def upload_roles_to_server(uri: str, roles_data: dict) -> bool:
    """Upload roles naar server voor server-side opslag."""
    try:
        ws = await open_ws(uri)
        payload = {
            "type": "upload_roles",
            "roles_data": roles_data
        }
        await ws.send(json.dumps(payload))

        # Wacht op response
        response = json.loads(await ws.recv())
        # Close websocket met timeout om hanging te voorkomen
        try:
            await asyncio.wait_for(ws.close(), timeout=3.0)
        except asyncio.TimeoutError:
            debug_log("WebSocket close timeout, connection might be stale", "SYNC")
        except Exception as e:
            debug_log(f"WebSocket close error: {e}", "SYNC")

        if response.get("type") == "roles_upload_success":
            debug_log("Roles succesvol geupload naar server", "SYNC")
            return True
        else:
            debug_log(f"Roles upload failed: {response.get('message', 'Unknown error')}", "SYNC")
            return False

    except Exception as exc:
        debug_log(f"Roles upload exception: {exc}", "ERROR")
        return False


async def download_roles_from_server(uri: str) -> dict | None:
    """Download roles van server."""
    try:
        ws = await open_ws(uri)
        payload = {
            "type": "download_roles"
        }
        await ws.send(json.dumps(payload))

        # Wacht op response
        response = json.loads(await ws.recv())
        # Close websocket met timeout om hanging te voorkomen
        try:
            await asyncio.wait_for(ws.close(), timeout=3.0)
        except asyncio.TimeoutError:
            debug_log("WebSocket close timeout, connection might be stale", "SYNC")
        except Exception as e:
            debug_log(f"WebSocket close error: {e}", "SYNC")

        if response.get("type") == "roles_data":
            debug_log("Roles succesvol gedownload van server", "SYNC")
            return response.get("roles_data")
        else:
            debug_log(f"Roles download failed: {response.get('message', 'Unknown error')}", "SYNC")
            return None

    except Exception as exc:
        debug_log(f"Roles download exception: {exc}", "ERROR")
        return None


async def upload_chat_message_to_server(uri: str, message: str, is_user: bool = True, timestamp: str = None) -> bool:
    """Upload een chat bericht naar server voor server-side opslag."""
    try:
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().isoformat()

        chat_message = {
            "message": message,
            "is_user": is_user,
            "timestamp": timestamp
        }

        ws = await open_ws(uri)
        payload = {
            "type": "upload_chat_message",
            "chat_message": chat_message
        }
        await ws.send(json.dumps(payload))

        # Wacht op response
        response = json.loads(await ws.recv())
        # Close websocket met timeout om hanging te voorkomen
        try:
            await asyncio.wait_for(ws.close(), timeout=3.0)
        except asyncio.TimeoutError:
            debug_log("WebSocket close timeout, connection might be stale", "SYNC")
        except Exception as e:
            debug_log(f"WebSocket close error: {e}", "SYNC")

        if response.get("type") == "chat_message_upload_success":
            debug_log("Chat message succesvol geupload naar server", "SYNC")
            return True
        else:
            debug_log(f"Chat message upload failed: {response.get('message', 'Unknown error')}", "SYNC")
            return False

    except Exception as exc:
        debug_log(f"Chat message upload exception: {exc}", "ERROR")
        return False


async def download_chat_history_from_server(uri: str) -> list:
    """Download chat history van server."""
    try:
        ws = await open_ws(uri)
        payload = {
            "type": "download_chat_history"
        }
        await ws.send(json.dumps(payload))

        # Wacht op response
        response = json.loads(await ws.recv())
        # Close websocket met timeout om hanging te voorkomen
        try:
            await asyncio.wait_for(ws.close(), timeout=3.0)
        except asyncio.TimeoutError:
            debug_log("WebSocket close timeout, connection might be stale", "SYNC")
        except Exception as e:
            debug_log(f"WebSocket close error: {e}", "SYNC")

        if response.get("type") == "chat_history_data":
            chat_history = response.get("chat_history", [])
            debug_log(f"Chat history gedownload: {len(chat_history)} berichten", "SYNC")
            return chat_history
        else:
            debug_log(f"Chat history download failed: {response.get('message', 'Unknown error')}", "SYNC")
            return []

    except Exception as exc:
        debug_log(f"Chat history download exception: {exc}", "ERROR")
        return []


def is_tailscale_ip(ip_address: str) -> bool:
    """Check of IP adres een Tailscale IP is (100.x.x.x range)."""
    if not ip_address:
        return False
    try:
        parts = ip_address.split('.')
        if len(parts) == 4 and parts[0] == '100':
            return True
    except ValueError:
        pass
    return False

def detect_tailscale_hosts() -> list:
    """Detecteer Tailscale hosts in het netwerk."""
    if not TAILSCALE_MODE or not AUTO_DETECT_TAILSCALE:
        return []

    tailscale_hosts = []
    try:
        # Probeer common Tailscale poorten
        for port in [8765, 80, 443, 8080]:
            test_ip = f"100.64.0.1:{port}"  # Algemeen Tailscale gateway IP
            try:
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((test_ip.split(':')[0], int(test_ip.split(':')[1])))
                if result == 0:
                    tailscale_hosts.append(test_ip)
                    print(f"🐉 Tailscale host gevonden: {test_ip}")
                sock.close()
            except OSError:
                pass
    except Exception as e:
        print(f"⚠️  Fout bij Tailscale detectie: {e}")

    return tailscale_hosts

def get_optimal_uri(host: str, port: int) -> str:
    """Bepaal optimale URI (ws vs wss) voor verbinding."""
    protocol = "ws://"
    if TAILSCALE_MODE and is_tailscale_ip(host) and USE_WSS_FOR_TAILSCALE:
        protocol = "wss://"
        print(f"🔒 Gebruik WSS voor Tailscale verbinding: {protocol}{host}:{port}")
    return f"{protocol}{host}:{port}"

async def check_server_connection(uri: str) -> bool:
    """Check verbinding met beveiligingsoverweging."""
    try:
        ws = await open_ws(uri)
        accepted = await wait_for_acceptance(ws, timeout=120)
        try:
            await asyncio.wait_for(ws.close(), timeout=3.0)
        except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
            pass
        return accepted
    except Exception as exc:
        print(f"Kan nog niet verbinden met Pi websocket: {exc}")
        return False


async def safe_send_visual_state(ws, emotion: str, text: str, audio_level: float):
    try:
        await send_visual_state(ws, emotion, text, audio_level)
    except websockets.exceptions.ConnectionClosed:
        pass


async def speak_with_visual_sync(ws, engine, text: str, emotion: str):
    """Speak met offline emotionele stem en visual sync."""
    done = threading.Event()

    def speak_worker():
        try:
            speak(engine, text, emotion)
        except Exception as exc:
            print(f"Stemfout: {exc}")
        finally:
            done.set()

    threading.Thread(target=speak_worker, daemon=True).start()

    duration = estimate_speech_duration(text)
    start_time = time.monotonic()
    # Deadline verhoogd naar +15 seconden voor langere antwoorden
    deadline = start_time + duration + 15.0
    try:
        while not done.is_set() and time.monotonic() < deadline:
            elapsed = time.monotonic() - start_time
            await safe_send_visual_state(ws, emotion, text, speech_level_at(elapsed, duration))
            await asyncio.sleep(0.06)
        if not done.is_set():
            try:
                engine.stop()
            except Exception as exc:
                print(f"Fout bij stoppen van stem: {exc}")
    finally:
        await safe_send_visual_state(ws, emotion, text, 0.0)


async def send_ssh_command(uri: str, command: str):
    """Stuur een SSH commando via websocket verbinding (client-side execution)."""
    if not SSH_ENABLED:
        print("SSH functionaliteit uitgeschakeld in settings")
        return None
        
    try:
        ws = await asyncio.wait_for(open_ws(uri), timeout=CONNECTION_TIMEOUT)
        try:
            # Wacht op per-verbinding goedkeuring (pending -> accepted/rejected)
            if not await wait_for_acceptance(ws, timeout=120):
                print("SSH afgebroken: verbinding niet goedgekeurd.")
                return None
            print("Verbinding geaccepteerd - stuur SSH commando voor client-uitvoering...")

            # Nu stuur het SSH commando
            payload = {"type": "ssh_command", "command": command}
            await ws.send(json.dumps(payload))
            
            # Wacht op ssh_execute bericht van server
            raw = await asyncio.wait_for(ws.recv(), timeout=MESSAGE_TIMEOUT)
            execute_data = json.loads(raw)
            
            if execute_data.get("type") == "ssh_execute":
                execute_command = execute_data.get("command", "")
                print(f"Commando ontvangen voor client-uitvoering: {execute_command}")
                
                # Voer commando lokaal uit op client
                try:
                    if platform.system() == "Windows":
                        if execute_command in ["dir", "cls", "type", "copy", "del", "move", "ren"]:
                            cmd_list = ["cmd", "/c", execute_command]
                        elif " " in execute_command:
                            cmd_list = ["cmd", "/c", execute_command]
                        else:
                            cmd_list = execute_command.split()
                            try:
                                result = subprocess.run(
                                    cmd_list,
                                    capture_output=True,
                                    text=True,
                                    timeout=SSH_TIMEOUT,
                                    shell=False
                                )
                            except FileNotFoundError:
                                cmd_list = ["cmd", "/c", execute_command]
                        result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=SSH_TIMEOUT, shell=False)
                    else:
                        result = subprocess.run(execute_command.split(), capture_output=True, text=True, timeout=SSH_TIMEOUT)
                    
                    output = f"Stdout: {result.stdout}\nStderr: {result.stderr}\nReturn: {result.returncode}"
                    print(f"Client output:\n{output}")
                    
                    # Stuur response terug naar server
                    await ws.send(json.dumps({
                        "type": "ssh_response",
                        "command": execute_command,
                        "output": output
                    }))
                    
                    return output
                except subprocess.TimeoutExpired:
                    output = f"Commando timeout ({SSH_TIMEOUT}s)"
                    print(f"Client error: {output}")
                    await ws.send(json.dumps({
                        "type": "ssh_response",
                        "command": execute_command,
                        "output": output
                    }))
                    return output
                except Exception as cmd_error:
                    output = f"Commando execution error: {cmd_error}"
                    print(f"Client error: {output}")
                    return output
            else:
                print(f"Onverwacht bericht: {execute_data}")
                return None
        finally:
            try:
                await asyncio.wait_for(ws.close(), timeout=3)
            except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
                pass
    except asyncio.TimeoutError:
        print(f"Connection timeout ({CONNECTION_TIMEOUT}s)")
        return None
    except Exception as exc:
        print(f"SSH command error: {exc}")
        return None


async def send_and_receive(uri: str, ollama_model: str, text: str, audio_level: float, tts):
    ws = await open_ws(uri)
    try:
        # Upload user message naar server
        await upload_chat_message_to_server(uri, text, is_user=True)

        # Wacht op per-verbinding goedkeuring (pending -> accepted/rejected)
        if not await wait_for_acceptance(ws, timeout=120):
            print("Chat afgebroken: verbinding niet goedgekeurd door de Pi.")
            return

        # Stuur het eigenlijke bericht
        payload = {"text": text, "model": ollama_model, "audio_level": audio_level}
        await ws.send(json.dumps(payload))
        raw = await ws.recv()
        data = json.loads(raw)

        if "error" in data:
            print(f"Fout: {data['error']}")
            return

        reply = data.get("reply", "")
        emotion = data.get("emotion", "neutral")
        print(f"Bot ({emotion}): {reply}")

        # Upload bot reply naar server
        await upload_chat_message_to_server(uri, reply, is_user=False)

        await speak_with_visual_sync(ws, tts, reply, emotion)
    finally:
        await safe_send_visual_state(ws, "neutral", "Ik ben klaar.", 0.0)
        try:
            await asyncio.wait_for(ws.close(), timeout=3)
        except (asyncio.TimeoutError, websockets.exceptions.WebSocketException):
            pass


async def test_ssh_command(uri: str):
    """Test functie om SSH commando's via websocket te testen."""
    command = input("Voer commando in (bijv. 'dir', 'ls', 'whoami'): ")
    print(f"Stuur commando via websocket: {command}")
    output = await send_ssh_command(uri, command)
    return output


async def handle_chat_turn(uri: str, ollama_model: str, text: str, audio_level: float, tts, speaker_rec=None, speaker_name=None, speaker_perms=None):
    try:
        # Check rechten op basis van spreker
        if speaker_perms:
            perms = speaker_perms.get_permissions(speaker_name, speaker_rec)

            # Check SSH toegang
            is_ssh_command = any(text.lower().startswith(prefix) for prefix in SSH_PREFIXES)
            if is_ssh_command and not perms.get("ssh_commands", True):
                print(f"🔐 {speaker_name or 'Onbekende'} heeft geen SSH toegang")
                # Sta chat nog toe voor onbekende sprekers
                if not speaker_name and perms.get("full_chat", True):
                    print("ℹ️  Chat toegang verleend (beperkte modus)")
                return

            # Check systeem commando's
            if any(word in text.lower() for word in ["verwijder", "delete", "format", "shutdown"]):
                if not perms.get("system_commands", True):
                    print(f"🔐 {speaker_name or 'Onbekende'} heeft geen systeem commando toegang")
                    # Sta chat nog toe voor onbekende sprekers
                    if not speaker_name and perms.get("full_chat", True):
                        print("ℹ️  Chat toegang verleend (beperkte modus)")
                    return

        # Voeg spreker info toe aan tekst indien bekend
        message_text = text
        if speaker_name:
            message_text = f"{speaker_name} zegt: {text}"
        else:
            message_text = f"Onbekende zegt: {text}"
            # Toon info dat het een onbekende spreker is
            print("👤 Onbekende spreker gedetecteerd")

        # Check of dit een SSH commando is
        is_ssh_command = any(text.lower().startswith(prefix) for prefix in SSH_PREFIXES)

        if is_ssh_command and SSH_ENABLED:
            # Verwijder prefix en stuur als SSH commando
            command = text
            for prefix in SSH_PREFIXES:
                if text.lower().startswith(prefix):
                    command = text[len(prefix):].strip()
                    break

            print(f"SSH commando gedetecteerd: {command}")
            output = await send_ssh_command(uri, command)
            if output:
                print("SSH output ontvangen")
            else:
                print("SSH commando mislukt")
            return

        # Normale Jarvis chat - altijd toegang voor onbekende sprekers
        await send_and_receive(uri, ollama_model, message_text, audio_level, tts)
    except (OSError, TimeoutError, websockets.exceptions.WebSocketException) as exc:
        print(f"Verbindingsfout, je kunt opnieuw proberen: {exc}")
    except Exception as exc:
        print(f"Vraag mislukt, je kunt opnieuw proberen: {exc}")
    finally:
        print("Klaar voor volgende vraag.")


async def press_to_record_loop(uri: str, transcriber, ollama_model: str, record_seconds: int, tts, language: str, speaker_rec=None, speaker_perms=None):
    print("Druk ENTER om op te nemen. Typ tekst direct om zonder opname te sturen. Typ 'q' om te stoppen.")
    while True:
        try:
            cmd = input("> ").strip()
        except KeyboardInterrupt:
            print("Gebruik 'q' om netjes te stoppen.")
            continue
        except EOFError:
            break

        if cmd.lower() == "q":
            break
        if cmd:
            print(f"Getypte tekst: {cmd}")
            await handle_chat_turn(uri, ollama_model, cmd, 0.0, tts, speaker_rec, None, speaker_perms)
            continue

        wav_path, audio_level = record_audio(record_seconds)

        # Speaker recognition
        speaker_name = None
        if speaker_rec:
            try:
                from scipy.io import wavfile
                sample_rate, audio_data = wavfile.read(wav_path)
                audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize
                speaker_name = speaker_rec.identify_speaker(audio_data, sample_rate)
                if speaker_name:
                    print(f"👤 Herkende spreker: {speaker_name}")
                else:
                    print("👤 Onbekende spreker (geen profiel kwam boven de drempel)")
            except Exception as e:
                print(f"⚠️  Speaker recognition mislukt (wordt genegeerd, verder als onbekend): {e}")

        text = transcriber.transcribe_wav(wav_path, language)
        if not text:
            print("Geen spraak herkend.")
            continue

        if speaker_name:
            print(f"{speaker_name} zei: {text}")
        else:
            print(f"Onbekende zei: {text}")

        await handle_chat_turn(uri, ollama_model, text, audio_level, tts, speaker_rec, speaker_name, speaker_perms)


async def hold_to_record_loop(uri: str, transcriber, ollama_model: str, tts, language: str, speaker_rec=None, speaker_perms=None):
    """HOLD mode: druk en houd SPACE om op te nemen, los om te versturen."""
    print("HOLD MODE: Druk en houd SPACE om op te nemen. Laat los om te versturen.")
    print("Typ tekst direct om zonder opname te sturen. Typ 'q' om te stoppen.")
    print("⚠️  Vereist 'keyboard' library: pip install keyboard")

    try:
        import keyboard
    except ImportError:
        print("⚠️  Keyboard library niet gevonden. Installeer met: pip install keyboard")
        print("Terugvallen op PRESS mode...")
        return await press_to_record_loop(uri, transcriber, ollama_model, DEFAULT_RECORD_SECONDS, tts, language, speaker_rec, speaker_perms)
    
    is_recording = False
    recording_data = []
    
    def on_space_press():
        nonlocal is_recording
        is_recording = True
        print("🎙️  Opname gestart... (laat SPACE los om te stoppen)")
    
    def on_space_release():
        nonlocal is_recording
        is_recording = False
        print("📤 Opname gestopt, audio wordt verwerkt...")
    
    keyboard.on_press_key("space", on_space_press)
    keyboard.on_release_key("space", on_space_release)
    
    try:
        while True:
            try:
                # Check voor tekst input (non-blocking)
                import sys
                import select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    cmd = input("> ").strip()
                    if cmd.lower() == "q":
                        break
                    if cmd:
                        print(f"Getypte tekst: {cmd}")
                        await handle_chat_turn(uri, ollama_model, cmd, 0.0, tts, speaker_rec, None, speaker_perms)
                        continue

                # Proces opname als gestopt
                if recording_data and not is_recording:
                    import numpy as np
                    print(f"Audio verwerken ({len(recording_data)} samples)...")

                    # Sla op als tijdelijk WAV bestand
                    temp_dir = tempfile.mkdtemp()
                    wav_path = f"{temp_dir}/hold_recording.wav"
                    audio_array = np.array(recording_data, dtype=np.float32)
                    write_wave_file(audio_array, wav_path)

                    # Speaker recognition
                    speaker_name = None
                    if speaker_rec:
                        try:
                            from scipy.io import wavfile
                            sample_rate, audio_data = wavfile.read(wav_path)
                            audio_data = audio_data.astype(np.float32) / 32768.0
                            speaker_name = speaker_rec.identify_speaker(audio_data, sample_rate)
                            if speaker_name:
                                print(f"👤 Herkende spreker: {speaker_name}")
                        except Exception as e:
                            print(f"⚠️  Speaker recognition mislukt (wordt genegeerd, verder als onbekend): {e}")

                    # Transcribeer
                    text = transcriber.transcribe_wav(wav_path, language)
                    audio_level = min(1.0, len(recording_data) / 500000.0)

                    if text:
                        if speaker_name:
                            print(f"{speaker_name} zei: {text}")
                        else:
                            print(f"Onbekende zei: {text}")
                        await handle_chat_turn(uri, ollama_model, text, audio_level, tts, speaker_rec, speaker_name, speaker_perms)
                    else:
                        print("Geen spraak herkend.")

                    # Cleanup
                    os.remove(wav_path)
                    os.rmdir(temp_dir)
                    recording_data = []
                
                # Start nieuwe opname als SPACE wordt ingedrukt
                if is_recording and not recording_data:
                    recording_data = []  # Start met lege list voor data chunks
                    recording_buffer = []  # Buffer voor numpy arrays
                    
                    def callback(indata, frames, time_info, status):
                        if is_recording:
                            recording_buffer.append(indata.copy())  # Direct numpy array copy
                    
                    try:
                        with sd.InputStream(callback=callback, channels=1, samplerate=16000):
                            while is_recording:
                                await asyncio.sleep(0.05)
                    except Exception as e:
                        print(f"Opname error: {e}")
                        is_recording = False
                    
                    # Combineer alle chunks na opname
                    if recording_buffer:
                        recording_data = np.concatenate(recording_buffer, axis=0).flatten().tolist()
                
                await asyncio.sleep(0.05)
                
            except (KeyboardInterrupt, EOFError):
                print("\nAfsluiten...")
                break
    finally:
        keyboard.unhook_all()


async def live_listen_loop(
    uri: str,
    transcriber,
    ollama_model: str,
    wake_word: str,
    tts,
    language: str,
    speaker_rec=None,
    speaker_perms=None,
    sample_rate: int = 16000,
):
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    wake_word = wake_word.strip().lower()
    print("Live luisteren gestart. Druk Ctrl+C om te stoppen.")
    if wake_word:
        print(f"Wake word actief: {wake_word}")

    def callback(indata, frames, time_info, status):
        if status:
            print(f"Audio waarschuwing: {status}")
        audio_queue.put(indata.copy().flatten())

    def collect_phrase() -> tuple[np.ndarray | None, float]:
        buffer = np.zeros((0,), dtype=np.float32)
        speech_started = False
        silence_chunks = 0
        min_samples = int(sample_rate * 0.8)
        max_samples = int(sample_rate * 6.0)
        speech_threshold = 0.018
        silence_threshold = 0.010

        while len(buffer) < max_samples:
            chunk = audio_queue.get()
            chunk_rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0

            if not speech_started:
                if chunk_rms < speech_threshold:
                    continue
                speech_started = True

            buffer = np.concatenate([buffer, chunk])

            if chunk_rms < silence_threshold:
                silence_chunks += 1
            else:
                silence_chunks = 0

            if len(buffer) >= min_samples and silence_chunks >= 5:
                break

        rms = float(np.sqrt(np.mean(np.square(buffer)))) if len(buffer) else 0.0
        level = max(0.0, min(1.0, rms * 10.0))

        if rms < speech_threshold:
            return None, level

        return buffer, level

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.float32, callback=callback):
        while True:
            audio, audio_level = await asyncio.to_thread(collect_phrase)
            if audio is None:
                continue

            text = await asyncio.to_thread(transcriber.transcribe_array, audio, language)
            text = text.strip()
            if not text:
                continue

            # Speaker recognition
            speaker_name = None
            if speaker_rec:
                try:
                    speaker_name = speaker_rec.identify_speaker(audio, sample_rate)
                    if speaker_name:
                        print(f"👤 Herkende spreker: {speaker_name}")
                except Exception as e:
                    print(f"⚠️  Speaker recognition mislukt (wordt genegeerd, verder als onbekend): {e}")

            if speaker_name:
                print(f"{speaker_name} zei: {text}")
            else:
                print(f"Onbekende zei: {text}")

            lowered = text.lower()

            if wake_word:
                if wake_word not in lowered:
                    continue
                wake_index = lowered.find(wake_word)
                text = text[wake_index + len(wake_word):].strip(" ,.!?")
                if not text:
                    print("Wake word gehoord, nog geen vraag erna.")
                    continue

            await handle_chat_turn(uri, ollama_model, text, audio_level, tts, speaker_rec, speaker_name, speaker_perms)


async def chat_loop(
    host: str,
    port: int,
    ollama_model: str,
    whisper_model_name: str,
    record_seconds: int,
    mode: str,
    wake_word: str,
    input_device: int | None,
    language: str,
):
    uri = get_optimal_uri(host, port)
    configure_input_device(input_device)
    input_name, output_name = get_audio_device_summary()

    print(f"Transcriptie laden: {whisper_model_name} ...")
    transcriber = create_transcriber(whisper_model_name)
    print("Transcriptie geladen.")

    # Initialiseer speaker recognition
    speaker_rec = SpeakerRecognition()
    speaker_perms = SpeakerPermissions()
    voices = speaker_rec.list_voices()
    if voices:
        print(f"🎤 Speaker recognition actief - Getrainde stemmen: {', '.join(voices)}")
    else:
        print("ℹ️  Geen getrainde stemmen - gebruik --voice-test om stemmen te registreren")

    # Toon rechten info
    print("🔐 Rechten systeem actief")
    print("   • Bekende sprekers: Volle toegang")
    print("   • Onbekende sprekers: Beperkte toegang (geen SSH/systeem commando's)")

    print(f"Microfoon van verbonden apparaat: {input_name}")
    print(f"Speakers van verbonden apparaat: {output_name}")
    tts = pyttsx3.init()
    configure_jarvis_voice(tts)
    print(f"Verbinden met {uri} ...")
    if not await check_server_connection(uri):
        print("Start eerst pi_app.py op de Raspberry Pi en probeer daarna opnieuw.")
        print("Op de Pi moet je zien: Websocket server klaar op 0.0.0.0:8765")
        return
    print(f"Klaar voor chat met {uri}")
    try:
        if mode == "live":
            await live_listen_loop(uri, transcriber, ollama_model, wake_word, tts, language, speaker_rec, speaker_perms)
        elif mode == "hold":
            await hold_to_record_loop(uri, transcriber, ollama_model, tts, language, speaker_rec, speaker_perms)
        else:
            await press_to_record_loop(uri, transcriber, ollama_model, record_seconds, tts, language, speaker_rec, speaker_perms)
    finally:
        tts.stop()
        sd.stop()
        # Cleanup memory en resources
        del speaker_rec
        del speaker_perms
        del transcriber
        del tts
        gc.collect()
        debug_log("Memory cleanup na chat_loop", "MEMORY")


async def voice_training_loop(voice_name: str, input_device: int | None, role: str = "user", language: str = "nl", whisper_model: str = "base", upload_to_server: bool = True, server_host: str = DEFAULT_HOST, server_port: int = DEFAULT_PORT):
    """Voice training loop voor het trainen van een nieuwe stem of extra samples toevoegen met transcripties."""
    configure_input_device(input_device)
    input_name, output_name = get_audio_device_summary()

    print(f"Microfoon: {input_name}")
    print(f"Speakers: {output_name}")
    print("=" * 50)

    speaker_rec = SpeakerRecognition()
    existing_samples = 0

    # Check of profile al bestaat
    if voice_name in speaker_rec.voice_profiles:
        existing_profile = speaker_rec.voice_profiles[voice_name]
        if isinstance(existing_profile, dict):
            existing_samples = existing_profile.get("sample_count", 1)
        print(f"ℹ️  Profile '{voice_name}' bestaat al met {existing_samples} opnames")
        print("ℹ️  Extra opnames worden toegevoegd voor betere nauwkeurigheid")
    else:
        print(f"ℹ️  Nieuw profile voor '{voice_name}' wordt aangemaakt")

    recordings_count = 0
    target_recordings = 5  # Aantal opnames voor training

    print(f"\n🎤 Training: {voice_name}")
    print(f"🎭 Rol: {role}")
    print(f"🌐 Taal: {language}")
    print(f"🤖 Whisper model: {whisper_model}")
    print(f"📤 Upload naar server: {upload_to_server}")
    print(f"Je hebt {target_recordings} opnames nodig voor deze sessie")
    print(f"Na sessie: totaal {existing_samples + target_recordings} opnames")
    print("📝 Je krijgt voor elke opname een zin om in te spreken")
    print("🎯 Spraakherkenning (Whisper) zal je zinnen transcriberen")
    print("=" * 50)

    while recordings_count < target_recordings:
        # Haal een training zin op
        sentence_index = existing_samples + recordings_count
        current_sentence = get_training_sentence(sentence_index, target_recordings)

        print(f"\nOpname {recordings_count + 1}/{target_recordings}")
        print("📝 Zeg deze zin:")
        print(f"   '{current_sentence}'")
        print("Druk ENTER om op te nemen...")

        # Wacht op enter
        await asyncio.to_thread(input)

        print("🔴 Opnemen... (zeg de bovenstaande zin duidelijk)")

        # Neem audio op - langere duur voor langere zinnen
        sample_rate = 16000
        duration = 10  # 10 seconden opname (voor langere zinnen)
        try:
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()  # Wacht tot opname klaar is

            print("✅ Opname voltooid")

            # Normaliseer audio
            audio = trim_and_normalize_audio(recording.squeeze(), sample_rate)

            # Transcribeer audio met Whisper
            print("🤖 Spraakherkenning bezig...")
            transcript = await transcribe_audio(audio, sample_rate, language, whisper_model)

            if transcript:
                print(f"📝 Transcript: '{transcript}'")
                # Vergelijk met verwachte zin
                similarity = len(set(current_sentence.lower().split()) & set(transcript.lower().split())) / len(set(current_sentence.lower().split() | set(transcript.lower().split())))
                print(f"🎯 Overeenkomst: {similarity:.1%}")
            else:
                print("⚠️  Kon geen transcript maken")

            # Train de stem met deze opname en rol
            if speaker_rec.train_voice(voice_name, audio, sample_rate, role, transcript):
                recordings_count += 1
                total_samples = speaker_rec.voice_profiles[voice_name].get("sample_count", recordings_count)
                print(f"✅ Opname {recordings_count}/{target_recordings} succesvol getraind (totaal: {total_samples})")
            else:
                print("❌ Training mislukt, probeer opnieuw")

            # Cleanup memory expliciet
            del recording
            del audio
            gc.collect()
            debug_log(f"Memory cleanup na opname {recordings_count}", "MEMORY")

        except Exception as e:
            print(f"❌ Fout bij opname: {e}")
            continue

    final_profile = speaker_rec.voice_profiles[voice_name]
    final_samples = final_profile.get("sample_count", target_recordings) if isinstance(final_profile, dict) else target_recordings
    final_transcripts = len(final_profile.get("transcripts", [])) if isinstance(final_profile, dict) else 0

    print("\n" + "=" * 50)
    if existing_samples > 0:
        print(f"🎉 Extra training compleet voor: {voice_name}")
        print(f"📊 Van {existing_samples} naar {final_samples} opnames")
    else:
        print(f"🎉 Training compleet voor: {voice_name}")
    print(f"🎭 Rol: {role}")
    print(f"✅ {recordings_count} nieuwe opnames verwerkt")
    print(f"🤖 {final_transcripts} transcripties opgeslagen")
    print(f"💾 Totaal: {final_samples} opnames opgeslagen in: voice_profiles.json")

    # Upload naar server indien gewenst
    if upload_to_server:
        print("\n📤 Voice profile uploaden naar server...")
        uri = get_optimal_uri(server_host, server_port)
        if await sync_voice_profile_to_server(uri, voice_name, speaker_rec):
            print(f"✅ Voice profile '{voice_name}' geupload naar server")
        else:
            print("⚠️  Voice profile upload mislukt, maar lokaal opgeslagen")

    print("\nGetrainde stemmen:")
    for voice in speaker_rec.list_voices():
        voice_data = speaker_rec.voice_profiles[voice]
        if isinstance(voice_data, dict):
            voice_role = voice_data.get("role", "user")
            sample_count = voice_data.get("sample_count", 1)
            transcript_count = len(voice_data.get("transcripts", []))
            print(f"  • {voice} (Rol: {voice_role}, Opnames: {sample_count}, Transcripties: {transcript_count})")
        else:
            print(f"  • {voice}")
    print("=" * 50)
    if existing_samples > 0:
        print("💡 Hoe meer opnames, hoe nauwkeuriger de herkenning wordt!")
    print("💡 Transcripties helpen bij spraakherkenning training voor jouw stem")
    print("Je kunt nu de app starten met normale spraakherkenning")

    # Cleanup memory en resources
    del speaker_rec
    gc.collect()
    debug_log("Memory cleanup na voice training", "MEMORY")


async def test_speaker_recognition(speaker_rec, input_device: int | None):
    """Test speaker recognition zonder verbinding met server."""
    configure_input_device(input_device)
    voices = speaker_rec.list_voices()

    if not voices:
        print("❌ Geen getrainde stemmen beschikbaar.")
        print("Train eerst een stem met --voice-test")
        return

    print("Test speaker recognition...")
    print("Druk ENTER om op te nemen en te testen wie er praat.")

    while True:
        try:
            cmd = input("> ").strip()
        except KeyboardInterrupt:
            print("Gebruik 'q' om netjes te stoppen.")
            continue
        except EOFError:
            break

        if cmd.lower() == "q":
            break
        if not cmd:
            print("🔴 Opnemen... (zeg iets, minimaal 3 seconden)")

            try:
                sample_rate = 16000
                duration = 5  # 5 seconden
                recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                sd.wait()

                audio = trim_and_normalize_audio(recording.squeeze(), sample_rate)
                speaker_name = speaker_rec.identify_speaker(audio, sample_rate)

                if speaker_name:
                    print(f"✅ Herkende: {speaker_name}")
                else:
                    print("❌ Onbekende spreker (niet genoeg overeenkomst of stem nog toevoegen)")

            except Exception as e:
                print(f"❌ Fout bij opname/herkenning: {e}")


def interactive_profile_selector(settings_mgr=None):
    """Interactief profile selector menu."""
    import sys
    import subprocess

    print("\n" + "=" * 60)
    print("🎯 JARVIS LAPTOP CLIENT - PROFILE SELECTOR")
    print("=" * 60)

    profiles = [
        {
            "name": "🌐 Normaal Gebruik (Aanbevolen)",
            "command": ["python", __file__, "--host", "localhost", "--port", "8765"],
            "description": "Standaard setup voor dagelijks gebruik"
        },
        {
            "name": "⚡ Snelste Speech Recognition",
            "command": ["python", __file__, "--host", "localhost", "--whisper-model", "tiny"],
            "description": "Gebruikt tiny model voor maximale snelheid"
        },
        {
            "name": "🎤 Live Modus met Wake Word",
            "command": ["python", __file__, "--host", "localhost", "--mode", "live", "--wake-word", "jarvis"],
            "description": "Luist continu voor 'jarvis' wake word"
        },
        {
            "name": "🎙️ Stem Training",
            "command": None,  # Requires user input for name
            "description": "Train een nieuwe stem met 5 opnames",
            "requires_input": True
        },
        {
            "name": "👤 Speaker Recognition Test",
            "command": ["python", __file__, "--host", "localhost", "--test-speaker"],
            "description": "Test of stem training werkt"
        },
        {
            "name": "🔐 Rechten Beheren",
            "command": ["python", __file__, "--host", "localhost", "--manage-permissions"],
            "description": "Bekijk en pas spreker rechten aan"
        },
        {
            "name": "🚨 Strict Onbekende Rechten",
            "command": ["python", __file__, "--host", "localhost", "--set-unknown-perms", '{"full_chat":false}'],
            "description": "Beperk onbekende sprekers volledig"
        },
        {
            "name": "🔧 SSH Commando Test",
            "command": ["python", __file__, "--host", "localhost", "--ssh-test"],
            "description": "Test SSH functionaliteit"
        },
    ]

    while True:
        print("\n📋 Beschikbare Profiles:\n")

        for i, profile in enumerate(profiles, 1):
            print(f"  [{i}] {profile['name']}")
            print(f"      {profile['description']}\n")

        print("  [0] 🚪 Exit")
        print("\n" + "-" * 60)
        print("Selecteer een profiel (0-8): ", end="")

        try:
            choice = input().strip()

            if choice == "0":
                print("👋 Tot ziens!")
                sys.exit(0)

            try:
                profile_num = int(choice) - 1
                if 0 <= profile_num < len(profiles):
                    profile = profiles[profile_num]

                    print(f"\n✅ Geselecteerd: {profile['name']}")
                    print(f"📝 {profile['description']}")

                    if profile.get("requires_input"):
                        print("\nVoer de naam in voor de stem training:")
                        name = input("Naam: ").strip()
                        if not name:
                            print("❌ Naam is verplicht")
                            continue

                        # Gebruik defaults voor language en whisper_model
                        command = ["python", __file__, "--host", "localhost", "--voice-test", "--voice-name", name,
                                   "--language", DEFAULT_LANGUAGE, "--whisper-model", DEFAULT_WHISPER_MODEL]
                    else:
                        command = profile["command"]

                    print(f"\n🚀 Starten met command: {' '.join(command)}")
                    print("-" * 60 + "\n")

                    subprocess.run(command)

                    print("\n" + "-" * 60)
                    print("✅ Profile voltooid!")
                    print("\nDruk ENTER voor menu of 0 om te exiten...")
                    next_choice = input().strip()

                    if next_choice == "0":
                        print("👋 Tot ziens!")
                        sys.exit(0)
                else:
                    print("❌ Ongeldige keuze. Probeer opnieuw.")
            except ValueError:
                print("❌ Voer een nummer in.")

        except KeyboardInterrupt:
            print("\n\n👋 Tot ziens!")
            sys.exit(0)
        except EOFError:
            print("\n\n👋 Tot ziens!")
            sys.exit(0)


def show_specific_help(topic: str):
    """Toon specifieke help voor een onderwerp."""
    help_topics = {
        "voice-test": """
🎤 VOICE TRAINING HELP
════════════════════════════════════════════════════════════════════

Beschrijving:
  Train een nieuwe stem met 5 voorgeschreven zinnen van 10 seconden voor speaker recognition.
  Het systeem leert de unieke kenmerken van je stem voor identificatie.

Gebruik:
  python laptop_app.py --host localhost --voice-test --voice-name "Jan"

Argumenten:
  --voice-test              Start voice training modus
  --voice-name NAAM        Naam voor de stem (verplicht)

Zinnen:
  Het systeem geeft je 5 verschillende zinnen om in te spreken:
  • Diverse klanken en woorden
  • Lange en korte zinnen
  • Cijfers en specifieke termen
  • Variatie in intonatie

Spraakherkenning:
  • Whisper transcribeert automatisch je opnames
  • Transcripties worden opgeslagen met je voice profile
  • Dit helpt het systeem jouw specifieke stem en uitspraak te leren
  • Transcripties worden vergeleken met verwachte zinnen (overeenkomst %)

Proces:
  1. Druk ENTER om opname te starten
  2. Zeg de getoonde zin duidelijk en rustig
  3. Wacht tot de opname klaar is (10 seconden)
  4. Herhaal voor alle 5 zinnen
  5. Profile wordt automatisch opgeslagen

Tips:
  • Gebruik duidelijke zinnen, niet alleen "ja" of "nee"
  • Spreek natuurlijk, niet geforceerd
  • Gebruik dezelfde stemtoon als tijdens normaal gebruik
  • Meer opnames = betere nauwkeurigheid

Bestanden:
  voice_profiles.json - Wordt automatisch aangemaakt met stem data

Na training:
  • Test met: --test-speaker
  • Gebruik in normale mode: speaker herkenning automatisch actief
        """,

        "test-speaker": """
👤 SPEAKER RECOGNITION TEST HELP
════════════════════════════════════════════════════════════════════

Beschrijving:
  Test of stem training werkt zonder verbinding met Raspberry Pi.
  Ideaal om te controleren of je stem correct is opgeslagen.

Gebruik:
  python laptop_app.py --host localhost --test-speaker

Argumenten:
  --test-speaker           Test speaker recognition zonder verbinding

Proces:
  1. Druk ENTER om opname te starten
  2. Zeg een zin van minimaal 3 seconden
  3. Systeem probeert je stem te identificeren
  4. Resultaat wordt getoond (herkend of onbekend)

Resultaten:
  ✅ Herkend: Systeem kent je stem
  ❌ Onbekende: Stem nog niet getraind of opname niet duidelijk

Probleemoplossing:
  • Onbekende spreker? Train opnieuw met --voice-test
  • Zorg voor goede audio kwaliteit
  • Spreek duidelijk en luid genoeg

Bestanden:
  voice_profiles.json - Moet getrainde stemmen bevatten
        """,

        "manage-permissions": """
🔐 PERMISSIONS MANAGEMENT HELP
════════════════════════════════════════════════════════════════════

Beschrijving:
  Bekijk en pas spreker rechten aan. Bekende sprekers hebben volledige toegang,
  onbekende sprekers hebben beperkte rechten.

Gebruik:
  python laptop_app.py --host localhost --manage-permissions

Argumenten:
  --manage-permissions     Toon huidige rechten per spreker

Wat je ziet:
  • Onbekende spreker rechten (standaard)
  • Bekende sprekers met specifieke rechten
  • Beschikbare opties en status

Standaard Rechten:
  Bekende sprekers:
    • full_chat: true (volledige chat toegang)
    • ssh_commands: true (SSH toegestaan)
    • system_commands: true (systeem commando's toegestaan)
    • file_operations: true (bestandsoperaties toegestaan)

  Onbekende sprekers:
    • full_chat: true (chat toegang)
    • ssh_commands: false (geen SSH)
    • system_commands: false (geen systeem commando's)
    • file_operations: false (geen bestandsoperaties)

Rechten Aanpassen:
  Via JSON: --set-unknown-perms '{"full_chat":false}'
  Handmatig: Bewerk speaker_permissions.json

Bestanden:
  speaker_permissions.json - Opslag van rechten
        """,

        "set-unknown-perms": """
🔒 UNKNOWN PERMISSIONS CONFIGURATION HELP
════════════════════════════════════════════════════════════════════

Beschrijving:
  Pas rechten aan voor onbekende sprekers via JSON string.
  Bepaal wat onbekenden wel en niet mogen doen.

Gebruik:
  python laptop_app.py --host localhost --set-unknown-perms '{"full_chat":true}'

Argumenten:
  --set-unknown-perms JSON JSON string met rechten configuratie

JSON Format:
  {
    "full_chat": true,           # Chat toegang (true/false)
    "ssh_commands": false,       # SSH commando's (true/false)
    "system_commands": false,    # Systeem commando's (true/false)
    "file_operations": false,    # Bestandsoperaties (true/false)
    "max_audio_level": 0.5,      # Max audio volume (0.0-1.0)
    "request_timeout": 15        # Timeout in seconden
  }

Voorbeelden:
  # Volledige toegang (niet aanbevolen)
  python laptop_app.py --host localhost --set-unknown-perms '{"full_chat":true,"ssh_commands":true}'

  # Alleen chat, geen gevaarlijke commando's (standaard)
  python laptop_app.py --host localhost --set-unknown-perms '{"full_chat":true,"ssh_commands":false}'

  # Geen toegang (strict)
  python laptop_app.py --host localhost --set-unknown-perms '{"full_chat":false}'

  # Specifieke settings
  python laptop_app.py --host localhost --set-unknown-perms '{"full_chat":true,"max_audio_level":0.3}'

Bestanden:
  speaker_permissions.json - Wordt automatisch bijgewerkt
        """,

        "ssh-test": """
🔧 SSH TESTING HELP
════════════════════════════════════════════════════════════════════

Beschrijving:
  Test SSH commando functionaliteit via websocket verbinding met Raspberry Pi.
  Controleer of Pi SSH toegang heeft en commando's kan uitvoeren.

Gebruik:
  python laptop_app.py --host localhost --ssh-test

Argumenten:
  --ssh-test               Test SSH commando's via websocket

Wat wordt getest:
  • WebSocket verbinding met Pi
  • SSH toegang op Raspberry Pi
  • Commando uitvoering via websocket
  • Output terugsturen naar client

SSH Commando Prefixes:
  Standard: "ssh "
  Alternatief: "run ", "exec "

Veilige Test Commando's:
  • ls (lijst bestanden)
  • pwd (toon huidige directory)
  • whoami (toon gebruiker)
  • date (toon datum/tijd)

Vereisten:
  • pi_app.py moet draaien op Raspberry Pi
  • SSH moet geactiveerd zijn op Pi
  • WebSocket server moet bereikbaar zijn

Probleemoplossing:
  • Geen verbinding? Check pi_app.py draait
  • SSH fout? Check SSH instellingen op Pi
  • Geen output? Check SSH commando syntax
        """,

        "ding": """
🔊 SPEECH RECOGNITION TRAINING HELP
════════════════════════════════════════════════════════════════════

Beschrijving:
  Train spraakherkenning op specifieke zinnen of woorden voor betere
  nauwkeurigheid in jouw specifieke spraakpatroon.

Status:
  ⚠️  Deze functionaliteit wordt binnenkort geïmplementeerd

Voorlopig Alternatief:
  Gebruik --voice-test voor speaker recognition training.
  Dit leert het systeem jouw unieke stemkenmerken kennen.

Toekomstige Functionaliteit:
  • Specifieke zinnen trainen (bijv. wachtwoorden)
  • Woorden met accenten trainen
  • Jargon of vaktermen trainen
  • Regionale uitspraak verbeteren

Houd updates in de gaten voor implementatie!
        """,

        "interactive": """
🎯 INTERACTIVE MODE HELP
════════════════════════════════════════════════════════════════════

Beschrijving:
  Start een interactief menu met klikbare profile opties.
  Kies door een nummer in te typen, geen command line argumenten nodig.

Gebruik:
  python laptop_app.py --interactive
  OF gewoon: python laptop_app.py (zonder argumenten)

Menu Opties:
  [1] 🌐 Normaal Gebruik - Standaard setup
  [2] ⚡ Snelste Speech Recognition - Tiny model
  [3] 🎤 Live Modus met Wake Word - Continu luisteren
  [4] 🎙️ Stem Training - Train nieuwe stem
  [5] 👤 Speaker Recognition Test - Test training
  [6] 🔐 Rechten Beheren - Pas rechten aan
  [7] 🚨 Strict Onbekende Rechten - Beperk onbekenden
  [8] 🔧 SSH Commando Test - Test SSH

Besturing:
  • Typ nummer (0-8) om te selecteren
  • [0] = Exit menu
  • Ctrl+C = Directe exit
  • ENTER = Bevestiging

Voordelen:
  • Geen lange commando's typen
  • Duidelijke beschrijvingen
  • Beginners vriendelijk
  • Handig voor nieuwe gebruikers
        """,

        "mode": """
🎛️ CHAT MODE HELP
════════════════════════════════════════════════════════════════════

Beschrijving:
  Kies hoe je met Jarvis wilt communiceren - verschillende opname methoden.

Gebruik:
  python laptop_app.py --host localhost --mode <modus>

Modi:
  press    (Aanbevolen voor beginners)
    Druk ENTER om op te nemen, los om te stoppen
    Controle over timing
    Bestand voor nieuwe gebruikers

  live     (Hands-free ervaring)
    Luist continu naar wake word
    Automatisch opnemen na wake word
    Ideaal voor langdurig gebruik

  hold     (Precieze controle)
    Houd SPACE vast om op te nemen
    Los om te versturen
    Maximale timing controle

Wake Word (voor live mode):
  Standaard: "jarvis"
  Aanpasbaar met: --wake-word "woord"

Voorbeelden:
  python laptop_app.py --host localhost --mode press
  python laptop_app.py --host localhost --mode live --wake-word hallo
  python laptop_app.py --host localhost --mode hold

Tips:
  • Begin met press mode
  • Gebruik live na training
  • Hold voor korte commando's
        """,

        "whisper-model": """
🎤 SPEECH RECOGNITION MODEL HELP
════════════════════════════════════════════════════════════════════

Beschrijving:
  Kies een Whisper model voor spraak-naar-tekst conversie.
  Balans tussen snelheid en nauwkeurigheid.

Gebruik:
  python laptop_app.py --host localhost --whisper-model <model>

Modellen:
  tiny    (Snelst, laagste nauwkeurigheid)
    • Snelheid: ⚡⚡⚡⚡⚡
    • Nauwkeurigheid: ★☆☆☆☆
    • CPU gebruik: Laag
    • Memory: ~39 MB
    • Aanbevolen voor: zwakke hardware

  base    (Balans - Standaard)
    • Snelheid: ⚡⚡⚡☆☆
    • Nauwkeurigheid: ★★★☆☆
    • CPU gebruik: Medium
    • Memory: ~74 MB
    • Aanbevolen voor: dagelijks gebruik

  small   (Betere nauwkeurigheid)
    • Snelheid: ⚡⚡☆☆☆
    • Nauwkeurigheid: ★★★★☆
    • CPU gebruik: Medium-Hoog
    • Memory: ~244 MB
    • Aanbevolen voor: betere transcriptie

  medium  (Hoge nauwkeurigheid)
    • Snelheid: ⚡☆☆☆☆
    • Nauwkeurigheid: ★★★★★
    • CPU gebruik: Hoog
    • Memory: ~769 MB
    • Aanbevolen voor: professioneel gebruik

  large   (Beste kwaliteit)
    • Snelheid: ☆☆☆☆☆
    • Nauwkeurigheid: ★★★★★
    • CPU gebruik: Zeer hoog
    • Memory: ~1550 MB
    • Aanbevolen voor: GPU systemen

Nederlands Ondersteuning:
  • Alle modellen ondersteunen Nederlands
  • --language nl voor beste resultaten
  • Tiny/Base meestal voldoende voor NL

Voorbeelden:
  python laptop_app.py --host localhost --whisper-model tiny
  python laptop_app.py --host localhost --whisper-model base --language nl
        """,
    }

    help_text = help_topics.get(topic)
    if help_text:
        print(help_text)
    else:
        print(f"❌ Geen specifieke help beschikbaar voor: {topic}")
        print("   Gebruik --help voor volledige help of --interactive voor menu")


def check_specific_help():
    """Check of specifieke help wordt gevraagd."""
    import sys

    # Check sys.argv voor --help met specifiek onderwerp
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        # Zoek naar specifieke onderwerpen
        help_args = {
            "--keuzemenu": "interactive",
            "--instellingen": "interactive",
            "--voice-test": "voice-test",
            "--test-speaker": "test-speaker",
            "--manage-permissions": "manage-permissions",
            "--set-unknown-perms": "set-unknown-perms",
            "--ssh-test": "ssh-test",
            "--ding": "ding",
            "--mode": "mode",
            "--whisper-model": "whisper-model",
        }

        for arg, topic in help_args.items():
            if arg in args:
                show_specific_help(topic)
                sys.exit(0)


def main():
    # Check voor specifieke help eerst
    check_specific_help()

    parser = argparse.ArgumentParser(
        description="Laptop client for Raspberry Pi bot with speaker recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📚 ALLE OPTIES:

🎯 INTERACTIEVE MODE:
  --keuzemenu              Start interactief keuzemenu voor instellingen en profiles
                          Selecteer profielen met nummerieke keuzes
                          Wijzig default instellingen via menu

🔧 SETTINGS MODE:
  --instellingen            Open uitgebreid instellingen menu
                          Wijzig alle configuraties (connection, AI, audio, permissions, etc.)
                          Categorieën: verbinding, AI, audio, speaker recognition, rechten, interface, performance, advanced

🌐 VERBINDING:
  --host HOST              IP adres van de Raspberry Pi (verplicht)
  --port PORT              WebSocket server poort (default: 8765)

🤖 AI MODEL:
  --ollama-model MODEL     Ollama AI model (default: phi3:mini)
                         Beschikbare modellen: phi3:mini, llama2, mistral, etc.

🎤 SPEECH RECOGNITION:
  --whisper-model MODEL    Speech recognition model (default: base)
                         Modellen: tiny (snelst), base (balans), small (nauwkeuriger)
  --record-seconds SEC    Opnameduur in seconden (default: 8)
  --language LANG          Taal voor transcriptie (default: nl)
                         Beschikbaar: nl, en, de, fr, es, it, etc.
  --input-device NUM       Audio input device nummer (default: auto)
                          Gebruik --list-devices om apparaten te zien

🎛️ CHAT MODUS:
  --mode MODE              Chat modus (default: press)
                         press = Druk ENTER om op te nemen
                         live   = Luist continu (wake word)
                         hold   = Houd SPACE vast om op te nemen
  --wake-word WORD        Wake word voor live modus (default: jarvis)

🎤 SPEAKER RECOGNITION:
  --voice-test             Start voice training modus
  --voice-name NAME        Naam voor de stem (verplicht met --voice-test)
                         Train 5 opnames met voorgeschreven zinnen voor betere nauwkeurigheid
                         Zinnen bevatten diverse klanken, cijfers en intonatie
                         Whisper transcribeert automatisch je opnames voor spraakherkenning training
  --voice-role ROLE       Rol voor de stem (default: user)
                         Beschikbaar: admin, user, guest, restricted
  --test-speaker           Test speaker recognition zonder verbinding
                          Test of stem training werkt

🔐 RECHTEN SYSTEEM:
  --manage-permissions     Beheer spreker rechten
                          Toon huidige rechten voor bekende/onbekende sprekers
  --set-unknown-perms JSON Pas onbekende rechten aan via JSON string
                          Voorbeeld: '{"full_chat":true,"ssh_commands":false}'

🔧 TESTING:
  --ssh-test               Test SSH commando's via websocket
  --ding TEXT              Spraakherkenning training (toekomstige functionaliteit)
  --debug                  Schakel volledige debug logging aan

📋 START PROFILES:
  # Interactief menu (eenvoudigst!)
  python laptop_app.py --keuzemenu
  OF gewoon: python laptop_app.py

  # Volledige instellingen menu
  python laptop_app.py --instellingen

  # Normaal gebruik (aanbevolen)
  python laptop_app.py --host localhost --port 8765

  # Snelste speech recognition
  python laptop_app.py --host localhost --whisper-model tiny

  # Live modus met wake word
  python laptop_app.py --host localhost --mode live --wake-word jarvis

  # Stem training
  python laptop_app.py --host localhost --voice-test --voice-name "Jan"

  # Speaker recognition test
  python laptop_app.py --host localhost --test-speaker

  # Rechten beheren
  python laptop_app.py --host localhost --manage-permissions

  # Strict onbekende rechten
  python laptop_app.py --host localhost --set-unknown-perms '{"full_chat":false}'

📁 BESTANDEN:
  settings.json               - Centrale configuratie (verbindinstellingen, rechten, AI instellingen)
  voice_profiles.json          - Stem embeddings (wordt automatisch aangemaakt)
  pretrained_models/           - AI model cache (SpeechBrain)

🔐 STANDAARD RECHTEN:
  Bekende sprekers:
    • full_chat: true (volle chat toegang)
    • ssh_commands: true (SSH commando's toegestaan)
    • system_commands: true (systeem commando's toegestaan)
    • file_operations: true (bestandsoperaties toegestaan)

  Onbekende sprekers:
    • full_chat: true (chat toegang)
    • ssh_commands: false (geen SSH toegang)
    • system_commands: false (geen systeem commando's)
    • file_operations: false (geen bestandsoperaties)
        """
    )
    parser.add_argument("--keuzemenu", action="store_true", help="Start interactief keuzemenu voor instellingen en profiles")
    parser.add_argument("--instellingen", action="store_true", help="Open uitgebreid instellingen menu")
    parser.add_argument("--host", required=False, help="IP van de Raspberry Pi (verplicht)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="WebSocket server poort (default: 8765)")
    parser.add_argument("--ollama-model", default=DEFAULT_MODEL, help="Ollama AI model (default: phi3:mini)")
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Speech recognition model (default: base)")
    parser.add_argument("--record-seconds", type=int, default=DEFAULT_RECORD_SECONDS, help="Opnameduur in seconden (default: 8)")
    parser.add_argument("--mode", choices=["live", "press", "hold"], default=DEFAULT_MODE, help="Chat modus: live=waak woord, press=ENTER om opnemen, hold=SPACE vasthouden (default: press)")
    parser.add_argument("--wake-word", default=DEFAULT_WAKE_WORD, help="Wake word voor live modus (default: jarvis)")
    parser.add_argument("--input-device", type=int, default=DEFAULT_INPUT_DEVICE, help="Audio input device nummer (default: auto)")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Taal voor transcriptie (default: nl)")
    parser.add_argument("--ssh-test", action="store_true", help="Test SSH commando's via websocket")
    parser.add_argument("--voice-test", action="store_true", help="Voice training modus - train stem met Whisper transcripties")
    parser.add_argument("--voice-name", type=str, help="Naam voor de stem bij training (verplicht met --voice-test)")
    parser.add_argument("--voice-role", type=str, default="user", help="Rol voor de stem bij training (default: user - beschikbare: admin, user, guest, restricted)")
    parser.add_argument("--ding", type=str, help="Spraakherkenning training - train specifieke zinnen of woorden")
    parser.add_argument("--test-speaker", action="store_true", help="Test speaker recognition zonder verbinding met Pi")
    parser.add_argument("--manage-permissions", action="store_true", help="Beheer spreker rechten - toon huidige rechten")
    parser.add_argument("--set-unknown-perms", type=str, help="Zet onbekende spreker rechten via JSON string")
    parser.add_argument("--debug", action="store_true", help="Schakel volledige debug logging aan")
    args = parser.parse_args()

    # Enable debug mode if flag
    if args.debug:
        global DEBUG_VERBOSE
        DEBUG_VERBOSE = True
        debug_log("DEBUG MODE GESTART VIA --debug FLAG", "SYSTEM")
        debug_log(f"Args: {vars(args)}", "SYSTEM")

    # Start interactief menu als --keuzemenu of geen host opgegeven
    if args.keuzemenu or not args.host:
        # Load settings voor keuzemenu
        settings_mgr = SettingsManager()
        interactive_profile_selector(settings_mgr)
        return

    # Start instellingen menu als --instellingen
    if args.instellingen:
        settings_mgr = SettingsManager()
        run_full_settings_menu(settings_mgr)
        return

    # Load settings voor defaults
    settings_mgr = SettingsManager()

    # Override command line args met settings
    default_args = settings_mgr.get_default_arguments()
    if not args.host:
        args.host = default_args["host"]
    if args.port == DEFAULT_PORT:
        args.port = default_args["port"]
    if args.ollama_model == DEFAULT_MODEL:
        args.ollama_model = default_args["ollama_model"]
    if args.whisper_model == DEFAULT_WHISPER_MODEL:
        args.whisper_model = default_args["whisper_model"]
    if args.record_seconds == DEFAULT_RECORD_SECONDS:
        args.record_seconds = default_args["record_seconds"]
    if args.language == DEFAULT_LANGUAGE:
        args.language = default_args["language"]
    if args.mode == DEFAULT_MODE:
        args.mode = default_args["mode"]
    if args.wake_word == DEFAULT_WAKE_WORD:
        args.wake_word = default_args["wake_word"]
    if args.input_device == DEFAULT_INPUT_DEVICE:
        args.input_device = default_args["input_device"]

    # Check verplichte argumenten na settings merge
    if not args.host:
        print("❌ Fout: --host is verplicht (gebruik --keuzemenu voor menu)")
        print("   Gebruik: python laptop_app.py --keuzemenu")
        print("   OF: python laptop_app.py --host localhost")
        sys.exit(1)

    uri = get_optimal_uri(args.host, args.port)

    if args.ssh_test:
        print("SSH TEST MODE - Commando's via websocket")
        print("=" * 50)
        asyncio.run(test_ssh_command(uri))


def run_full_settings_menu(settings_mgr):
    """Uitgebreid instellingen menu met alle categorieën."""
    import sys

    while True:
        print("\n" + "=" * 60)
        print("⚙️  VOLLEDIGE INSTELLINGEN MENU")
        print("=" * 60)

        print("\n📋 INSTELLING CATEGORIEËN:\n")

        print("  [1] 🌐 Verbindingsinstellingen")
        print("  [2] 🤖 AI Model Instellingen")
        print("  [3] 🎤 Audio Instellingen")
        print("  [4] 👤 Speaker Recognition")
        print("  [5] 🔐 Rechten (Bekende Sprekers)")
        print("  [6] 🔐 Rechten (Onbekende Sprekers)")
        print("  [7] 🎭 Rollen & Permissions")
        print("  [8] 🖥️  Interface Instellingen")
        print("  [9] 🚀 Performance Instellingen")
        print(" [10] 🔧 Advanced Instellingen")
        print(" [11] 📊 Overzicht Alle Instellingen")
        print(" [12] 💾 Export/Import Settings")
        print(" [0]  🚪 Exit")

        print("\n" + "-" * 60)
        print("Selecteer een categorie (0-12): ", end="")

        try:
            choice = input().strip()

            if choice == "0":
                print("👋 Tot ziens!")
                sys.exit(0)
            elif choice == "1":
                edit_connection_settings(settings_mgr)
            elif choice == "2":
                edit_ai_settings(settings_mgr)
            elif choice == "3":
                edit_audio_settings(settings_mgr)
            elif choice == "4":
                edit_speaker_recognition_settings(settings_mgr)
            elif choice == "5":
                edit_known_permissions(settings_mgr)
            elif choice == "6":
                edit_unknown_permissions(settings_mgr)
            elif choice == "7":
                edit_roles_management(settings_mgr)
            elif choice == "8":
                edit_interface_settings(settings_mgr)
            elif choice == "9":
                edit_performance_settings(settings_mgr)
            elif choice == "10":
                edit_advanced_settings(settings_mgr)
            elif choice == "11":
                show_all_settings_overview(settings_mgr)
            elif choice == "12":
                export_import_settings(settings_mgr)
            else:
                print("❌ Ongeldige keuze. Probeer opnieuw.")

        except KeyboardInterrupt:
            print("\n\n👋 Tot ziens!")
            sys.exit(0)
        except EOFError:
            print("\n\n👋 Tot ziens!")
            sys.exit(0)


def edit_connection_settings(settings_mgr):
    """Wijzig verbindingsinstellingen."""
    print("\n🌐 Verbindingsinstellingen")
    print("-" * 40)

    conn = settings_mgr.get_connection_settings()

    host = input(f"Host [{conn.get('host', 'localhost')}]: ").strip()
    port = input(f"Poort [{conn.get('port', 8765)}]: ").strip()
    auto = input(f"Auto Connect [{'true' if conn.get('auto_connect') else 'false'}]: ").strip().lower()

    new_settings = {}
    if host:
        new_settings["host"] = host
    if port:
        try:
            new_settings["port"] = int(port)
        except ValueError:
            print("⚠️  Ongeldige poort, gebruik huidige waarde")
    if auto in ['true', 'false']:
        new_settings["auto_connect"] = auto == 'true'

    if new_settings:
        settings_mgr.set_connection_settings(new_settings)
        print("✅ Verbindingsinstellingen bijgewerkt!")
    else:
        print("Geen wijzigingen.")


def edit_ai_settings(settings_mgr):
    """Wijzig AI instellingen."""
    print("\n🤖 AI Model Instellingen")
    print("-" * 40)

    ai = settings_mgr.get_ai_settings()

    ollama = input(f"Ollama Model [{ai.get('ollama_model', 'phi3:mini')}]: ").strip()
    whisper = input(f"Whisper Model [{ai.get('whisper_model', 'base')}]: ").strip()
    temp = input(f"Temperature [{ai.get('temperature', 0.7)}]: ").strip()
    tokens = input(f"Max Tokens [{ai.get('max_tokens', 1000)}]: ").strip()

    new_settings = {}
    if ollama:
        new_settings["ollama_model"] = ollama
    if whisper:
        new_settings["whisper_model"] = whisper
    if temp:
        try:
            new_settings["temperature"] = float(temp)
        except ValueError:
            print("⚠️  Ongeldige temperature, gebruik huidige waarde")
    if tokens:
        try:
            new_settings["max_tokens"] = int(tokens)
        except ValueError:
            print("⚠️  Ongeldige tokens, gebruik huidige waarde")

    if new_settings:
        settings_mgr.set_ai_settings(new_settings)
        print("✅ AI instellingen bijgewerkt!")
    else:
        print("Geen wijzigingen.")


def edit_audio_settings(settings_mgr):
    """Wijzig audio instellingen."""
    print("\n🎤 Audio Instellingen")
    print("-" * 40)

    audio = settings_mgr.get_audio_settings()

    mode = input(f"Mode [{audio.get('mode', 'press')}]: ").strip()
    lang = input(f"Taal [{audio.get('language', 'nl')}]: ").strip()
    duration = input(f"Opnameduur [{audio.get('record_seconds', 8)}]: ").strip()
    wake = input(f"Wake Word [{audio.get('wake_word', 'jarvis')}]: ").strip()
    sample = input(f"Sample Rate [{audio.get('sample_rate', 16000)}]: ").strip()
    noise = input(f"Noise Suppression [{'true' if audio.get('noise_suppression') else 'false'}]: ").strip().lower()

    new_settings = {}
    if mode:
        new_settings["mode"] = mode
    if lang:
        new_settings["language"] = lang
    if duration:
        try:
            new_settings["record_seconds"] = int(duration)
        except ValueError:
            print("⚠️  Ongeldige duur, gebruik huidige waarde")
    if wake:
        new_settings["wake_word"] = wake
    if sample:
        try:
            new_settings["sample_rate"] = int(sample)
        except ValueError:
            print("⚠️  Ongeldige sample rate, gebruik huidige waarde")
    if noise in ['true', 'false']:
        new_settings["noise_suppression"] = noise == 'true'

    if new_settings:
        settings_mgr.set_audio_settings(new_settings)
        print("✅ Audio instellingen bijgewerkt!")
    else:
        print("Geen wijzigingen.")


def edit_speaker_recognition_settings(settings_mgr):
    """Wijzig speaker recognition instellingen."""
    print("\n👤 Speaker Recognition Instellingen")
    print("-" * 40)

    speaker = settings_mgr.get_speaker_recognition_settings()

    threshold = input(f"Threshold [{speaker.get('threshold', 0.7)}]: ").strip()
    enabled = input(f"Enabled [{'true' if speaker.get('enabled') else 'false'}]: ").strip().lower()

    new_settings = {}
    if threshold:
        try:
            new_settings["threshold"] = float(threshold)
        except ValueError:
            print("⚠️  Ongeldige threshold, gebruik huidige waarde")
    if enabled in ['true', 'false']:
        new_settings["enabled"] = enabled == 'true'

    if new_settings:
        settings_mgr.set_audio_settings({"speaker_recognition": new_settings})
        print("✅ Speaker recognition instellingen bijgewerkt!")
    else:
        print("Geen wijzigingen.")


def edit_known_permissions(settings_mgr):
    """Wijzig rechten voor bekende sprekers."""
    print("\n🔐 Rechten (Bekende Sprekers)")
    print("-" * 40)

    perms = settings_mgr.get_permissions_settings()
    known = perms.get("known", {})

    full_chat = input(f"Full Chat [{'true' if known.get('full_chat') else 'false'}]: ").strip().lower()
    ssh = input(f"SSH Commands [{'true' if known.get('ssh_commands') else 'false'}]: ").strip().lower()
    system = input(f"System Commands [{'true' if known.get('system_commands') else 'false'}]: ").strip().lower()
    files = input(f"File Operations [{'true' if known.get('file_operations') else 'false'}]: ").strip().lower()
    audio = input(f"Max Audio Level [{known.get('max_audio_level', 1.0)}]: ").strip()
    timeout = input(f"Request Timeout [{known.get('request_timeout', 30)}]: ").strip()

    new_perms = known.copy()
    if full_chat in ['true', 'false']:
        new_perms["full_chat"] = full_chat == 'true'
    if ssh in ['true', 'false']:
        new_perms["ssh_commands"] = ssh == 'true'
    if system in ['true', 'false']:
        new_perms["system_commands"] = system == 'true'
    if files in ['true', 'false']:
        new_perms["file_operations"] = files == 'true'
    if audio:
        try:
            new_perms["max_audio_level"] = float(audio)
        except ValueError:
            print("⚠️  Ongeldige audio level, gebruik huidige waarde")
    if timeout:
        try:
            new_perms["request_timeout"] = int(timeout)
        except ValueError:
            print("⚠️  Ongeldige timeout, gebruik huidige waarde")

    if new_perms != known:
        settings_mgr.set_permissions({"known": new_perms})
        print("✅ Rechten bijgewerkt!")
    else:
        print("Geen wijzigingen.")


def edit_unknown_permissions(settings_mgr):
    """Wijzig rechten voor onbekende sprekers."""
    print("\n🔐 Rechten (Onbekende Sprekers)")
    print("-" * 40)

    perms = settings_mgr.get_permissions_settings()
    unknown = perms.get("unknown", {})

    full_chat = input(f"Full Chat [{'true' if unknown.get('full_chat') else 'false'}]: ").strip().lower()
    ssh = input(f"SSH Commands [{'true' if unknown.get('ssh_commands') else 'false'}]: ").strip().lower()
    system = input(f"System Commands [{'true' if unknown.get('system_commands') else 'false'}]: ").strip().lower()
    files = input(f"File Operations [{'true' if unknown.get('file_operations') else 'false'}]: ").strip().lower()
    audio = input(f"Max Audio Level [{unknown.get('max_audio_level', 0.5)}]: ").strip()
    timeout = input(f"Request Timeout [{unknown.get('request_timeout', 15)}]: ").strip()

    new_perms = unknown.copy()
    if full_chat in ['true', 'false']:
        new_perms["full_chat"] = full_chat == 'true'
    if ssh in ['true', 'false']:
        new_perms["ssh_commands"] = ssh == 'true'
    if system in ['true', 'false']:
        new_perms["system_commands"] = system == 'true'
    if files in ['true', 'false']:
        new_perms["file_operations"] = files == 'true'
    if audio:
        try:
            new_perms["max_audio_level"] = float(audio)
        except ValueError:
            print("⚠️  Ongeldige audio level, gebruik huidige waarde")
    if timeout:
        try:
            new_perms["request_timeout"] = int(timeout)
        except ValueError:
            print("⚠️  Ongeldige timeout, gebruik huidige waarde")

    if new_perms != unknown:
        settings_mgr.set_permissions({"unknown": new_perms})
        print("✅ Rechten bijgewerkt!")
    else:
        print("Geen wijzigingen.")


def edit_roles_management(settings_mgr):
    """Beheer rollen en hun permissions."""

    while True:
        print("\n🎭 ROLLEN & PERMISSIONS")
        print("-" * 40)

        perms = settings_mgr.get_permissions_settings()
        roles = perms.get("roles", {})

        print("Beschikbare Rollen:\n")

        role_list = list(roles.keys())
        for i, (role_name, role_data) in enumerate(roles.items(), 1):
            description = role_data.get("description", "Geen beschrijving")
            print(f"  [{i}] 🎭 {role_name}")
            print(f"      {description}")
            print(f"      SSH: {'✅' if role_data.get('ssh_commands') else '❌'} | Sys: {'✅' if role_data.get('system_commands') else '❌'}")
            print()

        print("  [N] 🔨 Nieuwe Rol Aanmaken")
        print("  [0] 🔙 Terug")

        print("\n" + "-" * 60)
        print("Kies een rol om te bewerken (of N voor nieuw, 0 voor terug): ", end="")

        choice = input().strip().upper()

        if choice == "0":
            return
        elif choice == "N":
            create_new_role(settings_mgr)
        else:
            try:
                role_num = int(choice)
                if 1 <= role_num <= len(role_list):
                    selected_role = role_list[role_num - 1]
                    edit_role_permissions(settings_mgr, selected_role)
                else:
                    print("❌ Ongeldige keuze.")
            except ValueError:
                print("❌ Voer een nummer in.")


def edit_role_permissions(settings_mgr, role_name: str):
    """Wijzig permissions voor een specifieke rol."""
    print(f"\n🎭 Rol: {role_name}")
    print("-" * 40)

    perms = settings_mgr.get_permissions_settings()
    roles = perms.get("roles", {})
    role_data = roles.get(role_name, {}).copy()

    print("Huidige Permissions:\n")
    print(f"  Beschrijving: {role_data.get('description', 'Geen beschrijving')}")
    print(f"  Full Chat: {'✅' if role_data.get('full_chat') else '❌'}")
    print(f"  SSH Commands: {'✅' if role_data.get('ssh_commands') else '❌'}")
    print(f"  System Commands: {'✅' if role_data.get('system_commands') else '❌'}")
    print(f"  File Operations: {'✅' if role_data.get('file_operations') else '❌'}")
    print(f"  Max Audio Level: {role_data.get('max_audio_level', 1.0)}")
    print(f"  Request Timeout: {role_data.get('request_timeout', 30)}")
    print()

    description = input(f"Nieuwe beschrijving [{role_data.get('description', '')}]: ").strip()
    full_chat = input(f"Full Chat [{'true' if role_data.get('full_chat') else 'false'}]: ").strip().lower()
    ssh = input(f"SSH Commands [{'true' if role_data.get('ssh_commands') else 'false'}]: ").strip().lower()
    system = input(f"System Commands [{'true' if role_data.get('system_commands') else 'false'}]: ").strip().lower()
    files = input(f"File Operations [{'true' if role_data.get('file_operations') else 'false'}]: ").strip().lower()
    audio = input(f"Max Audio Level [{role_data.get('max_audio_level', 1.0)}]: ").strip()
    timeout = input(f"Request Timeout [{role_data.get('request_timeout', 30)}]: ").strip()

    new_role_data = role_data.copy()
    if description:
        new_role_data["description"] = description
    if full_chat in ['true', 'false']:
        new_role_data["full_chat"] = full_chat == 'true'
    if ssh in ['true', 'false']:
        new_role_data["ssh_commands"] = ssh == 'true'
    if system in ['true', 'false']:
        new_role_data["system_commands"] = system == 'true'
    if files in ['true', 'false']:
        new_role_data["file_operations"] = files == 'true'
    if audio:
        try:
            new_role_data["max_audio_level"] = float(audio)
        except ValueError:
            print("⚠️  Ongeldige audio level, gebruik huidige waarde")
    if timeout:
        try:
            new_role_data["request_timeout"] = int(timeout)
        except ValueError:
            print("⚠️  Ongeldige timeout, gebruik huidige waarde")

    if new_role_data != role_data:
        # Update de rol in permissions
        if "roles" not in perms:
            perms["roles"] = {}
        perms["roles"][role_name] = new_role_data
        settings_mgr.set_permissions({"roles": perms["roles"]})
        print(f"✅ Rol '{role_name}' bijgewerkt!")
    else:
        print("Geen wijzigingen.")


def create_new_role(settings_mgr):
    """Maak een nieuwe rol aan."""
    print("\n🔨 NIEUWE ROL AANMAKEN")
    print("-" * 40)

    role_name = input("Rol naam (bijv. 'moderator', 'developer'): ").strip()
    if not role_name:
        print("❌ Rol naam is verplicht")
        return

    # Check of rol al bestaat
    perms = settings_mgr.get_permissions_settings()
    roles = perms.get("roles", {})
    if role_name in roles:
        print(f"❌ Rol '{role_name}' bestaat al")
        return

    description = input("Beschrijving van de rol: ").strip()
    print("\nStandaard permissions voor nieuwe rol:")
    print("Kies een basis rol om te kopiëren:")
    print("  [1] user - Standaard gebruiker")
    print("  [2] guest - Gast met beperkte rechten")
    print("  [3] restricted - Zeer beperkte toegang")
    print("  [4] admin - Volledige toegang")
    print("  [0] Custom - Start met lege permissions")

    base_choice = input("Kies basis rol: ").strip()

    base_role = {}
    if base_choice == "1":
        base_role = roles.get("user", {})
    elif base_choice == "2":
        base_role = roles.get("guest", {})
    elif base_choice == "3":
        base_role = roles.get("restricted", {})
    elif base_choice == "4":
        base_role = roles.get("admin", {})
    elif base_choice == "0":
        base_role = {
            "full_chat": True,
            "ssh_commands": False,
            "system_commands": False,
            "file_operations": False,
            "max_audio_level": 0.8,
            "request_timeout": 25
        }

    new_role = base_role.copy()
    new_role["description"] = description or "Custom rol"

    # Opslaan
    if "roles" not in perms:
        perms["roles"] = {}
    perms["roles"][role_name] = new_role
    settings_mgr.set_permissions({"roles": perms["roles"]})
    print(f"✅ Nieuwe rol '{role_name}' aangemaakt!")


def edit_interface_settings(settings_mgr):
    """Wijzig interface instellingen."""
    print("\n🖥️  Interface Instellingen")
    print("-" * 40)

    interface = settings_mgr.get_interface_settings()

    welcome = input(f"Show Welcome [{'true' if interface.get('show_welcome') else 'false'}]: ").strip().lower()
    timestamps = input(f"Show Timestamps [{'true' if interface.get('show_timestamps') else 'false'}]: ").strip().lower()
    feedback = input(f"Audio Feedback [{'true' if interface.get('audio_feedback') else 'false'}]: ").strip().lower()
    confirm = input(f"Confirm Dangerous [{'true' if interface.get('confirm_dangerous') else 'false'}]: ").strip().lower()

    new_settings = {}
    if welcome in ['true', 'false']:
        new_settings["show_welcome"] = welcome == 'true'
    if timestamps in ['true', 'false']:
        new_settings["show_timestamps"] = timestamps == 'true'
    if feedback in ['true', 'false']:
        new_settings["audio_feedback"] = feedback == 'true'
    if confirm in ['true', 'false']:
        new_settings["confirm_dangerous"] = confirm == 'true'

    if new_settings:
        settings_mgr.set_interface_settings(new_settings)
        print("✅ Interface instellingen bijgewerkt!")
    else:
        print("Geen wijzigingen.")


def edit_performance_settings(settings_mgr):
    """Wijzig performance instellingen."""
    print("\n🚀 Performance Instellingen")
    print("-" * 40)

    perf = settings_mgr.get_performance_settings()

    chunk = input(f"Audio Chunk Size [{perf.get('audio_chunk_size', 1024)}]: ").strip()
    cache = input(f"Cache Models [{'true' if perf.get('cache_models') else 'false'}]: ").strip().lower()
    lazy = input(f"Lazy Loading [{'true' if perf.get('lazy_loading') else 'false'}]: ").strip().lower()

    new_settings = {}
    if chunk:
        try:
            new_settings["audio_chunk_size"] = int(chunk)
        except ValueError:
            print("⚠️  Ongeldige chunk size, gebruik huidige waarde")
    if cache in ['true', 'false']:
        new_settings["cache_models"] = cache == 'true'
    if lazy in ['true', 'false']:
        new_settings["lazy_loading"] = lazy == 'true'

    if new_settings:
        settings_mgr.set_performance_settings(new_settings)
        print("✅ Performance instellingen bijgewerkt!")
    else:
        print("Geen wijzigingen.")


def edit_advanced_settings(settings_mgr):
    """Wijzig advanced instellingen."""
    print("\n🔧 Advanced Instellingen")
    print("-" * 40)

    adv = settings_mgr.get_advanced_settings()

    debug = input(f"Debug Mode [{'true' if adv.get('debug_mode') else 'false'}]: ").strip().lower()
    verbose = input(f"Verbose Logging [{'true' if adv.get('verbose_logging') else 'false'}]: ").strip().lower()
    save = input(f"Save Recordings [{'true' if adv.get('save_recordings') else 'false'}]: ").strip().lower()
    path = input(f"Recordings Path [{adv.get('recordings_path', 'recordings/')}]: ").strip()

    new_settings = {}
    if debug in ['true', 'false']:
        new_settings["debug_mode"] = debug == 'true'
    if verbose in ['true', 'false']:
        new_settings["verbose_logging"] = verbose == 'true'
    if save in ['true', 'false']:
        new_settings["save_recordings"] = save == 'true'
    if path:
        new_settings["recordings_path"] = path

    if new_settings:
        settings_mgr.set_advanced_settings(new_settings)
        print("✅ Advanced instellingen bijgewerkt!")
    else:
        print("Geen wijzigingen.")


def show_all_settings_overview(settings_mgr):
    """Toon overzicht van alle instellingen."""
    print("\n📊 OVERZICHT ALLE INSTELLINGEN")
    print("=" * 60)

    import json
    print(json.dumps(settings_mgr.settings, indent=2))

    print("\n" + "-" * 60)
    input("Druk ENTER om terug te gaan...")


def export_import_settings(settings_mgr):
    """Export/Import settings."""
    print("\n💾 Export/Import Settings")
    print("-" * 40)

    print("\nOpties:")
    print("  [1] Export settings naar JSON bestand")
    print("  [2] Import settings van JSON bestand")
    print("  [3] Reset naar standaardwaarden")
    print("  [0] Terug")

    choice = input("Kies een optie: ").strip()

    if choice == "1":
        filename = input("Export bestandsnaam [settings_export.json]: ").strip() or "settings_export.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(settings_mgr.settings, f, indent=2)
            print(f"✅ Settings geëxporteerd naar {filename}")
        except Exception as e:
            print(f"❌ Export mislukt: {e}")

    elif choice == "2":
        filename = input("Import bestandsnaam: ").strip()
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                imported = json.load(f)
            settings_mgr.settings = settings_mgr._deep_merge(settings_mgr.settings, imported)
            settings_mgr._save_settings()
            print(f"✅ Settings geïmporteerd van {filename}")
        except Exception as e:
            print(f"❌ Import mislukt: {e}")

    elif choice == "3":
        confirm = input("Weet je het zeker? (ja/nee): ").strip().lower()
        if confirm == "ja":
            # Verwijder settings file om defaults te gebruiken
            settings_mgr.settings_file.unlink(missing_ok=True)
            print("✅ Settings gereset naar standaardwaarden")
            # Laad defaults opnieuw
            settings_mgr.settings = settings_mgr._load_settings()
        else:
            print("Reset geannuleerd")

    elif choice == "0":
        return
    else:
        print("❌ Ongeldige keuze.")

def cleanup_on_exit():
    """Cleanup functie die wordt aangeroepen bij script afsluiten."""
    debug_log("Cleanup bij afsluiten", "MEMORY")
    cleanup_models()

# Registreer cleanup handler
atexit.register(cleanup_on_exit)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Onderbroken door gebruiker")
        cleanup_on_exit()
    except Exception as e:
        print(f"\n❌ Fatale fout: {e}")
