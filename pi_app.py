import argparse
import array
import asyncio
import base64
import binascii
import gc
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import numpy as np
from datetime import datetime
from io import BytesIO
import moderngl
import pygame
import requests
import websockets
from dataclasses import dataclass
from pathlib import Path
import glob

# Probeer AI libraries te laden voor echte speaker recognition
try:
    import torch
    from speechbrain.inference.speaker import SpeakerRecognition
    HAS_SPEAKER_AI = True
except ImportError:
    HAS_SPEAKER_AI = False
    print("⚠️  SpeechBrain of Torch niet gevonden. Speaker Recognition werkt in demo-modus.")

# Forceer GPU/Vulkan ondersteuning voor Ollama indien beschikbaar op dit systeem
os.environ["OLLAMA_VULKAN"] = "1"
os.environ["GGML_VULKAN"] = "1"
# Gebruik alle beschikbare cores voor inference op de Pi 5
os.environ["OMP_NUM_THREADS"] = "4"

# ═══════════════════════════════════════════════════════════════════════════════
# ⚙️  INSTELLINGEN - PAS DEZE AAN NAAR WENS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_GENERAL_MODEL = "Qwen3:8b"     # Snelle standaard voor tekst
DEFAULT_VISION_MODEL = "qwen2.5vl:7b"   # Voor scherm- en afbeeldingsanalyse
DEFAULT_CODER_MODEL = "qwen2.5-coder:3b" # Voor programmeer-gerelateerde vragen
IDLE_MODELS = ["Qwen3:8b", "qwen2.5vl:7b", "qwen2.5-coder:3b"] # Modellen die we in GPU-idle willen houden

# Aggressive RAM Management (Pi 5 Optimalisatie)
AGGRESSIVE_RAM_MODE = False  # Indien True, worden modellen na elk gebruik meteen verwijderd

# AI OFFloading (Bespaar ruimte op de Pi!)
OLLAMA_HOST = "127.0.0.1"  # Verander naar IP van je laptop (bijv. "192.168.x.x") om modellen daar te draaien
REMOTE_AI_HOST = None      # Verander naar IP van je laptop voor Speaker AI (bijv. "192.168.x.x")

# Paden
PROJECT_DIR = Path(__file__).parent
PI_TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
MEMINFO_PATH = Path("/proc/meminfo")

# ═══════════════════════════════════════════════════════════════════════════════
# 🛠️  HELPER CLASSES & FUNCTIES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VisualState:
    emotion: str = "neutral"
    previous_emotion: str = "neutral"
    text: str = ""
    level: float = 0.35
    transition: float = 1.0
    audio_level: float = 0.0
    active_model: str | None = None


@dataclass
class ButtonRect:
    x: int
    y: int
    w: int
    h: int
    action: str


class ConnectionGate:
    """Beheert per-verbinding goedkeuringen voor veiligheid."""

    @dataclass
    class PendingConnection:
        client_id: str
        label: str
        accept_event: asyncio.Event
        decision: str = "pending"  # pending, accepted, rejected

    def __init__(self):
        self.pending: dict[str, self.PendingConnection] = {}
        self.allowed_ips: set[str] = self._load_allowed_ips()
        self.lock = threading.Lock()

    def _load_allowed_ips(self) -> set[str]:
        path = PROJECT_DIR / "allowed_ips.json"
        if path.exists():
            try:
                return set(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                return set()
        return set()

    def _save_allowed_ips(self):
        path = PROJECT_DIR / "allowed_ips.json"
        path.write_text(json.dumps(list(self.allowed_ips)))

    def request(self, label: str, ip: str | None = None) -> PendingConnection:
        client_id = str(uuid.uuid4())[:8]
        conn = self.PendingConnection(client_id, label, asyncio.Event())
        with self.lock:
            self.pending[client_id] = conn
        return conn

    def approve(self, client_id: str, accept: bool, always: bool = False, ip: str | None = None):
        with self.lock:
            conn = self.pending.get(client_id)
            if not conn:
                return
            conn.decision = "accepted" if accept else "rejected"
            if accept and always and ip:
                self.allowed_ips.add(ip)
                self._save_allowed_ips()
            # Gebruik call_soon_threadsafe voor de event loop van de server
            # Deze methode wordt vanuit de GUI-thread aangeroepen.
            loop = getattr(self, "_server_loop", None)
            if loop:
                loop.call_soon_threadsafe(conn.accept_event.set)

    def is_allowed(self, ip: str) -> bool:
        return ip in self.allowed_ips

    def remove(self, client_id: str):
        with self.lock:
            if client_id in self.pending:
                del self.pending[client_id]

    def snapshot(self) -> list[PendingConnection]:
        with self.lock:
            return list(self.pending.values())

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self._server_loop = loop


class SharedState:
    def __init__(self):
        self._visual = VisualState()
        self._lock = threading.Lock()
        self.gate = ConnectionGate()

    def set(self, emotion: str, text: str, level: float):
        with self._lock:
            if self._visual.emotion != emotion:
                self._visual.previous_emotion = self._visual.emotion
                self._visual.transition = 0.0
            self._visual.emotion = emotion
            self._visual.text = text
            self._visual.level = level

    def get(self) -> VisualState:
        with self._lock:
            return self._visual

    def step_transition(self, dt: float):
        with self._lock:
            self._visual.transition = min(1.0, self._visual.transition + dt)

    def update_visual(self, emotion: str = None, text: str = None, audio_level: float = None, active_model: str = None):
        with self._lock:
            if emotion is not None:
                if self._visual.emotion != emotion:
                    self._visual.previous_emotion = self._visual.emotion
                    self._visual.transition = 0.0
                self._visual.emotion = emotion
            if text is not None:
                self._visual.text = text
            if audio_level is not None:
                self._visual.audio_level = audio_level
            if active_model is not None:
                self._visual.active_model = active_model

    def approve_top_connection(self, accept: bool, always: bool = False):
        pending = self.gate.snapshot()
        if pending:
            top = pending[0]
            # Voor de IP-allowlist moeten we eigenlijk de IP weten, maar de GUI-thread
            # heeft die nu niet direct. In een echte app zouden we die opslaan in PendingConnection.
            self.gate.approve(top.client_id, accept, always)

    def set_server_loop(self, loop: asyncio.AbstractEventLoop):
        self.gate.set_loop(loop)


# ═══════════════════════════════════════════════════════════════════════════════
# 🤖 AI CORE - OLLAMA INTEGRATIE
# ═══════════════════════════════════════════════════════════════════════════════

def detect_emotion(text: str) -> str:
    text = text.lower()
    if any(w in text for w in ["hoera", "geweldig", "blij", "leuk", "fijn", "top", "super"]):
        return "happy"
    if any(w in text for w in ["helaas", "jammer", "verdrietig", "slecht", "pijn", "moeilijk"]):
        return "sad"
    if any(w in text for w in ["boos", "irritant", "stom", "fout", "foutmelding", "verschrikkelijk"]):
        return "angry"
    return "neutral"


async def call_ollama_api_async(model: str, prompt: str, timeout: int = 60, images: list[str] = None, system_prompt: str = None) -> str:
    """Async variant van de Ollama API aanroep."""
    return await asyncio.to_thread(call_ollama_api, model, prompt, timeout, images, system_prompt)

def get_available_models() -> list[str]:
    """Haal een lijst van alle beschikbare Ollama modellen op."""
    try:
        resp = requests.get(f"http://{OLLAMA_HOST}:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            return [m["name"] for m in resp.json().get("models", [])]
    except:
        pass
    return []


def call_ollama_api(model: str, prompt: str, timeout: int = 60, images: list[str] = None, system_prompt: str = None) -> str:
    """Blocking Ollama API aanroep met automatische fallback."""
    url = f"http://{OLLAMA_HOST}:11434/api/chat"

    # Controleer eerst of het model überhaupt bestaat, anders direct fallback
    available = get_available_models()
    
    selected_model = model
    if model not in available and available:
        # Fallback volgorde: qwen3:8b -> qwen2.5-coder:3b -> qwen2.5vl:7b -> eerste in de lijst
        if DEFAULT_GENERAL_MODEL in available:
            selected_model = DEFAULT_GENERAL_MODEL
        elif DEFAULT_CODER_MODEL in available:
            selected_model = DEFAULT_CODER_MODEL
        elif DEFAULT_VISION_MODEL in available:
            selected_model = DEFAULT_VISION_MODEL
        else:
            selected_model = available[0]
        print(f"⚠️ Model '{model}' niet gevonden. Gebruik fallback: {selected_model}")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    msg = {"role": "user", "content": prompt}
    if images:
        msg["images"] = images
    messages.append(msg)

    payload = {
        "model": selected_model,
        "messages": messages,
        "stream": False,
        "keep_alive": 0 if AGGRESSIVE_RAM_MODE else "5m",
        "options": {
            "num_predict": 500,
            "temperature": 0.7
        }
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json().get("message", {}).get("content", "")
        
        # Specifieke foutafhandeling voor model mismatch (als tags check faalde)
        body = resp.text
        if resp.status_code == 404:
            print(f"Ollama kon model '{selected_model}' niet vinden (404).")
            if selected_model != DEFAULT_GENERAL_MODEL:
                print(f"Poging tot fallback naar {DEFAULT_GENERAL_MODEL}")
                return call_ollama_api(DEFAULT_GENERAL_MODEL, prompt, timeout, images, system_prompt)
        
        return f"Ollama fout (code {resp.status_code}): {body}"
    except Exception as e:
        return f"Verbindingsfout met Ollama: {e}"


async def analyze_screen_capture(image_base64: str, question: str = None, section: str = "full", model: str = DEFAULT_VISION_MODEL) -> tuple[str, str]:
    """Analyseert een screenshot met een Vision model."""
    if not question:
        question = "Beschrijf kort wat je ziet op dit scherm of deze afbeelding."

    # Verbeterde systeeminstructie voor Vision-modellen om directer te zijn
    system_instr = (
        "Je bent Jarvis, een behulpzame assistent die meekijkt op het scherm van de gebruiker. "
        "Geef DIRECT antwoord op de vraag van de gebruiker. Geef geen algemene beschrijving "
        "van het scherm als de gebruiker een specifieke vraag stelt. "
        "Houd je antwoord kort, krachtig en to-the-point."
    )

    try:
        answer = await asyncio.to_thread(call_ollama_api, model, question, 60, [image_base64], system_instr)
        return answer, detect_emotion(answer)
    except Exception as exc:
        return f"Schermafbeelding analyse mislukt: {exc}", "angry"


def unload_models():
    """Maakt GPU-geheugen vrij door modellen te unloaden."""
    try:
        # Dit vertelt Ollama om alles uit GPU/RAM te gooien
        requests.post("http://127.0.0.1:11434/api/chat", json={
            "model": "qwen3:8b", # Of een ander model dat geladen is
            "keep_alive": 0
        }, timeout=5)
        print("🧹 Modellen uit GPU/RAM verwijderd.")
    except:
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 GUI - PYGAME & MODERNGL WAVE VISUAL
# ═══════════════════════════════════════════════════════════════════════════════

EMOTION_COLORS = {
    "neutral": (100, 160, 255),
    "happy": (120, 255, 140),
    "sad": (140, 150, 180),
    "angry": (255, 100, 100),
}

EMOTION_LEVELS = {
    "neutral": 0.35,
    "happy": 0.85,
    "sad": 0.15,
    "angry": 1.0,
}

EMOTION_PROFILES = {
    "neutral": {"shape": 1.0, "speed": 1.0, "thickness": 10, "offset": 0},
    "happy": {"shape": 1.3, "speed": 1.8, "thickness": 14, "offset": -20},
    "sad": {"shape": 0.4, "speed": 0.5, "thickness": 6, "offset": 40},
    "angry": {"shape": 2.2, "speed": 3.5, "thickness": 18, "offset": -10},
}

STOP_KEYS = (pygame.K_LCTRL, pygame.K_RCTRL, pygame.K_ESCAPE)

ENERGY_VERTEX_SHADER = """
#version 120
attribute vec2 in_vert;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""
ENERGY_FRAGMENT_SHADER = """
#version 120
// Legacy GLSL (GL 2.1 / Pi 4/5 Mesa V3D) gebruikt gl_FragColor i.p.v. een out variabele.
// De wiskunde-body hieronder is ongewijzigd ten opzichte van de originele #version 330 shader.
#define fragColor gl_FragColor

uniform vec2 iResolution;
uniform float iTime;
uniform float iVolume;
uniform float iThinking;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2(0.0,0.0)), hash(i + vec2(1.0,0.0)), u.x),
               mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0,1.0)), u.x), u.y);
}

float fbm(vec2 p) {
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    mat2 rot = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < 4; ++i) {
        v += a * noise(p);
        p = rot * p * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

void main() {
    vec2 uv = (gl_FragCoord.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);
    float d = length(uv);
    float volume = clamp(iVolume, 0.0, 1.0);
    float thinking = clamp(iThinking, 0.0, 1.0);

    float bg_noise = fbm(uv * 2.2 + vec2(iTime * 0.08, -iTime * 0.05));
    vec3 bg_a = vec3(0.012, 0.020, 0.045);
    vec3 bg_b = vec3(0.015, 0.075, 0.090);
    vec3 background = mix(bg_a, bg_b, bg_noise * 0.65) + vec3(0.0, 0.025, 0.02) * exp(-d * 1.4);

    float calm_volume = smoothstep(0.02, 0.90, volume);
    float thinking_pulse = thinking * (0.55 + 0.45 * sin(iTime * 2.2));
    float energy_flares = fbm(uv * 3.4 - vec2(0.0, iTime * 0.55)) * (0.055 + calm_volume * 0.28 + thinking * 0.08);
    float core_radius = 0.37 + (calm_volume * 0.055) + thinking_pulse * 0.018;
    float shell = abs(d - core_radius - energy_flares);
    float core_glow = 0.007 / (shell + 0.004);
    float white_hot_center = exp(-d * (6.2 - calm_volume * 1.0));
    float ambient_glow = exp(-d * 2.5) * (0.14 + calm_volume * 0.42 + thinking * 0.16);

    float scan_angle = atan(uv.y, uv.x) + iTime * 1.4;
    float thinking_arc = thinking * smoothstep(0.035, 0.0, abs(d - 0.59)) * (0.35 + 0.65 * smoothstep(0.55, 0.95, sin(scan_angle * 3.0)));

    vec3 color_green = vec3(0.0, 1.0, 0.45);
    vec3 color_yellow = vec3(1.0, 0.9, 0.0);
    vec3 color_red = vec3(1.0, 0.1, 0.0);
    vec3 energy_color;

    if (calm_volume < 0.4) {
        energy_color = mix(color_green, color_yellow, calm_volume / 0.4);
    } else {
        energy_color = mix(color_yellow, color_red, clamp((calm_volume - 0.4) / 0.6, 0.0, 1.0));
    }

    vec3 rgb = background;
    rgb += energy_color * (core_glow * 1.25 + ambient_glow);
    rgb += vec3(0.8, 0.95, 1.0) * white_hot_center * 1.8;
    rgb += vec3(0.3, 0.9, 1.0) * thinking_arc;
    rgb = pow(rgb, vec3(0.75));
    fragColor = vec4(rgb, 1.0);
}
"""
TEXT_VERTEX_SHADER = """
#version 120
attribute vec2 in_vert;
attribute vec2 in_tex;
varying vec2 v_tex;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_tex = in_tex;
}
"""
TEXT_FRAGMENT_SHADER = """
#version 120
uniform sampler2D text_texture;
varying vec2 v_tex;
// Legacy GLSL gebruikt gl_FragColor; alias houdt de originele shader-body ongewijzigd.
#define fragColor gl_FragColor
void main() {
    // GLSL 1.20 kent de generieke texture() nog niet; texture2D is het equivalent.
    fragColor = texture2D(text_texture, v_tex);
}
"""

# Shaders voor effen gekleurde rechthoeken (knoppen in het GUI).
RECT_VERTEX_SHADER = """
#version 120
attribute vec2 in_vert;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""
RECT_FRAGMENT_SHADER = """
#version 120
uniform vec4 rect_color;
#define fragColor gl_FragColor
void main() {
    fragColor = rect_color;
}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 🌐 WEBSOCKET SERVER - COMMUNICATIE MET LAPTOP/MOBILE
# ═══════════════════════════════════════════════════════════════════════════════

WS_LOGGER = logging.getLogger("websockets")
WS_LOGGER.setLevel(logging.INFO)


def _get_peer_ip(peer) -> tuple[str | None, str]:
    if peer is None:
        return None, "onbekend"
    if isinstance(peer, tuple):
        ip = str(peer[0])
        return ip, ip
    return str(peer), str(peer)


def is_tailscale_ip(ip: str) -> bool:
    """Check of IP in de Tailscale range zit (100.64.0.0/10)."""
    if not ip: return False
    parts = ip.split('.')
    if len(parts) != 4: return False
    try:
        a, b = int(parts[0]), int(parts[1])
        return a == 100 and (64 <= b <= 127)
    except ValueError:
        return False


def detect_tailscale_ip() -> str | None:
    """Probeert het Tailscale IP van deze Pi te vinden."""
    try:
        # Gebruik de 'tailscale ip' command line tool
        output = subprocess.check_output(["tailscale", "ip", "-4"], text=True, timeout=2).strip()
        if output: return output
    except:
        pass

    # Fallback: scan interfaces
    try:
        import socket
        import fcntl
        import struct

        def get_ip_address(ifname):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return socket.inet_ntoa(fcntl.ioctl(
                s.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack('256s', ifname[:15].encode('utf-8'))
            )[20:24])

        for iface in ["tailscale0", "ts0"]:
            try:
                ip = get_ip_address(iface)
                if ip: return ip
            except: continue
    except:
        pass
    return None


def check_tailscale_status() -> dict:
    """Controleert of Tailscale correct draait en verbonden is."""
    status = {
        "installed": False,
        "running": False,
        "connected": False,
        "ip": None,
        "hostname": None,
        "errors": []
    }

    # Check if tailscale is installed
    if shutil.which("tailscale"):
        status["installed"] = True
    else:
        status["errors"].append("Tailscale niet gevonden in PATH")
        return status

    try:
        # Check if tailscaled daemon is running
        output = subprocess.check_output(["tailscale", "status", "--json"], text=True, timeout=3)
        data = json.loads(output)
        status["running"] = True
        status["connected"] = (data.get("BackendState") == "Running")
        status["ip"] = data.get("Self", {}).get("TailscaleIPs", [None])[0]
        status["hostname"] = data.get("Self", {}).get("HostName")
    except Exception as e:
        status["errors"].append(f"Fout bij status check: {e}")

    return status

def save_chat_history_entry(chat_message: dict):
    """Sla een bericht op in de server-side geschiedenis."""
    try:
        chat_history_file = PROJECT_DIR / "server_chat_history.json"
        chat_history = []
        if chat_history_file.exists():
            try:
                with open(chat_history_file, 'r', encoding='utf-8') as f:
                    chat_history = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        chat_history.append(chat_message)

        # Houd alleen de laatste 100 berichten
        if len(chat_history) > 100:
            chat_history = chat_history[-100:]

        with open(chat_history_file, 'w', encoding='utf-8') as f:
            json.dump(chat_history, f, indent=2)
    except Exception as e:
        print(f"⚠️ Kon chatgeschiedenis niet opslaan: {e}")


def get_server_sync_payload(status="accepted", client_id=None, peer_label=None):
    """Genereer de volledige payload voor client synchronisatie."""
    chat_history = []
    try:
        history_file = PROJECT_DIR / "server_chat_history.json"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
    except: pass

    voice_profiles = {}
    try:
        vp_file = PROJECT_DIR / "voice_profiles.json"
        if vp_file.exists():
            with open(vp_file, 'r', encoding='utf-8') as f:
                voice_profiles = json.load(f)
    except: pass

    speaker_perms = {}
    try:
        sp_file = PROJECT_DIR / "speaker_permissions.json"
        if sp_file.exists():
            with open(sp_file, 'r', encoding='utf-8') as f:
                speaker_perms = json.load(f)
    except: pass

    server_settings = {}
    try:
        settings_file = PROJECT_DIR / "settings.json"
        if settings_file.exists():
            with open(settings_file, 'r', encoding='utf-8') as f:
                server_settings = json.load(f)
    except: pass

    payload = {
        "status": status,
        "chat_history": chat_history[-50:], # Laatste 50 berichten
        "chat_history_count": len(chat_history),
        "voice_profiles": voice_profiles,
        "speaker_permissions": speaker_perms,
        "settings": server_settings
    }

    if client_id is not None: payload["client_id"] = client_id
    if peer_label is not None: payload["peer"] = peer_label

    return payload


SPEAKER_MODEL = None

def get_speaker_model():
    """Lazy loader voor het speaker model om RAM te sparen bij start."""
    global SPEAKER_MODEL
    if not HAS_SPEAKER_AI: return None
    if SPEAKER_MODEL is None:
        try:
            model_dir = PROJECT_DIR / "pretrained_models" / "spkrec-ecapa-voxceleb"
            SPEAKER_MODEL = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=model_dir
            )
            print("✅ Speaker Recognition model geladen.")
        except Exception as e:
            print(f"❌ Kon Speaker model niet laden: {e}")
    return SPEAKER_MODEL

def release_speaker_model():
    """Verwijdert het speaker model uit het geheugen."""
    global SPEAKER_MODEL
    if SPEAKER_MODEL is not None:
        del SPEAKER_MODEL
        SPEAKER_MODEL = None
        gc.collect()
        if HAS_SPEAKER_AI:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        print("🧹 Speaker Recognition model verwijderd uit geheugen.")

async def identify_speaker_logic(audio_data_base64: str) -> tuple[str, float]:
    """Voert de AI analyse uit (Lokaal of Remote op laptop)."""
    
    # Als er een remote host is ingesteld, stuur de audio daarheen (bespaart RAM/disk op Pi)
    if REMOTE_AI_HOST:
        try:
            url = f"http://{REMOTE_AI_HOST}:8766/identify"
            print(f"📡 Offloading stemherkenning naar {url}...")
            
            # We sturen de audio en de profielen mee zodat de laptop de vergelijking kan doen
            profiles_file = PROJECT_DIR / "voice_profiles.json"
            profiles_data = "{}"
            if profiles_file.exists():
                profiles_data = profiles_file.read_text(encoding='utf-8')
            
            resp = requests.post(url, json={
                "audio_data": audio_data_base64,
                "profiles": json.loads(profiles_data)
            }, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                name = data.get("name", "onbekend")
                conf = data.get("confidence", 0.0)
                print(f"✅ Remote resultaat: {name} ({int(conf*100)}%)")
                return name, conf
            else:
                print(f"⚠️ Remote server fout {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Kan geen verbinding maken met laptop op {REMOTE_AI_HOST}:8766. Staat laptop_app.py aan?")
        except Exception as e:
            print(f"⚠️ Remote AI herkenning faalde: {e}")
            # Fallback naar lokale herkenning indien beschikbaar
    
    model = get_speaker_model()
    
    # 1. Sla audio op naar tijdelijk bestand (.raw naar .wav)
    audio_bytes = base64.b64decode(audio_data_base64)
    temp_raw = PROJECT_DIR / f"temp_test_{uuid.uuid4().hex[:8]}.raw"
    temp_wav = temp_raw.with_suffix(".wav")
    
    try:
        temp_raw.write_bytes(audio_bytes)
        
        # Gebruik ffmpeg of sox om van raw naar wav te gaan indien nodig, 
        # maar SpeechBrain kan vaak direct wav lezen. 
        # Hier gaan we ervan uit dat de client 16bit mono 16kHz stuurt.
        import wave
        with wave.open(str(temp_wav), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)

        # 2. Laad bekende profielen
        profiles_file = PROJECT_DIR / "voice_profiles.json"
        if not profiles_file.exists():
            return "onbekend", 0.0
            
        profiles = json.loads(profiles_file.read_text())
        
        if not model:
            # Demo modus: als we geen AI hebben, doe een simpele check of er audio is
            if len(audio_bytes) > 1000:
                # Pak een willekeurige naam uit de lijst als 'gok'
                names = list(profiles.keys())
                if names:
                    # Varieer de score een beetje zodat het niet statisch lijkt (geeft 'levende' indruk)
                    import random
                    mock_conf = 0.40 + (random.random() * 0.10)
                    return names[0], mock_conf
            return "onbekend", 0.0

        # 3. Bereken embedding voor de nieuwe opname
        # (Dit is de 'echte' AI stap)
        test_embedding = model.encode_batch(model.load_audio(str(temp_wav)))
        
        best_name = "onbekend"
        best_score = 0.0
        
        # 4. Vergelijk met opgeslagen embeddings in profiles
        for name, data in profiles.items():
            stored_emb_list = data.get("embeddings") or data.get("embedding")
            if not stored_emb_list: continue
            
            # Convert list terug naar torch tensor
            stored_embedding = torch.tensor(stored_emb_list)
            
            # Bereken cosine similarity
            similarity = torch.nn.functional.cosine_similarity(test_embedding, stored_embedding)
            score = float(similarity.max())
            
            if score > best_score:
                best_score = score
                best_name = name
                
        # Drempelwaarde voor herkenning (meestal rond 0.25 - 0.35 voor ECAPA)
        if best_score < 0.25:
            return "onbekend", best_score
            
        return best_name, best_score
        
    except Exception as e:
        print(f"Error in identification: {e}")
        return "fout", 0.0
    finally:
        # Ruim tijdelijke bestanden op
        if temp_raw.exists(): temp_raw.unlink()
        if temp_wav.exists(): temp_wav.unlink()
        
        # User requested aggressive cleanup
        if AGGRESSIVE_RAM_MODE:
            release_speaker_model()

async def ws_handler(websocket, shared_state: SharedState, default_model: str):
    # Per-verbinding goedkeuring via het GUI (J/N) of auto-accept bij allowlist.
    try:
        peer = websocket.remote_address
    except Exception:
        peer = None
    peer_ip, peer_label = _get_peer_ip(peer)

    try:
        # Auto-accept Tailscale clients (100.x.x.x)
        if peer_ip and is_tailscale_ip(peer_ip):
            conn = shared_state.gate.request(peer_label, ip=peer_ip)
            print(f"Auto-accept (tailscale): {peer_label} (id={conn.client_id})")
            shared_state.update_visual(text=f"Verbonden (tailscale) {peer_label}", audio_level=0.0)
            try:
                payload = get_server_sync_payload("accepted", conn.client_id, peer_label)
                await websocket.send(json.dumps(payload))
            finally:
                shared_state.gate.remove(conn.client_id)
        # Auto-accept allowlist
        elif peer_ip and shared_state.gate.is_allowed(peer_ip):
            conn = shared_state.gate.request(peer_label, ip=peer_ip)
            print(f"Auto-accept (allowlist): {peer_label} (id={conn.client_id})")
            shared_state.update_visual(text=f"Verbonden (auto) {peer_label}", audio_level=0.0)
            try:
                payload = get_server_sync_payload("accepted", conn.client_id, peer_label)
                await websocket.send(json.dumps(payload))
            finally:
                shared_state.gate.remove(conn.client_id)
        else:
            # Vraag goedkeuring aan in het GUI.
            conn = shared_state.gate.request(peer_label, ip=peer_ip)
            print(f"Inkomende verbinding van {peer_label} (id={conn.client_id}) - wacht op goedkeuring in GUI")
            shared_state.update_visual(text=f"Verzoek van {peer_label} - Druk op JA, NEE of ALTIJD", audio_level=0.0)
            try:
                await websocket.send(json.dumps({
                    "status": "pending",
                    "client_id": conn.client_id,
                    "peer": peer_label,
                }))
                try:
                    await asyncio.wait_for(conn.accept_event.wait(), timeout=300)
                except asyncio.TimeoutError:
                    conn.decision = "rejected"

                if conn.decision == "accepted":
                    payload = get_server_sync_payload("accepted", conn.client_id, peer_label)
                    await websocket.send(json.dumps(payload))
                    print(f"Verbinding goedgekeurd: {peer_label} (id={conn.client_id})")
                    shared_state.update_visual(text=f"Verbonden met {peer_label}", audio_level=0.0)
                else:
                    await websocket.send(json.dumps({"status": "rejected", "client_id": conn.client_id}))
                    print(f"Verbinding geweigerd/verlopen: {peer_label} (id={conn.client_id})")
                    shared_state.update_visual(text="Verbinding geweigerd", audio_level=0.0)
                    return
            finally:
                shared_state.gate.remove(conn.client_id)

        async for message in websocket:
            data = json.loads(message)
            message_type = data.get("type", "request")

            if message_type == "visual":
                shared_state.update_visual(
                    emotion=data.get("emotion"),
                    text=data.get("text"),
                    audio_level=float(data.get("audio_level", 0.0) or 0.0),
                )
                continue

            if message_type in {"analyze_screen", "analyze_image"} or data.get("image_base64"):
                image_base64 = data.get("image_base64", "")
                # Fix: Check zowel 'question' als 'text' (sommige clients sturen 'text')
                question = data.get("question") or data.get("text", "")
                section = data.get("section", "full")
                model = data.get("model", DEFAULT_VISION_MODEL)
                think = data.get("think", False)

                if not model or model == default_model:
                    model = DEFAULT_VISION_MODEL

                # Als 'think' uit staat, forceer een snellere verwerking indien mogelijk
                # (Bij vision modellen is dit vaak beperkter dan bij tekstmodellen)

                shared_state.update_visual(text="Afbeelding analyseren...", audio_level=0.0, active_model=model)
                try:
                    # Sla vraag op in geschiedenis
                    save_chat_history_entry({
                        "message": question or "Afbeelding geüpload",
                        "is_user": True,
                        "timestamp": datetime.now().isoformat()
                    })

                    answer, emotion = await analyze_screen_capture(
                        image_base64,
                        question=question,
                        section=section,
                        model=model,
                    )

                    # Sla antwoord op in geschiedenis
                    save_chat_history_entry({
                        "message": answer,
                        "is_user": False,
                        "timestamp": datetime.now().isoformat()
                    })
                finally:
                    shared_state.update_visual(active_model=None)
                shared_state.set(emotion, answer, 0.0)
                await websocket.send(json.dumps({
                    "reply": answer,
                    "emotion": emotion,
                    "source": "screen",
                    "section": section,
                }))
                continue

            if message_type == "ssh_command":
                # Client vraagt toestemming voor commando-uitvoering op de CLIENT-zijde
                cmd = data.get("command", "")
                print(f"SSH commando verzoek ontvangen: {cmd}")
                # Voor nu sturen we altijd toestemming terug naar de laptop-client
                await websocket.send(json.dumps({
                    "type": "ssh_execute",
                    "command": cmd
                }))
                continue

            if message_type == "ssh_response":
                # Laptop heeft commando uitgevoerd en stuurt output terug naar server
                output = data.get("output", "")
                cmd = data.get("command", "")
                print(f"SSH output voor '{cmd}':\n{output}")
                shared_state.update_visual(text=f"SSH gereed: {cmd}", audio_level=0.0)
                continue

            # Voice profile sync endpoints
            if message_type == "upload_voice_profile":
                voice_name = data.get("voice_name")
                voice_data = data.get("voice_data")
                try:
                    # Sla voice profile op in server storage
                    voice_profiles_file = PROJECT_DIR / "voice_profiles.json"
                    voice_profiles = {}
                    if voice_profiles_file.exists():
                        with open(voice_profiles_file, 'r', encoding='utf-8') as f:
                            voice_profiles = json.load(f)

                    # Update of voeg voice profile toe
                    voice_profiles[voice_name] = voice_data

                    # Save naar file
                    with open(voice_profiles_file, 'w', encoding='utf-8') as f:
                        json.dump(voice_profiles, f, indent=2)

                    await websocket.send(json.dumps({"type": "voice_profile_upload_success", "voice_name": voice_name}))
                    continue
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": f"Voice profile upload failed: {exc}"}))
                    continue

            if message_type == "download_voice_profile":
                voice_name = data.get("voice_name")
                try:
                    voice_profiles_file = PROJECT_DIR / "voice_profiles.json"
                    if voice_profiles_file.exists():
                        with open(voice_profiles_file, 'r', encoding='utf-8') as f:
                            voice_profiles = json.load(f)

                        if voice_name in voice_profiles:
                            await websocket.send(json.dumps({
                                "type": "voice_profile_data",
                                "voice_name": voice_name,
                                "voice_data": voice_profiles[voice_name]
                            }))
                            continue
                        else:
                            await websocket.send(json.dumps({"type": "error", "message": f"Voice profile '{voice_name}' not found"}))
                            continue
                    else:
                        await websocket.send(json.dumps({"type": "error", "message": "No voice profiles found on server"}))
                        continue
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": f"Voice profile download failed: {exc}"}))
                    continue

            if message_type == "list_voice_profiles":
                try:
                    voice_profiles_file = PROJECT_DIR / "voice_profiles.json"
                    if voice_profiles_file.exists():
                        with open(voice_profiles_file, 'r', encoding='utf-8') as f:
                            voice_profiles = json.load(f)

                        await websocket.send(json.dumps({
                            "type": "voice_profiles_list",
                            "voice_profiles": voice_profiles
                        }))
                        continue
                    else:
                        await websocket.send(json.dumps({
                            "type": "voice_profiles_list",
                            "voice_profiles": {}
                        }))
                        continue
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": f"Failed to list voice profiles: {exc}"}))
                    continue

            # Settings sync endpoints
            if message_type == "upload_settings":
                settings_data = data.get("settings_data")
                try:
                    settings_file = PROJECT_DIR / "settings.json"
                    with open(settings_file, 'w', encoding='utf-8') as f:
                        json.dump(settings_data, f, indent=2)
                    await websocket.send(json.dumps({"type": "settings_upload_success"}))
                    continue
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": f"Settings upload failed: {exc}"}))
                    continue

            if message_type == "download_settings":
                try:
                    settings_file = PROJECT_DIR / "settings.json"
                    if settings_file.exists():
                        with open(settings_file, 'r', encoding='utf-8') as f:
                            settings_data = json.load(f)
                        # Voeg gedetecteerde Tailscale IP toe zodat clients (Android/iPhone)
                        # automatisch de VPN-host kunnen gebruiken wanneer beschikbaar.
                        try:
                            ts_ip = detect_tailscale_ip()
                            if ts_ip:
                                if isinstance(settings_data, dict):
                                    conn_data = settings_data.get("connection") if settings_data.get("connection") else {}
                                    conn_data["tailscale_host"] = ts_ip
                                    settings_data["connection"] = conn_data
                                print(f"📡 Added tailscale_host to settings: {ts_ip}")
                        except Exception as _:
                            pass

                        await websocket.send(json.dumps({
                            "type": "settings_data",
                            "settings_data": settings_data
                        }))
                        continue
                    else:
                        await websocket.send(json.dumps({"type": "error", "message": "No settings found on server"}))
                        continue
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": f"Settings download failed: {exc}"}))
                    continue

            # Roles sync endpoints
            if message_type == "upload_roles":
                roles_data = data.get("roles_data")
                try:
                    roles_file = PROJECT_DIR / "speaker_permissions.json"
                    with open(roles_file, 'w', encoding='utf-8') as f:
                        json.dump(roles_data, f, indent=2)
                    await websocket.send(json.dumps({"type": "roles_upload_success"}))
                    continue
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": f"Roles upload failed: {exc}"}))
                    continue

            if message_type == "download_roles":
                try:
                    roles_file = PROJECT_DIR / "speaker_permissions.json"
                    if roles_file.exists():
                        with open(roles_file, 'r', encoding='utf-8') as f:
                            roles_data = json.load(f)
                        await websocket.send(json.dumps({
                            "type": "roles_data",
                            "roles_data": roles_data
                        }))
                        continue
                    else:
                        await websocket.send(json.dumps({"type": "error", "message": "No roles found on server"}))
                        continue
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": f"Roles download failed: {exc}"}))
                    continue

            if message_type == "identify_speaker":
                audio_base64 = data.get("audio_data")
                shared_state.update_visual(text="Stem analyseren...", audio_level=0.5)
                try:
                    name, confidence = await identify_speaker_logic(audio_base64)
                    
                    # Geef resultaat terug
                    await websocket.send(json.dumps({
                        "type": "speaker_identified",
                        "name": name,
                        "confidence": confidence
                    }))
                    
                    if not HAS_SPEAKER_AI and name != "onbekend":
                        status_msg = f"Herkend (Gok): {name} ({int(confidence*100)}%)"
                    else:
                        status_msg = f"Herkend: {name} ({int(confidence*100)}%)" if name != "onbekend" else "Stem niet herkend"
                    shared_state.update_visual(text=status_msg, audio_level=0.0)
                    continue
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": f"Identification failed: {exc}"}))
                    continue

            # Chat history sync endpoints
            if message_type == "upload_chat_message":
                chat_message = data.get("chat_message")
                try:
                    save_chat_history_entry(chat_message)
                    await websocket.send(json.dumps({"type": "chat_message_upload_success"}))
                    continue
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": f"Chat message upload failed: {exc}"}))
                    continue

            if message_type == "download_chat_history":
                try:
                    chat_history_file = PROJECT_DIR / "server_chat_history.json"
                    if chat_history_file.exists():
                        with open(chat_history_file, 'r', encoding='utf-8') as f:
                            chat_history = json.load(f)
                        await websocket.send(json.dumps({
                            "type": "chat_history_data",
                            "chat_history": chat_history
                        }))
                        continue
                    else:
                        await websocket.send(json.dumps({
                            "type": "chat_history_data",
                            "chat_history": []
                        }))
                        continue
                except Exception as exc:
                    await websocket.send(json.dumps({"type": "error", "message": f"Chat history download failed: {exc}"}))
                    continue

            user_text = data.get("text", "").strip()
            if not user_text:
                await websocket.send(json.dumps({"error": "Lege tekst ontvangen"}))
                continue

            # Gebruik het door de client gevraagde model.
            model = data.get("model", default_model)
            think = data.get("think", False)

            # Als 'think' uit staat (snelle modus), gebruik qwen3:8b (de algemene default)
            # in plaats van het geforceerde coder model. Dit zorgt dat qwen3:8b
            # ook werkt zonder de vertraging van eventuele 'thinking' parameters.
            if not think and not data.get("model"):
                model = DEFAULT_GENERAL_MODEL

            # Sla gebruikersbericht op in geschiedenis
            save_chat_history_entry({
                "message": user_text,
                "is_user": True,
                "timestamp": datetime.now().isoformat()
            })

            try:
                shared_state.update_visual(text="AI denkt na...", audio_level=0.0, active_model=model)
                answer = await call_ollama_api_async(model, user_text)
            except Exception as exc:
                answer = f"Ollama fout: {exc}"
            finally:
                shared_state.update_visual(active_model=None)

            # Sla AI antwoord op in geschiedenis
            save_chat_history_entry({
                "message": answer,
                "is_user": False,
                "timestamp": datetime.now().isoformat()
            })

            emotion = detect_emotion(answer)
            shared_state.set(emotion, answer, 0.0)
            await websocket.send(json.dumps({"reply": answer, "emotion": emotion}))

    except websockets.exceptions.ConnectionClosed:
        print(f"ℹ️ Verbinding met {peer_label} ({peer_ip}) verbroken.")
    except Exception as e:
        print(f"⚠️ Onverwachte fout in ws_handler voor {peer_label}: {e}")


async def start_server(host: str, port: int, shared_state: SharedState, default_model: str):
    shared_state.set_server_loop(asyncio.get_running_loop())
    ts_ip = detect_tailscale_ip()
    ts_info = f" | Tailscale: {ts_ip}" if ts_ip else ""
    shared_state.update_visual(text=f"Websocket start op {host}:{port}{ts_info}", audio_level=0.0)
    print(f"Websocket server starten op {host}:{port}{ts_info}")

    async with websockets.serve(
            lambda ws: ws_handler(ws, shared_state, default_model),
            host,
            port,
            logger=WS_LOGGER,
            ping_interval=20,
            ping_timeout=120,
            max_size=20 * 1024 * 1024,  # 20MB — standaard is 1MB, te klein voor foto's/afbeeldingen
    ):
        shared_state.update_visual(text=f"Websocket klaar op {host}:{port}{ts_info}", audio_level=0.0)
        print(f"Websocket server klaar op {host}:{port}{ts_info}")
        # Luister op alle interfaces — dit is essentieel voor Tailscale (100.x.x.x)
        print(f"Server luistert op host={host} (0.0.0.0 = alle interfaces inclusief Tailscale)")
        def _print_tailscale_status():
            print("=" * 60)
            print("TAILSCALE STATUS CHECK")
            print("=" * 60)
            ts_status = check_tailscale_status()
            print(f"  Geïnstalleerd: {ts_status['installed']}")
            print(f"  tailscaled actief: {ts_status['running']}")
            print(f"  Tailscale IP: {ts_status['ip'] or '(niet gevonden)'}")
            print(f"  Hostname: {ts_status['hostname'] or '(onbekend)'}")
            print(f"  Verbonden met netwerk: {ts_status['connected']}")
            if ts_status["errors"]:
                print("  ⚠️  Problemen:")
                for err in ts_status["errors"]:
                    print(f"     - {err}")
            else:
                print("  ✅ Alles lijkt OK")
            print("=" * 60)
            if not ts_status["ip"]:
                print("💡 Tailscale installeren/opstarten:")
                print("   sudo apt update && curl -fsSL https://tailscale.com/install.sh | sh")
                print("   sudo systemctl enable --now tailscaled")
                print("   sudo tailscale up --accept-routes")
                print("=" * 60)

        threading.Thread(target=_print_tailscale_status, daemon=True).start()
        await asyncio.Future()


def build_wave_points(width: int, height: int, emotion: str, level: float, audio_level: float, phase: float):
    baseline = height // 2
    points = []
    samples = 720
    profile = EMOTION_PROFILES.get(emotion, EMOTION_PROFILES["neutral"])
    speaking_amount = max(0.0, min(1.0, audio_level))
    amplitude = 8 + (speaking_amount * 150)
    amplitude *= profile["shape"]
    baseline += profile["offset"]
    cycles = 5.0
    harmonic = 0.12 + (level * 0.08)

    for i in range(samples + 1):
        x = (width * i) / samples
        ratio = i / samples
        angle = (ratio * cycles * math.tau * profile["speed"]) + phase
        if speaking_amount <= 0.01:
            y = baseline
        else:
            y = baseline + (math.sin(angle) + (math.sin(angle * 2.0) * harmonic)) * amplitude
        points.append((x, y))

    return points


def draw_wave_layer(screen, points, color_main, thickness):
    color_glow = tuple(min(255, c + 40) for c in color_main)
    shadow_points = [(x, y + 2) for x, y in points]

    pygame.draw.lines(screen, (30, 45, 70), False, shadow_points, 22)
    pygame.draw.lines(screen, color_glow, False, points, 14)
    pygame.draw.aalines(screen, color_glow, False, points)
    pygame.draw.lines(screen, color_main, False, points, thickness)
    pygame.draw.aalines(screen, color_main, False, points)

    for x, y in points[::2]:
        pygame.draw.circle(screen, color_glow, (int(x), int(y)), 7)
        pygame.draw.circle(screen, color_main, (int(x), int(y)), max(3, thickness // 2))


def read_cpu_temperature() -> float | None:
    try:
        raw_value = PI_TEMP_PATH.read_text(encoding="utf-8").strip()
        return int(raw_value) / 1000.0
    except (OSError, ValueError):
        return None


_LAST_GPU_NS = 0
_LAST_FRAME_TIME = 0

def read_gpu_usage() -> float | None:
    """Berekent de GPU-belasting direct tussen GUI-frames door zonder de GUI te vertragen.

    Geeft altijd een float terug (bijv. 0.0 tot 100.0) of None bij rechtenproblemen.
    """
    global _LAST_GPU_NS, _LAST_FRAME_TIME

    current_gpu_ns = 0

    # scan alle actieve DRM engines
    for fd_path in glob.glob("/proc/[0-9]*/fdinfo/[0-9]*"):
        try:
            with open(fd_path, "r") as f:
                content = f.read()
                if "drm-engine-" in content:
                    for line in content.splitlines():
                        if "drm-engine-" in line:
                            clean_num = "".join(filter(str.isdigit, line))
                            if clean_num:
                                current_gpu_ns += int(clean_num)
        except (PermissionError, FileNotFoundError):
            continue

    current_time = time.time_ns()

    # Eerste frame-initialisatie
    if _LAST_FRAME_TIME == 0:
        _LAST_GPU_NS = current_gpu_ns
        _LAST_FRAME_TIME = current_time
        return 0.01  # Klein getal om GUI te activeren

    # Bereken het verschil met de vorige frame
    gpu_diff = current_gpu_ns - _LAST_GPU_NS
    time_diff = current_time - _LAST_FRAME_TIME

    # Voorkom deling door nul of negatieve tijd (zelden, maar kan bij snel herstarten)
    if time_diff <= 0:
        return 0.0

    # Sla huidige waarden op voor de volgende frame
    _LAST_GPU_NS = current_gpu_ns
    _LAST_FRAME_TIME = current_time

    # Berekening: (verschil in ns) / (verschil in tijd) * 100
    percentage = (gpu_diff / time_diff) * 100.0 / 2.3

    if any("v3d" in fd_path for fd_path in glob.glob("/proc/[0-9]*/fdinfo/[0-9]*")):
        percentage *= 1.5

    return min(percentage, 100.0)

def read_memory_usage() -> float | None:
    try:
        total = None
        available = None
        free = None
        buffers = None
        cached = None
        sreclaimable = None
        shmem = None

        with MEMINFO_PATH.open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1])
                elif line.startswith("MemFree:"):
                    free = int(line.split()[1])
                elif line.startswith("Buffers:"):
                    buffers = int(line.split()[1])
                elif line.startswith("Cached:"):
                    cached = int(line.split()[1])
                elif line.startswith("SReclaimable:"):
                    sreclaimable = int(line.split()[1])
                elif line.startswith("Shmem:"):
                    shmem = int(line.split()[1])

                if total is not None and available is not None:
                    break

        if total is None:
            return None

        if available is None:
            if free is None:
                return None
            available = free
            if buffers is not None:
                available += buffers
            if cached is not None:
                available += cached
            if sreclaimable is not None:
                available += sreclaimable
            if shmem is not None:
                available -= shmem

        available = max(0, min(available, total))
        return ((total - available) / total) * 100.0

    except (OSError, ValueError):
        return None


_CPU_USAGE_PATH = Path("/proc/stat")
_prev_cpu_idle: list[float] = [0.0, 0.0]
_prev_cpu_total: list[float] = [0.0, 0.0]
_prev_cpu_idx: int = 0


def read_cpu_usage() -> float | None:
    """Lees CPU-gebruik in procenten via /proc/stat (tussen twee samples)."""
    global _prev_cpu_idle, _prev_cpu_total, _prev_cpu_idx
    try:
        lines = _CPU_USAGE_PATH.read_text(encoding="utf-8").splitlines()
        cpu_parts = lines[0].split()
        if len(cpu_parts) < 5:
            return None
        user, nice, system, idle = (
            int(cpu_parts[1]), int(cpu_parts[2]),
            int(cpu_parts[3]), int(cpu_parts[4]),
        )
        total = user + nice + system + idle
        diff_idle = idle - _prev_cpu_idle[_prev_cpu_idx]
        diff_total = total - _prev_cpu_total[_prev_cpu_idx]
        _prev_cpu_idle[_prev_cpu_idx] = float(idle)
        _prev_cpu_total[_prev_cpu_idx] = float(total)
        _prev_cpu_idx = 1 - _prev_cpu_idx  # wissel slot voor volgende meting
        if diff_total <= 0:
            return None
        return ((diff_total - diff_idle) / diff_total) * 100.0
    except (OSError, ValueError, IndexError):
        return None

def temperature_color(gpu_usage: float | None) -> tuple[int, int, int]:
    if gpu_usage is None:
        return (170, 178, 190)
    if gpu_usage >= 75:
        return (255, 110, 110)
    if gpu_usage >= 60:
        return (255, 205, 90)
    return (150, 235, 190)

def usage_color(percent: float | None) -> tuple[int, int, int]:
    if percent is None:
        return (170, 178, 190)
    if percent >= 85:
        return (255, 110, 110)
    if percent >= 70:
        return (255, 205, 90)
    return (150, 235, 190)

def draw_system_stats(screen, small_font, temp_c, gpu_usage: float | None, memory_percent: float | None):
    width, _ = screen.get_size()
    temp_label = "-- C" if temp_c is None else f"{temp_c:.1f} C"
    mem_label = "RAM --%" if memory_percent is None else f"RAM {memory_percent:.0f}%"
    gpu_label = "GPU --%" if gpu_usage is None or gpu_usage <= 0 else f"GPU {gpu_usage:.0f}%"

    temp_text = small_font.render(temp_label, True, temperature_color(temp_c))
    mem_text = small_font.render(mem_label, True, usage_color(memory_percent))
    gpu_text = small_font.render(gpu_label, True, usage_color(gpu_usage))

    temp_rect = temp_text.get_rect(topright=(width - 26, 18))
    mem_rect = mem_text.get_rect(topright=(width - 26, temp_rect.bottom + 8))
    gpu_rect = gpu_text.get_rect(topright=(width - 26, mem_rect.bottom + 8))
    screen.blit(temp_text, temp_rect)
    screen.blit(mem_text, mem_rect)
    screen.blit(gpu_text, gpu_rect)

def draw_wave_screen(
        screen,
        font,
        small_font,
        state: VisualState,
        phase: float,
        temp_c: float | None,
        gpu_usage: float | None,
        memory_percent: float | None,
):
    width, height = screen.get_size()
    screen.fill((10, 14, 24))

    previous_emotion = state.previous_emotion
    current_emotion = state.emotion
    previous_level = EMOTION_LEVELS.get(previous_emotion, 0.35)
    previous_color = EMOTION_COLORS.get(previous_emotion, EMOTION_COLORS["neutral"])
    current_color = EMOTION_COLORS.get(current_emotion, EMOTION_COLORS["neutral"])
    previous_thickness = EMOTION_PROFILES.get(previous_emotion, EMOTION_PROFILES["neutral"])["thickness"]
    current_thickness = EMOTION_PROFILES.get(current_emotion, EMOTION_PROFILES["neutral"])["thickness"]

    previous_points = build_wave_points(width, height, previous_emotion, previous_level, state.audio_level, phase)
    current_points = build_wave_points(width, height, current_emotion, state.level, state.audio_level, phase)

    if previous_emotion != current_emotion and state.transition < 1.0:
        previous_mix = max(0.0, 1.0 - state.transition)
        current_mix = max(0.0, state.transition)
        previous_draw = tuple(int(c * previous_mix) for c in previous_color)
        current_draw = tuple(int(c * current_mix) for c in current_color)
        draw_wave_layer(screen, previous_points, previous_draw, previous_thickness)
        draw_wave_layer(screen, current_points, current_draw, current_thickness)
    else:
        draw_wave_layer(screen, current_points, current_color, current_thickness)

    title = font.render(state.emotion.upper(), True, (235, 240, 250))
    title_rect = title.get_rect(center=(width // 2, 80))
    screen.blit(title, title_rect)

    text = state.text[:110] + ("..." if len(state.text) > 110 else "")
    msg = small_font.render(text, True, (220, 226, 236))
    msg_rect = msg.get_rect(center=(width // 2, height - 70))
    screen.blit(msg, msg_rect)
    draw_system_stats(screen, small_font, temp_c, gpu_usage, memory_percent)


def create_energy_renderer(ctx):
    import array
    vertices = array.array(
        "f",
        [
            -1.0, -1.0,
            1.0, -1.0,
            -1.0, 1.0,
            -1.0, 1.0,
            1.0, -1.0,
            1.0, 1.0,
        ],
    )
    program = ctx.program(vertex_shader=ENERGY_VERTEX_SHADER, fragment_shader=ENERGY_FRAGMENT_SHADER)
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(program, [(vbo, "2f", "in_vert")])
    return program, vbo, vao

def check_ollama_install() -> bool:
    """Controleer of Ollama CLI correct is geïnstalleerd en toegankelijk is."""
    try:
        result = subprocess.run(["ollama", "-v"], capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        print(f"✅ Ollama CLI gevonden: {version}")
        return True
    except FileNotFoundError:
        print("❌ Ollama CLI niet gevonden. Zorg ervoor dat het is geïnstalleerd en in PATH staat.")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Fout bij het uitvoeren van 'ollama -v': {e.stderr.strip()}")
        exit(1)

def check_ollama_model(model_name: str) -> bool:
    """Controleer of een specifiek Ollama-model beschikbaar is."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        models = result.stdout.strip().splitlines()
        if model_name in models:
            print(f"✅ Ollama-model '{model_name}' is beschikbaar.")
            return True
        else:
            print(f"❌ Ollama-model '{model_name}' is niet beschikbaar. Beschikbare modellen: {', '.join(models)}")
            print("wilt u het model downloaden? (ja/nee)")
            user_input = input().strip().lower()
            if user_input in {"ja", "j", "yes", "y"}:
                print(f"📥 Downloaden van model '{model_name}'...")
                try:
                    subprocess.run(["ollama", "pull", model_name], check=True)
                    print(f"✅ Model '{model_name}' succesvol gedownload.")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"❌ Fout bij het downloaden van model '{model_name}': {e.stderr.strip()}")
                    exit(1)
            else:
                print("❌ Model niet gedownload.")
                return False
    except FileNotFoundError:
        print("❌ Ollama CLI niet gevonden. Zorg ervoor dat het is geïnstalleerd en in PATH staat.")
        exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Fout bij het uitvoeren van 'ollama list': {e.stderr.strip()}")
        exit(1)

def create_text_program(ctx):
    return ctx.program(vertex_shader=TEXT_VERTEX_SHADER, fragment_shader=TEXT_FRAGMENT_SHADER)


def create_rect_program(ctx):
    return ctx.program(vertex_shader=RECT_VERTEX_SHADER, fragment_shader=RECT_FRAGMENT_SHADER)


def render_rect(ctx, rect_program, x: int, y: int, w: int, h: int, r: float, g: float, b: float, a: float, screen_w: int, screen_h: int):
    """Teken een effen gekleurde rechthoek als twee driehoeken (OpenGL quad)."""
    import array
    left = (x / screen_w) * 2.0 - 1.0
    right = ((x + w) / screen_w) * 2.0 - 1.0
    top = 1.0 - (y / screen_h) * 2.0
    bottom = 1.0 - ((y + h) / screen_h) * 2.0
    vertices = array.array(
        "f",
        [
            left, bottom,
            right, bottom,
            left, top,
            left, top,
            right, bottom,
            right, top,
        ],
    )
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(rect_program, [(vbo, "2f", "in_vert")])
    rect_program["rect_color"].value = (r, g, b, a)
    vao.render(moderngl.TRIANGLES)
    vao.release()
    vbo.release()


def render_text(ctx, text_program, font, text: str, color, x: int, y: int, screen_w: int, screen_h: int, anchor: str = "topleft"):
    import array
    surface = font.render(text, True, color)
    text_w, text_h = surface.get_size()
    if anchor == "topright":
        x -= text_w
    elif anchor == "center":
        x -= text_w // 2
        y -= text_h // 2
    elif anchor == "bottomcenter":
        x -= text_w // 2
        y -= text_h

    data = pygame.image.tostring(surface, "RGBA", True)
    texture = ctx.texture((text_w, text_h), 4, data)
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

    left = (x / screen_w) * 2.0 - 1.0
    right = ((x + text_w) / screen_w) * 2.0 - 1.0
    top = 1.0 - (y / screen_h) * 2.0
    bottom = 1.0 - ((y + text_h) / screen_h) * 2.0
    vertices = array.array(
        "f",
        [
            left, bottom, 0.0, 0.0,
            right, bottom, 1.0, 0.0,
            left, top, 0.0, 1.0,
            left, top, 0.0, 1.0,
            right, bottom, 1.0, 0.0,
            right, top, 1.0, 1.0,
        ],
    )
    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(text_program, [(vbo, "2f 2f", "in_vert", "in_tex")])
    texture.use(location=0)
    text_program["text_texture"].value = 0
    vao.render(moderngl.TRIANGLES)
    vao.release()
    vbo.release()
    texture.release()


def render_overlay(ctx, text_program, rect_program, font, small_font, state: VisualState, temp_c, gpu_usage: float | None, memory_percent: float | None, cpu_usage: float | None, width: int, height: int, gate: ConnectionGate, tailscale_ip: str = "") -> list[ButtonRect]:
    """Rendert alle HUD-elementen. Retourneert een lijst van ButtonRects die
    de huidige knoppen op het scherm beschrijven (voor touch/mouse hit-testing)."""
    buttons: list[ButtonRect] = []
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    render_text(ctx, text_program, font, state.emotion.upper(), (235, 240, 250), width // 2, 80, width, height, "center")

    message = state.text[:110] + ("..." if len(state.text) > 110 else "")
    render_text(ctx, text_program, small_font, message, (220, 226, 236), width // 2, height - 64, width, height, "bottomcenter")

    # --- Rode X-knop linksboven om af te sluiten ---
    quit_size = 44
    quit_pad = 18
    quit_x = quit_pad
    quit_y = quit_pad
    render_rect(ctx, rect_program, quit_x, quit_y, quit_size, quit_size, 0.80, 0.18, 0.18, 0.90, width, height)
    render_text(ctx, text_program, small_font, "\u00d7", (255, 255, 255), quit_x + quit_size // 2, quit_y + quit_size // 2, width, height, "center")
    buttons.append(ButtonRect(quit_x, quit_y, quit_size, quit_size, "quit"))

    # --- Actief Model (onder de X-knop) ---
    if state.active_model:
        model_text = f"Model: {state.active_model}"
        model_color = (255, 210, 80) # Goud-geel voor actief werkend model
    else:
        model_text = "Model Is Idle"
        model_color = (150, 160, 180) # Grijs voor idle status

    render_text(ctx, text_program, small_font, model_text, model_color, quit_x, quit_y + quit_size + 12, width, height, "topleft")

    # --- Systeem-info rechtsboven ---
    temp_label = "-- C" if temp_c is None else f"{temp_c:.1f} C"
    gpu_label = "GPU --%" if gpu_usage is None or gpu_usage <= 0 else f"GPU {gpu_usage:.0f}%"
    mem_label = "RAM --%" if memory_percent is None else f"RAM {memory_percent:.0f}%"
    cpu_label = "CPU --%" if cpu_usage is None else f"CPU {cpu_usage:.0f}%"
    render_text(ctx, text_program, small_font, temp_label, temperature_color(temp_c), width - 26, 18, width, height, "topright")
    render_text(ctx, text_program, small_font, gpu_label, usage_color(gpu_usage), width - 26, 50, width, height, "topright")
    render_text(ctx, text_program, small_font, cpu_label, usage_color(cpu_usage), width - 26, 82, width, height, "topright")
    render_text(ctx, text_program, small_font, mem_label, usage_color(memory_percent), width - 26, 114, width, height, "topright")

    # Goedkeurings-knoppen links onderin (per-verbinding gate).
    if gate is not None:
        pending = gate.snapshot()
        if pending:
            top = pending[0]
            extra = f"  (+{len(pending) - 1} wachtend)" if len(pending) > 1 else ""
            render_text(ctx, text_program, small_font, f"Verzoek: {top.label}{extra}", (255, 220, 120), 26, height - 108, width, height, "topleft")

            # Knop-afmetingen en posities
            btn_w, btn_h, btn_gap = 140, 52, 16
            btn_y = height - 72
            btn_x0 = 26  # startpositie links

            # JA knop (groen)
            render_rect(ctx, rect_program, btn_x0, btn_y, btn_w, btn_h, 0.15, 0.70, 0.25, 0.85, width, height)
            render_text(ctx, text_program, font, "JA", (255, 255, 255), btn_x0 + btn_w // 2, btn_y + btn_h // 2, width, height, "center")
            buttons.append(ButtonRect(btn_x0, btn_y, btn_w, btn_h, "accept"))

            # NEE knop (rood)
            nee_x = btn_x0 + btn_w + btn_gap
            render_rect(ctx, rect_program, nee_x, btn_y, btn_w, btn_h, 0.80, 0.20, 0.20, 0.85, width, height)
            render_text(ctx, text_program, font, "NEE", (255, 255, 255), nee_x + btn_w // 2, btn_y + btn_h // 2, width, height, "center")
            buttons.append(ButtonRect(nee_x, btn_y, btn_w, btn_h, "reject"))

            # ALTIJD knop (blauw)
            alt_x = nee_x + btn_w + btn_gap
            render_rect(ctx, rect_program, alt_x, btn_y, btn_w, btn_h, 0.20, 0.45, 0.85, 0.85, width, height)
            render_text(ctx, text_program, font, "ALTIJD", (255, 255, 255), alt_x + btn_w // 2, btn_y + btn_h // 2, width, height, "center")
            buttons.append(ButtonRect(alt_x, btn_y, btn_w, btn_h, "always"))

    ctx.disable(moderngl.BLEND)
    return buttons


def run_wave_ui(shared_state: SharedState, host: str, port: int, model: str):
    pygame.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN | pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Pi Energy Core Visual")
    width, height = screen.get_size()
    ctx = moderngl.create_context(require=210)
    energy_program, energy_vbo, energy_vao = create_energy_renderer(ctx)
    text_program = create_text_program(ctx)
    rect_program = create_rect_program(ctx)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 34)
    small_font = pygame.font.SysFont("Arial", 26)
    refresh_rate_getter = getattr(pygame.display, "get_current_refresh_rate", None)
    refresh_rate = refresh_rate_getter() if callable(refresh_rate_getter) else 0
    target_fps = max(30, int(refresh_rate) + 5) if refresh_rate else 65

    # Tailscale IP éénmalig ophalen bij opstart
    ts_ip = detect_tailscale_ip()
    ts_info = f" | TS: {ts_ip}" if ts_ip else ""
    shared_state.set("neutral", f"Server actief op {host}:{port} ({model}){ts_info}", 0.0)

    shader_time = 0.0
    current_volume = 0.0
    temp_c = read_cpu_temperature()
    gpu_usage = read_gpu_usage()
    memory_percent = read_memory_usage()
    cpu_usage = read_cpu_usage()
    next_temp_read = 0.0
    previous_ticks = pygame.time.get_ticks() / 1000.0
    running = [True]

    current_buttons: list[ButtonRect] = []

    def hit_test_button(px: int, py: int) -> str | None:
        """Check of pixel-coördinaat (px, py) binnen een knop valt. Retourneert action of None."""
        for btn in current_buttons:
            if btn.x <= px <= btn.x + btn.w and btn.y <= py <= btn.y + btn.h:
                return btn.action
        return None

    def handle_button_action(action: str):
        if action == "quit":
            running[0] = False
        elif action == "accept":
            shared_state.approve_top_connection(accept=True, always=False)
        elif action == "reject":
            shared_state.approve_top_connection(accept=False)
        elif action == "always":
            shared_state.approve_top_connection(accept=True, always=True)

    while running[0]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running[0] = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pressed = pygame.key.get_pressed()
                if any(pressed[key] for key in STOP_KEYS):
                    running[0] = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_j:
                shared_state.approve_top_connection(accept=True)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                shared_state.approve_top_connection(accept=False)
            elif event.type == pygame.FINGERDOWN:
                px = int(event.x * width)
                py = int(event.y * height)
                action = hit_test_button(px, py)
                if action: handle_button_action(action)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                px, py = event.pos
                action = hit_test_button(px, py)
                if action: handle_button_action(action)

        state = shared_state.get()
        shared_state.step_transition(0.06)
        now = pygame.time.get_ticks() / 1000.0
        dt = max(0.0, now - previous_ticks)
        previous_ticks = now

        thinking_amount = 1.0 if "AI denkt na" in state.text else 0.0
        target_volume = max(0.0, min(1.0, state.audio_level))
        target_volume = target_volume * target_volume * 0.85
        smoothing = 0.18 if target_volume > current_volume else 0.10
        current_volume += (target_volume - current_volume) * smoothing
        shader_time += dt * (0.12 + thinking_amount * 0.45 + current_volume * 0.75)

        if now >= next_temp_read:
            temp_c = read_cpu_temperature()
            gpu_usage = read_gpu_usage()
            memory_percent = read_memory_usage()
            cpu_usage = read_cpu_usage()
            next_temp_read = now + 2.0
            
            # Periodieke RAM cleanup (elke 10s ongeveer)
            if int(now) % 10 == 0:
                gc.collect()

        ctx.clear(0.0, 0.0, 0.0, 1.0)
        energy_program["iResolution"].value = (width, height)
        energy_program["iTime"].value = shader_time
        energy_program["iVolume"].value = current_volume
        energy_program["iThinking"].value = thinking_amount
        energy_vao.render(moderngl.TRIANGLES)
        current_buttons = render_overlay(
            ctx, text_program, rect_program, font, small_font,
            state, temp_c, gpu_usage, memory_percent, cpu_usage, width, height,
            gate=shared_state.gate, tailscale_ip=ts_ip,
        )
        pygame.display.flip()
        clock.tick(target_fps)

    energy_vao.release()
    energy_vbo.release()
    energy_program.release()
    text_program.release()
    rect_program.release()
    pygame.quit()


def preload_models():
    """Laadt de belangrijkste modellen alvast in het geheugen."""
    if OLLAMA_HOST != "127.0.0.1" and OLLAMA_HOST != "localhost":
        print(f"ℹ️ Preloading overgeslagen (Ollama draait op remote host: {OLLAMA_HOST})")
        return

    def _worker():
        print("🚀 Modellen voorladen in GPU/RAM (idle stand)...")
        available = get_available_models()
        if not available:
            print("⚠️ Geen modellen gevonden in Ollama om voor te laden.")
            return

        for model in IDLE_MODELS:
            if model not in available:
                print(f"⏩ Overslaan: Model '{model}' niet gevonden op dit systeem.")
                continue
                
            try:
                # Gebruik de chat endpoint met een kort bericht om het model in het geheugen te trekken
                resp = requests.post("http://127.0.0.1:11434/api/chat", json={
                    "model": model, 
                    "messages": [{"role": "user", "content": "hi"}],
                    "keep_alive": "24h",
                    "stream": False
                }, timeout=60) # Verhoogd naar 60s voor trage Pi 5 laden
                
                if resp.status_code == 200:
                    print(f"✅ Model '{model}' staat nu in idle stand (GPU/RAM).")
                else:
                    print(f"⚠️ Kon model '{model}' niet voorladen (status {resp.status_code}).")
            except Exception as e:
                print(f"❌ Fout bij voorladen van '{model}': {e}")

    threading.Thread(target=_worker, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi bot server + wave visual")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_GENERAL_MODEL)
    args = parser.parse_args()

    shared_state = SharedState()

    def server_thread():
        try:
            asyncio.run(start_server(args.host, args.port, shared_state, args.model))
        except Exception as exc:
            message = f"Serverfout: {exc}"
            print(message)
            shared_state.update_visual(text=message, audio_level=0.0)

    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()

    try:
        check_ollama_install()
        check_ollama_model(args.model)
        preload_models()
        run_wave_ui(shared_state, args.host, args.port, args.model)
    finally:
        print("Sluit af...")
        unload_models()


if __name__ == "__main__":
    main()