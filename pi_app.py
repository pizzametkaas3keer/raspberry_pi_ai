import argparse
import array
import asyncio
import json
import logging
import math
import re
import shutil
import subprocess
import threading
import time
import uuid
import moderngl
import pygame
import requests
import websockets
from dataclasses import dataclass
from pathlib import Path
import glob
from jarvis_sandbox.pi_app import read_gpu_temperature


# Probeer optioneel `langdetect` te gebruiken voor brede taalherkenning.
try:
    from langdetect import detect, DetectorFactory

    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except Exception:
    LANGDETECT_AVAILABLE = False

# Mapping korte taalcode -> leesbare taalnaam (voor AI prompts)
LANG_NAME_MAP = {
    "nl": "Nederlands",
    "en": "English",
    "fr": "Français",
    "es": "Español",
    "de": "Deutsch",
    "pt": "Português",
    "it": "Italiano",
}

WS_LOGGER = logging.getLogger("pi_app.websockets")
WS_LOGGER.setLevel(logging.CRITICAL)
WS_LOGGER.propagate = False


OLLAMA_API = "http://127.0.0.1:11434/api/chat"
PROJECT_DIR = Path(__file__).resolve().parent
HOME_DIR = Path.home()
DEFAULT_WORK_DIR = HOME_DIR / "Desktop"
EMOTION_KEYWORDS = {
    "sad": ["sorry", "jammer", "helaas", "verdriet"],
    "happy": ["wow", "top", "geweldig", "yes", "blij"],
    "angry": ["boos", "fout", "nee", "stop"],
}
EMOTION_LEVELS = {
    "neutral": 0.28,
    "happy": 0.65,
    "sad": 0.12,
    "angry": 0.90,
}
EMOTION_COLORS = {
    "neutral": (140, 235, 255),
    "happy": (255, 210, 80),
    "sad": (80, 110, 210),
    "angry": (255, 110, 110),
}
EMOTION_PROFILES = {
    "neutral": {"speed": 1.0, "shape": 1.0, "offset": 0, "thickness": 6},
    "happy": {"speed": 1.35, "shape": 1.25, "offset": -6, "thickness": 7},
    "sad": {"speed": 0.65, "shape": 0.55, "offset": 24, "thickness": 5},
    "angry": {"speed": 1.8, "shape": 1.55, "offset": 0, "thickness": 8},
}
PI_TEMP_PATH = Path("/sys/class/thermal/thermal_zone0/temp")
MEMINFO_PATH = Path("/proc/meminfo")
ALLOWED_IPS_PATH = PROJECT_DIR / "allowed_ips.json"
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
    float core_radius = 0.20 + (calm_volume * 0.055) + thinking_pulse * 0.018;
    float shell = abs(d - core_radius - energy_flares);
    float core_glow = 0.005 / (shell + 0.002);
    float white_hot_center = exp(-d * (6.2 - calm_volume * 1.0));
    float ambient_glow = exp(-d * 2.5) * (0.14 + calm_volume * 0.42 + thinking * 0.16);

    float scan_angle = atan(uv.y, uv.x) + iTime * 1.4;
    float thinking_arc = thinking * smoothstep(0.035, 0.0, abs(d - 0.42)) * (0.35 + 0.65 * smoothstep(0.55, 0.95, sin(scan_angle * 3.0)));

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


@dataclass
class VisualState:
    emotion: str = "neutral"
    previous_emotion: str = "neutral"
    text: str = "Ik ben klaar."
    level: float = 0.35
    audio_level: float = 0.0
    transition: float = 0.0


@dataclass
class PendingConnection:
    """Een inkomende client-verbinding die wacht op goedkeuring via het GUI."""
    client_id: str
    label: str               # Korte beschrijving (peer-info) voor in het GUI
    ip: str = ""             # Alleen het IP-adres (zonder poort), voor de allowlist
    decision: str = "pending"  # "pending" | "accepted" | "rejected"
    # asyncio.Event wordt in de ws-handler aangemaakt (de gui-thread heeft zijn eigen loop,
    # dus het event leeft in de server-loop en wordt daar geset).
    accept_event: object = None


@dataclass
class ButtonRect:
    """Schermrechthoek van een klikbare knop (pixels, origine links-boven)."""
    x: int
    y: int
    w: int
    h: int
    action: str  # "accept", "reject", "always"


class ConnectionGate:
    """Beheert inkomende verbindingen die goedkeuring nodig hebben (draait in de asyncio server-loop).

    Onderhoudt daarnaast een allowlist van IP-adressen (zonder poort) die automatisch
    worden goedgekeurd, zodat een client die steeds via een andere poort komt na één
    keer 'ALTIJD' niet opnieuw om goedkeuring hoeft te vragen.
    """

    def __init__(self, allowlist_path: Path = ALLOWED_IPS_PATH):
        self._pending: dict[str, PendingConnection] = {}
        self._lock = threading.Lock()
        self._allowlist_path = allowlist_path
        self._allowed_ips: set[str] = self._load_allowed_ips()

    def _load_allowed_ips(self) -> set[str]:
        try:
            if self._allowlist_path.exists():
                data = json.loads(self._allowlist_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return {str(item) for item in data}
        except (OSError, ValueError):
            pass
        return set()

    def _save_allowed_ips(self):
        try:
            self._allowlist_path.write_text(
                json.dumps(sorted(self._allowed_ips), indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def is_allowed(self, ip: str) -> bool:
        with self._lock:
            return ip in self._allowed_ips

    def allow_ip(self, ip: str):
        """Voeg een IP (zonder poort) permanent toe aan de allowlist en sla op."""
        if not ip:
            return
        with self._lock:
            if ip not in self._allowed_ips:
                self._allowed_ips.add(ip)
                self._save_allowed_ips()

    def request(self, label: str, ip: str = "") -> PendingConnection:
        client_id = uuid.uuid4().hex[:8]
        conn = PendingConnection(
            client_id=client_id,
            label=label,
            ip=ip,
            accept_event=asyncio.Event(),
        )
        with self._lock:
            self._pending[client_id] = conn
        return conn

    def decide(self, accept: bool, always: bool = False) -> PendingConnection | None:
        """Bepaal de bovenste pending connection (FIFO) en signaleer de wachtende handler.

        Met always=True wordt het IP bovendien permanent vrijgegeven.
        """
        with self._lock:
            if not self._pending:
                return None
            client_id = next(iter(self._pending))
            conn = self._pending[client_id]
        # Beslissing + event worden in de server-loop geset via decide_async,
        # maar we zetten de waarde hier thread-safe zodat de render-thread leest.
        conn.decision = "accepted" if accept else "rejected"
        if always and accept and conn.ip:
            self.allow_ip(conn.ip)
        return conn

    def remove(self, client_id: str):
        with self._lock:
            self._pending.pop(client_id, None)

    def snapshot(self) -> list[PendingConnection]:
        with self._lock:
            return list(self._pending.values())


class SharedState:
    def __init__(self):
        self.data = VisualState()
        self.lock = threading.Lock()
        self.gate = ConnectionGate()
        self.server_loop: asyncio.AbstractEventLoop | None = None

    def set_server_loop(self, loop: asyncio.AbstractEventLoop):
        self.server_loop = loop

    def approve_top_connection(self, accept: bool, always: bool = False) -> bool:
        """Wordt vanuit de GUI-thread aangeroepen. Bepaalt de bovenste pending
        connectie en signaleert de wachtende ws-handler in de server-loop.

        Met always=True wordt het IP van deze connectie permanent vrijgegeven
        (alleen IP, geen poort).
        """
        conn = self.gate.decide(accept, always=always)
        if conn is None or self.server_loop is None:
            return False
        event = conn.accept_event
        if event is not None:
            self.server_loop.call_soon_threadsafe(event.set)
        return True

    def set(self, emotion: str, text: str, audio_level: float = 0.0):
        with self.lock:
            self.data.previous_emotion = self.data.emotion
            self.data.emotion = emotion
            self.data.text = text
            self.data.level = EMOTION_LEVELS.get(emotion, 0.35)
            self.data.audio_level = max(0.0, min(1.0, audio_level))
            self.data.transition = 0.0

    def update_visual(self, emotion: str | None = None, text: str | None = None, audio_level: float | None = None):
        with self.lock:
            if emotion and emotion != self.data.emotion:
                self.data.previous_emotion = self.data.emotion
                self.data.emotion = emotion
                self.data.level = EMOTION_LEVELS.get(emotion, self.data.level)
                self.data.transition = 0.0
            if text is not None:
                self.data.text = text
            if audio_level is not None:
                self.data.audio_level = max(0.0, min(1.0, audio_level))

    def get(self) -> VisualState:
        with self.lock:
            return VisualState(
                emotion=self.data.emotion,
                previous_emotion=self.data.previous_emotion,
                text=self.data.text,
                level=self.data.level,
                audio_level=self.data.audio_level,
                transition=self.data.transition,
            )

    def step_transition(self, amount: float):
        with self.lock:
            self.data.transition = min(1.0, self.data.transition + amount)


def detect_emotion(text: str) -> str:
    lower = text.lower()
    for emotion, words in EMOTION_KEYWORDS.items():
        if any(re.search(rf"\b{re.escape(word)}\b", lower) for word in words):
            return emotion
    return "neutral"


def exact_reply_from_prompt(prompt: str) -> str | None:
    lower = prompt.lower().strip()

    quoted = re.search(r"[\"']([^\"']{1,80})[\"']", prompt)
    if "alleen" in lower and quoted:
        return quoted.group(1).strip()

    match = re.search(
        r"\bzeg\s+(?:eens|is)?\s*([a-zA-Z0-9!?., -]{1,40}?)\s+alleen\b",
        prompt,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" .,!?")

    match = re.search(
        r"\balleen\s+([a-zA-Z0-9!?., -]{1,40}?)(?:\s+(?:verder|meer|niks|niets)\b|$)",
        prompt,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip(" .,!?")

    return None


def clean_tool_text(value: str) -> str:
    return value.strip().strip("\"'` ").replace("\x00", "")


def extract_quoted_text(text: str) -> str | None:
    match = re.search(r"[\"']([^\"']{1,500})[\"']", text)
    if match:
        return clean_tool_text(match.group(1))
    return None


def resolve_safe_path(raw_path: str | None, default_name: str = "") -> Path:
    DEFAULT_WORK_DIR.mkdir(parents=True, exist_ok=True)
    if not raw_path:
        target = DEFAULT_WORK_DIR / default_name if default_name else DEFAULT_WORK_DIR
    else:
        cleaned = clean_tool_text(raw_path)
        target = Path(cleaned).expanduser()
        if not target.is_absolute():
            target = DEFAULT_WORK_DIR / target

    resolved = target.resolve()
    allowed_roots = (HOME_DIR.resolve(), PROJECT_DIR.resolve())
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(f"Pad buiten veilige map geblokkeerd: {resolved}")
    return resolved


def extract_path_after_keywords(text: str, keywords: tuple[str, ...], default_name: str = "") -> Path:
    quoted = extract_quoted_text(text)
    if quoted:
        return resolve_safe_path(quoted, default_name)

    lower = text.lower()
    for keyword in keywords:
        index = lower.find(keyword)
        if index >= 0:
            candidate = text[index + len(keyword):].strip(" :.-")
            if candidate:
                return resolve_safe_path(candidate, default_name)
    return resolve_safe_path(None, default_name)


def extract_write_target(text: str) -> Path:
    lower = text.lower()
    for keyword in ("schrijf bestand", "maak bestand", "zet tekst in bestand"):
        index = lower.find(keyword)
        if index >= 0:
            candidate = text[index + len(keyword):].strip(" :.-")
            candidate = re.split(r"\bmet tekst\b|\bmet inhoud\b", candidate, maxsplit=1, flags=re.I)[0]
            candidate = candidate.strip(" :.-\"'")
            if candidate:
                return resolve_safe_path(candidate, "notitie.txt")
    return resolve_safe_path(None, "notitie.txt")


def format_command_output(command: list[str], timeout: int = 8) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    output = (result.stdout or result.stderr or "").strip()
    return output[:1200] or "Geen output."


def detect_language_simple(text: str) -> str:
    """Detecteer taalcode (ISO 639-1) van `text`.

    Als `langdetect` beschikbaar is gebruiken we dat; anders terugvallen
    op een eenvoudige heuristiek die 'nl' of 'en' teruggeeft.
    """
    if not text:
        return "en"
    if LANGDETECT_AVAILABLE:
        try:
            code = detect(text)
            # langdetect kan 'nl', 'en', 'fr', etc. teruggeven
            return code
        except Exception:
            pass

    # Fallback heuristiek (vooral voor NL/EN)
    lower = text.lower()
    if re.search(r"\bin het nederlands\b|\bnederlands\b|\bin nederlands\b", lower):
        return "nl"
    if re.search(r"\bin het engels\b|\benglish\b", lower):
        return "en"

    dutch_common = [" de ", " het ", " een ", "hoe", "wat", "waar", "waarom", "zoeken", "zoek"]
    english_common = [" the ", " how ", " what ", " why ", " search ", " find "]
    dutch_score = sum(1 for w in dutch_common if w in lower)
    eng_score = sum(1 for w in english_common if w in lower)
    return "nl" if dutch_score >= eng_score else "en"


def is_probably_shell_command(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if len(text) > 200:
        return False
    if "\n" in text:
        return True
    if re.search(r"[<>|;&$*~`\\]", text):
        return True

    lower = text.lower()
    if re.search(r"\b(how|what|hoe|wat|waar|waarom|welke|kun je|kunt u|geef|laat|beschrijf|vertel)\b", lower):
        return False

    first = text.split()[0]
    common_bin = {
        "ls", "cat", "grep", "echo", "pwd", "mkdir", "rm", "cp", "mv", "python", "pip",
        "uname", "whoami", "df", "du", "top", "htop", "ps", "sudo", "git", "find", "curl",
        "wget", "service", "systemctl", "bash", "sh",
    }
    if first in common_bin or first.startswith(("/", "./")) or first.endswith(".sh"):
        return True

    if re.search(r"\b(ls|cat|grep|echo|pwd|mkdir|rm|cp|mv|python|pip|uname|whoami|df|du|top|git)\b", text):
        return True

    return False


def extract_shell_command(text: str) -> str | None:
    """Haal een shell-command uit de gebruikersvraag als die expliciet gevraagd is."""
    if not text:
        return None
    cleaned = re.sub(
        r"\b(gebruik (de )?(command prompt|terminal)|run (in )?(de )?(command prompt|shell|terminal)|execute (in )?(de )?(shell|command prompt)|shell command|cmd prompt)\b",
        "",
        text,
        flags=re.I,
    ).strip()
    if not cleaned:
        return None

    quoted = re.search(r"[`\"]([^`\"]+)[`\"]", text)
    if quoted:
        return quoted.group(1).strip()

    if is_probably_shell_command(cleaned):
        return cleaned
    return None


def fix_shell_command(command: str, stderr: str, lang: str = "en") -> str | None:
    language_name = LANG_NAME_MAP.get(lang, lang)
    prompt = (
        f"De volgende shell command faalde met een foutmelding:\n{stderr}\n\n"
        f"Origineel commando: {command}\n"
        f"Verbeter dit commando zodat het de bedoeling uitvoert, geef alleen het gecorrigeerde commando terug,"
        f" zonder extra uitleg. Gebruik dezelfde taal ({language_name}) als de gebruiker waar nodig."
    )
    ai_response = call_ollama_api("phi3:mini", prompt)
    if not ai_response:
        return None
    corrected = ai_response.strip().splitlines()[0].strip()
    # Als de AI iets anders dan een plausibel commando teruggeeft, negeer het.
    if len(corrected) < 3 or any(keyword in corrected.lower() for keyword in ("error", "fout", "sorry", "can't", "kan niet")):
        return None
    return corrected


def execute_shell_command(command: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, shell=True)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if result.returncode != 0:
            return f"Command returned exit code {result.returncode}.\n{stderr or stdout}"
        return stdout or stderr or f"Command uitgevoerd met exitcode {result.returncode}, maar zonder output."
    except subprocess.TimeoutExpired as exc:
        return f"Command timed out na {timeout} seconden.\n{(exc.stdout or '').strip()}\n{(exc.stderr or '').strip()}"
    except Exception as exc:
        return f"Command uitvoeren mislukt: {exc}"


def call_ollama_api(model: str, prompt: str, timeout: int = 20) -> str:
    """Directe (laag-niveau) aanroep naar de Ollama REST API zonder lokale tool checks.
    Retourneert de tekstuele inhoud of een fallback-bericht bij fouten.
    """
    try:
        payload = {"model": model, "stream": False, "messages": [{"role": "user", "content": prompt}]}
        resp = requests.post(OLLAMA_API, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        return (msg.get("content", "") or "").strip() or ""
    except Exception:
        return ""


def search_online(query: str) -> str:
    # Detecteer taal van de zoekopdracht zodat we resultaten en samenvattingen
    # in de juiste taal teruggeven.
    lang = detect_language_simple(query)
    kl_value = f"{lang}-{lang}" if len(lang) == 2 else lang
    headers = {"Accept-Language": lang}
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
        # regio/taal hints voor DuckDuckGo
        "kl": kl_value,
        "lang": lang,
    }
    response = requests.get(
        "https://api.duckduckgo.com/",
        params=params,
        headers=headers,
        timeout=12,
    )
    print(response.url)
    response.raise_for_status()
    data = response.json()
    abstract = data.get("AbstractText") or data.get("Answer") or ""
    source = data.get("AbstractURL") or ""

    related = []
    for item in data.get("RelatedTopics", []):
        if isinstance(item, dict) and item.get("Text"):
            related.append(item["Text"])
        if len(related) >= 3:
            break

    parts = [part for part in [abstract, *related] if part]
    if not parts:
        return "Ik kon online niets duidelijks vinden."
    suffix = f"\nBron: {source}" if source else ""
    combined = "\n".join(parts)[:2000] + suffix

    # Vraag de AI om een korte samenvatting in de taal van de gebruiker
    language_name = LANG_NAME_MAP.get(lang, lang)
    ai_prompt = (
        f"Geef een korte samenvatting (maximaal twee korte zinnen) in {language_name} van de volgende zoekresultaten:\n{combined}"
    )
    ai_summary = call_ollama_api("phi3:mini", ai_prompt)
    if ai_summary:
        return ai_summary[:1200]
    return combined[:1200]


def set_system_volume(percent: int) -> str:
    percent = max(0, min(100, percent))
    mixers = [
        ["amixer", "sset", "Master", f"{percent}%"],
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
    ]
    for command in mixers:
        if shutil.which(command[0]):
            format_command_output(command)
            return f"Volume ingesteld op {percent}%."
    return "Ik kon geen ondersteunde volume-tool vinden op deze Pi."


def run_pi_tool(user_text: str) -> str | None:
    lower = user_text.lower()

    if any(word in lower for word in ("zoek online", "zoek op internet", "google", "online opzoeken", "internet")):
        query = re.sub(r"\b(zoek online|zoek op internet|google|online opzoeken|internet)\b", "", user_text, flags=re.I)
        query = clean_tool_text(query) or user_text
        results = search_online(query)
        # Gebruik AI om resultaten samen te vatten in de taal van de gebruiker
        lang = detect_language_simple(user_text)
        language_name = LANG_NAME_MAP.get(lang, lang)
        ai_prompt = (
            f"Je bent een hulpvaardige assistent. Geef een korte samenvatting (max twee zinnen) in {language_name} "
            f"van deze zoekresultaten voor de gebruiker:\n{results}"
        )
        ai_summary = call_ollama_api("phi3:mini", ai_prompt)
        return ai_summary or results

    if any(word in lower for word in ("temperatuur", "cpu temp", "cpu temperatuur")):
        temp = read_cpu_temperature()
        return "CPU temperatuur onbekend." if temp is None else f"CPU temperatuur is {temp:.1f} C."
    
    if any(word in lower for word in ("gpu temp", "gpu temperatuur")):
        temp = read_gpu_temperature()
        return "GPU temperatuur onbekend." if temp is None else f"GPU temperatuur is {temp:.1f} C."

    if any(word in lower for word in ("gpu usage", "gpu gebruik")):
        usage = read_gpu_usage()
        return "GPU gebruik onbekend." if usage is None else f"GPU gebruik is {usage:.1f}%."

    if any(word in lower for word in ("status", "systeem info", "pi info", "schijfruimte", "opslag")):
        uptime = format_command_output(["uptime"]) if shutil.which("uptime") else "uptime onbekend"
        disk = format_command_output(["df", "-h", str(HOME_DIR)]) if shutil.which("df") else "schijf onbekend"
        return f"Pi status:\n{uptime}\n\nOpslag:\n{disk}"

    if any(word in lower for word in ("ip adres", "ip-adres", "netwerk info")):
        if shutil.which("hostname"):
            return format_command_output(["hostname", "-I"])
        return "Ik kon het IP-adres niet ophalen."

    if "volume" in lower:
        match = re.search(r"(\d{1,3})\s*%?", lower)
        if match:
            return set_system_volume(int(match.group(1)))
        return "Zeg bijvoorbeeld: zet volume naar 50 procent."

    if any(word in lower for word in ("maak map", "maak een map", "nieuwe map", "map aan")):
        path = extract_path_after_keywords(user_text, ("maak een map", "maak map", "nieuwe map", "map aan"), "nieuwe_map")
        path.mkdir(parents=True, exist_ok=True)
        # Laat de AI een korte bevestiging of extra instructies geven
        lang = detect_language_simple(user_text)
        language_name = LANG_NAME_MAP.get(lang, lang)
        ai_prompt = (
            f"Bevestig kort in {language_name} dat de map is aangemaakt en geef één voorbeeldzin hoe de gebruiker ernaartoe kan navigeren: {path}"
        )
        ai_resp = call_ollama_api("phi3:mini", ai_prompt)
        return ai_resp or f"Map aangemaakt: {path}"

    if any(word in lower for word in ("lijst map", "toon map", "laat map zien", "wat staat er in")):
        path = extract_path_after_keywords(user_text, ("lijst map", "toon map", "laat map zien", "wat staat er in"), "")
        if not path.exists() or not path.is_dir():
            return f"Map bestaat niet: {path}"
        items = sorted(item.name + ("/" if item.is_dir() else "") for item in path.iterdir())
        listing = "Map is leeg." if not items else "\n".join(items[:80])
        lang = detect_language_simple(user_text)
        language_name = LANG_NAME_MAP.get(lang, lang)
        ai_prompt = f"Geef een korte, vriendelijke samenvatting in {language_name} van de inhoud van deze map:\n{listing}"
        ai_resp = call_ollama_api("phi3:mini", ai_prompt)
        return ai_resp or listing

    if any(word in lower for word in ("lees bestand", "toon bestand", "open bestand")):
        path = extract_path_after_keywords(user_text, ("lees bestand", "toon bestand", "open bestand"), "")
        if not path.exists() or not path.is_file():
            return f"Bestand bestaat niet: {path}"
        content = path.read_text(encoding="utf-8", errors="replace")[:2000]
        lang = detect_language_simple(user_text)
        language_name = LANG_NAME_MAP.get(lang, lang)
        ai_prompt = f"Vat kort samen in {language_name} wat er in het volgende bestand staat (max twee zinnen):\n{content}"
        ai_resp = call_ollama_api("phi3:mini", ai_prompt)
        return ai_resp or content

    if any(word in lower for word in ("schrijf bestand", "maak bestand", "zet tekst in bestand")):
        path = extract_write_target(user_text)
        content = extract_quoted_text(user_text) or "Aangemaakt door Jarvis."
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        lang = detect_language_simple(user_text)
        language_name = LANG_NAME_MAP.get(lang, lang)
        ai_prompt = f"Bevestig kort in {language_name} dat het bestand is opgeslagen: {path}"
        ai_resp = call_ollama_api("phi3:mini", ai_prompt)
        return ai_resp or f"Bestand opgeslagen: {path}"

    if any(word in lower for word in ("verwijder bestand", "delete bestand")):
        path = extract_path_after_keywords(user_text, ("verwijder bestand", "delete bestand"), "")
        if not path.exists() or not path.is_file():
            return f"Bestand bestaat niet: {path}"
        trash_dir = DEFAULT_WORK_DIR / ".trash"
        trash_dir.mkdir(parents=True, exist_ok=True)
        destination = trash_dir / f"{int(time.time())}_{path.name}"
        shutil.move(str(path), str(destination))
        lang = detect_language_simple(user_text)
        language_name = LANG_NAME_MAP.get(lang, lang)
        ai_prompt = f"Bevestig kort in {language_name} dat het bestand is verplaatst naar de prullenbak: {destination}"
        ai_resp = call_ollama_api("phi3:mini", ai_prompt)
        return ai_resp or f"Bestand verplaatst naar prullenbak: {destination}"

    command_text = extract_shell_command(user_text)
    if command_text:
        response = execute_shell_command(command_text)
        if "Command returned exit code" in response or "Command timed out" in response or "Command uitvoeren mislukt" in response:
            lang = detect_language_simple(user_text)
            fixed = fix_shell_command(command_text, response, lang=lang)
            if fixed and fixed != command_text:
                follow_up = execute_shell_command(fixed)
                return f"Fout gedetecteerd. Probeer gecorrigeerd commando:\n{fixed}\n\nUitkomst:\n{follow_up}"
        return response

    if any(word in lower for word in ("herstart ollama", "restart ollama")):
        if shutil.which("systemctl"):
            return format_command_output(["systemctl", "restart", "ollama"])
        return "systemctl is niet beschikbaar."

    return None

def ask_ollama(model: str, prompt: str, max_retries: int = 3) -> str:
    """
    Roept Ollama aan. Als het antwoord leeg is, probeert het opnieuw tot max_retries.
    Elke poging heeft een timeout van 60 seconden.
    """
    exact_reply = exact_reply_from_prompt(prompt)
    if exact_reply:
        return exact_reply

    tool_reply = run_pi_tool(prompt)
    if tool_reply:
        return tool_reply

    system_msg = (
        "Je bent een korte assistent die beknopt antwoord geeft. "
        "Antwoord in het Nederlands als de gebruiker Nederlands gebruikt, anders in het Engels. "
        "Geef maximaal twee korte zinnen. Reageer niet met code tenzij expliciet gevraagd."
    )
    ai_prompt = system_msg + "\n\nGebruikersvraag: " + prompt
    
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Probeer de API aan te roepen met een timeout van 60 seconden
            # Als de API langer dan 60s duurt, gooit hij een TimeoutError
            ai_resp = call_ollama_api(model, ai_prompt, timeout=60)
            
            # Controleer of het antwoord niet leeg is
            if ai_resp and ai_resp.strip():
                return ai_resp
            
            # Als we hier zijn, was het antwoord leeg of None
            print(f"Poging {retry_count + 1} gaf leeg antwoord. Proberen opnieuw...")
            
        except Exception as e:
            # Als er een error is (bijv. Timeout na 60s)
            print(f"Poging {retry_count + 1} mislukt: {e}. Proberen opnieuw...")
            ai_resp = None

        retry_count += 1
        
        # Wacht kort tussen pogingen (optioneel, bijv. 2 seconden)
        if retry_count < max_retries:
            time.sleep(2)

    # Als we alle pogingen hebben opgemaakt en niets gekregen:
    return "Ik heb even geen antwoord."


async def ask_ollama_async(model: str, prompt: str) -> str:
    return await asyncio.to_thread(ask_ollama, model, prompt)


def _get_peer_ip(peer) -> tuple[str, str]:
    """Retourneert (ip, label) uit een remote_address tuple. Label bevat ip:port."""
    if peer and isinstance(peer, (list, tuple)) and len(peer) >= 2:
        ip = str(peer[0])
        port = str(peer[1])
        return ip, f"{ip}:{port}"
    return "", "onbekend"

def is_tailscale_ip(ip_address: str) -> bool:
    """Controleer of een IP-adres in de Tailscale-range valt (100.x.x.x)."""
    try:
        parts = str(ip_address).split(".")
        return len(parts) == 4 and parts[0] == "100"
    except Exception:
        return False


def check_tailscale_status() -> dict:
    """Gedetailleerde Tailscale status check voor debugging.
    Retourneert een dict met status-info die in de logs wordt geprint.
    """
    status = {
        "installed": False,
        "running": False,
        "ip": "",
        "hostname": "",
        "connected": False,
        "errors": [],
    }

    # 1. Is tailscale geïnstalleerd?
    ts_bin = shutil.which("tailscale")
    tailscaled_bin = shutil.which("tailscaled")
    status["installed"] = bool(ts_bin or tailscaled_bin)

    if not status["installed"]:
        status["errors"].append("Tailscale niet gevonden in PATH — installeer via 'curl -fsSL https://tailscale.com/install.sh | sh'")
        return status

    # 2. Is tailscaled actief?
    if shutil.which("systemctl"):
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "tailscaled"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            status["running"] = result.stdout.strip() == "active"
            if not status["running"]:
                status["errors"].append(f"tailscaled service is niet actief (status: {result.stdout.strip()})")
                status["errors"].append("Start met: sudo systemctl start tailscaled && sudo systemctl enable tailscaled")
        except Exception as exc:
            status["errors"].append(f"Kon tailscaled status niet checken: {exc}")

    # 3. Tailscale IP ophalen
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            status["ip"] = result.stdout.strip()
        elif result.stderr:
            status["errors"].append(f"tailscale ip -4 error: {result.stderr.strip()}")
    except Exception as exc:
        status["errors"].append(f"tailscale ip -4 faalde: {exc}")

    # 4. Tailscale status (verbonden met netwerk?)
    try:
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            import json as _json
            ts_data = _json.loads(result.stdout)
            status["connected"] = ts_data.get("BackendState") == "Running"
            status["hostname"] = ts_data.get("Self", {}).get("HostName", "")
            if not status["connected"]:
                backend = ts_data.get("BackendState", "unknown")
                status["errors"].append(f"Tailscale backend state: {backend} — niet verbonden met Tailscale netwerk")
                status["errors"].append("Controleer: tailscale up --accept-routes")
    except Exception as exc:
        status["errors"].append(f"tailscale status faalde: {exc}")

    # 5. Controleer of poort 8765 bereikbaar is via Tailscale interface
    if status["ip"]:
        try:
            result = subprocess.run(
                ["ss", "-tlnp"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if ":8765" in result.stdout:
                pass  # poort luistert
            else:
                status["errors"].append("Poort 8765 lijkt niet te luisteren op een van de interfaces (geen match in 'ss -tlnp')")
        except Exception:
            pass

    return status


def detect_tailscale_ip() -> str:
    """Probeer het Tailscale IPv4-adres (100.x.x.x) van de Pi te vinden.

    Probeert achtereenvolgens:
      1. 'tailscale ip -4' commando
      2. Alle netwerkinterfaces scannen op 100.x.x.x adres
    """
    # Methode 1: tailscale CLI
    if shutil.which("tailscale"):
        try:
            result = subprocess.run(
                ["tailscale", "ip", "-4"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            ip = (result.stdout or "").strip()
            if ip and ip.startswith("100."):
                return ip
        except Exception:
            pass

    # Methode 2: scan netwerkinterfaces
    try:
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        for token in (result.stdout or "").split():
            token = token.strip()
            if token.startswith("100."):
                return token
    except Exception:
        pass

    return ""


async def ws_handler(websocket, shared_state: SharedState, default_model: str):
    # Per-verbinding goedkeuring via het GUI (J/N) of auto-accept bij allowlist.
    try:
        peer = websocket.remote_address
    except Exception:
        peer = None
    peer_ip, peer_label = _get_peer_ip(peer)

    # Auto-accept Tailscale clients (100.x.x.x) so mobile clients on Tailscale connect without GUI approval
    if peer_ip and is_tailscale_ip(peer_ip):
        conn = shared_state.gate.request(peer_label, ip=peer_ip)
        print(f"Auto-accept (tailscale): {peer_label} (id={conn.client_id})")
        shared_state.update_visual(text=f"Verbonden (tailscale) {peer_label}", audio_level=0.0)
        try:
            await websocket.send(json.dumps({
                "status": "accepted",
                "client_id": conn.client_id,
                "peer": peer_label,
            }))
        finally:
            shared_state.gate.remove(conn.client_id)
    elif peer_ip and shared_state.gate.is_allowed(peer_ip):
        conn = shared_state.gate.request(peer_label, ip=peer_ip)
        print(f"Auto-accept (allowlist): {peer_label} (id={conn.client_id})")
        shared_state.update_visual(text=f"Verbonden (auto) {peer_label}", audio_level=0.0)
        try:
            await websocket.send(json.dumps({
                "status": "accepted",
                "client_id": conn.client_id,
                "peer": peer_label,
            }))
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
                await websocket.send(json.dumps({"status": "accepted", "client_id": conn.client_id}))
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

        # Voice profile sync endpoints
        if message_type == "upload_voice_profile":
            voice_data = data.get("voice_data")
            voice_name = data.get("voice_name")
            try:
                # Sla voice profile op in server storage
                voice_profiles_file = PROJECT_DIR / "server_voice_profiles.json"
                voice_profiles = {}
                if voice_profiles_file.exists():
                    with open(voice_profiles_file, 'r') as f:
                        voice_profiles = json.load(f)

                # Update of voeg voice profile toe
                voice_profiles[voice_name] = voice_data

                # Save naar file
                with open(voice_profiles_file, 'w') as f:
                    json.dump(voice_profiles, f, indent=2)

                await websocket.send(json.dumps({"type": "voice_profile_upload_success", "voice_name": voice_name}))
                continue
            except Exception as exc:
                await websocket.send(json.dumps({"type": "error", "message": f"Voice profile upload failed: {exc}"}))
                continue

        if message_type == "download_voice_profile":
            voice_name = data.get("voice_name")
            try:
                voice_profiles_file = PROJECT_DIR / "server_voice_profiles.json"
                if voice_profiles_file.exists():
                    with open(voice_profiles_file, 'r') as f:
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
                voice_profiles_file = PROJECT_DIR / "server_voice_profiles.json"
                if voice_profiles_file.exists():
                    with open(voice_profiles_file, 'r') as f:
                        voice_profiles = json.load(f)

                    voice_names = list(voice_profiles.keys())
                    await websocket.send(json.dumps({
                        "type": "voice_profiles_list",
                        "voice_profiles": voice_names
                    }))
                    continue
                else:
                    await websocket.send(json.dumps({
                        "type": "voice_profiles_list",
                        "voice_profiles": []
                    }))
                    continue
            except Exception as exc:
                await websocket.send(json.dumps({"type": "error", "message": f"Failed to list voice profiles: {exc}"}))
                continue

        # Settings sync endpoints
        if message_type == "upload_settings":
            settings_data = data.get("settings_data")
            try:
                settings_file = PROJECT_DIR / "server_settings.json"
                with open(settings_file, 'w') as f:
                    json.dump(settings_data, f, indent=2)
                await websocket.send(json.dumps({"type": "settings_upload_success"}))
                continue
            except Exception as exc:
                await websocket.send(json.dumps({"type": "error", "message": f"Settings upload failed: {exc}"}))
                continue

        if message_type == "download_settings":
            try:
                settings_file = PROJECT_DIR / "server_settings.json"
                if settings_file.exists():
                    with open(settings_file, 'r') as f:
                        settings_data = json.load(f)
                    # Voeg gedetecteerde Tailscale IP toe zodat clients (Android/iPhone)
                    # automatisch de VPN-host kunnen gebruiken wanneer beschikbaar.
                    try:
                        ts_ip = detect_tailscale_ip()
                        if ts_ip:
                            if isinstance(settings_data, dict):
                                conn = settings_data.get("connection") if settings_data.get("connection") else {}
                                conn["tailscale_host"] = ts_ip
                                settings_data["connection"] = conn
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
                roles_file = PROJECT_DIR / "server_roles.json"
                with open(roles_file, 'w') as f:
                    json.dump(roles_data, f, indent=2)
                await websocket.send(json.dumps({"type": "roles_upload_success"}))
                continue
            except Exception as exc:
                await websocket.send(json.dumps({"type": "error", "message": f"Roles upload failed: {exc}"}))
                continue

        if message_type == "download_roles":
            try:
                roles_file = PROJECT_DIR / "server_roles.json"
                if roles_file.exists():
                    with open(roles_file, 'r') as f:
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

        # Chat history sync endpoints
        if message_type == "upload_chat_message":
            chat_message = data.get("chat_message")
            try:
                chat_history_file = PROJECT_DIR / "server_chat_history.json"
                chat_history = []
                if chat_history_file.exists():
                    with open(chat_history_file, 'r') as f:
                        chat_history = json.load(f)

                chat_history.append(chat_message)

                # Keep only last 100 messages
                if len(chat_history) > 100:
                    chat_history = chat_history[-100:]

                with open(chat_history_file, 'w') as f:
                    json.dump(chat_history, f, indent=2)
                await websocket.send(json.dumps({"type": "chat_message_upload_success"}))
                continue
            except Exception as exc:
                await websocket.send(json.dumps({"type": "error", "message": f"Chat message upload failed: {exc}"}))
                continue

        if message_type == "download_chat_history":
            try:
                chat_history_file = PROJECT_DIR / "server_chat_history.json"
                if chat_history_file.exists():
                    with open(chat_history_file, 'r') as f:
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

        model = data.get("model", default_model)
        try:
            shared_state.update_visual(text="AI denkt na...", audio_level=0.0)
            answer = await ask_ollama_async(model, user_text)
        except Exception as exc:
            answer = f"Ollama fout: {exc}"

        emotion = detect_emotion(answer)
        shared_state.set(emotion, answer, 0.0)
        await websocket.send(json.dumps({"reply": answer, "emotion": emotion}))


async def start_server(host: str, port: int, shared_state: SharedState, default_model: str):
    shared_state.set_server_loop(asyncio.get_running_loop())
    ts_ip = detect_tailscale_ip()
    ts_info = f" | Tailscale: {ts_ip}" if ts_ip else ""
    shared_state.update_visual(text=f"Websocket start op {host}:{port}{ts_info}", audio_level=0.0)
    print(f"Websocket server starten op {host}:{port}{ts_info}")

    # Gedetailleerde Tailscale status bij opstarten (voor debugging)
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

    async with websockets.serve(
        lambda ws: ws_handler(ws, shared_state, default_model),
        host,
        port,
        logger=WS_LOGGER,
        ping_interval=20,
        ping_timeout=120,
    ):
        shared_state.update_visual(text=f"Websocket klaar op {host}:{port}{ts_info}", audio_level=0.0)
        print(f"Websocket server klaar op {host}:{port}{ts_info}")
        # Luister op alle interfaces — dit is essentieel voor Tailscale (100.x.x.x)
        print(f"Server luistert op host={host} (0.0.0.0 = alle interfaces inclusief Tailscale)")
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

    # Scan alle actieve DRM engines
    for fd_path in glob.glob("/proc/[0-9]*/fdinfo/[0-9]*"):
        try:
            with open(fd_path, "r") as f:
                content = f.read()
                if "drm-engine-" in content:
                    for line in content.splitlines():
                        if "drm-engine-" in line:
                            # OPLOSSING: Haal ALLE cijfers uit de regel, niet alleen het laatste woord.
                            # Voorbeeld regel: "drm-engine-render:      1157436344 ns"
                            # filter haalt alleen de cijfers eruit: "1157436344"
                            clean_num = "".join(filter(str.isdigit, line))
                            
                            # Als er een getal in de regel staat (en het is niet leeg)
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
    # Let op: dit geeft een percentage van de totale CPU-tijd die aan GPU is besteed
    # Als dit >100% is (meerdere kernen), wordt het afgekapt op 100.
    percentage = (gpu_diff / time_diff) * 100.0 / 2.3
    
    # Debug output (optioneel, kan je later verwijderen)
    # print(f"GPU Usage: {percentage:.2f}% (diff: {gpu_diff}, time: {time_diff})")

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
        # Iowait en meer velden negeren we voor eenvoud; dat is nauwkeurig genoeg.
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


def create_text_program(ctx):
    return ctx.program(vertex_shader=TEXT_VERTEX_SHADER, fragment_shader=TEXT_FRAGMENT_SHADER)


def create_rect_program(ctx):
    return ctx.program(vertex_shader=RECT_VERTEX_SHADER, fragment_shader=RECT_FRAGMENT_SHADER)


def render_rect(ctx, rect_program, x: int, y: int, w: int, h: int, r: float, g: float, b: float, a: float, screen_w: int, screen_h: int):
    """Teken een effen gekleurde rechthoek als twee driehoeken (OpenGL quad)."""
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
    # Twee lijnen vormen een "X": \ en /
    render_text(ctx, text_program, small_font, "\u00d7", (255, 255, 255), quit_x + quit_size // 2, quit_y + quit_size // 2, width, height, "center")
    buttons.append(ButtonRect(quit_x, quit_y, quit_size, quit_size, "quit"))

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
    # Hergebruik de OpenGL-context van het pygame-venster (werkt op Pi 4/5 met de
    # Mesa V3D-driver, die maximaal OpenGL 2.1 levert). Een standalone context faalt
    # op de Pi ("cannot create context") en zou bovendien niet naar het venster renderen.
    # require=210 past bij de #version 120 shaders.
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
    cpu_usage = read_cpu_usage()  # eerste meting (returnt None want geen vorige sample)
    next_temp_read = 0.0
    previous_ticks = pygame.time.get_ticks() / 1000.0
    running = [True]  # list zodat geneste functies kunnen muteren

    # Houd de knoppen bij van de laatste frame voor hit-testing
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
            # Touchscreen: FINGERDOWN (pygame 2.x SDL_FINGER* events)
            # event.x en event.y zijn genormaliseerd 0.0-1.0
            elif event.type == pygame.FINGERDOWN:
                px = int(event.x * width)
                py = int(event.y * height)
                action = hit_test_button(px, py)
                if action:
                    handle_button_action(action)
            # Mouse: MOUSEBUTTONDOWN (voor debuggen op laptop, of USB-muis op Pi)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                px, py = event.pos
                action = hit_test_button(px, py)
                if action:
                    handle_button_action(action)

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


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi bot server + wave visual")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="phi3:mini")
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

    run_wave_ui(shared_state, args.host, args.port, args.model)


if __name__ == "__main__":
    main()


