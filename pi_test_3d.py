import pygame
import moderngl
import numpy as np
import sys
import pyaudio

# --- 1. AUDIO SETUP ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
current_volume = 0.0

# --- 2. PYGAME & MODERNGL SETUP ---
pygame.init()
WIDTH, HEIGHT = 800, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Sensitive Alert Energy Core")
clock = pygame.time.Clock()

ctx = moderngl.create_context()

# --- 3. VERTEX SHADER ---
VERTEX_SHADER = """
#version 330
in vec2 in_vert;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

# --- 4. FRAGMENT SHADER (Met Groen -> Geel -> Rood verloop) ---
FRAGMENT_SHADER = """
#version 330
out vec4 fragColor;

uniform vec2 iResolution;
uniform float iTime;
uniform float iVolume;

// 2D Noise voor plasma-stroom
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2(0.0,0.0)), hash(i + vec2(1.0,0.0)), u.x),
               mix(hash(i + vec2(0.0,1.0)), hash(i + vec2(1.0,1.0)), u.x), u.y);
}

// Gelaagde ruis (FBM) voor de plasma-armen
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
    // Normaliseer coördinaten naar het midden (-1.0 tot 1.0)
    vec2 uv = (gl_FragCoord.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);
    
    float d = length(uv);
    
    // Bereken de energie-ontladingen (reageert direct op iVolume)
    float energy_flares = fbm(uv * 5.0 - vec2(0.0, iTime * 2.0)) * (0.15 + iVolume * 0.6);
    
    // Basisstraal van de reactor-kern (reageert direct op volume)
    float core_radius = 0.22 + (iVolume * 0.12);
    
    // Elektrische pulserende schil rond de kern
    float shell = abs(d - core_radius - energy_flares);
    
    // Gevoelige plasma gloed formule
    float core_glow = 0.005 / (shell + 0.002);
    
    // De felle fusiereactor-kern in het exacte centrum
    float white_hot_center = exp(-d * (5.5 - iVolume * 2.5));
    
    // Atmosferische schokgolf-gloed die zacht naar buiten reikt
    float ambient_glow = exp(-d * 2.5) * (0.2 + iVolume * 0.8);
    
    // --- DYNAMISCHE KLEUR BEREKENING ---
    vec3 color_green = vec3(0.0, 1.0, 0.45); // Normaal / Rustig
    vec3 color_yellow = vec3(1.0, 0.9, 0.0); // Medium / Waarschuwing
    vec3 color_red = vec3(1.0, 0.1, 0.0);    // Extreem / Gevaar
    
    vec3 energy_color;
    
    // Maak een vloeiende overgang op basis van het volume
    if (iVolume < 0.4) {
        // Schaal volume van 0.0-0.4 naar een factor van 0.0-1.0 tussen groen en geel
        float factor = iVolume / 0.4;
        energy_color = mix(color_green, color_yellow, factor);
    } else {
        // Schaal volume van 0.4-1.0 naar een factor van 0.0-1.0 tussen geel en rood
        float factor = clamp((iVolume - 0.4) / 0.6, 0.0, 1.0);
        energy_color = mix(color_yellow, color_red, factor);
    }
    
    // Combineer alle energetische lichtberekeningen
    vec3 rgb = energy_color * (core_glow * 1.5 + ambient_glow);
    
    // Injecteer de superhete witte kern in het centrum
    rgb += vec3(0.8, 0.95, 1.0) * white_hot_center * 1.8;
    
    // HDR Toonmapping voor intense contrasten
    rgb = pow(rgb, vec3(0.75));
    
    fragColor = vec4(rgb, 1.0);
}
"""

# --- 5. OPENGL BUFFER OBJECTEN AANMAKEN ---
vertices = np.array([
    -1.0, -1.0,
    1.0, -1.0,
    -1.0,  1.0,
    -1.0,  1.0,
    1.0, -1.0,
    1.0,  1.0,
], dtype='f4')

program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
vbo = ctx.buffer(vertices.tobytes())
vao = ctx.vertex_array(program, [(vbo, '2f', 'in_vert')])

# --- 6. HOOFD LOOP ---
start_time = pygame.time.get_ticks()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Live audio input uitlezen en verwerken naar volume (0.0 - 1.0)
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
        
        # Microfoongevoeligheid
        target_volume = min(1.0, rms / 2200.0) 
        # Originele gevoelige smoothing behouden
        current_volume += (target_volume - current_volume) * 0.25
    except Exception:
        current_volume = 0.0

    current_time = (pygame.time.get_ticks() - start_time) / 1000.0

    # Data naar GPU sturen
    program['iResolution'].value = (WIDTH, HEIGHT)
    program['iTime'].value = current_time
    program['iVolume'].value = current_volume

    ctx.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLES)
    
    pygame.display.flip()
    clock.tick(70)

stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()
sys.exit()
